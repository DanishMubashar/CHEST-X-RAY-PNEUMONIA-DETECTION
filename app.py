import os
import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.cm as cm
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from huggingface_hub import hf_hub_download
from fpdf import FPDF
from datetime import datetime

# --- 1. Model Loading ---
REPO_ID = "DanishMubashar/chest-xray-pneumonia-detection-DenseNet121"
FILENAME = "pneumonia_detection_model.keras"
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
model = tf.keras.models.load_model(model_path)

# --- 2. Grad-CAM Logic ---
def get_gradcam(img_array, model, last_conv_layer_name="relu"):
    base_m = model.layers[0]
    grad_model = tf.keras.models.Model(
        [base_m.inputs], [base_m.get_layer(last_conv_layer_name).output, base_m.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, base_output = grad_model(img_array)
        x = base_output
        for layer in model.layers[1:]:
            x = layer(x)
        class_channel = x[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# --- 3. PDF Generation ---
def generate_pdf_report(name, age, gender, diagnosis, confidence, report_text, orig_img, heat_img):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "PneumoCare AI - Diagnostic Radiology Report", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
    pdf.line(10, 30, 200, 30)
    
    # Patient Info
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. Patient Information", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(60, 8, f"Name: {name}")
    pdf.cell(60, 8, f"Age: {age}")
    pdf.cell(60, 8, f"Gender: {gender}", ln=True)
    
    # Diagnosis
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. Clinical Observation (AI Analysis)", ln=True)
    pdf.set_font("Arial", size=11)
    res_text = f"Primary Prediction: {diagnosis} | Confidence Score: {confidence}%"
    pdf.cell(0, 8, res_text, ln=True)
    
    # Images
    orig_img.save("temp_orig.png")
    heat_img.save("temp_heat.png")
    pdf.image("temp_orig.png", x=10, y=85, w=90)
    pdf.image("temp_heat.png", x=105, y=85, w=90)
    
    # Report Text (Gemini)
    pdf.set_y(155)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "3. Detailed Medical Analysis & Advice", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, report_text)
    
    # Footer
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Signed by: Dr. Well", ln=True, align='R')
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "Disclaimer: This is an AI-generated report. Please correlate with clinical findings.", align='C')
    
    pdf_file = f"Report_{name.replace(' ','_')}.pdf"
    pdf.output(pdf_file)
    return pdf_file

# --- 4. Main Function ---
def process_all(input_img, name, age, gender):
    # Preprocess
    img_array = np.expand_dims(cv2.resize(input_img, (224, 224)).astype(np.float32) / 255.0, axis=0)
    
    # Prediction
    score = float(model.predict(img_array)[0][0])
    is_p = score > 0.5
    diagnosis = "PNEUMONIA" if is_p else "NORMAL"
    conf = round(score * 100, 2) if is_p else round((1-score)*100, 2)
    
    # Heatmap
    heatmap = get_gradcam(img_array, model)
    heatmap = cv2.resize(heatmap, (input_img.shape[1], input_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")(np.arange(256))[:, :3]
    jet_heatmap = (jet[heatmap] * 255).astype(np.uint8)
    superimposed = cv2.addWeighted(input_img, 0.6, jet_heatmap, 0.4, 0)
    heat_pil = Image.fromarray(superimposed)
    orig_pil = Image.fromarray(input_img)
    
    # Gemini Report
    api_key = os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
    template = """
    Role: Senior Pulmonologist (Dr. Well). 
    Context: Patient {name} ({gender}, Age {age}) has {diagnosis} with {confidence}% confidence.
    Task: Write a professional English radiology report.
    Include: 1. Findings summary, 2. Severity based on age {age}, 3. Specific medical advice, 4. Sample medication suggestions.
    Note: Be professional, empathetic, and direct.
    """
    prompt = PromptTemplate(template=template, input_variables=["name", "age", "gender", "diagnosis", "confidence"])
    report_content = llm.invoke(prompt.format(name=name, age=age, gender=gender, diagnosis=diagnosis, confidence=conf)).content
    
    # PDF
    pdf_link = generate_pdf_report(name, age, gender, diagnosis, conf, report_content, orig_pil, heat_pil)
    
    return heat_pil, diagnosis, f"{conf}%", report_content, pdf_link

# --- 5. Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🫁 PneumoCare AI: Professional Diagnostic System")
    gr.Markdown("Analyze Chest X-Rays and generate professional reports signed by **Dr. Well**.")
    
    with gr.Row():
        with gr.Column():
            name_in = gr.Textbox(label="Patient Name")
            age_in = gr.Number(label="Age")
            gender_in = gr.Radio(["Male", "Female", "Other"], label="Gender")
            img_in = gr.Image(label="Upload X-Ray")
            btn = gr.Button("Analyze & Generate Report", variant="primary")
            
        with gr.Column():
            heat_out = gr.Image(label="AI Infection Map (Grad-CAM)")
            diag_out = gr.Textbox(label="Diagnosis Result")
            conf_out = gr.Textbox(label="Confidence")
            report_out = gr.Markdown(label="Full Medical Analysis")
            pdf_out = gr.File(label="Download Official PDF Report")

    btn.click(process_all, [img_in, name_in, age_in, gender_in], [heat_out, diag_out, conf_out, report_out, pdf_out])

demo.launch()
