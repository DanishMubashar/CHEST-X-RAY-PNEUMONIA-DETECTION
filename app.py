import streamlit as st
import os
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.cm as cm
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate 
from huggingface_hub import hf_hub_download
from fpdf import FPDF
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="PneumoCare AI", page_icon="🫁", layout="wide")

# --- 1. Model Loading ---
REPO_ID = "DanishMubashar/chest-xray-pneumonia-detection-DenseNet121"
FILENAME = "pneumonia_detection_model.keras"
IMG_SIZE = (256, 256)

@st.cache_resource
def load_model():
    try:
        path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        # compile=False and safe_mode=False to bypass Keras 3 version conflicts
        model = tf.keras.models.load_model(path, compile=False, safe_mode=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- 2. Grad-CAM (Heatmap) Logic ---
def get_gradcam(img_array, model):
    try:
        last_conv_layer_name = "relu"
        target_source = model.layers[0] if hasattr(model.layers[0], 'get_layer') else model
        grad_model = tf.keras.models.Model(
            [target_source.inputs], [target_source.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()
    except:
        return np.zeros((7, 7))

# --- 3. PDF Report Generation ---
def generate_pdf(name, age, gender, diag, conf, report, orig_img, heat_img):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "PneumoCare AI Radiology Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Patient: {name} | Age: {age} | Gender: {gender}", ln=True)
    pdf.cell(0, 10, f"AI Diagnosis: {diag} ({conf})", ln=True)
    
    orig_img.save("temp_orig.png")
    heat_img.save("temp_heat.png")
    pdf.image("temp_orig.png", x=10, y=50, w=90)
    pdf.image("temp_heat.png", x=105, y=50, w=90)
    
    pdf.set_y(150)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Clinical Analysis:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, report)
    
    pdf_output = f"Report_{name}.pdf"
    pdf.output(pdf_output)
    return pdf_output

# --- 4. Streamlit UI ---
st.title("🫁 PneumoCare AI: Professional X-Ray Analysis")
st.markdown("---")

with st.sidebar:
    st.header("Patient Details")
    name = st.text_input("Full Name")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and name:
    col1, col2 = st.columns(2)
    
    # Process Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=uint8)
    image = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with col1:
        st.image(img_rgb, caption="Uploaded X-Ray", use_container_width=True)

    if st.button("Generate Diagnosis"):
        with st.spinner("AI is analyzing the X-ray..."):
            # Preprocessing 256x256
            img_res = cv2.resize(img_rgb, (256, 256))
            img_array = np.array(img_res) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            preds = model.predict(img_array)
            score = float(preds[0][0])
            is_p = score > 0.5
            diagnosis = "PNEUMONIA" if is_p else "NORMAL"
            confidence = f"{round(score*100 if is_p else (1-score)*100, 2)}%"
            
            # Heatmap
            heatmap = get_gradcam(img_array, model)
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            jet = cm.get_cmap("jet")(np.arange(256))[:, :3]
            jet_heatmap = (jet[heatmap] * 255).astype(np.uint8)
            superimposed = cv2.addWeighted(image, 0.6, jet_heatmap, 0.4, 0)
            super_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.image(super_rgb, caption="Diagnostic Heatmap", use_container_width=True)
                st.metric("Diagnosis", diagnosis)
                st.metric("Confidence", confidence)

            # LangChain Report
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
                report_content = llm.invoke(f"Write a medical report for {name}, {age} years old {gender}. AI detected {diagnosis} with {confidence} confidence.").content
                st.markdown("### 👨‍⚕️ Dr. Well's Analysis")
                st.write(report_content)
                
                # PDF Download
                pdf_file = generate_pdf(name, age, gender, diagnosis, confidence, report_content, Image.fromarray(img_rgb), Image.fromarray(super_rgb))
                with open(pdf_file, "rb") as f:
                    st.download_button("📥 Download Official Report", f, file_name=pdf_file)
            else:
                st.warning("GEMINI_API_KEY not found in Secrets.")
else:
    st.info("Please enter patient details and upload an X-ray to start.")
