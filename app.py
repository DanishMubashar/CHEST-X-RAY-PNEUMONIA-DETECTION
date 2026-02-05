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

# Page configuration
st.set_page_config(page_title="PneumoCare AI", page_icon="🫁", layout="wide")

# Custom CSS for professional look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 1. Model Loading ---
REPO_ID = "DanishMubashar/chest-xray-pneumonia-detection-DenseNet121"
FILENAME = "pneumonia_detection_model.keras"
IMG_SIZE = (256, 256)

@st.cache_resource
def load_model():
    try:
        path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        # compile=False and safe_mode=False for Keras 3 compatibility
        model = tf.keras.models.load_model(path, compile=False, safe_mode=False)
        return model
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None

model = load_model()

# --- 2. Grad-CAM Logic ---
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

# --- 3. PDF Function ---
def generate_pdf(name, age, gender, diag, conf, report, orig_img, heat_img):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "PneumoCare AI - Diagnostic Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Patient Name: {name}", ln=True)
    pdf.cell(0, 10, f"Age: {age} | Gender: {gender}", ln=True)
    pdf.cell(0, 10, f"AI Diagnosis: {diag} ({conf})", ln=True)
    
    orig_img.save("temp_orig.png")
    heat_img.save("temp_heat.png")
    pdf.image("temp_orig.png", x=10, y=60, w=90)
    pdf.image("temp_heat.png", x=105, y=60, w=90)
    
    pdf.set_y(160)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Radiology Analysis & Advice:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, report)
    
    file_path = f"Report_{name.replace(' ','_')}.pdf"
    pdf.output(file_path)
    return file_path

# --- 4. Main UI ---
st.title("🫁 PneumoCare AI: Chest X-Ray Diagnosis")
st.write("Upload a Chest X-ray image for AI-powered Pneumonia detection and automated reporting.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Patient Details")
    p_name = st.text_input("Name")
    p_age = st.number_input("Age", 1, 120, 30)
    p_gen = st.selectbox("Gender", ["Male", "Female", "Other"])
    uploaded_file = st.file_uploader("Upload X-Ray", type=["jpg", "png", "jpeg"])

if uploaded_file and p_name:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with col2:
        tab1, tab2 = st.tabs(["Analysis", "Detailed Report"])
        
        with tab1:
            if st.button("Run AI Diagnosis"):
                with st.spinner("Processing..."):
                    # Preprocess for 256x256
                    img_res = cv2.resize(img_rgb, (256, 256))
                    img_array = np.array(img_res) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Predict
                    preds = model.predict(img_array)
                    score = float(preds[0][0])
                    is_p = score > 0.5
                    diagnosis = "PNEUMONIA" if is_p else "NORMAL"
                    confidence = f"{round(score*100 if is_p else (1-score)*100, 2)}%"
                    
                    # Grad-CAM Heatmap
                    heatmap = get_gradcam(img_array, model)
                    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
                    heatmap = np.uint8(255 * heatmap)
                    jet = cm.get_cmap("jet")(np.arange(256))[:, :3]
                    jet_heatmap = (jet[heatmap] * 255).astype(np.uint8)
                    superimposed = cv2.addWeighted(image, 0.6, jet_heatmap, 0.4, 0)
                    super_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
                    
                    # Display Results
                    res_col1, res_col2 = st.columns(2)
                    res_col1.image(img_rgb, caption="Original X-Ray", use_container_width=True)
                    res_col2.image(super_rgb, caption="AI Heatmap", use_container_width=True)
                    
                    st.divider()
                    m1, m2 = st.columns(2)
                    m1.metric("Diagnosis", diagnosis, delta="Positive" if is_p else "Negative", delta_color="inverse")
                    m2.metric("Confidence", confidence)
                    
                    # Gemini Report
                    api_key = os.getenv("GEMINI_API_KEY")
                    if api_key:
                        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
                        prompt = f"Write a professional medical report for {p_name}, {p_age} year old {p_gen}. Finding: {diagnosis} ({confidence}). Include symptoms to watch and next steps."
                        report_text = llm.invoke(prompt).content
                        
                        with tab2:
                            st.markdown("### 👨‍⚕️ Dr. Well's AI Analysis")
                            st.write(report_text)
                            
                            pdf_file = generate_pdf(p_name, p_age, p_gen, diagnosis, confidence, report_text, Image.fromarray(img_rgb), Image.fromarray(super_rgb))
                            with open(pdf_file, "rb") as f:
                                st.download_button("📥 Download PDF Report", f, file_name=pdf_file)
                    else:
                        st.warning("GEMINI_API_KEY missing in Secrets!")
else:
    with col2:
        st.info("Patient details fill karein aur X-ray upload karein analysis shuru karne ke liye.")
