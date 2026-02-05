import streamlit as st
import os
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.cm as cm
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from huggingface_hub import hf_hub_download
from fpdf import FPDF
from datetime import datetime

# Page configuration
st.set_page_config(page_title="PneumoCare AI", page_icon="🫁", layout="wide")

# --- 1. Robust Model Loading (Fix for ndim=1 error) ---
REPO_ID = "DanishMubashar/chest-xray-pneumonia-detection-DenseNet121"
FILENAME = "pneumonia_detection_model.keras"
IMG_SIZE = (256, 256)

@st.cache_resource
def load_model_fixed():
    try:
        path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        # Keras 3 compatibility fix: safe_mode=False
        # Agar dense layer mismatch ho, toh aksar compile=False zaroori hota hai
        model = tf.keras.models.load_model(path, compile=False, safe_mode=False)
        
        # Ek dummy prediction run karte hain taake layers initialize ho jayein
        dummy_input = np.zeros((1, 256, 256, 3))
        model(dummy_input) 
        
        return model
    except Exception as e:
        st.error(f"Technical Error: {e}")
        return None

model = load_model_fixed()

# --- 2. Heatmap Logic ---
def get_gradcam(img_array, model):
    try:
        # DenseNet121 ki last conv layer dhundna
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

# --- 3. UI and Process ---
st.title("🫁 PneumoCare AI: Diagnostic Dashboard")

with st.sidebar:
    st.header("Patient Info")
    p_name = st.text_input("Name")
    p_age = st.number_input("Age", 1, 120, 30)
    p_gen = st.selectbox("Gender", ["Male", "Female", "Other"])
    uploaded_file = st.file_uploader("Upload X-Ray", type=["jpg", "png", "jpeg"])

if uploaded_file and p_name:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb, caption="Original X-Ray", use_container_width=True)

    if st.button("Analyze X-Ray"):
        if model is not None:
            with st.spinner("Model analyzing..."):
                # FIX: Preprocessing with correct float conversion
                img_res = cv2.resize(img_rgb, IMG_SIZE)
                img_array = np.array(img_res, dtype=np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
                
                # Model Prediction
                preds = model.predict(img_array, verbose=0)
                score = float(preds[0][0])
                
                is_p = score > 0.5
                diagnosis = "PNEUMONIA" if is_p else "NORMAL"
                confidence = f"{round(score*100 if is_p else (1-score)*100, 2)}%"
                
                # Heatmap
                heatmap = get_gradcam(img_array, model)
                heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
                heatmap = np.uint8(255 * heatmap)
                jet_heatmap = (cm.get_cmap("jet")(np.arange(256))[:, :3][heatmap] * 255).astype(np.uint8)
                superimposed = cv2.addWeighted(image, 0.6, jet_heatmap, 0.4, 0)
                
                with col2:
                    st.image(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB), caption="AI Visualization", use_container_width=True)
                    st.metric("Result", diagnosis)
                    st.metric("Confidence", confidence)
                
                # Report
                api_key = os.getenv("GEMINI_API_KEY")
                if api_key:
                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
                    report = llm.invoke(f"Write a short medical report for {p_name}, {p_age}y, {p_gen}. Diagnosis: {diagnosis} ({confidence}).").content
                    st.info(report)
                else:
                    st.warning("API Key missing.")
        else:
            st.error("Model is not initialized.")
