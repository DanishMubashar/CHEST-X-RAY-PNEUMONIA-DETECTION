import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.cm as cm
from PIL import Image

from huggingface_hub import hf_hub_download
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# ====================================
# CONFIG
# ====================================
HF_REPO_ID = "DanishMubashar/chest-xray-pneumonia-detection-DenseNet121"
HF_MODEL_FILE = "pneumonia_detection_model.keras"
LAST_CONV_LAYER = "conv5_block16_concat"

# ====================================
# LOAD MODEL FROM HUGGING FACE
# ====================================
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILE
    )
    return tf.keras.models.load_model(model_path)

model = load_model()

# ====================================
# IMAGE PREPROCESS
# ====================================
def preprocess_image(img):
    img = np.array(img)
    img = cv2.resize(img, (256, 256))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# ====================================
# GRAD-CAM
# ====================================
def make_gradcam_heatmap(img_array, model, last_conv_layer):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    return heatmap.numpy()

# ====================================
# LANGCHAIN REPORT
# ====================================
def generate_report(name, age, gender, result, confidence):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    prompt = PromptTemplate(
        input_variables=["name", "age", "gender", "result", "confidence"],
        template="""
You are Dr. Well, a pneumonia specialist.

Create a professional medical report.

Patient Name: {name}
Age: {age}
Gender: {gender}
Diagnosis Result: {result}
Confidence Level: {confidence}%

Include:
1. Diagnosis summary
2. Patient care instructions (according to age & gender)
3. Precautions
4. When to consult a doctor

End with a clear medical disclaimer.
"""
    )

    response = llm.invoke(
        prompt.format(
            name=name,
            age=age,
            gender=gender,
            result=result,
            confidence=confidence
        )
    )

    return response.content

# ====================================
# STREAMLIT UI
# ====================================
st.set_page_config("Pneumonia Detection | Dr. Well", layout="wide")

st.title("🫁 Chest X-Ray Pneumonia Detection System")
st.subheader("👨‍⚕️ Dr. Well — Pneumonia Specialist")

with st.form("patient_form"):
    name = st.text_input("Patient Name")
    age = st.number_input("Age", 1, 120)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    uploaded_file = st.file_uploader(
        "Upload Chest X-Ray Image (256×256)",
        type=["jpg", "png", "jpeg"]
    )
    submit = st.form_submit_button("Analyze X-Ray")

if submit and uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = preprocess_image(image)

    pred = model.predict(img_array)[0][0]
    is_pneumonia = pred > 0.5
    confidence = round(pred * 100 if is_pneumonia else (1 - pred) * 100, 2)

    col1, col2 = st.columns(2)
    col1.image(image, caption="Original Chest X-Ray", use_column_width=True)

    if is_pneumonia:
        heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER)
        heatmap = cv2.resize(heatmap, image.size)
        heatmap = cm.jet(heatmap)[..., :3]
        overlay = (heatmap * 255 * 0.4 + np.array(image)).astype(np.uint8)
        col2.image(overlay, caption="Grad-CAM Heatmap", use_column_width=True)
    else:
        col2.image(image, caption="No Infection Detected", use_column_width=True)

    result_text = "PNEUMONIA DETECTED" if is_pneumonia else "NORMAL"
    st.success(f"Diagnosis: {result_text} ({confidence}%)")

    report = generate_report(name, age, gender, result_text, confidence)

    st.markdown("## 📄 Medical Report")
    st.write(report)
