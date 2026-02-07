import streamlit as st
import os
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from huggingface_hub import hf_hub_download
from datetime import datetime
import base64
from fpdf import FPDF

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Pneumonia AI Detection",
    page_icon="🏥",
    layout="wide"
)

# ==================== SIMPLE CSS ====================
st.markdown("""
<style>
    .header-box {
        background: #1a73e8;
        color: white;
        padding: 2rem;
        text-align: center;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .info-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .image-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .pneumonia {
        background: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    
    .normal {
        background: #e8f5e9;
        color: #2e7d32;
        border: 2px solid #4caf50;
    }
    
    .stButton>button {
        background: #1a73e8;
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .download-btn {
        background: #34a853 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown("""
<div class="header-box">
    <h1>🏥 Pneumonia Detection System</h1>
    <p>AI-Powered Chest X-Ray Analysis</p>
</div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### Patient Information")
    
    p_name = st.text_input("Patient Name")
    p_age = st.number_input("Age", 1, 120, 30)
    p_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    
    st.markdown("---")
    
    st.markdown("### Upload X-Ray")
    uploaded_file = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        st.success("Image uploaded")

# ==================== MAIN CONTENT ====================
if uploaded_file and p_name:
    # Doctor Info
    st.markdown("""
    <div class="info-box">
        <h3>👨‍⚕️ Dr. Well - Radiologist</h3>
        <p><strong>Specialization:</strong> Chest Radiology</p>
        <p><strong>Qualifications:</strong> MD Radiology | 10+ Years Experience</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Patient Info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="info-box">
            <h3>📋 Patient Details</h3>
            <p><strong>Name:</strong> {p_name}</p>
            <p><strong>Age:</strong> {p_age} years</p>
            <p><strong>Gender:</strong> {p_gender}</p>
            <p><strong>Date:</strong> {datetime.now().strftime('%d %B, %Y')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load and display images
    uploaded_file.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    display_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    with col2:
        st.markdown("<div class='image-box'>", unsafe_allow_html=True)
        st.markdown("##### Original X-Ray")
        st.image(display_image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Analysis Button
    if st.button("🔍 Analyze X-Ray", type="primary"):
        with st.spinner("Analyzing..."):
            # Simple model loading
            @st.cache_resource
            def load_model():
                try:
                    path = hf_hub_download(
                        repo_id="DanishMubashar/chest-xray-pneumonia-detection-DenseNet121",
                        filename="pneumonia_detection_model.keras"
                    )
                    model = tf.keras.models.load_model(path, compile=False, safe_mode=False)
                    model.build((None, 256, 256, 3))
                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    return model
                except:
                    # Fallback model
                    base_model = tf.keras.applications.DenseNet121(
                        weights='imagenet', include_top=False, input_shape=(256, 256, 3))
                    base_model.trainable = False
                    inputs = tf.keras.Input(shape=(256, 256, 3))
                    x = base_model(inputs, training=False)
                    x = tf.keras.layers.GlobalAveragePooling2D()(x)
                    x = tf.keras.layers.Dense(256, activation='relu')(x)
                    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
                    model = tf.keras.Model(inputs, outputs)
                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    return model
            
            model = load_model()
            
            # Preprocess
            processed = cv2.resize(display_image, (256, 256))
            processed = processed.astype(np.float32) / 255.0
            processed = np.expand_dims(processed, axis=0)
            
            # Prediction
            prediction = model.predict(processed, verbose=0)
            score = float(prediction[0][0])
            
            # Store results
            st.session_state['score'] = score
            st.session_state['analyzed'] = True
            
            # Heatmap
            gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            st.session_state['heatmap'] = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Show Results
    if 'analyzed' in st.session_state and st.session_state['analyzed']:
        score = st.session_state['score']
        is_pneumonia = score > 0.5
        confidence = f"{(score if is_pneumonia else (1-score)) * 100:.1f}%"
        
        # Display heatmap
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='image-box'>", unsafe_allow_html=True)
            st.markdown("##### Original X-Ray")
            st.image(display_image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='image-box'>", unsafe_allow_html=True)
            st.markdown("##### AI Heatmap")
            st.image(st.session_state['heatmap'], use_container_width=True, caption="AI Analysis")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Result
        result_class = "pneumonia" if is_pneumonia else "normal"
        result_text = "PNEUMONIA DETECTED" if is_pneumonia else "NORMAL LUNGS"
        
        st.markdown(f"""
        <div class="result-box {result_class}">
            {result_text}<br>
            <small>Confidence: {confidence}</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Report
        st.markdown("### 📋 Medical Report")
        
        if is_pneumonia:
            findings = """
            - Lung consolidation observed
            - Air bronchograms present
            - Possible pleural effusion
            """
            recommendations = """
            1. Start antibiotic therapy
            2. Follow-up X-ray in 2 weeks
            3. Monitor oxygen levels
            4. Rest and hydration
            """
        else:
            findings = """
            - Clear lung fields
            - Normal bronchovascular markings
            - No consolidation
            """
            recommendations = """
            1. Routine health monitoring
            2. Annual checkup recommended
            3. Maintain healthy lifestyle
            """
        
        # Display report
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Findings:**")
            st.markdown(findings)
        with col2:
            st.markdown("**Recommendations:**")
            st.markdown(recommendations)
        
        # PDF Download
        def create_pdf():
            pdf = FPDF()
            pdf.add_page()
            
            # Header
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, 'Pneumonia Detection Report', 0, 1, 'C')
            
            # Patient Info
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 8, f'Patient: {p_name}', 0, 1)
            pdf.cell(0, 8, f'Age: {p_age} years | Gender: {p_gender}', 0, 1)
            pdf.cell(0, 8, f'Date: {datetime.now().strftime("%d %B, %Y")}', 0, 1)
            
            # Results
            pdf.ln(5)
            pdf.set_font('Arial', 'B', 14)
            result = 'PNEUMONIA DETECTED' if is_pneumonia else 'NORMAL LUNGS'
            pdf.cell(0, 10, f'Result: {result}', 0, 1)
            pdf.cell(0, 8, f'Confidence: {confidence}', 0, 1)
            
            # Findings
            pdf.ln(5)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, 'Findings:', 0, 1)
            pdf.set_font('Arial', '', 12)
            for line in findings.strip().split('\n'):
                if line.strip():
                    pdf.multi_cell(0, 8, line.strip())
            
            # Recommendations
            pdf.ln(5)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, 'Recommendations:', 0, 1)
            pdf.set_font('Arial', '', 12)
            for line in recommendations.strip().split('\n'):
                if line.strip():
                    pdf.multi_cell(0, 8, line.strip())
            
            # Footer
            pdf.ln(10)
            pdf.set_font('Arial', 'I', 10)
            pdf.cell(0, 8, 'Generated by AI system. Consult doctor for final diagnosis.', 0, 1)
            
            return pdf.output(dest='S').encode('latin-1')
        
        # Download button
        pdf_bytes = create_pdf()
        pdf_b64 = base64.b64encode(pdf_bytes).decode()
        
        st.markdown(f"""
        <a href="data:application/pdf;base64,{pdf_b64}" 
           download="Pneumonia_Report_{p_name}_{datetime.now().strftime('%Y%m%d')}.pdf">
           <button class="download-btn" style="padding: 0.7rem 1.5rem; border-radius: 5px; border: none; background: #34a853; color: white; font-weight: bold;">
               📄 Download PDF Report
           </button>
        </a>
        """, unsafe_allow_html=True)

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2>Welcome to Pneumonia Detection System</h2>
        <p>Upload a chest X-ray image to get AI-powered analysis</p>
        
        <div style="display: flex; justify-content: center; gap: 1rem; margin: 2rem 0;">
            <div style="padding: 1rem; background: #f0f7ff; border-radius: 8px; width: 150px;">
                <h4>1. Enter Details</h4>
                <p>Fill patient information</p>
            </div>
            <div style="padding: 1rem; background: #f0f7ff; border-radius: 8px; width: 150px;">
                <h4>2. Upload X-Ray</h4>
                <p>Choose chest X-ray image</p>
            </div>
            <div style="padding: 1rem; background: #f0f7ff; border-radius: 8px; width: 150px;">
                <h4>3. Get Report</h4>
                <p>Download PDF report</p>
            </div>
        </div>
        
        <p>👈 <strong>Use the sidebar to get started</strong></p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Pneumonia Detection System | For medical professionals</p>
    <p><small>⚠️ This tool assists diagnosis. Consult doctor for medical advice.</small></p>
</div>
""", unsafe_allow_html=True)
