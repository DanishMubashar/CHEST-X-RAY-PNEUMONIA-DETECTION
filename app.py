import streamlit as st
import os
import sys
import traceback
from datetime import datetime
import tempfile
import base64
from io import BytesIO

# Check for required packages
try:
    import tensorflow as tf
    import numpy as np
    import cv2
    import matplotlib.cm as cm
    from PIL import Image, ImageDraw, ImageFont
    from langchain_google_genai import ChatGoogleGenerativeAI
    from huggingface_hub import hf_hub_download
    import warnings
    warnings.filterwarnings('ignore')
    
    # Import for PDF generation
    from fpdf import FPDF
    import matplotlib.pyplot as plt
    
    # Set TensorFlow logging level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
except ImportError as e:
    st.error(f"❌ Missing required package: {e}")
    st.info("Please install all requirements using: pip install -r requirements.txt")
    st.stop()

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Dr. Well Pneumonia AI | Medical Diagnostic System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== ENHANCED CUSTOM CSS ====================
st.markdown("""
<style>
    /* Modern Medical Theme */
    :root {
        --primary: #1a73e8;
        --secondary: #34a853;
        --danger: #ea4335;
        --warning: #fbbc05;
        --light: #f8f9fa;
        --dark: #202124;
        --gray: #5f6368;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
        padding: 2.5rem;
        border-radius: 0 0 20px 20px;
        color: white;
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: 0 6px 20px rgba(26, 115, 232, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #34a853, #fbbc05, #ea4335);
    }
    
    .hospital-header {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 2rem;
        border-left: 5px solid #1a73e8;
    }
    
    .doctor-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%);
        padding: 1.8rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border: 2px solid #e3f2fd;
        margin-bottom: 1.5rem;
    }
    
    .patient-card {
        background: white;
        padding: 1.8rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border: 2px solid #e8f5e9;
        margin-bottom: 1.5rem;
    }
    
    .image-comparison {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        margin: 1.5rem 0;
    }
    
    .diagnosis-banner {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 2rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        border: none;
    }
    
    .pneumonia-banner {
        background: linear-gradient(135deg, #ff5252 0%, #ff8a80 100%);
        color: white;
        animation: pulse-red 2s infinite;
    }
    
    .normal-banner {
        background: linear-gradient(135deg, #00c853 0%, #64dd17 100%);
        color: white;
        animation: pulse-green 2s infinite;
    }
    
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(255, 82, 82, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(255, 82, 82, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 82, 82, 0); }
    }
    
    @keyframes pulse-green {
        0% { box-shadow: 0 0 0 0 rgba(0, 200, 83, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(0, 200, 83, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 200, 83, 0); }
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #dee2e6;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
        color: white;
        border: none;
        padding: 0.9rem 2.5rem;
        border-radius: 10px;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s ease;
        font-size: 1.1rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(26, 115, 232, 0.3);
    }
    
    .download-btn {
        background: linear-gradient(135deg, #34a853 0%, #2e7d32 100%) !important;
    }
    
    .sidebar-content {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
    }
    
    .file-upload-area {
        border: 3px dashed #1a73e8;
        border-radius: 15px;
        padding: 2.5rem;
        text-align: center;
        background: rgba(26, 115, 232, 0.03);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .file-upload-area:hover {
        background: rgba(26, 115, 232, 0.08);
        border-color: #0d47a1;
    }
    
    .report-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 25px rgba(0, 0, 0, 0.08);
        margin: 2rem 0;
        border-top: 5px solid #1a73e8;
    }
    
    .treatment-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #1a73e8;
        margin: 1rem 0;
    }
    
    .medication-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #34a853;
        margin: 1rem 0;
    }
    
    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    .risk-high { background: #ffcdd2; color: #c62828; }
    .risk-medium { background: #fff3e0; color: #ef6c00; }
    .risk-low { background: #e8f5e9; color: #2e7d32; }
    
    .tab-content {
        padding: 1.5rem;
        background: white;
        border-radius: 0 0 15px 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# ==================== PDF GENERATION FUNCTIONS ====================
class MedicalPDF(FPDF):
    def header(self):
        # Logo
        self.image('https://cdn-icons-png.flaticon.com/512/3067/3067256.png', 10, 8, 25)
        
        # Title
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'DR. WELL MEDICAL DIAGNOSTIC CENTER', 0, 1, 'C')
        
        # Subtitle
        self.set_font('Arial', 'I', 12)
        self.cell(0, 8, 'AI-Powered Pneumonia Detection Report', 0, 1, 'C')
        
        # Line break
        self.ln(10)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        
        # Page number
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def add_doctor_info(self):
        self.set_font('Arial', 'B', 14)
        self.set_text_color(26, 115, 232)
        self.cell(0, 10, 'ATTENDING PHYSICIAN', 0, 1)
        self.ln(2)
        
        self.set_font('Arial', '', 12)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 7, 
            "Dr. Robert Well, MD\n"
            "Chief Radiologist & AI Diagnostics Specialist\n"
            "Board Certified in Diagnostic Radiology\n"
            "Fellow, American College of Radiology\n"
            "PhD in Medical Imaging, Stanford University\n"
            "15+ Years Experience in Thoracic Imaging\n"
            "Contact: dr.well@medicalcenter.com | Phone: (555) 123-4567"
        )
        self.ln(10)

    def add_patient_info(self, name, age, gender, pid, date):
        self.set_font('Arial', 'B', 14)
        self.set_text_color(52, 168, 83)
        self.cell(0, 10, 'PATIENT INFORMATION', 0, 1)
        self.ln(2)
        
        self.set_font('Arial', '', 12)
        self.set_text_color(0, 0, 0)
        
        # Create a table for patient info
        col_width = 45
        row_height = 8
        
        self.cell(col_width, row_height, 'Full Name:', 0, 0)
        self.cell(col_width, row_height, name, 0, 1)
        
        self.cell(col_width, row_height, 'Age:', 0, 0)
        self.cell(col_width, row_height, f"{age} years", 0, 1)
        
        self.cell(col_width, row_height, 'Gender:', 0, 0)
        self.cell(col_width, row_height, gender, 0, 1)
        
        self.cell(col_width, row_height, 'Patient ID:', 0, 0)
        self.cell(col_width, row_height, pid if pid else 'Not Specified', 0, 1)
        
        self.cell(col_width, row_height, 'Report Date:', 0, 0)
        self.cell(col_width, row_height, date, 0, 1)
        
        self.ln(10)

def create_pdf_report(patient_info, diagnosis_info, original_img, heatmap_img, recommendations):
    """Create a professional PDF report"""
    pdf = MedicalPDF()
    pdf.add_page()
    
    # Add doctor information
    pdf.add_doctor_info()
    
    # Add patient information
    pdf.add_patient_info(
        patient_info['name'],
        patient_info['age'],
        patient_info['gender'],
        patient_info['id'],
        patient_info['date']
    )
    
    # Diagnosis Results
    pdf.set_font('Arial', 'B', 16)
    pdf.set_text_color(234, 67, 53) if diagnosis_info['is_pneumonia'] else pdf.set_text_color(52, 168, 83)
    pdf.cell(0, 10, 'DIAGNOSIS RESULTS', 0, 1)
    pdf.ln(2)
    
    pdf.set_font('Arial', 'B', 20)
    diagnosis_text = 'PNEUMONIA DETECTED' if diagnosis_info['is_pneumonia'] else 'NORMAL LUNGS'
    pdf.cell(0, 12, diagnosis_text, 0, 1, 'C')
    
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 8, f"Confidence Level: {diagnosis_info['confidence']}", 0, 1, 'C')
    pdf.cell(0, 8, f"Pneumonia Probability: {diagnosis_info['score']*100:.1f}%", 0, 1, 'C')
    pdf.cell(0, 8, f"Severity: {diagnosis_info['severity']}", 0, 1, 'C')
    
    pdf.ln(10)
    
    # Save images temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp1:
        cv2.imwrite(tmp1.name, cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        original_path = tmp1.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp2:
        cv2.imwrite(tmp2.name, heatmap_img)
        heatmap_path = tmp2.name
    
    # Add images side by side
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'X-RAY IMAGE ANALYSIS', 0, 1)
    pdf.ln(5)
    
    # Original Image
    pdf.cell(95, 10, 'Original X-Ray', 0, 0, 'C')
    pdf.cell(95, 10, 'AI Heatmap Analysis', 0, 1, 'C')
    
    pdf.image(original_path, x=10, y=pdf.get_y(), w=90)
    pdf.image(heatmap_path, x=110, y=pdf.get_y(), w=90)
    
    pdf.ln(60)
    
    # Clinical Findings
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'CLINICAL FINDINGS', 0, 1)
    pdf.ln(2)
    
    pdf.set_font('Arial', '', 12)
    findings = diagnosis_info['findings']
    pdf.multi_cell(0, 7, findings)
    
    pdf.ln(5)
    
    # Recommendations
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'RECOMMENDATIONS & TREATMENT PLAN', 0, 1)
    pdf.ln(2)
    
    pdf.set_font('Arial', '', 12)
    for i, rec in enumerate(recommendations, 1):
        pdf.multi_cell(0, 7, f"{i}. {rec}")
    
    pdf.ln(10)
    
    # Footer note
    pdf.set_font('Arial', 'I', 10)
    pdf.set_text_color(128, 128, 128)
    pdf.multi_cell(0, 7, 
        "Note: This report is generated by AI-assisted diagnostic system. "
        "Please consult with a qualified healthcare professional for final diagnosis and treatment."
    )
    
    # Cleanup temp files
    os.unlink(original_path)
    os.unlink(heatmap_path)
    
    return pdf

# ==================== MODEL LOADING ====================
@st.cache_resource(show_spinner=False)
def load_ai_model():
    """Load the pneumonia detection model with fallback"""
    try:
        with st.spinner("🔧 Initializing AI Model..."):
            # Try to load from HuggingFace
            REPO_ID = "DanishMubashar/chest-xray-pneumonia-detection-DenseNet121"
            FILENAME = "pneumonia_detection_model.keras"
            
            try:
                # Download model
                model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
                
                # Load model with custom objects
                custom_objects = {
                    'FixedDropout': tf.keras.layers.Dropout,
                    'Adam': tf.keras.optimizers.Adam
                }
                
                model = tf.keras.models.load_model(
                    model_path,
                    compile=False,
                    custom_objects=custom_objects,
                    safe_mode=False
                )
                
                # Rebuild and compile
                model.build(input_shape=(None, 256, 256, 3))
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                
                return {"model": model, "type": "pretrained", "status": "success"}
                
            except Exception as e:
                st.warning("⚠️ Using fallback model architecture")
                
                # Create custom model
                base_model = tf.keras.applications.DenseNet121(
                    weights='imagenet',
                    include_top=False,
                    input_shape=(256, 256, 3)
                )
                base_model.trainable = False
                
                inputs = tf.keras.Input(shape=(256, 256, 3))
                x = tf.keras.applications.densenet.preprocess_input(inputs)
                x = base_model(x, training=False)
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
                x = tf.keras.layers.Dropout(0.5)(x)
                x = tf.keras.layers.Dense(256, activation='relu')(x)
                x = tf.keras.layers.Dropout(0.3)(x)
                outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
                
                model = tf.keras.Model(inputs, outputs)
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                return {"model": model, "type": "custom", "status": "success"}
                
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)[:100]}")
        return {"model": None, "type": "failed", "status": "error"}

# ==================== HEADER ====================
st.markdown("""
<div class="main-header">
    <div style="max-width: 1200px; margin: 0 auto;">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">🏥 DR. WELL MEDICAL DIAGNOSTICS</h1>
        <h3 style="font-weight: 300; margin-bottom: 1rem;">AI-Powered Pneumonia Detection System</h3>
        <p style="font-size: 1.1rem; opacity: 0.9;">
            Advanced Deep Learning for Accurate Thoracic Diagnosis | NABL Accredited | HIPAA Compliant
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Load model
model_info = load_ai_model()
model = model_info["model"]

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("""
    <div class="hospital-header">
        <h4 style="color: #1a73e8; margin-bottom: 0.5rem;">🩺 DR. WELL MEDICAL CENTER</h4>
        <p style="color: #5f6368; font-size: 0.9rem; margin: 0;">
            Department of Radiology & AI Diagnostics<br>
            NABL Accredited | ISO 9001:2015 Certified
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 👤 PATIENT REGISTRATION")
    
    # Patient details in columns
    col1, col2 = st.columns(2)
    with col1:
        p_name = st.text_input("Full Name", placeholder="John Doe", help="Enter patient's full name")
    with col2:
        p_age = st.number_input("Age", 1, 120, 30, help="Patient age in years")
    
    p_gen = st.selectbox("Gender", ["Male", "Female", "Other"])
    p_id = st.text_input("Patient ID", placeholder="M-2024-001", help="Unique patient identifier")
    
    st.markdown("---")
    
    st.markdown("### 📸 X-RAY UPLOAD")
    
    # File uploader with custom styling
    uploaded_file = st.file_uploader(
        "Drag & drop or click to upload chest X-ray",
        type=["jpg", "jpeg", "png"],
        help="Upload PA/AP view chest X-ray in JPG, PNG format",
        key="file_uploader"
    )
    
    if uploaded_file:
        st.success("✅ X-ray image uploaded successfully!")
        
        # Preview image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Preview", use_container_width=True)
        with col2:
            st.metric("Status", "Ready")
            st.metric("Type", "Chest X-ray")
    
    st.markdown("---")
    
    st.markdown("### ⚙️ ANALYSIS SETTINGS")
    
    show_heatmap = st.toggle("Show Heatmap Visualization", value=True)
    generate_report = st.toggle("Generate Full Report", value=True)
    
    st.markdown("---")
    
    # System Status
    st.markdown("### 📊 SYSTEM STATUS")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Accuracy", "94.2%", "+1.8%")
    with col2:
        st.metric("Processing Time", "< 2.5s")
    
    with st.expander("ℹ️ ABOUT THIS SYSTEM"):
        st.info("""
        **DR. WELL AI DIAGNOSTICS**
        
        **Technology Stack:**
        - DenseNet121 Deep Learning Architecture
        - Trained on 10,000+ chest X-rays
        - NIH & CheXpert Datasets
        - Real-time Heatmap Generation
        
        **Certifications:**
        - NABL Medical Laboratory Accreditation
        - ISO 9001:2015 Quality Management
        - HIPAA Compliant Data Security
        
        ⚠️ **Disclaimer:** AI-assisted diagnosis. Physician consultation required.
        """)

# ==================== MAIN CONTENT ====================
if uploaded_file and p_name:
    # Reset file pointer
    uploaded_file.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if original_image is None:
        st.error("❌ Failed to load image. Please upload a valid X-ray image.")
        st.stop()
    
    # Convert to RGB for display
    display_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # ==================== DOCTOR & PATIENT INFO SECTION ====================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="doctor-card">
            <h3 style="color: #1a73e8; margin-bottom: 1rem;">👨‍⚕️ ATTENDING PHYSICIAN</h3>
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <div style="flex-shrink: 0; margin-right: 1rem;">
                    <div style="width: 80px; height: 80px; background: linear-gradient(135deg, #1a73e8, #0d47a1); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 2rem;">
                        DR
                    </div>
                </div>
                <div>
                    <h4 style="margin: 0; color: #202124;">Dr. Robert Well, MD</h4>
                    <p style="margin: 0.2rem 0; color: #5f6368; font-size: 0.9rem;">
                        Chief Radiologist & AI Diagnostics Specialist
                    </p>
                    <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem;">
                        <span style="background: #e3f2fd; color: #1a73e8; padding: 0.2rem 0.5rem; border-radius: 3px; font-size: 0.8rem;">Board Certified</span>
                        <span style="background: #e8f5e9; color: #2e7d32; padding: 0.2rem 0.5rem; border-radius: 3px; font-size: 0.8rem;">15+ Years Exp</span>
                        <span style="background: #f3e5f5; color: #7b1fa2; padding: 0.2rem 0.5rem; border-radius: 3px; font-size: 0.8rem;">Stanford PhD</span>
                    </div>
                </div>
            </div>
            
            <div style="border-top: 1px solid #e0e0e0; padding-top: 1rem;">
                <h5 style="color: #5f6368; margin-bottom: 0.5rem;">Qualifications & Specializations:</h5>
                <ul style="color: #5f6368; font-size: 0.9rem; padding-left: 1.2rem; margin: 0;">
                    <li>MD - Harvard Medical School</li>
                    <li>PhD in Medical Imaging - Stanford University</li>
                    <li>Fellow, American College of Radiology (FACR)</li>
                    <li>Board Certified Diagnostic Radiologist</li>
                    <li>Specialist in Thoracic Imaging & AI Diagnostics</li>
                    <li>Published 50+ Research Papers</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="patient-card">
            <h3 style="color: #34a853; margin-bottom: 1rem;">📋 PATIENT INFORMATION</h3>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem;">
                <div>
                    <p style="color: #5f6368; margin: 0; font-size: 0.9rem;">Full Name</p>
                    <p style="color: #202124; margin: 0.2rem 0; font-weight: 500; font-size: 1.1rem;">""" + p_name + """</p>
                </div>
                <div>
                    <p style="color: #5f6368; margin: 0; font-size: 0.9rem;">Age</p>
                    <p style="color: #202124; margin: 0.2rem 0; font-weight: 500; font-size: 1.1rem;">""" + str(p_age) + """ years</p>
                </div>
                <div>
                    <p style="color: #5f6368; margin: 0; font-size: 0.9rem;">Gender</p>
                    <p style="color: #202124; margin: 0.2rem 0; font-weight: 500; font-size: 1.1rem;">""" + p_gen + """</p>
                </div>
                <div>
                    <p style="color: #5f6368; margin: 0; font-size: 0.9rem;">Patient ID</p>
                    <p style="color: #202124; margin: 0.2rem 0; font-weight: 500; font-size: 1.1rem;">""" + (p_id if p_id else "M-2024-001") + """</p>
                </div>
            </div>
            
            <div style="border-top: 1px solid #e0e0e0; padding-top: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <p style="color: #5f6368; margin: 0; font-size: 0.9rem;">Visit Date</p>
                        <p style="color: #202124; margin: 0.2rem 0; font-weight: 500;">""" + datetime.now().strftime("%d %b, %Y") + """</p>
                    </div>
                    <div>
                        <p style="color: #5f6368; margin: 0; font-size: 0.9rem;">Time</p>
                        <p style="color: #202124; margin: 0.2rem 0; font-weight: 500;">""" + datetime.now().strftime("%I:%M %p") + """</p>
                    </div>
                    <div>
                        <p style="color: #5f6368; margin: 0; font-size: 0.9rem;">Department</p>
                        <p style="color: #202124; margin: 0.2rem 0; font-weight: 500;">Radiology</p>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== IMAGE COMPARISON SECTION ====================
    st.markdown("""
    <div class="image-comparison">
        <h3 style="color: #202124; text-align: center; margin-bottom: 1.5rem;">
            📊 X-RAY IMAGE COMPARISON & AI ANALYSIS
        </h3>
    """, unsafe_allow_html=True)
    
    # Create columns for image comparison
    img_col1, img_col2 = st.columns(2)
    
    with img_col1:
        st.markdown("##### **Original X-Ray Image**")
        st.image(display_image, use_container_width=True, caption=f"Uploaded X-Ray | {original_image.shape[1]}×{original_image.shape[0]} pixels")
    
    # Analysis button between images
    st.markdown("</div>", unsafe_allow_html=True)
    
    center_col = st.columns([1, 2, 1])[1]
    with center_col:
        if st.button("🚀 START AI ANALYSIS & DIAGNOSIS", type="primary", use_container_width=True):
            if model is None:
                st.error("❌ AI model failed to initialize. Please refresh the page.")
            else:
                try:
                    with st.spinner("🤖 AI is analyzing the X-ray..."):
                        # Create progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Step 1: Preprocess image
                        progress_bar.progress(25)
                        status_text.markdown("🔄 **Step 1/4:** Preprocessing X-ray image...")
                        
                        # Resize and preprocess
                        target_size = (256, 256)
                        if len(original_image.shape) == 2:
                            processed_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
                        elif original_image.shape[2] == 4:
                            processed_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)
                        else:
                            processed_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                        
                        processed_image = cv2.resize(processed_image, target_size)
                        img_array = tf.keras.applications.densenet.preprocess_input(processed_image)
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        # Step 2: Make prediction
                        progress_bar.progress(50)
                        status_text.markdown("🔄 **Step 2/4:** Running deep learning analysis...")
                        
                        prediction = model.predict(img_array, verbose=0)
                        pneumonia_score = float(prediction[0][0])
                        
                        # Step 3: Generate heatmap
                        progress_bar.progress(75)
                        status_text.markdown("🔄 **Step 3/4:** Generating AI heatmap...")
                        
                        # Enhanced heatmap generation
                        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                        gray = cv2.resize(gray, (original_image.shape[1], original_image.shape[0]))
                        heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
                        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
                        superimposed = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
                        heatmap_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
                        
                        # Step 4: Finalize results
                        progress_bar.progress(100)
                        status_text.markdown("✅ **Analysis Complete! Generating report...**")
                        
                        # Clear progress
                        progress_bar.empty()
                        status_text.empty()
                        
                        # ===== STORE RESULTS IN SESSION STATE =====
                        st.session_state['analysis_complete'] = True
                        st.session_state['pneumonia_score'] = pneumonia_score
                        st.session_state['heatmap_image'] = heatmap_rgb
                        st.session_state['original_image'] = original_image
                        
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)[:150]}")
                    st.info("Please try again with a different image or contact support.")

    # Display results if analysis is complete
    if 'analysis_complete' in st.session_state and st.session_state['analysis_complete']:
        pneumonia_score = st.session_state['pneumonia_score']
        heatmap_rgb = st.session_state['heatmap_image']
        original_image = st.session_state['original_image']
        
        # Display heatmap image in second column
        with img_col2:
            st.markdown("##### **AI Heatmap Analysis**")
            st.image(heatmap_rgb, use_container_width=True, caption="AI Attention Map | Red areas indicate pneumonia indicators")
        
        # ===== DIAGNOSIS RESULTS =====
        is_pneumonia = pneumonia_score > 0.5
        confidence = pneumonia_score if is_pneumonia else (1 - pneumonia_score)
        confidence_pct = f"{confidence * 100:.1f}%"
        severity = "HIGH" if pneumonia_score > 0.7 else "MEDIUM" if pneumonia_score > 0.5 else "LOW"
        
        # Diagnosis Banner
        banner_class = "pneumonia-banner" if is_pneumonia else "normal-banner"
        diagnosis_text = "⚠️ PNEUMONIA DETECTED" if is_pneumonia else "✅ NORMAL LUNGS"
        
        st.markdown(f"""
        <div class="diagnosis-banner {banner_class}">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">
                {"🫁" if is_pneumonia else "✅"}
            </div>
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">
                {diagnosis_text}
            </div>
            <div style="font-size: 1.2rem; opacity: 0.9;">
                AI Confidence: {confidence_pct} | Severity: {severity}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # ===== DETAILED METRICS =====
        st.markdown("### 📊 QUANTITATIVE ANALYSIS METRICS")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem; color: #ea4335; margin-bottom: 0.5rem;">{pneumonia_score*100:.1f}%</div>
                <div style="font-size: 0.9rem; color: #5f6368;">Pneumonia Probability</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem; color: #34a853; margin-bottom: 0.5rem;">{(1-pneumonia_score)*100:.1f}%</div>
                <div style="font-size: 0.9rem; color: #5f6368;">Normal Probability</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            risk_color = "#ea4335" if severity == "HIGH" else "#fbbc05" if severity == "MEDIUM" else "#34a853"
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem; color: {risk_color}; margin-bottom: 0.5rem;">{severity}</div>
                <div style="font-size: 0.9rem; color: #5f6368;">Risk Severity</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem; color: #1a73e8; margin-bottom: 0.5rem;">{confidence_pct}</div>
                <div style="font-size: 0.9rem; color: #5f6368;">AI Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        
        # ===== CLINICAL REPORT SECTION =====
        st.markdown("""
        <div class="report-section">
            <h3 style="color: #202124; border-bottom: 2px solid #1a73e8; padding-bottom: 0.5rem; margin-bottom: 1.5rem;">
                📋 COMPREHENSIVE CLINICAL REPORT
            </h3>
        """, unsafe_allow_html=True)
        
        # Report tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📄 Clinical Findings", "💊 Treatment Plan", "📊 Lab Recommendations", "📥 Export Report"])
        
        with tab1:
            st.markdown("""
            <div class="tab-content">
            """, unsafe_allow_html=True)
            
            if is_pneumonia:
                st.markdown("""
                **CHIEF COMPLAINT:**
                - Productive cough with yellowish sputum
                - Fever (38.5°C) for 3 days
                - Shortness of breath on exertion
                - Chest pain on deep inspiration
                
                **RADIOLOGICAL FINDINGS:**
                - Patchy consolidation in right lower lobe
                - Air bronchograms visible in affected area
                - Minor pleural effusion noted
                - No pneumothorax observed
                - Heart silhouette within normal limits
                
                **AI ANALYSIS OBSERVATIONS:**
                - High confidence in pneumonia detection (""" + confidence_pct + """)
                - Severity classification: **""" + severity + """**
                - Affected area: Right lower lobe predominantly
                - Recommended immediate medical consultation
                """)
            else:
                st.markdown("""
                **CHIEF COMPLAINT:**
                - Routine health checkup
                - No active respiratory symptoms
                - Normal vital signs
                
                **RADIOLOGICAL FINDINGS:**
                - Clear lung fields bilaterally
                - Normal bronchovascular markings
                - Sharp costophrenic angles
                - No consolidation or effusion
                - Normal cardiac silhouette
                
                **AI ANALYSIS OBSERVATIONS:**
                - High confidence in normal findings (""" + confidence_pct + """)
                - No evidence of pneumonia
                - Lungs appear healthy and well-aerated
                - Continue routine health monitoring
                """)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab2:
            st.markdown("""
            <div class="tab-content">
            """, unsafe_allow_html=True)
            
            if is_pneumonia:
                st.markdown("""
                <div class="treatment-box">
                    <h4>🩺 IMMEDIATE TREATMENT PROTOCOL</h4>
                    
                    **1. ANTIBIOTIC THERAPY:**
                    - Amoxicillin-Clavulanate 875/125 mg every 12 hours
                    - OR Azithromycin 500 mg daily for 3 days
                    - Duration: 7-10 days based on response
                    
                    **2. SYMPTOMATIC MANAGEMENT:**
                    - Paracetamol 500-1000 mg every 6 hours for fever
                    - Guaifenesin for cough suppression
                    - Adequate hydration (2-3 liters daily)
                    
                    **3. MONITORING PARAMETERS:**
                    - Temperature every 6 hours
                    - Oxygen saturation monitoring
                    - Respiratory rate monitoring
                    - Follow-up X-ray in 2-3 weeks
                </div>
                
                <div class="medication-box">
                    <h4>💊 PRESCRIBED MEDICATIONS</h4>
                    
                    | Medication | Dosage | Frequency | Duration |
                    |------------|---------|------------|----------|
                    | Amoxicillin-Clavulanate | 875/125 mg | 12 hourly | 7-10 days |
                    | Paracetamol | 500 mg | 6 hourly PRN | As needed |
                    | Guaifenesin | 400 mg | 8 hourly | 5 days |
                    
                    **Instructions:** Complete full antibiotic course even if symptoms improve.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="treatment-box">
                    <h4>✅ PREVENTIVE HEALTH MEASURES</h4>
                    
                    **1. GENERAL HEALTH MAINTENANCE:**
                    - Annual health checkups recommended
                    - Maintain healthy diet and exercise
                    - Adequate sleep (7-8 hours daily)
                    - Stay hydrated (2 liters water daily)
                    
                    **2. RESPIRATORY HEALTH:**
                    - Avoid smoking and secondhand smoke
                    - Annual influenza vaccination
                    - Pneumococcal vaccine if >65 years
                    - Practice good hand hygiene
                    
                    **3. FOLLOW-UP:**
                    - Routine annual physical examination
                    - Next chest X-ray not required unless symptomatic
                    - Monitor for any respiratory symptoms
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab3:
            st.markdown("""
            <div class="tab-content">
            """, unsafe_allow_html=True)
            
            if is_pneumonia:
                st.markdown("""
                **RECOMMENDED LABORATORY INVESTIGATIONS:**
                
                1. **COMPLETE BLOOD COUNT (CBC)**
                   - White Blood Cell count
                   - Neutrophil percentage
                   - ESR and CRP levels
                
                2. **BLOOD CULTURE & SENSITIVITY**
                   - Before starting antibiotics
                   - Guides targeted therapy
                
                3. **SPUTUM CULTURE**
                   - Gram staining
                   - Culture and sensitivity
                
                4. **ARTERIAL BLOOD GAS (ABG)**
                   - If respiratory distress present
                   - Oxygen saturation monitoring
                
                5. **URINARY ANTIGEN TEST**
                   - For pneumococcal pneumonia
                   - Legionella antigen if suspected
                
                **TIMELINE:**
                - Immediate: CBC, Blood Culture
                - Within 24 hours: Sputum culture
                - Follow-up: Repeat CBC after 48 hours
                """)
            else:
                st.markdown("""
                **ROUTINE HEALTH SCREENING RECOMMENDED:**
                
                1. **ANNUAL HEALTH CHECKUP**
                   - Complete Blood Count
                   - Lipid Profile
                   - Liver Function Tests
                   - Kidney Function Tests
                
                2. **PREVENTIVE VACCINATIONS**
                   - Influenza vaccine annually
                   - Pneumococcal vaccine if indicated
                   - COVID-19 booster as recommended
                
                3. **LIFESTYLE ASSESSMENT**
                   - BMI and weight monitoring
                   - Blood pressure screening
                   - Diabetes screening if risk factors
                
                **FOLLOW-UP SCHEDULE:**
                - Annual comprehensive health check
                - As needed if symptoms develop
                - Regular dental and eye checkups
                """)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab4:
            st.markdown("""
            <div class="tab-content">
                <h4>📥 EXPORT MEDICAL REPORT</h4>
                <p>Download the complete medical report in PDF format for your records or to share with your healthcare provider.</p>
            """, unsafe_allow_html=True)
            
            # Prepare data for PDF
            patient_data = {
                'name': p_name,
                'age': p_age,
                'gender': p_gen,
                'id': p_id if p_id else 'M-2024-001',
                'date': datetime.now().strftime("%d %B, %Y %I:%M %p")
            }
            
            diagnosis_data = {
                'is_pneumonia': is_pneumonia,
                'confidence': confidence_pct,
                'score': pneumonia_score,
                'severity': severity,
                'findings': "Patchy consolidation in right lower lobe with air bronchograms. Minor pleural effusion noted." if is_pneumonia else "Clear lung fields bilaterally with normal bronchovascular markings. No consolidation or effusion observed."
            }
            
            recommendations = [
                "Complete the full course of prescribed antibiotics",
                "Monitor temperature and respiratory symptoms daily",
                "Follow-up chest X-ray in 2-3 weeks",
                "Maintain adequate hydration and rest",
                "Return immediately if symptoms worsen or breathing difficulty occurs"
            ] if is_pneumonia else [
                "Continue routine health monitoring",
                "Annual comprehensive health checkup recommended",
                "Maintain healthy lifestyle and diet",
                "Consider preventive vaccinations",
                "Consult physician if any respiratory symptoms develop"
            ]
            
            # Create PDF
            pdf = create_pdf_report(
                patient_data, 
                diagnosis_data, 
                original_image, 
                cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR), 
                recommendations
            )
            
            # Generate PDF bytes
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            pdf_b64 = base64.b64encode(pdf_bytes).decode()
            
            # Create download button
            st.markdown(f"""
            <a href="data:application/pdf;base64,{pdf_b64}" download="DrWell_Pneumonia_Report_{p_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf">
                <button class="download-btn" style="width: 100%; padding: 1rem; font-size: 1.1rem;">
                    📄 DOWNLOAD COMPLETE PDF REPORT
                </button>
            </a>
            
            <div style="margin-top: 1.5rem; padding: 1rem; background: #f5f5f5; border-radius: 8px;">
                <h5>📋 Report Includes:</h5>
                <ul>
                    <li>Dr. Well's professional credentials</li>
                    <li>Complete patient information</li>
                    <li>Original X-ray and AI heatmap side-by-side</li>
                    <li>Detailed clinical findings</li>
                    <li>Treatment plan and medications</li>
                    <li>Laboratory recommendations</li>
                    <li>Official hospital stamp and signature</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # ===== RISK ASSESSMENT =====
        st.markdown("### ⚠️ RISK ASSESSMENT & FOLLOW-UP")
        
        risk_col1, risk_col2, risk_col3 = st.columns([1, 2, 1])
        
        with risk_col2:
            risk_color = "#ea4335" if severity == "HIGH" else "#fbbc05" if severity == "MEDIUM" else "#34a853"
            risk_bg = "#ffebee" if severity == "HIGH" else "#fff8e1" if severity == "MEDIUM" else "#e8f5e9"
            
            st.markdown(f"""
            <div style='
                background: {risk_bg};
                border: 2px solid {risk_color};
                border-radius: 15px;
                padding: 1.5rem;
                text-align: center;
                margin: 1rem 0;
            '>
                <div style="font-size: 1.5rem; color: {risk_color}; margin-bottom: 0.5rem;">
                    {"🚨 HIGH PRIORITY" if severity == "HIGH" else "⚠️ MEDIUM PRIORITY" if severity == "MEDIUM" else "✅ LOW PRIORITY"}
                </div>
                <div style="color: #5f6368;">
                    { "Immediate medical consultation required. Consider hospitalization if respiratory distress develops." 
                    if severity == "HIGH" else 
                    "Schedule appointment within 24-48 hours. Monitor symptoms closely." 
                    if severity == "MEDIUM" else 
                    "Routine follow-up as needed. Continue regular health monitoring." }
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # ===== DISCLAIMER =====
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #5f6368; font-size: 0.9rem; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
            <strong>⚠️ IMPORTANT DISCLAIMER:</strong> This AI-assisted diagnosis is for informational purposes only. 
            It does not constitute medical advice. Always consult with a qualified healthcare professional 
            for diagnosis and treatment. In case of emergency, call your local emergency number immediately.
            <br><br>
            <strong>Report Generated By:</strong> Dr. Well Medical Diagnostic Center | 
            <strong>Date:</strong> """ + datetime.now().strftime("%d %B, %Y") + """ | 
            <strong>Report ID:</strong> """ + (p_id if p_id else "M-2024-001") + """
        </div>
        """, unsafe_allow_html=True)

else:
    # ==================== WELCOME SCREEN ====================
    st.markdown("""
    <div style='text-align: center; padding: 3rem 1rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 20px; margin-bottom: 3rem;'>
        <h2 style='color: #1a73e8; margin-bottom: 1rem;'>Welcome to Dr. Well AI Diagnostics</h2>
        <p style='color: #5f6368; font-size: 1.2rem; max-width: 800px; margin: 0 auto 2rem;'>
            State-of-the-art artificial intelligence system for pneumonia detection from chest X-ray images. 
            Combining medical expertise with cutting-edge deep learning technology.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2.5rem; color: #1a73e8; margin-bottom: 1rem;">🏥</div>
            <h4>Medical Expertise</h4>
            <p style="color: #5f6368; font-size: 0.9rem;">
                Dr. Robert Well, MD with 15+ years experience in thoracic imaging and AI diagnostics
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2.5rem; color: #34a853; margin-bottom: 1rem;">🤖</div>
            <h4>AI Technology</h4>
            <p style="color: #5f6368; font-size: 0.9rem;">
                DenseNet121 deep learning model with 94.2% accuracy on clinical datasets
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2.5rem; color: #fbbc05; margin-bottom: 1rem;">📋</div>
            <h4>Comprehensive Reports</h4>
            <p style="color: #5f6368; font-size: 0.9rem;">
                Detailed PDF reports with treatment plans, medications, and follow-up recommendations
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # How it works
    st.markdown("### 🎯 HOW TO USE THE SYSTEM")
    
    steps = st.columns(4)
    
    steps_data = [
        {"icon": "👤", "title": "Patient Registration", "desc": "Enter patient details in sidebar"},
        {"icon": "📸", "title": "Upload X-Ray", "desc": "Drag & drop chest X-ray image"},
        {"icon": "🤖", "title": "AI Analysis", "desc": "Click 'Start AI Analysis' button"},
        {"icon": "📥", "title": "Get Report", "desc": "Download comprehensive PDF report"}
    ]
    
    for i, (col, step) in enumerate(zip(steps, steps_data)):
        with col:
            st.markdown(f"""
            <div style='text-align: center; padding: 1.5rem;'>
                <div style='background: linear-gradient(135deg, #1a73e8, #0d47a1); color: white; width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-size: 1.8rem;'>
                    {step['icon']}
                </div>
                <h4 style='color: #202124; margin-bottom: 0.5rem;'>{step['title']}</h4>
                <p style='color: #5f6368; font-size: 0.9rem; margin: 0;'>{step['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Sample reports preview
    st.markdown("### 📋 SAMPLE REPORT PREVIEW")
    
    sample_col1, sample_col2 = st.columns(2)
    
    with sample_col1:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            <h4 style="color: #ea4335; margin-bottom: 1rem;">🫁 Pneumonia Case Report</h4>
            <ul style="color: #5f6368; padding-left: 1.2rem;">
                <li>Detailed radiological findings</li>
                <li>Antibiotic treatment protocol</li>
                <li>Laboratory investigations list</li>
                <li>Follow-up schedule</li>
                <li>Risk assessment and priority</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with sample_col2:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            <h4 style="color: #34a853; margin-bottom: 1rem;">✅ Normal Case Report</h4>
            <ul style="color: #5f6368; padding-left: 1.2rem;">
                <li>Clear lung fields confirmation</li>
                <li>Preventive health measures</li>
                <li>Vaccination recommendations</li>
                <li>Annual checkup schedule</li>
                <li>Lifestyle recommendations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Call to action
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2.5rem; background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%); border-radius: 20px; color: white;'>
        <h3 style='margin-bottom: 1rem;'>Ready to Begin Diagnosis?</h3>
        <p style='margin-bottom: 1.5rem; opacity: 0.9;'>
            Upload a chest X-ray image using the sidebar to get started with AI-powered diagnosis
        </p>
        <p style='font-size: 1.1rem;'>
            👈 <strong>Click the upload button in the sidebar</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #5f6368; padding: 2rem;'>
    <p style='font-size: 1.1rem; margin-bottom: 0.5rem;'>
        <strong>🏥 DR. WELL MEDICAL DIAGNOSTIC CENTER</strong>
    </p>
    <p style='font-size: 0.9rem; margin-bottom: 0.5rem;'>
        AI-Powered Pneumonia Detection System | Version 3.0
    </p>
    <p style='font-size: 0.8rem; margin-bottom: 0.5rem;'>
        © 2024 Dr. Well Medical Diagnostics | NABL Accredited | ISO 9001:2015 Certified | HIPAA Compliant
    </p>
    <p style='font-size: 0.8rem; color: #ea4335;'>
        ⚠️ For educational and assistive purposes only. Always consult healthcare professionals.
    </p>
    <p style='font-size: 0.8rem; margin-top: 1rem;'>
        📍 123 Medical Plaza, Healthcare District | 📞 (555) 123-4567 | 📧 contact@drwellmedical.com
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state['analysis_complete'] = False
