"""
SWASTHYA AI — MONAI Medical Imaging Lab
Streamlit Rapid-Prototyping Healthcare AI Application.
Features:
- MONAI Preprocessing (transforms, channel format, resize, normalization)
- Pre-loaded Demo Scans with transparent 'DEMO DATA — NOT FOR CLINICAL USE' badges
- Chest X-ray AI (DenseNet121 & Grad-CAM)
- Ultrasound AI (Classification & MONAI UNet Segmentation)
- Strict compliance: never fabricates medical predictions; transparent demo mode.
- ReportLab PDF Diagnostic Export
"""

import sys
import os
from pathlib import Path
from PIL import Image
import torch
import streamlit as st

# Configure page layout
st.set_page_config(
    page_title="SWASTHYA AI | MONAI Medical Lab",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure local imports resolve
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from ai.xray import XRayEngine
from ai.ultrasound import UltrasoundEngine
from utils.report import create_medical_pdf

# Initialize single-instance engines
@st.cache_resource
def get_xray_engine():
    return XRayEngine()

@st.cache_resource
def get_ultrasound_engine():
    return UltrasoundEngine()

xray_engine = get_xray_engine()
us_engine = get_ultrasound_engine()

# Detect compute hardware
device_name = "CUDA" if torch.cuda.is_available() else "CPU"

# Session state initialization
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = True

if "selected_modality" not in st.session_state:
    st.session_state.selected_modality = "Chest X-ray"

if "active_image" not in st.session_state:
    st.session_state.active_image = None

if "active_sample_title" not in st.session_state:
    st.session_state.active_sample_title = ""

if "current_result" not in st.session_state:
    st.session_state.current_result = None

if "us_task_mode" not in st.session_state:
    st.session_state.us_task_mode = "Classification"

sample_dir = CURRENT_DIR / "data" / "sample"

# Custom aesthetic styling
st.markdown(
    """
    <style>
    .stMetric {
        background-color: #1e293b;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #334155;
    }
    .status-card {
        background-color: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .check-item {
        color: #10b981;
        font-weight: 600;
        font-size: 13px;
        margin: 2px 0;
    }
    .demo-badge {
        background-color: #f59e0b;
        color: #000000;
        font-size: 10px;
        font-weight: 800;
        padding: 2px 6px;
        border-radius: 4px;
        letter-spacing: 0.5px;
    }
    .warning-badge {
        background-color: #3f1d1d;
        color: #fca5a5;
        font-size: 10px;
        font-weight: 700;
        padding: 3px 6px;
        border-radius: 4px;
        text-align: center;
        margin: 4px 0 8px 0;
        border: 1px solid #7f1d1d;
    }
    .demo-sample-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("## 🏥 SWASTHYA AI")
    st.caption("MONAI Medical Imaging Lab")
    st.markdown("---")

    st.markdown("### Medical Modality")
    modality_index = 0 if st.session_state.selected_modality == "Chest X-ray" else 1
    selected_mod = st.radio(
        "Select diagnostic imaging modality:",
        ["Chest X-ray", "Ultrasound"],
        index=modality_index,
        label_visibility="collapsed",
        key="sidebar_modality_radio"
    )
    if selected_mod != st.session_state.selected_modality:
        st.session_state.selected_modality = selected_mod
        st.session_state.active_image = None
        st.session_state.current_result = None
        st.rerun()

    st.markdown("---")
    st.markdown("### System Telemetry")
    st.markdown(f"**AI Engine:** MONAI + PyTorch")
    st.markdown(f"**Device:** {device_name}")
    st.markdown(f"**X-ray Model:** {'Configured' if xray_engine.is_configured() else 'Not configured'}")
    st.markdown(f"**Ultrasound Model:** {'Configured' if us_engine.is_configured('classification') else 'Not configured'}")

    st.markdown("---")
    st.session_state.demo_mode = st.toggle("Enable Demo Mode", value=st.session_state.demo_mode)
    if st.session_state.demo_mode:
        st.caption("🟡 Demo mode active — exercises full MONAI pipelines with transparent test data.")
    else:
        st.caption("🟢 Real model mode active — requires verified trained weights.")

    st.markdown("---")
    st.caption("Privacy: 100% Local Inference • Zero External APIs")

# ----------------- MAIN HEADER -----------------
st.markdown("# 🏥 SWASTHYA AI")
st.markdown("#### MONAI Medical Imaging Intelligence")

# Top Banner: Feature Cards
f1, f2, f3 = st.columns(3)
with f1:
    st.markdown(
        """
        <div class="status-card">
            <h4 style="margin:0; color:#38bdf8;">🫁 Chest X-ray</h4>
            <p style="margin:4px 0 0 0; color:#94a3b8; font-size:13px;">AI-assisted multi-pathology detection using MONAI DenseNet121.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
with f2:
    st.markdown(
        """
        <div class="status-card">
            <h4 style="margin:0; color:#38bdf8;">🔬 Ultrasound</h4>
            <p style="margin:4px 0 0 0; color:#94a3b8; font-size:13px;">Classification & MONAI UNet lesion contour segmentation.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
with f3:
    st.markdown(
        """
        <div class="status-card">
            <h4 style="margin:0; color:#38bdf8;">🧠 Explainable AI</h4>
            <p style="margin:4px 0 0 0; color:#94a3b8; font-size:13px;">Convolutional Grad-CAM heatmaps & boundary overlays.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Small Pipeline Visualization
st.markdown(
    """
    <div style="background-color:#1e293b; padding:8px 16px; border-radius:6px; font-size:12px; color:#cbd5e1; text-align:center; margin-bottom:14px; border:1px solid #334155;">
        <b>DIAGNOSTIC PIPELINE:</b> &nbsp;
        UPLOAD &nbsp; ➔ &nbsp;
        <b>MONAI PREPROCESSING</b> &nbsp; ➔ &nbsp;
        <b>AI MODEL</b> &nbsp; ➔ &nbsp;
        <b>EXPLAINABILITY</b> &nbsp; ➔ &nbsp;
        <b>CLINICAL REPORT</b>
    </div>
    """,
    unsafe_allow_html=True
)

# Top Model Status Bar
sc1, sc2, sc3, sc4 = st.columns(4)
sc1.metric("AI Engine", "MONAI + PyTorch")
sc2.metric("Hardware Device", device_name)
sc3.metric("X-ray Model", "Configured" if xray_engine.is_configured() else "Not configured")
sc4.metric("Ultrasound Model", "Configured" if us_engine.is_configured("classification") else "Not configured")

# Demo Mode Warning Banner
if st.session_state.demo_mode:
    st.markdown(
        """
        <div style="background-color: #451a03; border-left: 4px solid #f59e0b; padding: 10px 16px; border-radius: 4px; margin: 12px 0;">
            <b style="color: #fbbf24;">🟡 DEMO MODE</b> &nbsp;
            <span style="color: #fef3c7; font-size: 13px;">This prototype is for demonstration purposes. Never present demo output as a real medical diagnosis.</span>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# ----------------- SECTION: PRE-LOADED DEMO SCANS -----------------
st.markdown("## PRE-LOADED DEMO SCANS")
st.markdown("<p style='color:#94a3b8; font-size:14px; margin-top:-10px; margin-bottom:16px;'>Test the imaging pipelines instantly using bundled demonstration images.</p>", unsafe_allow_html=True)

demo_col1, demo_col2, demo_col3, demo_col4 = st.columns(4)

# Sample 1: Chest X-ray — Normal
with demo_col1:
    xray_norm_path = sample_dir / "xray_normal.png"
    if xray_norm_path.exists():
        st.image(str(xray_norm_path), use_container_width=True)
    st.markdown(
        """
        <div class="demo-sample-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="color:#38bdf8; font-size:11px; font-weight:700;">Chest X-ray</span>
                <span class="demo-badge">DEMO SAMPLE</span>
            </div>
            <h4 style="margin:4px 0; font-size:13px; color:#f8fafc;">Normal</h4>
            <div style="font-size:11px; color:#94a3b8; margin-bottom:4px;"><b>Data type:</b> Synthetic Demo</div>
            <div class="warning-badge">DEMO DATA — NOT FOR CLINICAL USE</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("Use Sample", key="btn_use_demo_xray_norm", use_container_width=True):
        st.session_state.selected_modality = "Chest X-ray"
        st.session_state.active_image = Image.open(xray_norm_path)
        st.session_state.active_sample_title = "Chest X-ray — Normal"
        with st.spinner("Processing Chest X-ray sample with MONAI..."):
            st.session_state.current_result = xray_engine.analyze(st.session_state.active_image, demo_mode=st.session_state.demo_mode)
        st.rerun()

# Sample 2: Chest X-ray — Pneumonia Demo
with demo_col2:
    xray_pneu_path = sample_dir / "xray_pneumonia.png"
    if xray_pneu_path.exists():
        st.image(str(xray_pneu_path), use_container_width=True)
    st.markdown(
        """
        <div class="demo-sample-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="color:#38bdf8; font-size:11px; font-weight:700;">Chest X-ray</span>
                <span class="demo-badge">DEMO SAMPLE</span>
            </div>
            <h4 style="margin:4px 0; font-size:13px; color:#f8fafc;">Pneumonia Demo</h4>
            <div style="font-size:11px; color:#94a3b8; margin-bottom:4px;"><b>Data type:</b> Synthetic Demo</div>
            <div class="warning-badge">DEMO DATA — NOT FOR CLINICAL USE</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("Use Sample", key="btn_use_demo_xray_pneu", use_container_width=True):
        st.session_state.selected_modality = "Chest X-ray"
        st.session_state.active_image = Image.open(xray_pneu_path)
        st.session_state.active_sample_title = "Chest X-ray — Pneumonia Demo"
        with st.spinner("Processing Chest X-ray sample with MONAI..."):
            st.session_state.current_result = xray_engine.analyze(st.session_state.active_image, demo_mode=st.session_state.demo_mode)
        st.rerun()

# Sample 3: Ultrasound — Normal B-mode Demo
with demo_col3:
    us_norm_path = sample_dir / "ultrasound_normal.png"
    if us_norm_path.exists():
        st.image(str(us_norm_path), use_container_width=True)
    st.markdown(
        """
        <div class="demo-sample-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="color:#38bdf8; font-size:11px; font-weight:700;">Ultrasound</span>
                <span class="demo-badge">DEMO SAMPLE</span>
            </div>
            <h4 style="margin:4px 0; font-size:13px; color:#f8fafc;">Normal B-mode Demo</h4>
            <div style="font-size:11px; color:#94a3b8; margin-bottom:4px;"><b>Data type:</b> Synthetic Demo</div>
            <div class="warning-badge">DEMO DATA — NOT FOR CLINICAL USE</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("Use Sample", key="btn_use_demo_us_norm", use_container_width=True):
        st.session_state.selected_modality = "Ultrasound"
        st.session_state.us_task_mode = "Classification"
        st.session_state.active_image = Image.open(us_norm_path)
        st.session_state.active_sample_title = "Ultrasound — Normal B-mode Demo"
        with st.spinner("Processing Ultrasound sample with MONAI..."):
            st.session_state.current_result = us_engine.analyze(st.session_state.active_image, task="Classification", demo_mode=st.session_state.demo_mode)
        st.rerun()

# Sample 4: Ultrasound — Nodule Demo
with demo_col4:
    us_les_path = sample_dir / "ultrasound_lesion.png"
    if us_les_path.exists():
        st.image(str(us_les_path), use_container_width=True)
    st.markdown(
        """
        <div class="demo-sample-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="color:#38bdf8; font-size:11px; font-weight:700;">Ultrasound</span>
                <span class="demo-badge">DEMO SAMPLE</span>
            </div>
            <h4 style="margin:4px 0; font-size:13px; color:#f8fafc;">Nodule Demo</h4>
            <div style="font-size:11px; color:#94a3b8; margin-bottom:4px;"><b>Data type:</b> Synthetic Demo</div>
            <div class="warning-badge">DEMO DATA — NOT FOR CLINICAL USE</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("Use Sample", key="btn_use_demo_us_les", use_container_width=True):
        st.session_state.selected_modality = "Ultrasound"
        st.session_state.us_task_mode = "Segmentation"
        st.session_state.active_image = Image.open(us_les_path)
        st.session_state.active_sample_title = "Ultrasound — Nodule Demo"
        with st.spinner("Processing Ultrasound lesion sample with MONAI..."):
            st.session_state.current_result = us_engine.analyze(st.session_state.active_image, task="Segmentation", demo_mode=st.session_state.demo_mode)
        st.rerun()

st.markdown("---")

# ----------------- ACTIVE MODALITY PIPELINE EXECUTION -----------------
current_mod = st.session_state.selected_modality

if current_mod == "Chest X-ray":
    st.markdown("## 🫁 Chest X-ray Analysis")

    source_col1, source_col2 = st.columns([3, 1])
    with source_col1:
        uploaded_xray = st.file_uploader(
            "Upload Chest X-ray (PNG/JPG/JPEG):",
            type=["png", "jpg", "jpeg"],
            key="custom_xray_upload"
        )
        if uploaded_xray is not None:
            st.session_state.active_image = Image.open(uploaded_xray)
            st.session_state.active_sample_title = f"Uploaded: {uploaded_xray.name}"
            st.session_state.current_result = None

    with source_col2:
        if st.session_state.active_sample_title:
            st.info(f"**Loaded Image:**\n{st.session_state.active_sample_title}")
            if "Demo" in st.session_state.active_sample_title or "Normal" in st.session_state.active_sample_title:
                st.caption("Data type: Synthetic Demo (Technical Test Data)")

    if st.session_state.active_image is not None:
        c_left, c_right = st.columns([1, 1])
        with c_left:
            st.markdown("### Original X-ray")
            st.image(st.session_state.active_image, caption=st.session_state.active_sample_title or "Input Scan", use_container_width=True)

        with c_right:
            st.markdown("<br/><br/>", unsafe_allow_html=True)
            analyze_xray = st.button("⚡ Analyze X-ray", type="primary", use_container_width=True, key="btn_run_xray_analysis")

            if analyze_xray:
                with st.spinner("Executing MONAI preprocessing & diagnostic pipeline..."):
                    st.session_state.current_result = xray_engine.analyze(
                        st.session_state.active_image,
                        demo_mode=st.session_state.demo_mode
                    )

        # If analysis result exists
        if st.session_state.current_result is not None:
            res = st.session_state.current_result
            st.markdown("---")

            # MONAI Preprocessing Verification Checklist
            st.markdown("### MONAI Preprocessing Verification")
            tele = res.get("telemetry", {})
            p_cols = st.columns(4)
            p_cols[0].markdown(f"<div class='check-item'>✓ Image loaded ({tele.get('orig_size')})</div>", unsafe_allow_html=True)
            p_cols[1].markdown(f"<div class='check-item'>✓ Resized ({tele.get('resized')})</div>", unsafe_allow_html=True)
            p_cols[2].markdown(f"<div class='check-item'>✓ Intensity normalized ({tele.get('normalized')})</div>", unsafe_allow_html=True)
            p_cols[3].markdown(f"<div class='check-item'>✓ Tensor created ({tele.get('tensor_shape')})</div>", unsafe_allow_html=True)

            st.markdown("<br/>", unsafe_allow_html=True)

            # Analysis Result Card
            st.markdown("### AI ANALYSIS RESULT")
            if res["status"] in ("MODEL_NOT_CONFIGURED", "NOT_CONFIGURED"):
                st.markdown(
                    """
                    <div style="background-color: #1e293b; border-left: 4px solid #f59e0b; padding: 14px 18px; border-radius: 8px; margin: 12px 0 16px 0; border: 1px solid #334155;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="color: #fbbf24; font-weight: 700; font-size: 14px;">⚠️ Model not configured — preprocessing/demo pipeline only.</span>
                            <span style="background-color: #f59e0b; color: #000; font-size: 10px; font-weight: 800; padding: 2px 6px; border-radius: 4px;">DEMO DATA — NOT FOR CLINICAL USE</span>
                        </div>
                        <p style="color: #cbd5e1; font-size: 13px; margin: 4px 0 0 0;">
                            The MONAI preprocessing pipeline (channel formatting, bilinear resize to (224, 224), min-max intensity normalization, and PyTorch float32 tensor conversion) was executed and verified.
                        </p>
                        <p style="color: #94a3b8; font-size: 12px; margin: 6px 0 0 0;">
                            <b>Strict Clinical Integrity:</b> In compliance with medical AI safety standards, disease predictions and probabilities are <u>never fabricated</u> without a validated, loaded neural network model.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Modality", "Chest X-ray")
                rc2.metric("Pipeline", "MONAI Preprocessor")
                rc3.metric("Model Status", "Not Configured")
                rc4.metric("Execution Time", f"{res['inference_time_sec']} sec")

                st.markdown("---")
                st.markdown("## MONAI Scan Inspection")
                t_orig, t_proc, t_guide = st.tabs(["Original Radiograph", "MONAI Preprocessed Tensor Slice", "Model Weights Setup Guide"])
                imgs = res.get("images", {})
                with t_orig:
                    st.image(imgs.get("original", st.session_state.active_image), caption=st.session_state.active_sample_title or "Original Input", use_container_width=True)
                with t_proc:
                    if imgs.get("preprocessed"):
                        st.image(imgs.get("preprocessed"), caption="Normalized 224×224 MONAI Tensor Representation (PyTorch float32)", use_container_width=True)
                with t_guide:
                    st.info(f"**Weights File Location:**\n`{res.get('instructions')}`\n\nPlace a trained DenseNet121 checkpoint (`.pth`) at this path to enable real neural diagnostic inference and Grad-CAM generation.")

                # Report Generation
                st.markdown("---")
                st.markdown("### Diagnostic Preprocessing Report")
                if st.button("📄 Generate Report (PDF)", key="btn_pdf_xray_export"):
                    pdf_bytes = create_medical_pdf(
                        modality="Chest X-ray",
                        model_name=res["model_name"],
                        status="Preprocessing Pipeline Verified (Model Not Configured)",
                        inference_time=res["inference_time_sec"],
                        findings=[],
                        original_image=st.session_state.active_image,
                        viz_image=imgs.get("preprocessed"),
                        is_demo=True
                    )
                    st.download_button(
                        label="⬇️ Download SWASTHYA AI Analysis Report (PDF)",
                        data=pdf_bytes,
                        file_name="Swasthya_Chest_XRay_Report.pdf",
                        mime="application/pdf",
                        key="dl_xray_pdf_btn"
                    )
                    st.success("Report generated successfully!")
            else:
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Modality", "Chest X-ray")
                rc2.metric("Model", res["model_name"].split()[0])
                rc3.metric("Status", "Real Pretrained Model")
                rc4.metric("Inference", f"{res['inference_time_sec']} sec")

                # Findings Table
                st.markdown("## AI Findings")
                for f in res["findings"]:
                    cols = st.columns([3, 4, 2])
                    cols[0].markdown(f"**{f['finding']}**")
                    with cols[1]:
                        st.progress(f["probability"], text=f"{f['probability']*100:.1f}%")
                    status_color = "#ef4444" if f["status"] == "HIGH" else ("#f59e0b" if f["status"] == "MODERATE" else "#10b981")
                    cols[2].markdown(f"<span style='color:{status_color}; font-weight:700;'>{f['status']}</span>", unsafe_allow_html=True)

                # Explainability Section
                st.markdown("---")
                st.markdown("## AI Visualization (Explainability)")
                t_orig, t_heat, t_over = st.tabs(["Original", "Heatmap", "Overlay"])
                imgs = res.get("images", {})
                with t_orig:
                    st.image(imgs.get("original", st.session_state.active_image), caption="Original Radiograph", use_container_width=True)
                with t_heat:
                    st.image(imgs.get("heatmap"), caption="Grad-CAM Activation Heatmap", use_container_width=True)
                with t_over:
                    st.image(imgs.get("overlay"), caption="Grad-CAM Blended Overlay", use_container_width=True)

                # Report Generation
                st.markdown("---")
                st.markdown("### Diagnostic Report")
                if st.button("📄 Generate Report (PDF)", key="btn_pdf_xray_export"):
                    pdf_bytes = create_medical_pdf(
                        modality="Chest X-ray",
                        model_name=res["model_name"],
                        status="Real Pretrained Model",
                        inference_time=res["inference_time_sec"],
                        findings=res["findings"],
                        original_image=st.session_state.active_image,
                        viz_image=imgs.get("overlay"),
                        is_demo=False
                    )
                    st.download_button(
                        label="⬇️ Download SWASTHYA AI Analysis Report (PDF)",
                        data=pdf_bytes,
                        file_name="Swasthya_Chest_XRay_Report.pdf",
                        mime="application/pdf",
                        key="dl_xray_pdf_btn"
                    )
                    st.success("Report generated successfully!")


# ----------------- MODALITY: ULTRASOUND -----------------
else:
    st.markdown("## 🔬 Ultrasound Analysis")

    task_choice = st.radio(
        "Select Ultrasound Diagnostic Task:",
        ["Classification", "Segmentation"],
        index=0 if st.session_state.us_task_mode == "Classification" else 1,
        horizontal=True,
        key="us_task_selector"
    )
    st.session_state.us_task_mode = task_choice

    u_col1, u_col2 = st.columns([3, 1])
    with u_col1:
        uploaded_us = st.file_uploader(
            "Upload Ultrasound Scan (PNG/JPG/JPEG):",
            type=["png", "jpg", "jpeg"],
            key="custom_us_upload"
        )
        if uploaded_us is not None:
            st.session_state.active_image = Image.open(uploaded_us)
            st.session_state.active_sample_title = f"Uploaded: {uploaded_us.name}"
            st.session_state.current_result = None

    with u_col2:
        if st.session_state.active_sample_title:
            st.info(f"**Loaded Image:**\n{st.session_state.active_sample_title}")
            if "Demo" in st.session_state.active_sample_title:
                st.caption("Data type: Synthetic Demo (Technical Test Data)")

    if st.session_state.active_image is not None:
        c_left_us, c_right_us = st.columns([1, 1])
        with c_left_us:
            st.markdown("### Original Ultrasound")
            st.image(st.session_state.active_image, caption=st.session_state.active_sample_title or "B-Mode Sonogram", use_container_width=True)

        with c_right_us:
            st.markdown("<br/><br/>", unsafe_allow_html=True)
            analyze_us = st.button(f"⚡ Analyze Ultrasound ({task_choice})", type="primary", use_container_width=True, key="btn_run_us_analysis")

            if analyze_us:
                with st.spinner(f"Executing MONAI {task_choice} pipeline..."):
                    st.session_state.current_result = us_engine.analyze(
                        st.session_state.active_image,
                        task=task_choice,
                        demo_mode=st.session_state.demo_mode
                    )

        if st.session_state.current_result is not None:
            res_us = st.session_state.current_result
            st.markdown("---")

            # MONAI Preprocessing Verification Checklist
            st.markdown("### MONAI Preprocessing Verification")
            tele_us = res_us.get("telemetry", {})
            up_cols = st.columns(4)
            up_cols[0].markdown(f"<div class='check-item'>✓ Image loaded ({tele_us.get('orig_size')})</div>", unsafe_allow_html=True)
            up_cols[1].markdown(f"<div class='check-item'>✓ Resized ({tele_us.get('resized')})</div>", unsafe_allow_html=True)
            up_cols[2].markdown(f"<div class='check-item'>✓ Intensity normalized ({tele_us.get('normalized')})</div>", unsafe_allow_html=True)
            up_cols[3].markdown(f"<div class='check-item'>✓ Tensor created ({tele_us.get('tensor_shape')})</div>", unsafe_allow_html=True)

            st.markdown("<br/>", unsafe_allow_html=True)

            if res_us["status"] in ("MODEL_NOT_CONFIGURED", "NOT_CONFIGURED"):
                st.markdown(
                    f"""
                    <div style="background-color: #1e293b; border-left: 4px solid #f59e0b; padding: 14px 18px; border-radius: 8px; margin: 12px 0 16px 0; border: 1px solid #334155;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="color: #fbbf24; font-weight: 700; font-size: 14px;">⚠️ Model not configured — preprocessing/demo pipeline only.</span>
                            <span style="background-color: #f59e0b; color: #000; font-size: 10px; font-weight: 800; padding: 2px 6px; border-radius: 4px;">DEMO DATA — NOT FOR CLINICAL USE</span>
                        </div>
                        <p style="color: #cbd5e1; font-size: 13px; margin: 4px 0 0 0;">
                            The MONAI ultrasound preprocessing pipeline (channel formatting, bilinear resize to (256, 256), min-max intensity normalization, and PyTorch float32 tensor conversion) was executed and verified.
                        </p>
                        <p style="color: #94a3b8; font-size: 12px; margin: 6px 0 0 0;">
                            <b>Strict Clinical Integrity:</b> In compliance with medical AI safety standards, sonographic lesion findings and segmentations are <u>never fabricated</u> without a validated, loaded neural network model.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Modality", f"Ultrasound ({task_choice})")
                rc2.metric("Pipeline", "MONAI Preprocessor")
                rc3.metric("Model Status", "Not Configured")
                rc4.metric("Execution Time", f"{res_us['inference_time_sec']} sec")

                st.markdown("---")
                st.markdown("## MONAI Scan Inspection")
                t_orig, t_proc, t_guide = st.tabs(["Original Sonogram", "MONAI Preprocessed Tensor Slice", "Model Weights Setup Guide"])
                imgs_us = res_us.get("images", {})
                with t_orig:
                    st.image(imgs_us.get("original", st.session_state.active_image), caption=st.session_state.active_sample_title or "Original Input", use_container_width=True)
                with t_proc:
                    if imgs_us.get("preprocessed"):
                        st.image(imgs_us.get("preprocessed"), caption="Normalized 256×256 MONAI Tensor Representation (PyTorch float32)", use_container_width=True)
                with t_guide:
                    st.info(f"**Weights File Location:**\n`{res_us.get('instructions')}`\n\nPlace validated ultrasound weights checkpoint (`.pth`) at this path to enable real neural diagnostic inference.")

                # Report Generation
                st.markdown("---")
                st.markdown("### Diagnostic Preprocessing Report")
                if st.button("📄 Generate Report (PDF)", key="btn_pdf_us_export"):
                    pdf_bytes = create_medical_pdf(
                        modality=f"Ultrasound ({task_choice})",
                        model_name=res_us["model_name"],
                        status="Preprocessing Pipeline Verified (Model Not Configured)",
                        inference_time=res_us["inference_time_sec"],
                        findings=[],
                        original_image=st.session_state.active_image,
                        viz_image=imgs_us.get("preprocessed"),
                        is_demo=True
                    )
                    st.download_button(
                        label="⬇️ Download SWASTHYA AI Analysis Report (PDF)",
                        data=pdf_bytes,
                        file_name="Swasthya_Ultrasound_Report.pdf",
                        mime="application/pdf",
                        key="dl_us_pdf_btn"
                    )
                    st.success("Report generated successfully!")
            else:
                st.markdown("### AI ANALYSIS RESULT")
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Modality", f"Ultrasound ({task_choice})")
                rc2.metric("Model", res_us["model_name"].split()[0])
                rc3.metric("Status", "Real Pretrained Model")
                rc4.metric("Inference", f"{res_us['inference_time_sec']} sec")

                # Findings & Visualization
                st.markdown("## AI Findings")
                if task_choice == "Segmentation":
                    st.metric("Segmented Lesion Area Percentage", f"{res_us.get('lesion_area_pct', 0.0)}%")
                    st.metric("Lesion Pixel Count", f"{res_us.get('lesion_pixels', 0):,} pixels")

                    st.markdown("---")
                    st.markdown("## AI Visualization (UNet Lesion Masking)")
                    imgs_us = res_us.get("images", {})
                    t_orig, t_mask, t_over = st.tabs(["Original", "Mask", "Overlay"])
                    with t_orig:
                        st.image(imgs_us.get("original", st.session_state.active_image), caption="Original Sonogram", use_container_width=True)
                    with t_mask:
                        st.image(imgs_us.get("mask"), caption="Binary UNet Lesion Mask", use_container_width=True)
                    with t_over:
                        st.image(imgs_us.get("overlay"), caption="Contour & Alpha Blended Lesion Fill", use_container_width=True)

                    viz_for_rep = imgs_us.get("overlay")
                else:
                    for f in res_us["findings"]:
                        cols = st.columns([3, 4, 2])
                        cols[0].markdown(f"**{f['finding']}**")
                        with cols[1]:
                            st.progress(f["probability"], text=f"{f['probability']*100:.1f}%")
                        status_color = "#ef4444" if f["status"] == "HIGH" else "#10b981"
                        cols[2].markdown(f"<span style='color:{status_color}; font-weight:700;'>{f['status']}</span>", unsafe_allow_html=True)

                    st.markdown("---")
                    st.markdown("## AI Visualization (Explainability)")
                    imgs_us = res_us.get("images", {})
                    t_orig, t_heat, t_over = st.tabs(["Original", "Heatmap", "Overlay"])
                    with t_orig:
                        st.image(imgs_us.get("original", st.session_state.active_image), caption="Original Sonogram", use_container_width=True)
                    with t_heat:
                        st.image(imgs_us.get("heatmap"), caption="Activation Heatmap", use_container_width=True)
                    with t_over:
                        st.image(imgs_us.get("overlay"), caption="Blended Heatmap Overlay", use_container_width=True)

                    viz_for_rep = imgs_us.get("overlay")

                # Report Generation
                st.markdown("---")
                st.markdown("### Diagnostic Report")
                if st.button("📄 Generate Report (PDF)", key="btn_pdf_us_export"):
                    pdf_bytes = create_medical_pdf(
                        modality=f"Ultrasound ({task_choice})",
                        model_name=res_us["model_name"],
                        status="Real Pretrained Model",
                        inference_time=res_us["inference_time_sec"],
                        findings=res_us["findings"],
                        original_image=st.session_state.active_image,
                        viz_image=viz_for_rep,
                        is_demo=False
                    )
                    st.download_button(
                        label="⬇️ Download SWASTHYA AI Analysis Report (PDF)",
                        data=pdf_bytes,
                        file_name="Swasthya_Ultrasound_Report.pdf",
                        mime="application/pdf",
                        key="dl_us_pdf_btn"
                    )
                    st.success("Report generated successfully!")

