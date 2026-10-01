"""
Chest X-Ray Diagnostic Page
Handles radiograph upload, MONAI pipeline execution, Grad-CAM visualization,
and PDF report export.
"""

import io
from datetime import datetime
from pathlib import Path
from PIL import Image
import streamlit as st

from ai.xray.inference import XRayPipeline, XRayResult
from ai.preprocessing import MedicalPreprocessor
from utils.report import create_medical_pdf
from app.components.image_viewer import render_image_viewer
from app.components.prediction_card import (
    render_model_badge,
    render_unconfigured_instructions,
    render_findings_table,
    render_model_telemetry,
)
from app.components.heatmap_viewer import render_gradcam_viewer
from app.components.report_viewer import render_report_section


def render_xray_page(demo_mode: bool):
    """Renders the Chest X-Ray AI diagnostic interface."""
    st.markdown("## 🫁 Chest X-Ray Diagnostic Analysis")
    st.caption("Deep-learning multi-finding radiograph analysis and explainability powered by MONAI DenseNet121.")

    pipeline = XRayPipeline()
    sample_dir = Path(__file__).resolve().parent.parent.parent / "data" / "sample"

    # Input Selection: Upload or Preloaded Sample
    selected_sample = st.session_state.get("selected_sample_path", "")
    has_sample = bool(selected_sample and "xray" in selected_sample)

    input_source = st.radio(
        "Choose Scan Source:",
        ["Upload Radiograph", "Load Bundled Sample"],
        index=1 if has_sample else 0,
        horizontal=True,
        key="xray_source_radio"
    )

    image_to_analyze = None
    scan_filename = "xray_scan.png"

    if input_source == "Upload Radiograph":
        uploaded_file = st.file_uploader(
            "Upload Chest X-Ray (PNG, JPG, TIFF, DICOM)",
            type=["png", "jpg", "jpeg", "tiff", "tif", "bmp"],
            help="Images are processed strictly on your local hardware.",
            key="xray_uploader"
        )
        if uploaded_file is not None:
            image_to_analyze = uploaded_file
            scan_filename = uploaded_file.name
    else:
        sample_options = ["Chest X-ray — Patient Scan (Uploaded)", "Chest X-ray — Normal", "Chest X-ray — Pneumonia Demo"]
        default_idx = 0
        if "pneumonia" in selected_sample:
            default_idx = 2
        elif "normal" in selected_sample:
            default_idx = 1
        elif "chest_user" in selected_sample:
            default_idx = 0

        sample_choice = st.selectbox(
            "Select Bundled Demonstration Image:",
            sample_options,
            index=default_idx,
            key="xray_sample_select"
        )
        if "Patient Scan" in sample_choice:
            sample_file = sample_dir / "xray_chest_user.jpg"
        elif "Normal" in sample_choice:
            sample_file = sample_dir / "xray_normal.png"
        else:
            sample_file = sample_dir / "xray_pneumonia.png"

        if sample_file.exists():
            image_to_analyze = str(sample_file)
            scan_filename = sample_file.name
            st.image(str(sample_file), caption=f"Loaded Demo Sample: {sample_choice}", width=240)
            st.caption("Data source: Clinical Radiograph Scan • DEMO MODE — LOCAL AI ANALYSIS")

    st.markdown("<br/>", unsafe_allow_html=True)
    analyze_btn = st.button(
        "⚡ Analyze Chest Radiograph",
        type="primary",
        disabled=(image_to_analyze is None),
        key="btn_analyze_xray"
    )

    auto_run = st.session_state.get("auto_analyze", False) and image_to_analyze is not None
    if auto_run:
        st.session_state.auto_analyze = False

    analyze_clicked = analyze_btn or auto_run


    if analyze_clicked and image_to_analyze is not None:
        try:
            if isinstance(image_to_analyze, (str, Path)):
                pil_img = Image.open(image_to_analyze).convert("RGB")
            else:
                image_to_analyze.seek(0)
                pil_img = Image.open(image_to_analyze).convert("RGB")
        except Exception as e:
            st.error(f"❌ Failed to load radiograph: {e}")
            return

        with st.spinner("Executing MONAI preprocessing pipeline..."):
            preprocessor = MedicalPreprocessor(spatial_size=(224, 224))
            input_tensor, telemetry = preprocessor.process(pil_img)
            preprocessed_pil = preprocessor.tensor_to_pil(input_tensor)

        # Preprocessing status display
        st.markdown("---")
        st.markdown("### MONAI Preprocessing Verification")
        p_cols = st.columns(4)
        p_cols[0].markdown(f"<div class='check-item'>✓ Image loaded ({telemetry.get('orig_size')})</div>", unsafe_allow_html=True)
        p_cols[1].markdown(f"<div class='check-item'>✓ Resized ({telemetry.get('resized')})</div>", unsafe_allow_html=True)
        p_cols[2].markdown(f"<div class='check-item'>✓ Intensity normalized ({telemetry.get('normalized')})</div>", unsafe_allow_html=True)
        p_cols[3].markdown(f"<div class='check-item'>✓ Tensor created ({telemetry.get('tensor_shape')})</div>", unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # If a real trained model is configured, run real inference
        if pipeline.is_model_configured():
            with st.spinner("Running real MONAI DenseNet121 inference..."):
                result: XRayResult = pipeline.run(
                    image_input=image_to_analyze,
                    filename=scan_filename,
                    demo_mode=False
                )

            # Handle Error / Validation failures
            if result.status == "ERROR":
                st.error(f"❌ Validation / Inference Error: {result.message}")
                return

            # Record to session history
            if "analysis_history" not in st.session_state:
                st.session_state.analysis_history = []
            st.session_state.analysis_history.append({
                "timestamp": result.timestamp,
                "modality": "Chest X-Ray",
                "finding": result.top_finding,
                "probability": result.top_probability,
                "status": result.status,
            })

            # Display Diagnostic Output
            st.markdown("---")
            render_model_badge(result.status, result.is_demo)

            # Telemetry Row
            render_model_telemetry(
                model_name=result.model_name,
                version=result.model_version,
                device=result.device_name,
                inference_time_ms=result.inference_time_ms
            )

            st.markdown("---")

            # Layout: Findings Table and Original Image
            col_img, col_findings = st.columns([1, 1])

            with col_img:
                render_image_viewer(
                    image=result.original_image,
                    title="Original Radiograph",
                    metadata=result.validation_meta,
                    allow_windowing=True,
                    key_prefix="xray_main"
                )

            with col_findings:
                render_findings_table(result.findings)

            st.markdown("---")

            # Grad-CAM Explainability Section
            render_gradcam_viewer(
                original=result.original_image,
                heatmap=result.heatmap,
                overlay=result.overlay,
                title="Grad-CAM Pathology Localization"
            )

            st.markdown("---")

            # Clinical Report Generation
            reports_dir = str(Path(__file__).resolve().parent.parent.parent / "reports")
            render_report_section(
                modality="Chest X-Ray",
                findings=result.findings,
                original_image=result.original_image,
                viz_image=result.overlay,
                viz_caption="Grad-CAM Attention Overlay",
                model_name=result.model_name,
                model_version=result.model_version,
                timestamp=result.timestamp,
                is_demo=result.is_demo,
                reports_dir=reports_dir
            )
        else:
            # If no trained model is configured, show preprocessing status & never fabricate findings
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
            rc4.metric("Engine", "PyTorch (float32)")

            st.markdown("---")
            st.markdown("## MONAI Scan Inspection")
            t_orig, t_proc, t_guide = st.tabs(["Original Radiograph", "MONAI Preprocessed Tensor Slice", "Model Weights Setup Guide"])
            with t_orig:
                st.image(pil_img, caption=scan_filename or "Original Input Scan", use_container_width=True)
            with t_proc:
                st.image(preprocessed_pil, caption="Normalized 224×224 MONAI Tensor Representation (PyTorch float32)", use_container_width=True)
            with t_guide:
                render_unconfigured_instructions(
                    "Chest X-ray model weights are not configured.",
                    f"Place a trained DenseNet121 checkpoint (.pth) at:\nswasthya-monai-imaging/models/xray/chexnet_monai_densenet121.pth\nto enable real neural diagnostic inference and Grad-CAM generation."
                )

            # Record to session history
            if "analysis_history" not in st.session_state:
                st.session_state.analysis_history = []
            st.session_state.analysis_history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "modality": "Chest X-Ray",
                "finding": "Preprocessing Verified (Model Not Configured)",
                "probability": 0.0,
                "status": "MODEL_NOT_CONFIGURED",
            })

            # Report Generation
            st.markdown("---")
            st.markdown("### Diagnostic Preprocessing Report")
            if st.button("📄 Generate Report (PDF)", key="btn_pdf_xray_export"):
                pdf_bytes = create_medical_pdf(
                    modality="Chest X-ray",
                    model_name="Swasthya-MONAI-DenseNet121 (Preprocessing Only)",
                    status="Preprocessing Pipeline Verified (Model Not Configured)",
                    inference_time=0.05,
                    findings=[],
                    original_image=pil_img,
                    viz_image=preprocessed_pil,
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
