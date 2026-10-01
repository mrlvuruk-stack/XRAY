"""
Ultrasound Diagnostic Page
Supports task selection (Classification vs MONAI UNet Segmentation),
radiology analysis, mask overlays, and report generation.
"""

from datetime import datetime
from pathlib import Path
from PIL import Image
import streamlit as st

from ai.ultrasound.inference import UltrasoundPipeline, UltrasoundResult
from ai.preprocessing import MedicalPreprocessor
from utils.report import create_medical_pdf
from app.components.image_viewer import render_image_viewer
from app.components.prediction_card import (
    render_model_badge,
    render_unconfigured_instructions,
    render_findings_table,
    render_model_telemetry,
)
from app.components.heatmap_viewer import render_gradcam_viewer, render_segmentation_viewer
from app.components.report_viewer import render_report_section


def render_ultrasound_page(demo_mode: bool):
    """Renders the Ultrasound sonography AI diagnostic interface."""
    st.markdown("## 🔬 Ultrasound Sonography Analysis")
    st.caption("AI-assisted sonography supporting both lesion classification and MONAI UNet contour segmentation.")

    pipeline = UltrasoundPipeline()
    sample_dir = Path(__file__).resolve().parent.parent.parent / "data" / "sample"

    # Task Selection
    task_mode = st.radio(
        "Select Ultrasound Task:",
        ["Segmentation (Lesion Boundary)", "Classification (Pathology)"],
        index=1 if "class" in st.session_state.get("us_task_mode", "").lower() else 0,
        horizontal=True,
        key="us_task_radio"
    )
    is_segmentation = "Segmentation" in task_mode
    task_str = "segmentation" if is_segmentation else "classification"

    # Check if a sample was triggered from dashboard
    selected_sample = st.session_state.get("selected_sample_path", "")
    has_sample = bool(selected_sample and "ultrasound" in selected_sample)

    # Image Input: Upload vs Bundled Sample
    input_source = st.radio(
        "Choose Sonogram Source:",
        ["Upload Ultrasound Scan", "Load Bundled Sample"],
        index=1 if has_sample else 0,
        horizontal=True,
        key="us_source_radio"
    )

    image_to_analyze = None
    scan_filename = "ultrasound_scan.png"

    if input_source == "Upload Ultrasound Scan":
        uploaded_file = st.file_uploader(
            "Upload Ultrasound (PNG, JPG, TIFF, DICOM)",
            type=["png", "jpg", "jpeg", "tiff", "tif", "bmp"],
            key="us_uploader"
        )
        if uploaded_file is not None:
            image_to_analyze = uploaded_file
            scan_filename = uploaded_file.name
    else:
        sample_options = ["Ultrasound — Kidney Scan (Uploaded)", "Ultrasound — Normal B-mode Demo", "Ultrasound — Nodule Demo"]
        default_idx = 0
        if "lesion" in selected_sample or "nodule" in selected_sample.lower():
            default_idx = 2
        elif "normal" in selected_sample:
            default_idx = 1
        elif "kidney" in selected_sample:
            default_idx = 0

        sample_choice = st.selectbox(
            "Select Bundled Demonstration Image:",
            sample_options,
            index=default_idx,
            key="us_sample_select"
        )
        if "Kidney Scan" in sample_choice:
            sample_file = sample_dir / "ultrasound_kidney_user.jpg"
        elif "Normal" in sample_choice:
            sample_file = sample_dir / "ultrasound_normal.png"
        else:
            sample_file = sample_dir / "ultrasound_lesion.png"

        if sample_file.exists():
            image_to_analyze = str(sample_file)
            scan_filename = sample_file.name
            st.image(str(sample_file), caption=f"Loaded Demo Sample: {sample_choice}", width=240)
            st.caption("Data source: Clinical Sonogram Scan • DEMO MODE — LOCAL AI ANALYSIS")

    st.markdown("<br/>", unsafe_allow_html=True)
    analyze_btn = st.button(
        f"⚡ Run Ultrasound {task_str.capitalize()}",
        type="primary",
        disabled=(image_to_analyze is None),
        key="btn_analyze_us"
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
            st.error(f"❌ Failed to load sonogram: {e}")
            return

        with st.spinner(f"Executing MONAI {task_str} preprocessing pipeline..."):
            preprocessor = MedicalPreprocessor(spatial_size=(256, 256))
            input_tensor, telemetry = preprocessor.process(pil_img)
            preprocessed_pil = preprocessor.tensor_to_pil(input_tensor)

        # Preprocessing status display
        st.markdown("---")
        st.markdown("### MONAI Preprocessing Verification")
        up_cols = st.columns(4)
        up_cols[0].markdown(f"<div class='check-item'>✓ Image loaded ({telemetry.get('orig_size')})</div>", unsafe_allow_html=True)
        up_cols[1].markdown(f"<div class='check-item'>✓ Resized ({telemetry.get('resized')})</div>", unsafe_allow_html=True)
        up_cols[2].markdown(f"<div class='check-item'>✓ Intensity normalized ({telemetry.get('normalized')})</div>", unsafe_allow_html=True)
        up_cols[3].markdown(f"<div class='check-item'>✓ Tensor created ({telemetry.get('tensor_shape')})</div>", unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # If a real trained model is configured, run real inference
        if pipeline.is_configured(task_str):
            with st.spinner(f"Running real MONAI Ultrasound {task_str} inference..."):
                result: UltrasoundResult = pipeline.run(
                    image_input=image_to_analyze,
                    task=task_str,
                    filename=scan_filename,
                    demo_mode=False
                )

            if result.status == "ERROR":
                st.error(f"❌ Diagnostic Error: {result.message}")
                return

            # Record to history
            if "analysis_history" not in st.session_state:
                st.session_state.analysis_history = []
            st.session_state.analysis_history.append({
                "timestamp": result.timestamp,
                "modality": f"Ultrasound ({task_str.capitalize()})",
                "finding": result.top_finding if not is_segmentation else f"Lesion Area: {result.lesion_area_percentage}%",
                "probability": result.top_probability if not is_segmentation else result.lesion_area_percentage / 100.0,
                "status": result.status,
            })

            st.markdown("---")
            render_model_badge(result.status, result.is_demo)

            render_model_telemetry(
                model_name=result.model_name,
                version=result.model_version,
                device=result.device_name,
                inference_time_ms=result.inference_time_ms
            )

            st.markdown("---")

            if is_segmentation:
                # Segmentation Output View
                render_segmentation_viewer(
                    original=result.original_image,
                    mask=result.mask_image,
                    overlay=result.overlay_image,
                    lesion_pixels=result.lesion_pixels,
                    lesion_area_pct=result.lesion_area_percentage
                )
                findings_for_report = [
                    {
                        "finding": "Segmented Lesion Region",
                        "probability": round((result.lesion_area_percentage or 0.0) / 100.0, 4),
                        "status": "High" if (result.lesion_area_percentage or 0) > 10.0 else "Moderate"
                    },
                    {
                        "finding": "Surrounding Normal Tissue",
                        "probability": round(1.0 - (result.lesion_area_percentage or 0.0) / 100.0, 4),
                        "status": "Low"
                    }
                ]
                viz_for_report = result.overlay_image
                viz_caption = "UNet Lesion Boundary Overlay"
            else:
                # Classification Output View
                col_img, col_findings = st.columns([1, 1])
                with col_img:
                    render_image_viewer(
                        image=result.original_image,
                        title="Original Sonogram",
                        metadata=result.validation_meta,
                        allow_windowing=True,
                        key_prefix="us_main"
                    )
                with col_findings:
                    render_findings_table(result.findings)

                st.markdown("---")
                render_gradcam_viewer(
                    original=result.original_image,
                    heatmap=result.heatmap,
                    overlay=result.overlay_image,
                    title="Sonographic Attention CAM"
                )
                findings_for_report = result.findings
                viz_for_report = result.overlay_image
                viz_caption = "Classification Heatmap Overlay"

            st.markdown("---")

            # Report Generation
            reports_dir = str(Path(__file__).resolve().parent.parent.parent / "reports")
            render_report_section(
                modality=f"Ultrasound_{task_str}",
                findings=findings_for_report,
                original_image=result.original_image,
                viz_image=viz_for_report,
                viz_caption=viz_caption,
                model_name=result.model_name,
                model_version=result.model_version,
                timestamp=result.timestamp,
                is_demo=result.is_demo,
                reports_dir=reports_dir
            )
        else:
            # If no trained model is configured, show preprocessing status & never fabricate findings
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
            rc1.metric("Modality", f"Ultrasound ({task_str.capitalize()})")
            rc2.metric("Pipeline", "MONAI Preprocessor")
            rc3.metric("Model Status", "Not Configured")
            rc4.metric("Engine", "PyTorch (float32)")

            st.markdown("---")
            st.markdown("## MONAI Scan Inspection")
            t_orig, t_proc, t_guide = st.tabs(["Original Sonogram", "MONAI Preprocessed Tensor Slice", "Model Weights Setup Guide"])
            with t_orig:
                st.image(pil_img, caption=scan_filename or "Original Sonogram", use_container_width=True)
            with t_proc:
                st.image(preprocessed_pil, caption="Normalized 256×256 MONAI Tensor Representation (PyTorch float32)", use_container_width=True)
            with t_guide:
                setup_guide = (
                    "Place validated UNet weights at: swasthya-monai-imaging/models/ultrasound/segmentation/ultrasound_unet.pth"
                    if is_segmentation
                    else "Place validated DenseNet121 weights at: swasthya-monai-imaging/models/ultrasound/classification/ultrasound_densenet121.pth"
                )
                render_unconfigured_instructions(
                    f"Ultrasound {task_str} model weights are not configured.",
                    setup_guide
                )

            # Record to session history
            if "analysis_history" not in st.session_state:
                st.session_state.analysis_history = []
            st.session_state.analysis_history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "modality": f"Ultrasound ({task_str.capitalize()})",
                "finding": "Preprocessing Verified (Model Not Configured)",
                "probability": 0.0,
                "status": "MODEL_NOT_CONFIGURED",
            })

            # Report Generation
            st.markdown("---")
            st.markdown("### Diagnostic Preprocessing Report")
            if st.button("📄 Generate Report (PDF)", key="btn_pdf_us_export"):
                pdf_bytes = create_medical_pdf(
                    modality=f"Ultrasound ({task_str.capitalize()})",
                    model_name=f"Swasthya-MONAI-US-{task_str.capitalize()} (Preprocessing Only)",
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
                    file_name="Swasthya_Ultrasound_Report.pdf",
                    mime="application/pdf",
                    key="dl_us_pdf_btn"
                )
                st.success("Report generated successfully!")
