"""
Dashboard UI Page
Central control center displaying modality statuses, compute hardware, and demo mode flags.
"""

import streamlit as st
from pathlib import Path
from PIL import Image

from ai.common.device import get_device_info
from ai.xray.inference import XRayPipeline
from ai.ultrasound.inference import UltrasoundPipeline


def render_dashboard(demo_mode: bool):
    """Renders the comprehensive medical imaging operations dashboard."""
    st.markdown("## 🏥 Medical Imaging Operations Dashboard")
    st.caption("AI-assisted multi-modal diagnostic radiologic computing powered by MONAI & PyTorch.")

    # Device & Infrastructure Overview
    dev_info = get_device_info()
    dev_name = dev_info["device_name"]
    dev_type = dev_info["device_type"]
    cuda_avail = dev_info["cuda_available"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Compute Device",
            value=f"{dev_type}",
            delta="CUDA Accelerated" if cuda_avail else "CPU Mode"
        )
    with col2:
        st.metric(
            label="Hardware Engine",
            value=dev_name[:16] if dev_name != "CPU" else "Host CPU",
            help=f"Processor: {dev_name}"
        )
    with col3:
        st.metric(
            label="MONAI Core",
            value="v1.6.0",
            help="Medical Open Network for AI"
        )
    with col4:
        st.metric(
            label="Operation Mode",
            value="DEMO SIMULATION" if demo_mode else "CLINICAL READY",
            delta="⚠️ Simulation Active" if demo_mode else "Real Weights Only"
        )

    st.markdown("---")

    # Modality Architecture Cards
    st.markdown("### 🧬 Modality Pipelines")

    xray_pipe = XRayPipeline()
    us_pipe = UltrasoundPipeline()

    xray_configured = xray_pipe.is_model_configured()
    us_cls_configured = us_pipe.is_configured("classification")
    us_seg_configured = us_pipe.is_configured("segmentation")

    m_col1, m_col2 = st.columns(2)

    with m_col1:
        st.markdown(
            f"""
            <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 18px; margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h3 style="margin: 0; color: #38bdf8;">🫁 Chest X-Ray AI</h3>
                    <span style="background-color: {'#065f46' if xray_configured else ('#78350f' if demo_mode else '#7f1d1d')}; color: white; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 700;">
                        {'REAL WEIGHTS' if xray_configured else ('DEMO READY' if demo_mode else 'NOT CONFIGURED')}
                    </span>
                </div>
                <p style="color: #cbd5e1; font-size: 13px; margin: 8px 0 12px 0;">
                    DenseNet121 architecture for multi-pathology detection (Pneumonia, Cardiomegaly, Pleural Effusion, Atelectasis, etc.) with MONAI Grad-CAM explainability.
                </p>
                <div style="font-size: 12px; color: #94a3b8;">
                    <strong>Input:</strong> 224×224 Grayscale &nbsp;|&nbsp; <strong>Backbone:</strong> MONAI DenseNet121
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m_col2:
        st.markdown(
            f"""
            <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 18px; margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h3 style="margin: 0; color: #38bdf8;">🔬 Ultrasound Sonography AI</h3>
                    <span style="background-color: {'#065f46' if (us_cls_configured or us_seg_configured) else ('#78350f' if demo_mode else '#7f1d1d')}; color: white; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 700;">
                        {'REAL WEIGHTS' if (us_cls_configured or us_seg_configured) else ('DEMO READY' if demo_mode else 'NOT CONFIGURED')}
                    </span>
                </div>
                <p style="color: #cbd5e1; font-size: 13px; margin: 8px 0 12px 0;">
                    Dual-task sonography pipeline supporting both 3-class pathology classification and MONAI UNet lesion segmentation with area quantification.
                </p>
                <div style="font-size: 12px; color: #94a3b8;">
                    <strong>Input:</strong> 256×256 B-Mode &nbsp;|&nbsp; <strong>Backbone:</strong> MONAI UNet + DenseNet
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Future Modalities Roadmap (CT & MRI)
    st.markdown("### 🔭 Future Modality Architecture (Modular Extensibility)")
    fut_col1, fut_col2 = st.columns(2)

    with fut_col1:
        st.markdown(
            """
            <div style="background-color: #0f172a; border: 1px dashed #475569; border-radius: 10px; padding: 16px;">
                <h4 style="margin: 0; color: #94a3b8;">🧠 Magnetic Resonance Imaging (MRI)</h4>
                <p style="color: #64748b; font-size: 12px; margin: 6px 0;">
                    Planned 3D volumetric segmentation using MONAI V-Net & SwinUNETR for multi-parametric neuroimaging (T1, T2, FLAIR).
                </p>
                <span style="color: #0284c7; font-size: 11px; font-weight: 600;">Status: Architectural Interface Ready</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with fut_col2:
        st.markdown(
            """
            <div style="background-color: #0f172a; border: 1px dashed #475569; border-radius: 10px; padding: 16px;">
                <h4 style="margin: 0; color: #94a3b8;">🩻 Computed Tomography (CT)</h4>
                <p style="color: #64748b; font-size: 12px; margin: 6px 0;">
                    Planned Hounsfield Unit (HU) windowing and 3D pulmonary nodule detection with MONAI 3D DenseNet.
                </p>
                <span style="color: #0284c7; font-size: 11px; font-weight: 600;">Status: Architectural Interface Ready</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # User-Provided Clinical Scans Section
    usr_xray = sample_dir / "xray_chest_user.jpg"
    usr_us = sample_dir / "ultrasound_kidney_user.jpg"

    if usr_xray.exists() or usr_us.exists():
        st.markdown("## 📥 PATIENT SCANS READY FOR ANALYSIS")
        st.markdown("<p style='color:#94a3b8; font-size:14px; margin-top:-10px; margin-bottom:16px;'>High-resolution patient scans loaded and ready for immediate AI diagnosis.</p>", unsafe_allow_html=True)
        u_col1, u_col2 = st.columns(2)

        with u_col1:
            if usr_xray.exists():
                st.image(str(usr_xray), use_container_width=True)
                st.markdown(
                    """
                    <div style="background-color: #1e293b; border: 1px solid #0284c7; border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-size: 11px; font-weight: 700; color: #38bdf8; text-transform: uppercase;">Chest Radiograph</span>
                            <span style="background-color: #0284c7; color: #fff; font-size: 10px; font-weight: 800; padding: 2px 6px; border-radius: 4px;">USER UPLOAD</span>
                        </div>
                        <h4 style="margin: 0 0 4px 0; font-size: 14px; color: #f8fafc;">Patient Chest & Rib Radiograph</h4>
                        <div style="font-size: 11px; color: #94a3b8; margin-bottom: 6px;"><b>Source:</b> Clinical Right Hemithorax Radiograph</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if st.button("⚡ Analyze Patient Chest X-Ray", key="dash_use_usr_xray", use_container_width=True, type="primary"):
                    st.session_state.main_nav_choice = "🫁 Chest X-ray"
                    st.session_state.selected_sample_path = str(usr_xray)
                    st.session_state.selected_sample_name = "Chest X-ray — Patient Scan (Uploaded)"
                    st.session_state.auto_analyze = True
                    st.rerun()

        with u_col2:
            if usr_us.exists():
                st.image(str(usr_us), use_container_width=True)
                st.markdown(
                    """
                    <div style="background-color: #1e293b; border: 1px solid #0284c7; border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-size: 11px; font-weight: 700; color: #38bdf8; text-transform: uppercase;">Ultrasound</span>
                            <span style="background-color: #0284c7; color: #fff; font-size: 10px; font-weight: 800; padding: 2px 6px; border-radius: 4px;">USER UPLOAD</span>
                        </div>
                        <h4 style="margin: 0 0 4px 0; font-size: 14px; color: #f8fafc;">Patient Kidney Sonogram (Doppler & B-Mode)</h4>
                        <div style="font-size: 11px; color: #94a3b8; margin-bottom: 6px;"><b>Source:</b> Multi-quadrant Renal Sonogram</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if st.button("⚡ Analyze Patient Kidney Ultrasound", key="dash_use_usr_us", use_container_width=True, type="primary"):
                    st.session_state.main_nav_choice = "🔬 Ultrasound"
                    st.session_state.selected_sample_path = str(usr_us)
                    st.session_state.selected_sample_name = "Ultrasound — Kidney Scan (Uploaded)"
                    st.session_state.us_task_mode = "Segmentation"
                    st.session_state.auto_analyze = True
                    st.rerun()

        st.markdown("---")

    # Pre-loaded Demo Scans
    st.markdown("## PRE-LOADED DEMO SCANS")
    st.markdown("<p style='color:#94a3b8; font-size:14px; margin-top:-10px; margin-bottom:16px;'>Test the imaging pipelines instantly using bundled demonstration images.</p>", unsafe_allow_html=True)

    s_col1, s_col2, s_col3, s_col4 = st.columns(4)

    card_style = """
    <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 12px; margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span style="font-size: 11px; font-weight: 700; color: #38bdf8; text-transform: uppercase;">{modality}</span>
            <span style="background-color: #f59e0b; color: #000; font-size: 10px; font-weight: 800; padding: 2px 6px; border-radius: 4px;">DEMO SAMPLE</span>
        </div>
        <h4 style="margin: 0 0 4px 0; font-size: 14px; color: #f8fafc;">{name}</h4>
        <div style="font-size: 11px; color: #94a3b8; margin-bottom: 6px;"><b>Data type:</b> Synthetic Demo</div>
        <div style="background-color: #3f1d1d; color: #fca5a5; font-size: 10px; font-weight: 700; padding: 3px 6px; border-radius: 4px; text-align: center; margin-bottom: 6px; border: 1px solid #7f1d1d;">
            DEMO DATA — NOT FOR CLINICAL USE
        </div>
    </div>
    """

    with s_col1:
        norm_xray = sample_dir / "xray_normal.png"
        if norm_xray.exists():
            st.image(str(norm_xray), use_container_width=True)
            st.markdown(card_style.format(modality="Chest X-ray", name="Normal"), unsafe_allow_html=True)
            if st.button("Use Sample", key="dash_use_xray_norm", use_container_width=True):
                st.session_state.main_nav_choice = "🫁 Chest X-ray"
                st.session_state.selected_sample_path = str(norm_xray)
                st.session_state.selected_sample_name = "Chest X-ray — Normal"
                st.session_state.auto_analyze = True
                st.rerun()

    with s_col2:
        pneu_xray = sample_dir / "xray_pneumonia.png"
        if pneu_xray.exists():
            st.image(str(pneu_xray), use_container_width=True)
            st.markdown(card_style.format(modality="Chest X-ray", name="Pneumonia Demo"), unsafe_allow_html=True)
            if st.button("Use Sample", key="dash_use_xray_pneu", use_container_width=True):
                st.session_state.main_nav_choice = "🫁 Chest X-ray"
                st.session_state.selected_sample_path = str(pneu_xray)
                st.session_state.selected_sample_name = "Chest X-ray — Pneumonia Demo"
                st.session_state.auto_analyze = True
                st.rerun()

    with s_col3:
        us_norm = sample_dir / "ultrasound_normal.png"
        if us_norm.exists():
            st.image(str(us_norm), use_container_width=True)
            st.markdown(card_style.format(modality="Ultrasound", name="Normal B-mode Demo"), unsafe_allow_html=True)
            if st.button("Use Sample", key="dash_use_us_norm", use_container_width=True):
                st.session_state.main_nav_choice = "🔬 Ultrasound"
                st.session_state.selected_sample_path = str(us_norm)
                st.session_state.selected_sample_name = "Ultrasound — Normal B-mode Demo"
                st.session_state.us_task_mode = "Classification"
                st.session_state.auto_analyze = True
                st.rerun()

    with s_col4:
        us_les = sample_dir / "ultrasound_lesion.png"
        if us_les.exists():
            st.image(str(us_les), use_container_width=True)
            st.markdown(card_style.format(modality="Ultrasound", name="Nodule Demo"), unsafe_allow_html=True)
            if st.button("Use Sample", key="dash_use_us_les", use_container_width=True):
                st.session_state.main_nav_choice = "🔬 Ultrasound"
                st.session_state.selected_sample_path = str(us_les)
                st.session_state.selected_sample_name = "Ultrasound — Nodule Demo"
                st.session_state.us_task_mode = "Segmentation"
                st.session_state.auto_analyze = True
                st.rerun()


