"""
Swasthya MONAI Medical Imaging Lab - Streamlit Main Application
Standalone Medical AI Platform powered by MONAI + PyTorch.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment configuration
load_dotenv(PROJECT_ROOT / ".env")

from ai.common.device import get_device_info
from ai.xray.inference import XRayPipeline
from ai.ultrasound.inference import UltrasoundPipeline
from app.ui.dashboard import render_dashboard
from app.ui.xray_page import render_xray_page
from app.ui.ultrasound_page import render_ultrasound_page

# Set Streamlit Page Config
st.set_page_config(
    page_title="Swasthya AI | Medical Imaging Lab",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern medical UI aesthetics
st.markdown(
    """
    <style>
    /* Clean typography & header styling */
    .main-header {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #0284c7;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .stMetric {
        background-color: #1e293b;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #334155;
    }
    .stMetric label {
        color: #94a3b8 !important;
        font-size: 13px !important;
    }
    .stMetric div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }
    /* Sidebar aesthetic */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
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
        padding: 12px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize Session State
if "main_nav_choice" not in st.session_state:
    st.session_state.main_nav_choice = "🏠 Dashboard"

if "demo_mode" not in st.session_state:
    env_demo = os.getenv("DEMO_MODE", "true").lower() in ("true", "1", "yes")
    st.session_state.demo_mode = env_demo

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []


def render_sidebar():
    """Renders the standardized hospital laboratory sidebar."""
    with st.sidebar:
        # Branding
        st.markdown(
            """
            <div style="padding: 10px 0 16px 0;">
                <h1 style="color: #38bdf8; margin: 0; font-size: 24px; font-weight: 800; letter-spacing: 0.5px;">SWASTHYA AI</h1>
                <p style="color: #94a3b8; margin: 2px 0 0 0; font-size: 13px; font-weight: 500;">Medical Imaging Lab</p>
                <div style="margin-top: 8px; display: inline-block; background-color: #0369a1; color: white; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 700;">
                    MONAI + PyTorch Engine
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        # Navigation
        st.caption("CLINICAL NAVIGATION")
        nav_choice = st.radio(
            "Go to",
            [
                "🏠 Dashboard",
                "🫁 Chest X-ray",
                "🔬 Ultrasound",
                "📋 Analysis History",
                "⚙️ Settings",
            ],
            key="main_nav_choice",
            label_visibility="collapsed"
        )


        st.markdown("---")

        # Mode Indicator & Quick Toggle
        st.caption("SYSTEM OPERATING MODE")
        st.session_state.demo_mode = st.toggle(
            "Demo Mode (Simulation)",
            value=st.session_state.demo_mode,
            help="When ON, exercises full MONAI pipelines with realistic simulated clinical findings. When OFF, requires verified model weight files."
        )

        if st.session_state.demo_mode:
            st.markdown(
                """
                <div style="background-color: #451a03; border-left: 3px solid #f59e0b; padding: 6px 10px; border-radius: 4px; margin-top: 6px;">
                    <span style="color: #fbbf24; font-size: 11px; font-weight: 700;">⚠️ DEMO ACTIVE</span>
                    <p style="color: #fef3c7; font-size: 11px; margin: 2px 0 0 0;">Simulated results only.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="background-color: #064e3b; border-left: 3px solid #10b981; padding: 6px 10px; border-radius: 4px; margin-top: 6px;">
                    <span style="color: #34d399; font-size: 11px; font-weight: 700;">● REAL WEIGHTS MODE</span>
                    <p style="color: #d1fae5; font-size: 11px; margin: 2px 0 0 0;">Strict verification active.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")

        # Hardware Engine Telemetry
        dev_info = get_device_info()
        st.caption("COMPUTE DIAGNOSTICS")
        st.markdown(
            f"""
            <div style="font-size: 12px; color: #cbd5e1; line-height: 1.6;">
                <div><b>Device:</b> {dev_info['device_type']}</div>
                <div><b>Hardware:</b> {dev_info['device_name'][:18]}</div>
                <div><b>PyTorch:</b> {dev_info['torch_version'].split('+')[0]}</div>
                <div><b>Privacy:</b> 100% Local Inference</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.caption("⚕️ Swasthya AI Clinical Support Prototype")

    return nav_choice


def render_history_page():
    """Renders past analyses completed during the active session."""
    st.markdown("## 📋 Session Analysis History")
    st.caption("Real-time local audit trail of diagnostic runs performed in this session.")

    history = st.session_state.get("analysis_history", [])
    if not history:
        st.info("No scans analyzed in this session yet. Upload a radiograph or sonogram to start.")
        return

    st.markdown(f"**Total Analyses Performed:** {len(history)}")

    for idx, item in enumerate(reversed(history)):
        with st.expander(f"{item['timestamp']} — {item['modality']}: {item['finding']}", expanded=(idx == 0)):
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Modality:** {item['modality']}")
            c2.markdown(f"**Top Finding:** {item['finding']}")
            c3.markdown(f"**Confidence:** {item['probability'] * 100:.1f}%")
            st.caption(f"Engine State: {item['status']}")


def render_settings_page():
    """Renders system configuration and model weights setup guide."""
    st.markdown("## ⚙️ System Settings & Model Weights Guide")
    st.caption("Manage hardware acceleration, model paths, and learn how to install real weights.")

    st.markdown("### 🎛️ Execution Configuration")
    demo_val = st.toggle("Enable Demo Mode (Global)", value=st.session_state.demo_mode, key="settings_demo_toggle")
    st.session_state.demo_mode = demo_val

    st.markdown("---")
    st.markdown("### 📦 Model Weight Setup Instructions")

    t1, t2 = st.tabs(["Chest X-Ray Weights", "Ultrasound Weights"])

    with t1:
        st.markdown(
            """
            #### Chest X-Ray Model (MONAI DenseNet121)
            1. Download a pretrained DenseNet121 weights checkpoint (e.g. CheXNet or NIH ChestX-ray14).
            2. Place the `.pth` file at:
               ```
               swasthya-monai-imaging/models/xray/chexnet_monai_densenet121.pth
               ```
            3. Turn **Demo Mode OFF** in the sidebar or `.env`.
            4. The pipeline will automatically verify and load the real model.
            """
        )

    with t2:
        st.markdown(
            """
            #### Ultrasound Classification & Segmentation Models
            1. **Classification (DenseNet121):**
               Place checkpoint at:
               ```
               swasthya-monai-imaging/models/ultrasound/classification/ultrasound_densenet121.pth
               ```
            2. **Segmentation (MONAI UNet):**
               Place checkpoint at:
               ```
               swasthya-monai-imaging/models/ultrasound/segmentation/ultrasound_unet.pth
               ```
            """
        )

    st.markdown("---")
    st.markdown("### 🔒 Privacy & Data Protection")
    st.success(
        """
        - **Local Processing:** Images are stored temporarily in local memory or designated local cache.
        - **Zero Cloud Leakage:** No data or patient scans are transmitted to external AI APIs.
        - **Safe Filenames:** Filenames are sanitized to prevent directory traversal and system path disclosure.
        """
    )


def main():
    nav = render_sidebar()

    if nav == "🏠 Dashboard":
        render_dashboard(st.session_state.demo_mode)
    elif nav == "🫁 Chest X-ray":
        render_xray_page(st.session_state.demo_mode)
    elif nav == "🔬 Ultrasound":
        render_ultrasound_page(st.session_state.demo_mode)
    elif nav == "📋 Analysis History":
        render_history_page()
    elif nav == "⚙️ Settings":
        render_settings_page()


if __name__ == "__main__":
    main()
