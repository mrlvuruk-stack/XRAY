"""
Prediction Card Component
Visualizes classification findings table, risk statuses, model information,
and clearly distinguishes REAL MODEL vs DEMO MODE vs MODEL NOT CONFIGURED.
"""

from typing import List, Dict, Any, Optional
import streamlit as st


def render_model_badge(status: str, is_demo: bool):
    """Renders prominent status indicator."""
    if status == "REAL_MODEL":
        st.markdown(
            """
            <div style="background-color: #0e3a24; border: 1px solid #10b981; border-radius: 8px; padding: 10px 16px; margin-bottom: 15px;">
                <span style="color: #34d399; font-weight: 700; font-size: 14px;">● REAL PRETRAINED MODEL</span>
                <p style="color: #d1fae5; margin: 4px 0 0 0; font-size: 13px;">Inference executed using verified neural network weights on local hardware.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif status == "DEMO_MODE" or is_demo:
        st.markdown(
            """
            <div style="background-color: #3b2503; border: 2px solid #f59e0b; border-radius: 8px; padding: 12px 18px; margin-bottom: 16px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="color: #fbbf24; font-size: 18px;">⚠️</span>
                    <strong style="color: #fbbf24; font-size: 15px; letter-spacing: 0.5px;">DEMO RESULT — NOT A REAL MEDICAL PREDICTION</strong>
                </div>
                <p style="color: #fef3c7; margin: 6px 0 0 0; font-size: 13px; line-height: 1.4;">
                    This analysis was generated in demonstration mode using simulated inference.
                    It has <strong>not</strong> been calculated by trained clinical weights.
                    Do not use for medical triage, diagnosis, or patient care.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif status == "MODEL_NOT_CONFIGURED":
        st.markdown(
            """
            <div style="background-color: #1e293b; border-left: 4px solid #f59e0b; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px; border: 1px solid #334155;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="color: #fbbf24; font-weight: 700; font-size: 14px;">⚠️ Model not configured — preprocessing/demo pipeline only.</span>
                    <span style="background-color: #f59e0b; color: #000; font-size: 10px; font-weight: 800; padding: 2px 6px; border-radius: 4px;">DEMO DATA — NOT FOR CLINICAL USE</span>
                </div>
                <p style="color: #cbd5e1; margin: 6px 0 0 0; font-size: 13px;">
                    MONAI preprocessing transforms completed and verified. In accordance with strict clinical integrity standards, medical predictions are <u>never fabricated</u> without a validated, loaded neural network model.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )



def render_unconfigured_instructions(message: str, instructions: Optional[str]):
    """Renders clean guide for configuring weights."""
    st.error(message or "Model is not configured.")
    if instructions:
        st.info("### How to add model weights")
        st.code(instructions.strip(), language="markdown")


def render_findings_table(findings: List[Dict[str, Any]]):
    """
    Renders standard Finding | Probability | Status table with progress bars.
    """
    st.markdown("### AI Findings")

    # Header
    cols = st.columns([3, 3, 2])
    cols[0].markdown("**Finding**")
    cols[1].markdown("**Probability**")
    cols[2].markdown("**Status**")
    st.divider()

    for item in findings:
        f_cols = st.columns([3, 3, 2])
        finding = item.get("finding", "Unknown")
        prob = float(item.get("probability", 0.0))
        status = item.get("status", "Low")

        # Color-coded badge
        if status == "High" or prob >= 0.50:
            badge_html = f"<span style='background-color: #ef4444; color: white; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 12px;'>HIGH RISK</span>"
        elif status == "Moderate" or prob >= 0.25:
            badge_html = f"<span style='background-color: #f59e0b; color: black; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 12px;'>MODERATE</span>"
        else:
            badge_html = f"<span style='background-color: #10b981; color: white; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 12px;'>NORMAL / LOW</span>"

        f_cols[0].markdown(f"**{finding}**")
        with f_cols[1]:
            st.progress(min(max(prob, 0.0), 1.0), text=f"{prob * 100:.1f}%")
        f_cols[2].markdown(badge_html, unsafe_allow_html=True)


def render_model_telemetry(
    model_name: str,
    version: str,
    device: str,
    inference_time_ms: float
):
    """Renders technical metadata block."""
    st.markdown("### Model Information")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Architecture", model_name.split()[0])
    c2.metric("Version", version)
    c3.metric("Compute Device", device.replace("Device: ", ""))
    c4.metric("Inference Time", f"{inference_time_ms:.1f} ms")
