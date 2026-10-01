"""
Heatmap and Mask Viewer Component
Renders Grad-CAM heatmaps, overlays, and segmentation masks.
Supports side-by-side and tabbed visualizations with opacity controls.
"""

from typing import Optional
from PIL import Image
import streamlit as st


def render_gradcam_viewer(
    original: Image.Image,
    heatmap: Image.Image,
    overlay: Image.Image,
    title: str = "Grad-CAM Explainability"
):
    """
    Renders [Original] [Heatmap] [Overlay] visualization cards.
    """
    st.markdown(f"### {title}")
    st.caption("Visualizing spatial convolutional activations contributing to AI classification findings.")

    tab1, tab2 = st.tabs(["Side-by-Side Comparison", "Individual Views"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Original Scan**")
            st.image(original, use_container_width=True)
        with col2:
            st.markdown("**Activation Heatmap**")
            st.image(heatmap, use_container_width=True)
        with col3:
            st.markdown("**Grad-CAM Overlay**")
            st.image(overlay, use_container_width=True)

    with tab2:
        view_choice = st.radio(
            "Select View",
            ["Original Scan", "Heatmap Only", "Blended Overlay"],
            horizontal=True,
            key=f"{title}_radio"
        )
        if view_choice == "Original Scan":
            st.image(original, caption="Input Scan", use_container_width=True)
        elif view_choice == "Heatmap Only":
            st.image(heatmap, caption="Color Normalized Heatmap", use_container_width=True)
        else:
            st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)


def render_segmentation_viewer(
    original: Image.Image,
    mask: Image.Image,
    overlay: Image.Image,
    lesion_pixels: Optional[int] = None,
    lesion_area_pct: Optional[float] = None
):
    """
    Renders [Original] [Mask] [Overlay] visualization cards for Ultrasound Segmentation.
    """
    st.markdown("### Lesion Segmentation (MONAI UNet)")

    if lesion_area_pct is not None:
        c1, c2 = st.columns(2)
        c1.metric("Segmented Lesion Area", f"{lesion_area_pct:.2f}%")
        c2.metric("Lesion Pixel Count", f"{lesion_pixels:,}" if lesion_pixels else "N/A")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Original Ultrasound**")
        st.image(original, use_container_width=True)
    with col2:
        st.markdown("**Binary Lesion Mask**")
        st.image(mask, use_container_width=True)
    with col3:
        st.markdown("**Contour & Fill Overlay**")
        st.image(overlay, use_container_width=True)
