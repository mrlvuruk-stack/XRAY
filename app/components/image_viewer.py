"""
Image Viewer Component
Renders medical scans with metadata inspection and contrast adjustments.
"""

from typing import Optional, Dict, Any
from PIL import Image, ImageEnhance
import streamlit as st


def render_image_viewer(
    image: Image.Image,
    title: str = "Diagnostic Scan",
    metadata: Optional[Dict[str, Any]] = None,
    allow_windowing: bool = True,
    key_prefix: str = "img_view"
):
    """
    Renders a diagnostic image with optional intensity/contrast adjustment controls.
    """
    st.markdown(f"#### {title}")

    col1, col2 = st.columns([3, 1])

    with col2:
        if allow_windowing:
            st.caption("Windowing / Display Controls")
            contrast_factor = st.slider(
                "Contrast", 0.5, 2.5, 1.0, 0.1, key=f"{key_prefix}_contrast"
            )
            brightness_factor = st.slider(
                "Brightness", 0.5, 2.0, 1.0, 0.1, key=f"{key_prefix}_bright"
            )
        else:
            contrast_factor = 1.0
            brightness_factor = 1.0

        if metadata:
            st.caption("Scan Telemetry")
            st.markdown(
                f"""
                - **Resolution:** {metadata.get('width', 'N/A')} × {metadata.get('height', 'N/A')}
                - **Format:** {metadata.get('format', 'N/A')}
                - **Color Mode:** {metadata.get('mode', 'N/A')}
                - **File Size:** {metadata.get('size_kb', 'N/A')} KB
                """
            )

    with col1:
        # Apply windowing adjustments
        display_img = image
        if contrast_factor != 1.0:
            enhancer = ImageEnhance.Contrast(display_img)
            display_img = enhancer.enhance(contrast_factor)
        if brightness_factor != 1.0:
            enhancer = ImageEnhance.Brightness(display_img)
            display_img = enhancer.enhance(brightness_factor)

        st.image(
            display_img,
            caption=f"{title} (Adjusted)",
            use_container_width=True
        )
