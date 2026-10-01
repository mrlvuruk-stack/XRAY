"""
UI Pages Package
"""

from .dashboard import render_dashboard
from .xray_page import render_xray_page
from .ultrasound_page import render_ultrasound_page

__all__ = [
    "render_dashboard",
    "render_xray_page",
    "render_ultrasound_page",
]
