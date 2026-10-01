"""
Simple ReportLab PDF Report Generator
Generates clinical prototype analysis reports with findings and medical disclaimers.
"""

import io
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

DISCLAIMER_TEXT = (
    "AI-generated output is intended for research/prototype and clinical decision-support demonstration purposes. "
    "It is not a definitive diagnosis and must be reviewed by a qualified healthcare professional."
)


def create_medical_pdf(
    modality: str,
    model_name: str,
    status: str,
    inference_time: float,
    findings: List[Dict[str, Any]],
    original_image: Optional[Image.Image],
    viz_image: Optional[Image.Image],
    is_demo: bool = True
) -> bytes:
    """Creates a standardized clinical summary PDF and returns the bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0284c7"),
        alignment=0,
    )
    meta_style = ParagraphStyle(
        "ReportMeta",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#0f172a"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#475569"),
        alignment=1,
    )

    story = []

    # Header
    story.append(Paragraph("SWASTHYA AI", title_style))
    story.append(Paragraph("Medical Imaging Intelligence Analysis Report", styles["Normal"]))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=10))

    if is_demo:
        demo_banner = [[Paragraph("<b>⚠️ DEMONSTRATION RESULT — NOT A MEDICAL DIAGNOSIS</b>", ParagraphStyle("W", textColor=colors.HexColor("#b45309"), alignment=1))]]
        t_demo = Table(demo_banner, colWidths=[540])
        t_demo.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef3c7")),
            ("BORDER", (0, 0), (-1, -1), 1, colors.HexColor("#f59e0b")),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t_demo)
        story.append(Spacer(1, 10))

    # Meta Table
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_data = [
        [Paragraph(f"<b>Modality:</b> {modality}", meta_style), Paragraph(f"<b>Timestamp:</b> {timestamp_str}", meta_style)],
        [Paragraph(f"<b>Model:</b> {model_name}", meta_style), Paragraph(f"<b>Status:</b> {status}", meta_style)],
        [Paragraph(f"<b>Inference Time:</b> {inference_time}s", meta_style), Paragraph("<b>Engine:</b> MONAI + PyTorch", meta_style)],
    ]
    t_meta = Table(meta_data, colWidths=[270, 270])
    t_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BORDER", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 14))

    # Side-by-Side Images
    with tempfile.TemporaryDirectory() as tmp:
        p_tmp = Path(tmp)
        row_imgs = []

        if original_image:
            orig_p = p_tmp / "orig.png"
            original_image.save(orig_p, "PNG")
            row_imgs.append(RLImage(str(orig_p), width=240, height=240))

        if viz_image:
            viz_p = p_tmp / "viz.png"
            viz_image.save(viz_p, "PNG")
            row_imgs.append(RLImage(str(viz_p), width=240, height=240))

        if len(row_imgs) == 2:
            t_img = Table([
                [Paragraph("<b>Input Medical Scan</b>", meta_style), Paragraph("<b>AI Visualization / Overlay</b>", meta_style)],
                [row_imgs[0], row_imgs[1]]
            ], colWidths=[270, 270])
            t_img.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            story.append(t_img)
            story.append(Spacer(1, 14))

        # Findings Table
        if findings:
            story.append(Paragraph("<b>AI Diagnostic Findings</b>", styles["Heading3"]))
            story.append(Spacer(1, 4))
            rows = [["Finding / Condition", "Probability / Area", "Stratification"]]
            for f in findings[:8]:
                rows.append([
                    Paragraph(f.get("finding", "Unknown"), table_cell),
                    f"{f.get('probability', 0.0)*100:.1f}%",
                    f.get("status", "NORMAL")
                ])
            t_find = Table(rows, colWidths=[240, 150, 150])
            t_find.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(t_find)
            story.append(Spacer(1, 16))
        else:
            story.append(Paragraph("<b>AI Diagnostic Findings</b>", styles["Heading3"]))
            story.append(Spacer(1, 4))
            unconf_p = Paragraph(
                "<b>Status:</b> Model not configured — preprocessing/demo pipeline only.<br/>"
                "<i>Strict Clinical Integrity: No medical predictions or condition probabilities were fabricated. "
                "Connect verified neural network weights to perform real inference.</i>",
                table_cell
            )
            story.append(unconf_p)
            story.append(Spacer(1, 16))


        # Mandatory Disclaimer
        t_disc = Table([[Paragraph(f"<b>MANDATORY MEDICAL DISCLAIMER:</b><br/>{DISCLAIMER_TEXT}", disclaimer_style)]], colWidths=[540])
        t_disc.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
            ("BORDER", (0, 0), (-1, -1), 0.8, colors.HexColor("#94a3b8")),
            ("PADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(t_disc)

        doc.build(story)
        return buffer.getvalue()
