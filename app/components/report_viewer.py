"""
Report Viewer and PDF Generation Component
Generates professional clinical diagnostic analysis PDF reports using ReportLab.
Includes metadata, findings table, images, side-by-side overlays, and mandatory medical disclaimers.
"""

import os
import io
import tempfile
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
from PIL import Image
import streamlit as st

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

DISCLAIMER_TEXT = (
    "AI-generated output is intended for clinical decision support and research/prototype purposes. "
    "It is not a definitive diagnosis and must be reviewed by a qualified healthcare professional."
)


def generate_pdf_report(
    patient_id: str,
    study_id: str,
    modality: str,
    findings: List[Dict[str, Any]],
    original_image: Optional[Image.Image],
    viz_image: Optional[Image.Image],
    viz_caption: str,
    model_name: str,
    model_version: str,
    timestamp: str,
    is_demo: bool = False,
    output_dir: Optional[str] = None,
) -> bytes:
    """
    Constructs a professional ReportLab PDF report and returns the PDF bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0284c7"),
    )
    demo_style = ParagraphStyle(
        "DemoWarning",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#b45309"),
        alignment=1,
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#475569"),
        alignment=1,
    )

    story = []

    # Hospital / Lab Header
    story.append(Paragraph("SWASTHYA AI", title_style))
    story.append(Paragraph("Medical Imaging Analysis Laboratory Report", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0284c7"), spaceAfter=12))

    # Prominent Demo Warning Banner if Demo Mode
    if is_demo:
        demo_data = [[
            Paragraph("⚠️ DEMO RESULT — NOT A REAL MEDICAL PREDICTION — SIMULATED PROTOTYPE OUTPUT ONLY", demo_style)
        ]]
        demo_table = Table(demo_data, colWidths=[540])
        demo_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef3c7")),
            ("BORDER", (0, 0), (-1, -1), 1.5, colors.HexColor("#f59e0b")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(demo_table)
        story.append(Spacer(1, 10))

    # Study and Patient Metadata Table
    meta_data = [
        [
            Paragraph(f"<b>Patient ID:</b> {patient_id or 'Anonymous / Clinical Subject'}", body_style),
            Paragraph(f"<b>Study ID:</b> {study_id or 'ST-2026-X01'}", body_style),
        ],
        [
            Paragraph(f"<b>Modality:</b> {modality.upper()}", body_style),
            Paragraph(f"<b>Inference Timestamp:</b> {timestamp}", body_style),
        ],
        [
            Paragraph(f"<b>AI Model:</b> {model_name}", body_style),
            Paragraph(f"<b>Model Version:</b> {model_version}", body_style),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[270, 270])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BORDER", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # Diagnostic Images (Side by Side)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        img_elements = []

        if original_image:
            orig_tmp = tmp_path / "orig.png"
            original_image.save(orig_tmp, format="PNG")
            img_elements.append(
                [Paragraph("<b>Original Medical Scan</b>", body_style),
                 RLImage(str(orig_tmp), width=230, height=230)]
            )

        if viz_image:
            viz_tmp = tmp_path / "viz.png"
            viz_image.save(viz_tmp, format="PNG")
            img_elements.append(
                [Paragraph(f"<b>{viz_caption}</b>", body_style),
                 RLImage(str(viz_tmp), width=230, height=230)]
            )

        if len(img_elements) == 2:
            img_table = Table([
                [img_elements[0][0], img_elements[1][0]],
                [img_elements[0][1], img_elements[1][1]],
            ], colWidths=[270, 270])
            img_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
            ]))
            story.append(img_table)
            story.append(Spacer(1, 14))

        # Findings Summary Table
        story.append(Paragraph("AI Diagnostic Findings & Probability Analysis", section_heading))
        story.append(Spacer(1, 6))

        if findings:
            table_rows = [["Finding / Pathology", "Confidence Score", "Risk Stratification"]]
            for f in findings[:8]:  # Top findings
                prob_val = f.get("probability", 0.0)
                status_val = f.get("status", "Low")
                table_rows.append([
                    f.get("finding", "Unknown"),
                    f"{prob_val * 100:.1f}% ({prob_val:.4f})",
                    status_val.upper(),
                ])

            t_findings = Table(table_rows, colWidths=[240, 150, 150])
            t_findings.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ]))
            story.append(t_findings)
            story.append(Spacer(1, 16))

        # Medical Disclaimer Box
        disclaimer_data = [[
            Paragraph(f"<b>MANDATORY MEDICAL DISCLAIMER:</b><br/>{DISCLAIMER_TEXT}", disclaimer_style)
        ]]
        t_disc = Table(disclaimer_data, colWidths=[540])
        t_disc.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
            ("BORDER", (0, 0), (-1, -1), 1, colors.HexColor("#94a3b8")),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(t_disc)

        doc.build(story)
        pdf_bytes = buffer.getvalue()

    # Save to disk if output_dir specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"Swasthya_{modality}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = os.path.join(output_dir, filename)
        try:
            with open(filepath, "wb") as f:
                f.write(pdf_bytes)
        except Exception:
            pass

    return pdf_bytes


def render_report_section(
    modality: str,
    findings: List[Dict[str, Any]],
    original_image: Optional[Image.Image],
    viz_image: Optional[Image.Image],
    viz_caption: str,
    model_name: str,
    model_version: str,
    timestamp: str,
    is_demo: bool = False,
    reports_dir: str = "reports"
):
    """Renders Streamlit UI section to input metadata and download the clinical PDF report."""
    st.markdown("### Generate Clinical Report")
    st.caption("Export a standardized PDF report with verified scan findings, overlays, and clinical disclaimers.")

    c1, c2 = st.columns(2)
    patient_id = c1.text_input("Patient / Subject ID (Optional)", placeholder="e.g. PT-94021")
    study_id = c2.text_input("Study Accession ID (Optional)", placeholder="e.g. ACC-2026-001")

    if st.button("📄 Generate & Download PDF Report", key=f"btn_pdf_{modality}"):
        with st.spinner("Compiling PDF diagnostic report..."):
            pdf_data = generate_pdf_report(
                patient_id=patient_id,
                study_id=study_id,
                modality=modality,
                findings=findings,
                original_image=original_image,
                viz_image=viz_image,
                viz_caption=viz_caption,
                model_name=model_name,
                model_version=model_version,
                timestamp=timestamp,
                is_demo=is_demo,
                output_dir=reports_dir
            )
            st.download_button(
                label="⬇️ Download Medical Imaging Report (PDF)",
                data=pdf_data,
                file_name=f"Swasthya_{modality.lower()}_report.pdf",
                mime="application/pdf",
                key=f"dl_pdf_{modality}"
            )
            st.success("Report successfully generated!")
