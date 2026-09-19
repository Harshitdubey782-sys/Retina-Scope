import streamlit as st
import torch
import timm
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import geopandas as gpd
from datetime import datetime
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
import plotly.express as px

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ============================================================
# RETINASCOPE
# Explainable AI for Diabetic Retinopathy Screening
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/efficientnet_b0_best.pth"
DEVICE = torch.device("cpu")

CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative"
}

GRADE_DESCRIPTIONS = {
    0: "No signs of diabetic retinopathy detected",
    1: "Mild non-proliferative diabetic retinopathy",
    2: "Moderate non-proliferative diabetic retinopathy",
    3: "Severe non-proliferative diabetic retinopathy",
    4: "Proliferative diabetic retinopathy"
}


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RetinaScope",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PREMIUM UI
# ============================================================

st.markdown(
    """
<style>

/* ==========================================================
   GLOBAL
   ========================================================== */

.stApp {
    background:
        radial-gradient(
            circle at 12% 4%,
            rgba(37, 99, 235, 0.12),
            transparent 27%
        ),
        radial-gradient(
            circle at 88% 12%,
            rgba(124, 58, 237, 0.10),
            transparent 30%
        ),
        #070a11;
}

.block-container {
    max-width: 1380px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}


/* ==========================================================
   TYPOGRAPHY
   ========================================================== */

h1,
h2,
h3,
h4 {
    color: #f8fafc !important;
}

p {
    color: #94a3b8;
}


/* ==========================================================
   HERO
   ========================================================== */

.hero-eyebrow {
    color: #60a5fa;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.20em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}

.hero-title {
    color: #f8fafc;
    font-size: 3.15rem;
    line-height: 1.04;
    font-weight: 850;
    letter-spacing: -0.045em;
    margin-bottom: 0.8rem;
}

.hero-description {
    max-width: 880px;
    color: #94a3b8;
    font-size: 1rem;
    line-height: 1.7;
}


/* ==========================================================
   DIVIDERS
   ========================================================== */

hr {
    border-color: rgba(148, 163, 184, 0.11) !important;
}


/* ==========================================================
   SECTION HEADINGS
   ========================================================== */

.section-kicker {
    color: #60a5fa;
    font-size: 0.70rem;
    font-weight: 800;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}

.section-heading {
    color: #f8fafc;
    font-size: 1.45rem;
    font-weight: 800;
    margin-bottom: 0.25rem;
}

.section-description {
    color: #64748b;
    font-size: 0.88rem;
    margin-bottom: 1rem;
}


/* ==========================================================
   UPLOADER
   ========================================================== */

[data-testid="stFileUploader"] {
    background:
        linear-gradient(
            145deg,
            rgba(15, 23, 42, 0.85),
            rgba(10, 15, 27, 0.85)
        );
    border: 1px solid rgba(96, 165, 250, 0.18);
    border-radius: 18px;
    padding: 1rem;
}

[data-testid="stFileUploaderDropzone"] {
    background: rgba(15, 23, 42, 0.55);
    border: 1px dashed rgba(96, 165, 250, 0.35);
    border-radius: 14px;
}


/* ==========================================================
   ASSESSMENT CARDS
   ========================================================== */

.assessment-card {
    min-height: 175px;
    background:
        linear-gradient(
            145deg,
            rgba(17, 27, 48, 0.96),
            rgba(9, 15, 27, 0.96)
        );
    border: 1px solid rgba(148, 163, 184, 0.13);
    border-radius: 18px;
    padding: 1.35rem 1.4rem;
    box-shadow:
        0 14px 40px rgba(0, 0, 0, 0.18);
}

.assessment-label {
    color: #64748b;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.13em;
    text-transform: uppercase;
}

.assessment-value {
    color: #f8fafc;
    font-size: 2rem;
    line-height: 1.1;
    font-weight: 850;
    margin-top: 0.7rem;
}

.assessment-subtitle {
    color: #94a3b8;
    font-size: 0.88rem;
    margin-top: 0.35rem;
}


/* ==========================================================
   STATUS CARDS
   ========================================================== */

.status-card {
    min-height: 175px;
    border-radius: 18px;
    padding: 1.35rem 1.4rem;
}

.status-card.referable {
    background:
        linear-gradient(
            145deg,
            rgba(127, 29, 29, 0.25),
            rgba(69, 10, 10, 0.18)
        );
    border: 1px solid rgba(248, 113, 113, 0.28);
}

.status-card.normal {
    background:
        linear-gradient(
            145deg,
            rgba(20, 83, 45, 0.25),
            rgba(5, 46, 22, 0.18)
        );
    border: 1px solid rgba(74, 222, 128, 0.22);
}

.status-label {
    color: #64748b;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.13em;
    text-transform: uppercase;
}

.status-value {
    font-size: 1.75rem;
    font-weight: 850;
    margin-top: 0.7rem;
}

.status-card.referable .status-value {
    color: #fca5a5;
}

.status-card.normal .status-value {
    color: #86efac;
}

.status-description {
    color: #94a3b8;
    font-size: 0.83rem;
    margin-top: 0.45rem;
}


/* ==========================================================
   CONFIDENCE
   ========================================================== */

.confidence-bar {
    margin-top: 0.85rem;
}


/* ==========================================================
   IMAGE DISPLAY
   ========================================================== */

[data-testid="stImage"] {
    border-radius: 16px;
    overflow: hidden;
}


/* ==========================================================
   SIDEBAR
   ========================================================== */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #0c101a 0%,
            #090c13 100%
        );
    border-right: 1px solid rgba(148, 163, 184, 0.10);
}

section[data-testid="stSidebar"] h2 {
    color: #f8fafc !important;
}


/* ==========================================================
   SIDEBAR METRICS
   ========================================================== */

section[data-testid="stSidebar"] [data-testid="stMetric"] {
    background:
        linear-gradient(
            145deg,
            rgba(17, 27, 48, 0.95),
            rgba(9, 15, 27, 0.95)
        );
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 16px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}


/* ==========================================================
   MAIN METRICS
   ========================================================== */

[data-testid="stMetric"] {
    background:
        linear-gradient(
            145deg,
            rgba(17, 27, 48, 0.96),
            rgba(9, 15, 27, 0.96)
        );
    border: 1px solid rgba(148, 163, 184, 0.13);
    border-radius: 18px;
    padding: 1.2rem;
}

[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
}

[data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-weight: 800 !important;
}


/* ==========================================================
   PROGRESS
   ========================================================== */

.stProgress > div > div {
    border-radius: 999px;
}


/* ==========================================================
   TABS
   ========================================================== */

button[data-baseweb="tab"] {
    color: #94a3b8 !important;
    font-weight: 700;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #60a5fa !important;
}


/* ==========================================================
   ALERTS
   ========================================================== */

[data-testid="stAlert"] {
    border-radius: 14px;
}


/* ==========================================================
   FOOTER
   ========================================================== */

.app-footer {
    text-align: center;
    color: #475569;
    font-size: 0.74rem;
    line-height: 1.7;
    padding-top: 2.5rem;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=5
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model


model = load_model()



# ============================================================
# PATIENT SCREENING REPORT
# ============================================================

def generate_screening_report(patient, original_image, heatmap, probabilities,
                              predicted_class, confidence):
    """Create a professional PDF screening report in memory."""
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="RetinaScope Patient Screening Report",
        author="RetinaScope"
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontSize=20,
        leading=24, alignment=TA_CENTER, textColor=colors.HexColor("#0F172A"),
        spaceAfter=3
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle", parent=styles["Normal"], fontSize=9,
        leading=12, alignment=TA_CENTER, textColor=colors.HexColor("#64748B"),
        spaceAfter=10
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"], fontSize=11,
        leading=14, textColor=colors.HexColor("#2563EB"),
        spaceBefore=7, spaceAfter=5
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=8.5,
        leading=12, textColor=colors.HexColor("#334155")
    )
    small_style = ParagraphStyle(
        "Small", parent=styles["Normal"], fontSize=7,
        leading=9, textColor=colors.HexColor("#64748B")
    )

    story = [
        Paragraph("RETINASCOPE", title_style),
        Paragraph("Diabetic Retinopathy Screening Report", subtitle_style),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1")),
        Spacer(1, 5)
    ]

    patient_rows = [
        ["Case ID", patient["Case ID"], "Screening Date", patient["Screening Date"]],
        ["Patient Name", patient["Patient Name"], "Age", patient["Age"]],
        ["Sex", patient["Sex"], "Screening Centre", patient["Screening Centre"]],
        ["Eye Examined", patient["Eye Examined"], "Image ID", patient["Image ID"]],
        ["Diabetes Type", patient["Diabetes Type"], "Duration", patient["Diabetes Duration"]],
        ["Previous DR", patient["Previous DR"], "Previous Eye Exam", patient["Previous Eye Exam"]],
    ]

    story += [
        Paragraph("1. Patient Information", section_style),
        Table(patient_rows, colWidths=[30*mm, 57*mm, 34*mm, 57*mm],
              style=TableStyle([
                  ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#F1F5F9")),
                  ("BACKGROUND",(2,0),(2,-1),colors.HexColor("#F1F5F9")),
                  ("TEXTCOLOR",(0,0),(-1,-1),colors.HexColor("#334155")),
                  ("FONTNAME",(0,0),(-1,-1),"Helvetica"),
                  ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
                  ("FONTNAME",(2,0),(2,-1),"Helvetica-Bold"),
                  ("FONTSIZE",(0,0),(-1,-1),8),
                  ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#CBD5E1")),
                  ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                  ("TOPPADDING",(0,0),(-1,-1),5),
                  ("BOTTOMPADDING",(0,0),(-1,-1),5),
              ]))
    ]

    referable = predicted_class >= 2
    status = "REFERABLE DR" if referable else "NON-REFERABLE"
    status_color = "#DC2626" if referable else "#16A34A"
    recommendation = (
        "Further evaluation by a qualified eye-care professional is recommended."
        if referable else
        "Routine screening / follow-up according to appropriate clinical guidance."
    )

    story += [
        Paragraph("2. AI Screening Result", section_style),
        Table([
            ["Predicted Grade", f"Grade {predicted_class} — {CLASS_NAMES[predicted_class]}"],
            ["Model Confidence", f"{confidence * 100:.2f}%"],
            ["Screening Status", status],
            ["Screening Recommendation", recommendation],
        ], colWidths=[48*mm, 130*mm], style=TableStyle([
            ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#F1F5F9")),
            ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
            ("FONTNAME",(1,0),(1,-1),"Helvetica"),
            ("FONTSIZE",(0,0),(-1,-1),8.5),
            ("TEXTCOLOR",(0,0),(-1,-1),colors.HexColor("#334155")),
            ("TEXTCOLOR",(1,2),(1,2),colors.HexColor(status_color)),
            ("FONTNAME",(1,2),(1,2),"Helvetica-Bold"),
            ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#CBD5E1")),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("TOPPADDING",(0,0),(-1,-1),6),
            ("BOTTOMPADDING",(0,0),(-1,-1),6),
        ]))
    ]

    # Save image bytes for ReportLab
    def image_bytes(img):
        b = BytesIO()
        Image.fromarray(np.asarray(img).astype(np.uint8)).save(b, format="PNG")
        b.seek(0)
        return b

    orig_bytes = BytesIO()
    original_image.save(orig_bytes, format="PNG")
    orig_bytes.seek(0)

    heat_bytes = image_bytes(heatmap)

    story += [
        Paragraph("3. Explainability", section_style),
        Table([
            [RLImage(orig_bytes, width=82*mm, height=62*mm),
             RLImage(heat_bytes, width=82*mm, height=62*mm)],
            [Paragraph("<b>Original Fundus Image</b>", small_style),
             Paragraph("<b>Grad-CAM Attention</b>", small_style)]
        ], colWidths=[89*mm, 89*mm], style=TableStyle([
            ("ALIGN",(0,0),(-1,-1),"CENTER"),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("BOX",(0,0),(-1,-2),0.5,colors.HexColor("#CBD5E1")),
            ("INNERGRID",(0,0),(-1,-2),0.4,colors.HexColor("#E2E8F0")),
            ("BOTTOMPADDING",(0,0),(-1,-1),4),
        ])),
        Paragraph(
            "Grad-CAM highlights image regions that contributed to the selected model prediction. "
            "It represents model attention and is not a confirmed lesion map.",
            small_style
        )
    ]

    prob_rows = [["Severity", "Probability"]]
    for i in range(5):
        prob_rows.append([f"Grade {i} — {CLASS_NAMES[i]}", f"{float(probabilities[0, i]) * 100:.2f}%"])

    story += [
        Paragraph("4. Class Probability Distribution", section_style),
        Table(prob_rows, colWidths=[125*mm, 53*mm], style=TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#E2E8F0")),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),8),
            ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#CBD5E1")),
            ("ALIGN",(1,1),(1,-1),"RIGHT"),
            ("TOPPADDING",(0,0),(-1,-1),4),
            ("BOTTOMPADDING",(0,0),(-1,-1),4),
        ]))
    ]

    story += [
        Paragraph("5. Model Information", section_style),
        Paragraph(
            "Model: EfficientNet-B0 &nbsp;|&nbsp; Task: 5-class DR classification "
            "&nbsp;|&nbsp; Training dataset: APTOS 2019 &nbsp;|&nbsp; Explainability: Grad-CAM",
            body_style
        ),
        Spacer(1, 4),
        Paragraph(
            "Validation reference: 80.22% 5-class accuracy, 65.03% macro F1, "
            "90.27% referable sensitivity, 92.18% referable specificity.",
            body_style
        ),
        Spacer(1, 7),
        HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#CBD5E1")),
        Spacer(1, 5),
        Paragraph(
            "<b>IMPORTANT:</b> RetinaScope is a research and demonstration prototype. "
            "This AI-assisted screening output is not a standalone clinical diagnosis and "
            "should not replace evaluation by a qualified healthcare professional. "
            "Reported performance is based on an internal APTOS validation split and does "
            "not establish clinical validity.",
            small_style
        ),
        Spacer(1, 5),
        Paragraph("Generated by RetinaScope · Explainable AI for Diabetic Retinopathy Screening", small_style)
    ]

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()



# ============================================================
# DUAL-EYE PATIENT SCREENING REPORT
# ============================================================

def generate_dual_eye_report(patient, right_data, left_data):
    """Create a professional PDF report for bilateral screening."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=15*mm, leftMargin=15*mm,
        topMargin=14*mm, bottomMargin=14*mm,
        title="RetinaScope Dual-Eye Screening Report",
        author="RetinaScope"
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DualTitle", parent=styles["Title"], fontSize=20, leading=24,
        alignment=TA_CENTER, textColor=colors.HexColor("#0F172A"), spaceAfter=3
    )
    subtitle_style = ParagraphStyle(
        "DualSubtitle", parent=styles["Normal"], fontSize=9, leading=12,
        alignment=TA_CENTER, textColor=colors.HexColor("#64748B"), spaceAfter=10
    )
    section_style = ParagraphStyle(
        "DualSection", parent=styles["Heading2"], fontSize=11, leading=14,
        textColor=colors.HexColor("#2563EB"), spaceBefore=7, spaceAfter=5
    )
    body_style = ParagraphStyle(
        "DualBody", parent=styles["Normal"], fontSize=8.5, leading=12,
        textColor=colors.HexColor("#334155")
    )
    small_style = ParagraphStyle(
        "DualSmall", parent=styles["Normal"], fontSize=7, leading=9,
        textColor=colors.HexColor("#64748B")
    )

    story = [
        Paragraph("RETINASCOPE", title_style),
        Paragraph("Dual-Eye Diabetic Retinopathy Screening Report", subtitle_style),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1")),
        Spacer(1, 5),
    ]

    patient_rows = [
        ["Case ID", patient["Case ID"], "Screening Date", patient["Screening Date"]],
        ["Patient Name", patient["Patient Name"], "Age", patient["Age"]],
        ["Sex", patient["Sex"], "Screening Centre", patient["Screening Centre"]],
        ["Eye Examined", "Both", "Right Image ID", patient["Right Image ID"]],
        ["Diabetes Type", patient["Diabetes Type"], "Left Image ID", patient["Left Image ID"]],
        ["Duration", patient["Diabetes Duration"], "Previous DR", patient["Previous DR"]],
        ["Previous Eye Exam", patient["Previous Eye Exam"], "", ""],
    ]
    story += [
        Paragraph("1. Patient Information", section_style),
        Table(patient_rows, colWidths=[30*mm,57*mm,34*mm,57*mm],
              style=TableStyle([
                  ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#F1F5F9")),
                  ("BACKGROUND",(2,0),(2,-1),colors.HexColor("#F1F5F9")),
                  ("FONTNAME",(0,0),(-1,-1),"Helvetica"),
                  ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
                  ("FONTNAME",(2,0),(2,-1),"Helvetica-Bold"),
                  ("FONTSIZE",(0,0),(-1,-1),8),
                  ("TEXTCOLOR",(0,0),(-1,-1),colors.HexColor("#334155")),
                  ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#CBD5E1")),
                  ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                  ("TOPPADDING",(0,0),(-1,-1),5),
                  ("BOTTOMPADDING",(0,0),(-1,-1),5),
              ]))
    ]

    def result(d):
        pred = d["predicted_class"]
        return (
            f"Grade {pred} — {CLASS_NAMES[pred]}",
            f"{d['confidence']*100:.2f}%",
            "REFERABLE DR" if pred >= 2 else "NON-REFERABLE"
        )

    rr = result(right_data)
    lr = result(left_data)
    story += [
        Paragraph("2. AI Screening Result — Right vs Left Eye", section_style),
        Table([
            ["", "Right Eye", "Left Eye"],
            ["Predicted Grade", rr[0], lr[0]],
            ["Model Confidence", rr[1], lr[1]],
            ["Screening Status", rr[2], lr[2]],
        ], colWidths=[55*mm,62*mm,62*mm], style=TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#E2E8F0")),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTNAME",(0,1),(0,-1),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),8.5),
            ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#CBD5E1")),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("TOPPADDING",(0,0),(-1,-1),6),
            ("BOTTOMPADDING",(0,0),(-1,-1),6),
        ])),
        Spacer(1,5),
        Paragraph(
            "Inter-eye severity comparison describes the two AI screening outputs. "
            "It is not a clinical assessment of disease asymmetry.", small_style
        )
    ]

    def img_bytes(img):
        b = BytesIO()
        Image.fromarray(np.asarray(img).astype(np.uint8)).save(b, format="PNG")
        b.seek(0)
        return b

    ro = BytesIO(); right_data["image"].save(ro, format="PNG"); ro.seek(0)
    lo = BytesIO(); left_data["image"].save(lo, format="PNG"); lo.seek(0)
    rh = img_bytes(right_data["heatmap"])
    lh = img_bytes(left_data["heatmap"])

    story += [
        Paragraph("3. Dual-Eye Explainability", section_style),
        Table([
            [Paragraph("<b>Right Eye</b>", small_style), Paragraph("<b>Left Eye</b>", small_style)],
            [RLImage(ro, width=76*mm, height=57*mm), RLImage(lo, width=76*mm, height=57*mm)],
            [RLImage(rh, width=76*mm, height=57*mm), RLImage(lh, width=76*mm, height=57*mm)],
            [Paragraph("Original Fundus", small_style), Paragraph("Original Fundus", small_style)],
            [Paragraph("Grad-CAM Attention", small_style), Paragraph("Grad-CAM Attention", small_style)],
        ], colWidths=[89*mm,89*mm], style=TableStyle([
            ("ALIGN",(0,0),(-1,-1),"CENTER"),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#CBD5E1")),
            ("INNERGRID",(0,0),(-1,-1),0.4,colors.HexColor("#E2E8F0")),
            ("TOPPADDING",(0,0),(-1,-1),4),
            ("BOTTOMPADDING",(0,0),(-1,-1),4),
        ])),
        Paragraph(
            "Grad-CAM highlights regions that contributed to each selected prediction. "
            "Highlighted regions represent model attention, not confirmed lesions.", small_style
        )
    ]

    prob_rows = [["Severity", "Right Eye", "Left Eye"]]
    for i in range(5):
        prob_rows.append([
            f"Grade {i} — {CLASS_NAMES[i]}",
            f"{float(right_data['probabilities'][0,i])*100:.2f}%",
            f"{float(left_data['probabilities'][0,i])*100:.2f}%"
        ])

    story += [
        Paragraph("4. Class Probability Comparison", section_style),
        Table(prob_rows, colWidths=[95*mm,41*mm,41*mm], style=TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#E2E8F0")),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),8),
            ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#CBD5E1")),
            ("ALIGN",(1,1),(-1,-1),"RIGHT"),
            ("TOPPADDING",(0,0),(-1,-1),4),
            ("BOTTOMPADDING",(0,0),(-1,-1),4),
        ])),
        Paragraph("5. Model Information", section_style),
        Paragraph(
            "Model: EfficientNet-B0 &nbsp;|&nbsp; Task: 5-class DR classification "
            "&nbsp;|&nbsp; Training dataset: APTOS 2019 &nbsp;|&nbsp; Explainability: Grad-CAM",
            body_style
        ),
        Spacer(1,4),
        Paragraph(
            "Validation reference: 80.22% 5-class accuracy, 65.03% macro F1, "
            "90.27% referable sensitivity, 92.18% referable specificity.", body_style
        ),
        Spacer(1,7),
        HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#CBD5E1")),
        Spacer(1,5),
        Paragraph(
            "<b>IMPORTANT:</b> RetinaScope is a research and demonstration prototype. "
            "This AI-assisted screening output is not a standalone clinical diagnosis and "
            "should not replace evaluation by a qualified healthcare professional. "
            "Reported performance is based on an internal APTOS validation split and does "
            "not establish clinical validity.", small_style
        ),
        Spacer(1,5),
        Paragraph("Generated by RetinaScope · Explainable AI for Diabetic Retinopathy Screening", small_style)
    ]
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = np.array(
        image.convert("RGB")
    )

    # Resize
    image = cv2.resize(
        image,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    # CLAHE
    lab = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2LAB
    )

    l_channel, a_channel, b_channel = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_channel = clahe.apply(
        l_channel
    )

    lab = cv2.merge(
        [l_channel, a_channel, b_channel]
    )

    image = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2RGB
    )

    # Light denoising
    image = cv2.GaussianBlur(
        image,
        (3, 3),
        0
    )

    # Visualization image
    rgb_image = image.astype(
        np.float32
    ) / 255.0

    # Tensor
    tensor = torch.from_numpy(
        rgb_image
    ).permute(2, 0, 1).float()

    # ImageNet normalization
    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    tensor = (tensor - mean) / std

    tensor = tensor.unsqueeze(0)

    return rgb_image, tensor


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    input_tensor,
    predicted_class,
    rgb_image
):

    target_layers = [
        model.conv_head
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    targets = [
        ClassifierOutputTarget(
            predicted_class
        )
    ]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    return visualization



# ============================================================
# DISTRICT RESOURCE PLANNING
# ============================================================

PLANNING_PATH = "planning_data.csv"
SHAPEFILE_PATH = "district_shapes/output.shp"

@st.cache_data
def load_planning_data():
    return pd.read_csv(PLANNING_PATH)

@st.cache_data
def load_district_shapes():
    gdf = gpd.read_file(SHAPEFILE_PATH)
    if gdf.crs is not None:
        gdf = gdf.to_crs(epsg=4326)
    return gdf

def normalize_name(series):
    return (
        series.astype(str)
        .str.upper()
        .str.replace("&", "AND", regex=False)
        .str.replace(r"[^A-Z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

@st.cache_data(show_spinner=False)
def build_map_data(selected_state="All States"):
    planning = load_planning_data().copy()
    shapes = load_district_shapes().copy()

    # Normalize names for a robust state + district join.
    shapes["_state_key"] = normalize_name(shapes["statename"])
    shapes["_district_key"] = normalize_name(shapes["distname"])

    planning["_state_key"] = normalize_name(planning["state"])
    planning["_district_key"] = normalize_name(planning["district"])

    # IMPORTANT: filter before the merge so the browser receives only the
    # selected state's geometries instead of the complete India shapefile.
    if selected_state != "All States":
        state_key = normalize_name(pd.Series([selected_state])).iloc[0]
        shapes = shapes[shapes["_state_key"] == state_key].copy()
        planning = planning[planning["_state_key"] == state_key].copy()

    merged = shapes.merge(
        planning,
        on=["_state_key", "_district_key"],
        how="inner",
        suffixes=("_shape", "_plan")
    )

    merged = merged.dropna(subset=["priority"]).copy()

    if merged.empty:
        return merged

    # Reduce geometry complexity for much faster Plotly rendering.
    merged["geometry"] = merged.geometry.simplify(
        tolerance=0.005,
        preserve_topology=True
    )

    merged["map_id"] = merged.index.astype(str)
    return merged


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## RetinaScope")

    st.caption(
        "Explainable AI for Diabetic Retinopathy Screening"
    )

    st.divider()

    st.markdown("### MODEL")

    st.write("EfficientNet-B0")

    st.caption(
        "5-class severity classification"
    )

    st.divider()

    st.markdown("### VALIDATION PERFORMANCE")

    st.metric(
        "5-Class Accuracy",
        "80.22%"
    )

    st.metric(
        "Referable Sensitivity",
        "90.27%"
    )

    st.metric(
        "Referable Specificity",
        "92.18%"
    )

    st.divider()

    st.markdown("### SCREENING LOGIC")

    st.caption(
        "Grades 0–1 → Non-Referable"
    )

    st.caption(
        "Grades 2–4 → Referable"
    )

    st.divider()

    st.caption(
        "Research prototype. Not a clinical diagnostic device."
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero-eyebrow">
RETINASCOPE · AI SCREENING
</div>

<div class="hero-title">
Explainable Diabetic<br>
Retinopathy Screening
</div>

<div class="hero-description">
AI-assisted retinal image analysis with five-level severity
classification, confidence estimation, and visual Grad-CAM
explanations. Designed as a research and demonstration prototype.
</div>
""",
    unsafe_allow_html=True
)

st.divider()



# ============================================================
# PATIENT DETAILS
# ============================================================

st.markdown(
    '<div class="section-kicker">01 · PATIENT</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-heading">Patient & Screening Details</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-description">Enter the screening details that will appear on the generated patient report.</div>',
    unsafe_allow_html=True
)

case_id = st.text_input(
    "Case ID",
    value=f"RS-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    help="Automatically generated. You can replace it with your institution's case ID."
)

p1, p2, p3 = st.columns(3, gap="medium")
with p1:
    patient_name = st.text_input("Patient Name", placeholder="Enter patient name")
with p2:
    patient_age = st.number_input("Age", min_value=0, max_value=120, value=0, step=1)
with p3:
    patient_sex = st.selectbox("Sex", ["Female", "Male", "Other", "Prefer not to say"])

p4, p5, p6 = st.columns(3, gap="medium")
with p4:
    screening_centre = st.text_input("Screening Centre", placeholder="Centre / Hospital name")
with p5:
    eye_examined = st.selectbox("Eye Examined", ["Right", "Left", "Both", "Unknown"])
with p6:
    diabetes_type = st.selectbox("Diabetes Type", ["Type 1", "Type 2", "Other", "Unknown"])

p7, p8, p9 = st.columns(3, gap="medium")
with p7:
    diabetes_duration = st.text_input("Diabetes Duration", placeholder="e.g. 8 years")
with p8:
    previous_dr = st.selectbox("Previous DR History", ["Yes", "No", "Unknown"])
with p9:
    previous_eye_exam = st.selectbox("Previous Eye Examination", ["Yes", "No", "Unknown"])

st.divider()

# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-kicker">01 · INPUT</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-heading">Upload Fundus Image</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">Upload a retinal fundus photograph in JPG, JPEG, or PNG format.</div>',
    unsafe_allow_html=True
)

screening_mode = st.radio(
    "Screening Mode",
    ["Single Eye", "Both Eyes"],
    horizontal=True,
    help="Analyse one eye or upload separate right- and left-eye fundus images."
)

if screening_mode == "Single Eye":
    uploaded_file = st.file_uploader(
        "Choose a retinal fundus image",
        type=["jpg", "jpeg", "png"],
        key="single_eye_upload"
    )
    right_eye_file = None
    left_eye_file = None
else:
    up_right, up_left = st.columns(2, gap="medium")
    with up_right:
        right_eye_file = st.file_uploader(
            "Right Eye — Fundus Image",
            type=["jpg", "jpeg", "png"],
            key="right_eye_upload"
        )
    with up_left:
        left_eye_file = st.file_uploader(
            "Left Eye — Fundus Image",
            type=["jpg", "jpeg", "png"],
            key="left_eye_upload"
        )
    uploaded_file = None


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def run_eye_analysis(uploaded, generate_cam=True):
    """Run RetinaScope inference for one eye, with optional Grad-CAM."""
    img = Image.open(uploaded).convert("RGB")
    rgb, tensor = preprocess_image(img)

    with torch.inference_mode():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0, pred].item()

    cam = generate_gradcam(tensor, pred, rgb) if generate_cam else None

    return {
        "image": img,
        "rgb_image": rgb,
        "input_tensor": tensor,
        "probabilities": probs,
        "predicted_class": pred,
        "confidence": conf,
        "heatmap": cam,
        "image_id": uploaded.name,
    }


def run_dual_eye_analysis(right_uploaded, left_uploaded):
    """Batch EfficientNet inference for both eyes, then generate Grad-CAM."""
    right_img = Image.open(right_uploaded).convert("RGB")
    left_img = Image.open(left_uploaded).convert("RGB")

    right_rgb, right_tensor = preprocess_image(right_img)
    left_rgb, left_tensor = preprocess_image(left_img)

    # One EfficientNet forward pass for both eyes.
    batch = torch.cat([right_tensor, left_tensor], dim=0)

    with torch.inference_mode():
        output = model(batch)
        probabilities = torch.softmax(output, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

    right_pred = int(predictions[0].item())
    left_pred = int(predictions[1].item())
    right_conf = float(probabilities[0, right_pred].item())
    left_conf = float(probabilities[1, left_pred].item())

    # Grad-CAM is still the expensive CPU operation, but runs only after
    # the fast shared inference pass has completed.
    right_heatmap = generate_gradcam(
        right_tensor, right_pred, right_rgb
    )
    left_heatmap = generate_gradcam(
        left_tensor, left_pred, left_rgb
    )

    right_data = {
        "image": right_img,
        "rgb_image": right_rgb,
        "input_tensor": right_tensor,
        "probabilities": probabilities[0:1],
        "predicted_class": right_pred,
        "confidence": right_conf,
        "heatmap": right_heatmap,
        "image_id": right_uploaded.name,
    }

    left_data = {
        "image": left_img,
        "rgb_image": left_rgb,
        "input_tensor": left_tensor,
        "probabilities": probabilities[1:2],
        "predicted_class": left_pred,
        "confidence": left_conf,
        "heatmap": left_heatmap,
        "image_id": left_uploaded.name,
    }

    return right_data, left_data


def render_eye_result(data, label="Screening Result"):
    """Render the normal single-eye result UI."""
    pred = data["predicted_class"]
    conf = data["confidence"]
    probs = data["probabilities"]

    st.markdown('<div class="section-kicker">02 · AI ASSESSMENT</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-heading">{label}</div>', unsafe_allow_html=True)

    grade_col, confidence_col, screening_col = st.columns(3, gap="medium")

    with grade_col:
        st.markdown(
            f"""<div class="assessment-card">
            <div class="assessment-label">PREDICTED SEVERITY</div>
            <div class="assessment-value">Grade {pred}</div>
            <div class="assessment-subtitle">{CLASS_NAMES[pred]}</div>
            </div>""", unsafe_allow_html=True
        )

    with confidence_col:
        st.markdown(
            f"""<div class="assessment-card">
            <div class="assessment-label">MODEL CONFIDENCE</div>
            <div class="assessment-value">{conf*100:.2f}%</div>
            <div class="assessment-subtitle">Confidence for predicted class</div>
            </div>""", unsafe_allow_html=True
        )
        st.progress(float(conf))

    with screening_col:
        if pred >= 2:
            st.markdown(
                """<div class="status-card referable">
                <div class="status-label">SCREENING ASSESSMENT</div>
                <div class="status-value">REFERABLE DR</div>
                <div class="status-description">Predicted severity falls within Grade 2–4.</div>
                </div>""", unsafe_allow_html=True
            )
        else:
            st.markdown(
                """<div class="status-card normal">
                <div class="status-label">SCREENING ASSESSMENT</div>
                <div class="status-value">NON-REFERABLE</div>
                <div class="status-description">Predicted severity falls within Grade 0–1.</div>
                </div>""", unsafe_allow_html=True
            )

    st.divider()
    st.markdown('<div class="section-kicker">CLINICAL CONTEXT</div>', unsafe_allow_html=True)
    st.markdown(f"**Predicted classification:** {GRADE_DESCRIPTIONS[pred]}")
    st.divider()

    st.markdown('<div class="section-kicker">03 · EXPLAINABILITY</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Visual AI Explanation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-description">Grad-CAM highlights image regions that contributed to the selected prediction.</div>',
        unsafe_allow_html=True
    )

    tab_original, tab_gradcam, tab_compare = st.tabs(["Original Image", "Grad-CAM", "Side-by-Side"])
    with tab_original:
        st.image(data["image"], use_container_width=True)
    with tab_gradcam:
        st.image(data["heatmap"], use_container_width=True)
        st.caption("Warmer regions indicate stronger model attention.")
    with tab_compare:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown("#### Original Retina")
            st.image(data["image"], use_container_width=True)
        with c2:
            st.markdown("#### Grad-CAM Attention")
            st.image(data["heatmap"], use_container_width=True)

    st.divider()
    st.markdown('<div class="section-kicker">04 · MODEL OUTPUT</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Class Probability Distribution</div>', unsafe_allow_html=True)

    for i in range(5):
        probability = float(probs[0, i])
        a, b = st.columns([6,1], gap="medium")
        with a:
            st.caption(f"Grade {i} · {CLASS_NAMES[i]}")
            st.progress(probability)
        with b:
            st.markdown(f"**{probability*100:.2f}%**")


if screening_mode == "Single Eye":

    if uploaded_file is None:
        st.info("Upload a retinal fundus image above to begin screening.")
        st.markdown(
            """<div class="assessment-card">
            <div class="assessment-label">ANALYSIS PIPELINE</div>
            <div class="assessment-value">Image → AI → Explanation</div>
            <div class="assessment-subtitle">
            Upload a fundus image, receive a severity prediction, and inspect the Grad-CAM regions
            that contributed to the model's decision.
            </div></div>""",
            unsafe_allow_html=True
        )
    else:
        with st.spinner("Analysing retinal image..."):
            single_data = run_eye_analysis(uploaded_file)

        render_eye_result(single_data)

        st.divider()
        st.markdown('<div class="section-kicker">05 · PATIENT REPORT</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Generate Screening Report</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-description">Create a professional PDF containing patient details, AI results, Grad-CAM explanation and model information.</div>',
            unsafe_allow_html=True
        )

        patient_data = {
            "Case ID": case_id,
            "Patient Name": patient_name if patient_name.strip() else "Not provided",
            "Age": str(patient_age),
            "Sex": patient_sex,
            "Screening Date": datetime.now().strftime("%d %b %Y, %H:%M"),
            "Screening Centre": screening_centre if screening_centre.strip() else "Not provided",
            "Eye Examined": eye_examined,
            "Image ID": single_data["image_id"],
            "Diabetes Type": diabetes_type,
            "Diabetes Duration": diabetes_duration if diabetes_duration.strip() else "Not provided",
            "Previous DR": previous_dr,
            "Previous Eye Exam": previous_eye_exam,
        }

        report_bytes = generate_screening_report(
            patient_data, single_data["image"], single_data["heatmap"],
            single_data["probabilities"], single_data["predicted_class"], single_data["confidence"]
        )

        st.download_button(
            label="📄 Generate & Download Screening Report",
            data=report_bytes,
            file_name=f"RetinaScope_Report_{case_id}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

else:

    if right_eye_file is None or left_eye_file is None:
        st.info("Upload both the Right Eye and Left Eye fundus images to begin dual-eye screening.")
    else:
        with st.spinner("Running AI analysis on both eyes..."):
            right_data, left_data = run_dual_eye_analysis(
                right_eye_file,
                left_eye_file
            )

        st.markdown('<div class="section-kicker">02 · DUAL-EYE ASSESSMENT</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Right Eye vs Left Eye</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-description">Both images are evaluated together by EfficientNet-B0, followed by individual Grad-CAM explanations.</div>',
            unsafe_allow_html=True
        )

        rc, lc = st.columns(2, gap="medium")
        for col, label, data in [(rc, "Right Eye", right_data), (lc, "Left Eye", left_data)]:
            with col:
                st.markdown(f"### 👁 {label}")
                st.image(data["image"], use_container_width=True)
                st.metric(
                    "Predicted Grade",
                    f"Grade {data['predicted_class']} · {CLASS_NAMES[data['predicted_class']]}"
                )
                st.metric("Model Confidence", f"{data['confidence']*100:.2f}%")
                if data["predicted_class"] >= 2:
                    st.error("REFERABLE DR")
                else:
                    st.success("NON-REFERABLE")

        st.divider()
        rgrade = right_data["predicted_class"]
        lgrade = left_data["predicted_class"]
        diff = abs(rgrade - lgrade)

        st.markdown('<div class="section-kicker">03 · INTER-EYE ANALYSIS</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Severity Comparison</div>', unsafe_allow_html=True)

        a, b, c = st.columns(3, gap="medium")
        with a:
            st.metric("Right Eye", f"Grade {rgrade}")
        with b:
            st.metric("Left Eye", f"Grade {lgrade}")
        with c:
            st.metric("Grade Difference", str(diff))

        if diff == 0:
            st.success("Both eyes received the same predicted severity grade.")
        else:
            higher = "Right Eye" if rgrade > lgrade else "Left Eye"
            st.warning(f"Different predicted severity levels detected. {higher} has the higher predicted grade.")

        st.caption(
            "Inter-eye comparison describes model outputs for the two uploaded images. "
            "It is not a clinical diagnosis of disease asymmetry."
        )

        st.divider()
        st.markdown('<div class="section-kicker">04 · DUAL-EYE EXPLAINABILITY</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Model Attention — Both Eyes</div>', unsafe_allow_html=True)

        a, b = st.columns(2, gap="medium")
        with a:
            st.markdown("#### Right Eye · Grad-CAM")
            st.image(right_data["heatmap"], use_container_width=True)
        with b:
            st.markdown("#### Left Eye · Grad-CAM")
            st.image(left_data["heatmap"], use_container_width=True)

        st.caption(
            "Grad-CAM highlights regions that contributed to each selected prediction. "
            "Highlighted regions represent model attention, not confirmed lesions."
        )

        st.divider()
        st.markdown('<div class="section-kicker">05 · MODEL OUTPUT</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Probability Comparison</div>', unsafe_allow_html=True)

        for i in range(5):
            rp = float(right_data["probabilities"][0,i])
            lp = float(left_data["probabilities"][0,i])
            st.markdown(f"**Grade {i} · {CLASS_NAMES[i]}**")
            a, b = st.columns(2, gap="medium")
            with a:
                st.caption(f"Right Eye · {rp*100:.2f}%")
                st.progress(rp)
            with b:
                st.caption(f"Left Eye · {lp*100:.2f}%")
                st.progress(lp)

        st.divider()
        st.markdown('<div class="section-kicker">06 · PATIENT REPORT</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Generate Dual-Eye Screening Report</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-description">Create a PDF containing both-eye results, inter-eye comparison, Grad-CAM explanations and probability distributions.</div>',
            unsafe_allow_html=True
        )

        dual_patient_data = {
            "Case ID": case_id,
            "Patient Name": patient_name if patient_name.strip() else "Not provided",
            "Age": str(patient_age),
            "Sex": patient_sex,
            "Screening Date": datetime.now().strftime("%d %b %Y, %H:%M"),
            "Screening Centre": screening_centre if screening_centre.strip() else "Not provided",
            "Right Image ID": right_data["image_id"],
            "Left Image ID": left_data["image_id"],
            "Diabetes Type": diabetes_type,
            "Diabetes Duration": diabetes_duration if diabetes_duration.strip() else "Not provided",
            "Previous DR": previous_dr,
            "Previous Eye Exam": previous_eye_exam,
        }

        dual_report_bytes = generate_dual_eye_report(dual_patient_data, right_data, left_data)

        st.download_button(
            label="📄 Generate & Download Dual-Eye Report",
            data=dual_report_bytes,
            file_name=f"RetinaScope_DualEye_Report_{case_id}.pdf",
            mime="application/pdf",
            use_container_width=True
        )


# ========================================================
    # VALIDATION PERFORMANCE
    # ========================================================

    st.markdown(
        '<div class="section-kicker">07 · MODEL PERFORMANCE</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-heading">Validation Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">Performance measured on the held-out APTOS validation split.</div>',
        unsafe_allow_html=True
    )


    perf1, perf2, perf3, perf4 = st.columns(
        4,
        gap="medium"
    )


    with perf1:

        st.metric(
            "5-Class Accuracy",
            "80.22%"
        )


    with perf2:

        st.metric(
            "Macro F1",
            "65.03%"
        )


    with perf3:

        st.metric(
            "Referable Sensitivity",
            "90.27%"
        )


    with perf4:

        st.metric(
            "Referable Specificity",
            "92.18%"
        )


    st.divider()



    # ========================================================
    # RESOURCE PLANNING
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-kicker">08 · RESOURCE PLANNING</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-heading">District Screening Capacity</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-description">'
        'District-level planning view using Census 2011 population data and '
        'transparent prototype estimates for screening demand and capacity.'
        '</div>',
        unsafe_allow_html=True
    )

    try:
        planning_df = load_planning_data()

        state_col, priority_col = st.columns(2, gap="medium")

        with state_col:
            states = ["All States"] + sorted(
                planning_df["state"].dropna().unique().tolist()
            )
            selected_state = st.selectbox(
                "State", states, key="planning_state"
            )

        with priority_col:
            priorities = ["All Priorities", "Critical", "High", "Moderate", "Low"]
            selected_priority = st.selectbox(
                "Priority", priorities, key="planning_priority"
            )

        filtered = planning_df.copy()

        if selected_state != "All States":
            filtered = filtered[filtered["state"] == selected_state]

        if selected_priority != "All Priorities":
            filtered = filtered[filtered["priority"] == selected_priority]

        # Summary
        c1, c2, c3, c4 = st.columns(4, gap="medium")

        with c1:
            st.metric("Districts", f"{len(filtered):,}")

        with c2:
            st.metric(
                "Annual Demand",
                f"{int(filtered['estimated_annual_screening_demand'].sum()):,}"
            )

        with c3:
            st.metric(
                "Capacity Gap",
                f"{int(filtered['capacity_gap'].sum()):,}"
            )

        with c4:
            st.metric(
                "High / Critical",
                f"{int(filtered['priority'].isin(['High', 'Critical']).sum()):,}"
            )

        # ---------------- TABLE ----------------
        # Render the planning table BEFORE the map so district data remains
        # immediately visible even if browser-side map rendering is slow.
        st.markdown("#### Priority Districts")

        table = filtered[
            [
                "state",
                "district",
                "estimated_annual_screening_demand",
                "estimated_screening_capacity",
                "capacity_gap",
                "priority",
            ]
        ].sort_values(
            "capacity_gap",
            ascending=False
        )

        st.dataframe(
            table.rename(columns={
                "state": "State",
                "district": "District",
                "estimated_annual_screening_demand": "Annual Demand",
                "estimated_screening_capacity": "Capacity",
                "capacity_gap": "Capacity Gap",
                "priority": "Priority",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # ---------------- MAP ----------------
        try:
            map_df = build_map_data(selected_state)

            if selected_priority != "All Priorities":
                map_df = map_df[
                    map_df["priority"] == selected_priority
                ].copy()

            if not map_df.empty:
                geojson = map_df.__geo_interface__

                priority_order = [
                    "Low",
                    "Moderate",
                    "High",
                    "Critical"
                ]

                map_df["priority"] = pd.Categorical(
                    map_df["priority"],
                    categories=priority_order,
                    ordered=True,
                )

                fig = px.choropleth(
                    map_df,
                    geojson=geojson,
                    locations="map_id",
                    featureidkey="properties.map_id",
                    color="priority",
                    category_orders={"priority": priority_order},
                    color_discrete_map={
                        "Low": "#1f9d55",
                        "Moderate": "#b7d84b",
                        "High": "#f2a93b",
                        "Critical": "#d64545",
                    },
                    hover_name="district",
                    hover_data={
                        "state": True,
                        "priority": True,
                        "capacity_gap": ":,",
                        "estimated_annual_screening_demand": ":,",
                        "estimated_screening_capacity": ":,",
                        "priority_score": ":.1f",
                        "map_id": False,
                    },
                    labels={
                        "priority": "Priority",
                        "capacity_gap": "Capacity Gap",
                        "estimated_annual_screening_demand": "Annual Demand",
                        "estimated_screening_capacity": "Capacity",
                        "priority_score": "Priority Score",
                    },
                )

                fig.update_geos(
                    fitbounds="locations",
                    visible=False,
                    bgcolor="rgba(0,0,0,0)",
                )

                fig.update_layout(
                    height=520,
                    margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#cbd5e1"),
                    legend=dict(
                        title="Priority",
                        orientation="h",
                        yanchor="bottom",
                        y=1.01,
                        xanchor="right",
                        x=1,
                    ),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displayModeBar": False,
                        "responsive": True,
                    },
                )
            else:
                st.info("No mapped districts match the selected filters.")

        except Exception as map_error:
            st.warning(
                f"District map unavailable: {map_error}"
            )

        st.caption(
            "Planning estimates are prototype assumptions derived from "
            "district-level Census 2011 population data; they are not observed "
            "healthcare capacity statistics."
        )

    except Exception:
        st.error(
            "Planning data could not be loaded. Make sure planning_data.csv "
            "is in the project root."
        )


    # ========================================================
    # MEDICAL DISCLAIMER
    # ========================================================

    st.warning(
        "Research prototype only. The AI output and Grad-CAM "
        "visualization are not a clinical diagnosis and should "
        "not replace evaluation by a qualified healthcare professional. "
        "Reported performance is based on an internal APTOS "
        "validation split and does not establish clinical validity."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="app-footer">

<b>RETINASCOPE</b> · EXPLAINABLE AI FOR DIABETIC RETINOPATHY SCREENING

<br>

EfficientNet-B0 · Grad-CAM · APTOS 2019

<br>

Research & Demonstration Prototype

</div>
""",
    unsafe_allow_html=True
)