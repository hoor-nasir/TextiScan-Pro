
import io
import json
import math
import sqlite3
import time
import wave
from datetime import datetime
from html import escape
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "best.pt"

DB_PATH = ROOT / "textiscan_history.db"

st.set_page_config(
    page_title="TextiScan Pro | Command Center",
    page_icon="🔍",
    layout="wide",
)


# ============================================================
# THEME AND CUSTOM CSS
# ============================================================

# IMPORTANT:
# HTML starts at the beginning of the string.
# This prevents Streamlit from displaying HTML as a code block.

st.markdown(
    """<style>
.stApp {
    background: linear-gradient(130deg, #0a1018, #101d2a);
    color: #e5f4fa;
}

[data-testid="stSidebar"] {
    background: #0e1c29;
    border-right: 1px solid #244252;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.6rem;
}

h1, h2, h3 {
    color: #58e5f1 !important;
}

/* SIDEBAR BRANDING */

.brand {
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 8px 0 16px;
    border-bottom: 1px solid #2b4050;
    margin-bottom: 13px;
}

.brand-symbol {
    width: 45px;
    height: 45px;
    flex: 0 0 45px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 11px;
    background: #173b49;
    border: 1px solid #2c8497;
    color: #57e6ef;
}

.brand-symbol svg {
    width: 25px;
    height: 25px;
}

.brand-name {
    font-size: 21px;
    font-weight: 800;
    white-space: nowrap;
    letter-spacing: -0.65px;
    color: #66e8f0;
    line-height: 1.2;
}

.brand-name span {
    color: #ffffff;
}

.brand small {
    display: block;
    color: #9eb5c4;
    letter-spacing: 1.2px;
    font-size: 9px;
    margin-top: 5px;
    white-space: nowrap;
}

.sidebar-group {
    color: #f0f8ff;
    font-size: 14px;
    font-weight: 700;
    margin: 20px 0 11px;
    display: flex;
    align-items: center;
    gap: 9px;
}

.sidebar-group svg {
    width: 18px;
    height: 18px;
    stroke: #58dce8;
    stroke-width: 2;
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
}

.engine-status {
    display: flex;
    align-items: center;
    gap: 9px;
    color: #b9f5cd;
    font-weight: 600;
    font-size: 13px;
    margin: 13px 0 7px;
}

.engine-status .dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #4ade80;
    box-shadow: 0 0 0 3px #183e33;
}

.engine-meta {
    color: #92aab9;
    font-size: 11px;
    margin-bottom: 14px;
}

[data-testid="stSidebar"] .stButton button {
    background: #193544;
    border: 1px solid #2d7081;
    color: #78e8f0;
    font-weight: 650;
    border-radius: 9px;
}

[data-testid="stSidebar"] .stButton button:hover {
    border-color: #67e8f9;
    color: #e8ffff;
    background: #214758;
}

[data-testid="stSidebar"] hr {
    border-color: #2b4050;
}

/* MAIN HEADER */

.hero {
    padding: 14px 0 18px;
    border-bottom: 1px solid #294454;
    margin-bottom: 18px;
}

.hero h1 {
    font-size: 44px;
    margin: 0;
    color: #e9fcff !important;
    letter-spacing: -1px;
}

.hero p {
    color: #9bb6c6;
    margin-top: 8px;
}

.eyebrow {
    font-size: 11px;
    letter-spacing: 2.5px;
    color: #44dce9;
    font-weight: 700;
}

/* SUPPORTED DEFECT CLASSES */

.class-tile {
    border: 1px solid #2c4556;
    background: #142535;
    border-radius: 10px;
    padding: 13px;
    min-height: 91px;
}

.class-tile b {
    color: #dffaff;
    font-size: 14px;
}

.class-tile small {
    color: #99b1c2;
    display: block;
    margin-top: 7px;
}

[data-testid="stExpander"] {
    border: 1px solid #294657 !important;
    border-radius: 12px !important;
    background: #101f2d !important;
}

/* METRIC CARDS */

[data-testid="stMetric"] {
    background: #132433;
    border: 1px solid #2c4556;
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: #9cb8c8;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #e8f9ff;
}

[data-testid="stTabs"] button {
    font-weight: 600;
}

/* INSPECTION STATUS */

.pass {
    padding: 14px;
    background: #133d30;
    color: #80f2b7;
    border: 1px solid #287854;
    border-radius: 10px;
}

.alert {
    padding: 14px;
    background: #42242c;
    color: #ff9ba7;
    border: 1px solid #a44855;
    border-radius: 10px;
}

/* INSPECTION SUMMARY */

.result-summary {
    background: #132638;
    border: 1px solid #2e5263;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 16px 0;
}

.result-summary .rs-title {
    color: #a9c0ce;
    font-size: 12px;
    letter-spacing: 1.3px;
    text-transform: uppercase;
}

.result-summary .rs-value {
    color: #eaf9ff;
    font-size: 23px;
    font-weight: 750;
    margin-top: 4px;
}

.result-summary .rs-meta {
    color: #a8bdcb;
    font-size: 13px;
    margin-top: 8px;
}
</style>""",
    unsafe_allow_html=True,
)


# ============================================================
# DATABASE
# ============================================================

def db_connect():
    conn = sqlite3.connect(DB_PATH, timeout=10)

    conn.execute(
        """CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            inspected_at TEXT NOT NULL,
            filename TEXT NOT NULL,
            status TEXT NOT NULL,
            defect_count INTEGER NOT NULL,
            latency_ms REAL NOT NULL,
            threshold REAL NOT NULL,
            predictions TEXT NOT NULL,
            feedback TEXT DEFAULT 'Not reviewed',
            notes TEXT DEFAULT ''
        )"""
    )

    conn.commit()
    return conn


def save_inspection(
    filename,
    count,
    latency,
    threshold,
    predictions,
):
    with db_connect() as conn:
        cursor = conn.execute(
            """INSERT INTO inspections
            (
                inspected_at,
                filename,
                status,
                defect_count,
                latency_ms,
                threshold,
                predictions
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                filename,
                (
                    "DEFECT DETECTED"
                    if count
                    else "NO DEFECT DETECTED"
                ),
                count,
                latency,
                threshold,
                json.dumps(predictions),
            ),
        )

        conn.commit()
        return cursor.lastrowid


def read_history():
    with db_connect() as conn:
        return pd.read_sql_query(
            "SELECT * FROM inspections ORDER BY id DESC",
            conn,
        )


def update_feedback(record_id, feedback, notes):
    with db_connect() as conn:
        conn.execute(
            """UPDATE inspections
            SET feedback=?, notes=?
            WHERE id=?""",
            (
                feedback,
                notes,
                int(record_id),
            ),
        )
        conn.commit()


# ============================================================
# LOAD YOLO MODEL
# ============================================================

@st.cache_resource
def load_model(path):
    return YOLO(str(path))


# ============================================================
# INDUSTRIAL BUZZER
# ============================================================

@st.cache_data
def alarm_audio():
    rate = 22050
    duration = 0.35

    samples = bytearray()

    for i in range(int(rate * duration)):
        t = i / rate

        envelope = max(
            0,
            min(
                1,
                t / 0.02,
                (duration - t) / 0.04,
            ),
        )

        value = int(
            17000
            * envelope
            * (
                math.sin(2 * math.pi * 830 * t)
                + 0.25
                * math.sin(2 * math.pi * 1250 * t)
            )
        )

        value = max(
            -32768,
            min(32767, value),
        )

        samples += value.to_bytes(
            2,
            "little",
            signed=True,
        )

    output = io.BytesIO()

    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples)

    return output.getvalue()


def buzzer():
    st.audio(
        alarm_audio(),
        format="audio/wav",
        autoplay=True,
    )


# ============================================================
# IMAGE DETECTION
# ============================================================

def inspect(
    image,
    model,
    conf,
    iou,
    show_classes=True,
):
    start = time.perf_counter()

    result = model.predict(
        image,
        conf=conf,
        iou=iou,
        agnostic_nms=True,
        verbose=False,
    )[0]

    latency = (
        time.perf_counter() - start
    ) * 1000

    annotated = image.copy()
    predictions = []

    for box in result.boxes:
        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].cpu().tolist(),
        )

        name = result.names[
            int(box.cls[0])
        ]

        confidence = float(box.conf[0])

        predictions.append(
            {
                "class": name,
                "confidence": round(
                    confidence,
                    4,
                ),
                "box": [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
            }
        )

        # Red defect bounding box

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3,
        )

        # Red filled label with white text

        label = (
            f"{name} | {confidence:.0%}"
            if show_classes
            else "DEFECT DETECTED"
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 2

        (text_w, text_h), baseline = (
            cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness,
            )
        )

        img_h, img_w = annotated.shape[:2]

        label_x = min(
            max(0, x1),
            max(0, img_w - text_w - 14),
        )

        if y1 >= text_h + 20:
            label_y = y1 - text_h - 15
        else:
            label_y = min(
                img_h - text_h - 12,
                y2 + 6,
            )

        label_y = max(
            0,
            label_y,
        )

        cv2.rectangle(
            annotated,
            (
                label_x,
                label_y,
            ),
            (
                min(
                    img_w - 1,
                    label_x + text_w + 12,
                ),
                min(
                    img_h - 1,
                    label_y + text_h + 12,
                ),
            ),
            (0, 0, 190),
            -1,
        )

        cv2.putText(
            annotated,
            label,
            (
                label_x + 6,
                label_y + text_h + 5,
            ),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return annotated, predictions, latency


def png_bytes(image):
    success, encoded = cv2.imencode(
        ".png",
        image,
    )

    if not success:
        raise ValueError(
            "Could not encode image."
        )

    return encoded.tobytes()


# ============================================================
# PDF REPORT GENERATION
# ============================================================

def make_pdf(
    filename,
    original,
    annotated,
    predictions,
    latency,
    threshold,
):
    output = io.BytesIO()

    pdf = canvas.Canvas(
        output,
        pagesize=A4,
    )

    width, height = A4

    pdf.setFillColor(
        colors.HexColor("#0f2532")
    )

    pdf.rect(
        0,
        height - 100,
        width,
        100,
        fill=1,
        stroke=0,
    )

    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 22)

    pdf.drawString(
        40,
        height - 49,
        "TEXTISCAN PRO",
    )

    pdf.setFont("Helvetica", 10)

    pdf.drawString(
        40,
        height - 69,
        "AI Textile Quality Inspection Report",
    )

    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica", 10)

    status = (
        "DEFECT DETECTED - REVIEW"
        if predictions
        else "NO DEFECT DETECTED"
    )

    lines = [
        (
            "Generated: "
            f"{datetime.now():%Y-%m-%d %H:%M:%S}"
        ),
        f"File: {filename[:75]}",
        f"Status: {status}",
        (
            f"Detections: {len(predictions)} | "
            f"Threshold: {threshold:.2f} | "
            f"Latency: {latency:.1f} ms"
        ),
    ]

    y = height - 125

    for line in lines:
        pdf.drawString(
            40,
            y,
            line,
        )
        y -= 19

    def draw_photo(
        image,
        x,
        top_y,
        max_w=245,
        max_h=205,
    ):
        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        h, w = rgb.shape[:2]

        scale = min(
            max_w / w,
            max_h / h,
        )

        dw = w * scale
        dh = h * scale

        pdf.drawImage(
            ImageReader(
                Image.fromarray(rgb)
            ),
            x,
            top_y - dh,
            dw,
            dh,
        )

    pdf.setFont(
        "Helvetica-Bold",
        11,
    )

    pdf.drawString(
        40,
        y - 10,
        "ORIGINAL FABRIC",
    )

    pdf.drawString(
        310,
        y - 10,
        "AI DETECTION RESULT",
    )

    draw_photo(
        original,
        40,
        y - 25,
    )

    draw_photo(
        annotated,
        310,
        y - 25,
    )

    y -= 260

    pdf.setFont(
        "Helvetica-Bold",
        11,
    )

    pdf.drawString(
        40,
        y,
        "DETECTED DEFECTS",
    )

    y -= 20

    pdf.setFont(
        "Helvetica",
        10,
    )

    if not predictions:
        pdf.drawString(
            40,
            y,
            "No predictions above the selected confidence threshold.",
        )

    else:
        for idx, pred in enumerate(
            predictions[:15],
            1,
        ):
            line = (
                f'{idx}. {pred["class"]} - '
                f'{pred["confidence"]:.1%} confidence'
            )

            pdf.drawString(
                40,
                y,
                line,
            )

            y -= 17

            if y < 65:
                pdf.showPage()
                y = height - 50

    pdf.setFont(
        "Helvetica-Oblique",
        9,
    )

    pdf.drawString(
        40,
        35,
        "Prototype result: no detection does not guarantee defect-free fabric.",
    )

    pdf.save()

    return output.getvalue()


# ============================================================
# UPDATED SIDEBAR
# ============================================================

with st.sidebar:

    # FIXED: HTML begins at column zero in the string.

    st.markdown(
        """<div class="brand">
<div class="brand-symbol">
<svg viewBox="0 0 24 24" aria-hidden="true">
<path d="M8 3H5a2 2 0 0 0-2 2v3M16 3h3a2 2 0 0 1 2 2v3M3 16v3a2 2 0 0 0 2 2h3M21 16v3a2 2 0 0 1-2 2h-3M4 12h16M9 9l-2 3 2 3M15 9l2 3-2 3" stroke="currentColor" stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
</div>
<div>
<div class="brand-name">TextiScan <span>Pro</span></div>
<small>INDUSTRIAL AI INSPECTION</small>
</div>
</div>
<div class="engine-status">
<span class="dot"></span>
AI Engine Online
</div>
<div class="engine-meta">
BDFD9 · YOLOv8 · 9 defect classes
</div>""",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        """<div class="sidebar-group">
<svg viewBox="0 0 24 24">
<path d="M4 7h16M4 17h16M9 4v6M15 14v6"/>
</svg>
Detection Settings
</div>""",
        unsafe_allow_html=True,
    )

    conf = st.slider(
        "Confidence threshold",
        0.05,
        0.95,
        0.40,
        0.05,
    )

    iou = st.slider(
        "Overlap suppression (IoU)",
        0.10,
        0.80,
        0.30,
        0.05,
    )

    show_classes = st.toggle(
        "Show class names and confidence",
        True,
    )

    st.divider()

    st.markdown(
        """<div class="sidebar-group">
<svg viewBox="0 0 24 24">
<path d="M11 5 6 9H3v6h3l5 4V5ZM15 9a5 5 0 0 1 0 6M18 6a9 9 0 0 1 0 12"/>
</svg>
Audio Alerts
</div>""",
        unsafe_allow_html=True,
    )

    sound_on = st.toggle(
        "Defect Buzzer",
        False,
    )

    if st.button(
        "🔊 Test Buzzer",
        use_container_width=True,
    ):
        buzzer()

    st.caption(
        "Audio autoplay may be blocked by browser settings."
    )

    st.divider()

    st.caption(
        "Inspection history is stored locally "
        "in textiscan_history.db."
    )


# ============================================================
# INITIALIZE MODEL
# ============================================================

if not MODEL_PATH.exists():
    st.error(
        f"Model not found: {MODEL_PATH}"
    )
    st.stop()

try:
    model = load_model(
        str(MODEL_PATH)
    )

except Exception as exc:
    st.error(
        f"Unable to load model: {exc}"
    )
    st.stop()


# ============================================================
# MAIN HEADER
# ============================================================

# FIXED: No leading indentation in HTML.

st.markdown(
    """<div class="hero">
<div class="eyebrow">
AI-POWERED TEXTILE QUALITY ASSURANCE
</div>
<h1>TextiScan Pro</h1>
<p>
Industrial Inspection Command Center ·
Defect Detection · Quality Analytics
</p>
</div>""",
    unsafe_allow_html=True,
)


# ============================================================
# SUPPORTED DEFECT CLASSES
# ============================================================

CLASS_DESCRIPTIONS = {
    "DropStitch": "Missing or interrupted stitch",
    "Hole": "Opening or puncture in the fabric",
    "Horizontal Mark": "Crosswise irregular line or mark",
    "Knot": "Visible knot in the yarn",
    "Oil Spot": "Oil or grease contamination",
    "Setup": "Fabric setup or pattern irregularity",
    "Slub": "Thick or uneven yarn section",
    "Vertical LineMark": "Lengthwise irregular line or mark",
    "Yurn Dust": "Dust or fiber contamination",
}

with st.expander(
    f"🧵 Supported Defect Classes ({len(model.names)}) "
    "· Explore the AI model"
):
    st.caption(
        "These are the classes the trained model can predict. "
        "Detection is not guaranteed on every fabric."
    )

    class_names = [
        model.names[i]
        for i in sorted(model.names)
    ]

    for offset in range(
        0,
        len(class_names),
        3,
    ):
        tiles = st.columns(3)

        for j, name in enumerate(
            class_names[offset:offset + 3]
        ):
            with tiles[j]:
                display_name = (
                    "Yarn Dust"
                    if name == "Yurn Dust"
                    else name
                )

                description = (
                    CLASS_DESCRIPTIONS.get(
                        name,
                        "Textile surface irregularity",
                    )
                )

                st.markdown(
                    '<div class="class-tile">'
                    f'<b>{offset + j + 1:02d} · '
                    f'{escape(display_name)}</b>'
                    f'<small>{escape(description)}</small>'
                    '</div>',
                    unsafe_allow_html=True,
                )


# ============================================================
# INSPECTION OVERVIEW
# ============================================================

overview = read_history()

inspections_total = len(overview)

flagged_total = (
    int(
        (overview["defect_count"] > 0).sum()
    )
    if inspections_total
    else 0
)

detections_total = (
    int(
        overview["defect_count"].sum()
    )
    if inspections_total
    else 0
)

pass_rate = (
    (
        inspections_total - flagged_total
    )
    / inspections_total
    * 100
    if inspections_total
    else None
)

avg_latency = (
    float(
        overview["latency_ms"].mean()
    )
    if inspections_total
    else None
)

st.markdown(
    "### Inspection Overview"
)

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Total inspections",
    inspections_total,
)

k2.metric(
    "Defects detected",
    detections_total,
)

k3.metric(
    "No-detection rate",
    (
        f"{pass_rate:.1f}%"
        if pass_rate is not None
        else "—"
    ),
)

k4.metric(
    "Avg. processing time",
    (
        f"{avg_latency:.1f} ms"
        if avg_latency is not None
        else "—"
    ),
)

st.caption(
    "Overview reflects saved inspections only. "
    "No-detection rate is not a verified quality pass rate."
)


# ============================================================
# NAVIGATION TABS
# ============================================================

tab_image, tab_history, tab_analytics, tab_camera = (
    st.tabs(
        [
            "🔍 Image Inspection",
            "🗂️ Inspection History",
            "📊 Analytics",
            "📹 Live Camera",
        ]
    )
)


# ============================================================
# IMAGE INSPECTION
# ============================================================

with tab_image:

    st.subheader(
        "Batch Quality Inspection"
    )

    uploaded = st.file_uploader(
        "Upload a fabric image",
        type=[
            "png",
            "jpg",
            "jpeg",
            "bmp",
            "webp",
        ],
    )

    if uploaded is not None:

        original = cv2.imdecode(
            np.frombuffer(
                uploaded.getvalue(),
                np.uint8,
            ),
            cv2.IMREAD_COLOR,
        )

        if original is None:
            st.error(
                "Cannot read this image."
            )

        else:
            annotated, predictions, latency = (
                inspect(
                    original,
                    model,
                    conf,
                    iou,
                    show_classes,
                )
            )

            count = len(predictions)

            col1, col2 = st.columns(
                2,
                gap="medium",
            )

            with col1:
                st.markdown(
                    "#### 01 / ORIGINAL FABRIC"
                )

                st.image(
                    cv2.cvtColor(
                        original,
                        cv2.COLOR_BGR2RGB,
                    ),
                    use_container_width=True,
                )

            with col2:
                st.markdown(
                    "#### 02 / AI INSPECTION RESULT"
                )

                st.image(
                    cv2.cvtColor(
                        annotated,
                        cv2.COLOR_BGR2RGB,
                    ),
                    use_container_width=True,
                )

            # STATUS ALERT

            if count:
                st.markdown(
                    '<div class="alert">'
                    '🚨 DEFECT DETECTED — '
                    f'{count} prediction(s). '
                    'REVIEW REQUIRED.'
                    '</div>',
                    unsafe_allow_html=True,
                )

            else:
                st.markdown(
                    '<div class="pass">'
                    '✓ NO DEFECT DETECTED '
                    'AT CURRENT THRESHOLD'
                    '</div>',
                    unsafe_allow_html=True,
                )

            # INSPECTION METRICS

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Defects detected",
                count,
            )

            m2.metric(
                "Inference latency",
                f"{latency:.1f} ms",
            )

            m3.metric(
                "Confidence threshold",
                f"{conf:.2f}",
            )

            # INSPECTION RESULT SUMMARY

            if predictions:

                top_prediction = max(
                    predictions,
                    key=lambda p: p["confidence"],
                )

                st.markdown(
                    '<div class="result-summary">'
                    '<div class="rs-title">'
                    'Inspection result · Review required'
                    '</div>'
                    '<div class="rs-value">'
                    f'{escape(top_prediction["class"])} '
                    f'· {top_prediction["confidence"]:.0%} confidence'
                    '</div>'
                    '<div class="rs-meta">'
                    f'{count} total prediction(s) · '
                    'Highest-confidence class shown'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

            else:
                st.markdown(
                    '<div class="result-summary">'
                    '<div class="rs-title">'
                    'Inspection result'
                    '</div>'
                    '<div class="rs-value">'
                    'No defect detected'
                    '</div>'
                    '<div class="rs-meta">'
                    'No predictions above the selected '
                    'confidence threshold.'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

            # INDIVIDUAL CONFIDENCE

            st.markdown(
                "#### Individual Defect Confidence"
            )

            if predictions:

                for index, item in enumerate(
                    predictions,
                    1,
                ):
                    st.write(
                        f'**{index}. {item["class"]}** '
                        f'— {item["confidence"]:.1%}'
                    )

                    st.progress(
                        min(
                            1.0,
                            item["confidence"],
                        )
                    )

            else:
                st.caption(
                    "No predictions above threshold."
                )

            # INDUSTRIAL BUZZER

            if sound_on and count:

                alarm_id = (
                    f"{uploaded.name}-"
                    f"{len(uploaded.getvalue())}-"
                    f"{conf}-{iou}"
                )

                if (
                    st.session_state.get(
                        "last_alarm_id"
                    )
                    != alarm_id
                ):
                    buzzer()

                    st.session_state.last_alarm_id = (
                        alarm_id
                    )

            # DOWNLOAD AND SAVE ACTIONS

            d1, d2, d3 = st.columns(3)

            with d1:
                st.download_button(
                    "⬇ Download annotated PNG",
                    png_bytes(annotated),
                    "textiscan_result.png",
                    "image/png",
                    use_container_width=True,
                )

            with d2:

                report = make_pdf(
                    uploaded.name,
                    original,
                    annotated,
                    predictions,
                    latency,
                    conf,
                )

                st.download_button(
                    "📄 Download PDF report",
                    report,
                    "textiscan_report.pdf",
                    "application/pdf",
                    use_container_width=True,
                )

            with d3:

                if st.button(
                    "💾 Save inspection to history",
                    use_container_width=True,
                ):

                    record_id = save_inspection(
                        uploaded.name,
                        count,
                        latency,
                        conf,
                        predictions,
                    )

                    st.success(
                        f"Inspection #{record_id} saved."
                    )

            st.caption(
                "Save inspection to history before leaving "
                "this image if you want a permanent record."
            )

    else:
        st.info(
            "Upload an image to begin inspection."
        )


# ============================================================
# INSPECTION HISTORY
# ============================================================

with tab_history:

    st.subheader(
        "Saved Inspection History"
    )

    history = read_history()

    if history.empty:

        st.info(
            "No inspections saved yet. "
            "Inspect an image and click "
            "Save inspection to history."
        )

    else:

        st.dataframe(
            history[
                [
                    "id",
                    "inspected_at",
                    "filename",
                    "status",
                    "defect_count",
                    "latency_ms",
                    "feedback",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇ Export history CSV",
            history.to_csv(
                index=False
            ).encode("utf-8"),
            "textiscan_history.csv",
            "text/csv",
        )

        st.markdown(
            "#### Manual Review & Feedback"
        )

        selected_id = st.selectbox(
            "Choose inspection ID",
            history["id"].tolist(),
            format_func=lambda value: (
                f"#{value} — "
                + history.loc[
                    history["id"] == value,
                    "filename",
                ].iloc[0]
            ),
        )

        selected = history.loc[
            history["id"] == selected_id
        ].iloc[0]

        options = [
            "Not reviewed",
            "Correct detection",
            "False alarm",
            "Missed defect",
            "Needs further review",
        ]

        current = (
            selected["feedback"]
            if selected["feedback"] in options
            else "Not reviewed"
        )

        with st.form(
            "feedback_form"
        ):

            feedback = st.selectbox(
                "Review result",
                options,
                index=options.index(current),
            )

            notes = st.text_area(
                "Reviewer notes",
                value=selected["notes"] or "",
                placeholder=(
                    "Example: Slub was visible "
                    "but the model missed it."
                ),
            )

            if st.form_submit_button(
                "Save feedback"
            ):

                update_feedback(
                    selected_id,
                    feedback,
                    notes,
                )

                st.success(
                    "Feedback saved."
                )

                st.rerun()

        st.caption(
            "Feedback is for evaluation only; "
            "it does not automatically retrain the model."
        )


# ============================================================
# ANALYTICS
# ============================================================

with tab_analytics:

    st.subheader(
        "Inspection Analytics"
    )

    history = read_history()

    if history.empty:

        st.info(
            "No inspection history yet. "
            "Save your first inspection "
            "to populate the charts."
        )

    else:

        total = len(history)

        flagged = int(
            (
                history["defect_count"] > 0
            ).sum()
        )

        defects = int(
            history["defect_count"].sum()
        )

        reviewed = int(
            (
                history["feedback"]
                != "Not reviewed"
            ).sum()
        )

        k1, k2, k3, k4 = st.columns(4)

        k1.metric(
            "Total inspections",
            total,
        )

        k2.metric(
            "Flagged inspections",
            flagged,
        )

        k3.metric(
            "Defect predictions",
            defects,
        )

        k4.metric(
            "Reviewed",
            reviewed,
        )

        # INSPECTION ACTIVITY

        left, right = st.columns(
            [3, 2],
            gap="large",
        )

        with left:

            st.markdown(
                "#### Inspection Activity"
            )

            timeline = history.copy()

            timeline["Date"] = pd.to_datetime(
                timeline["inspected_at"]
            ).dt.date

            daily = (
                timeline.groupby("Date")
                .size()
                .rename("Inspections")
            )

            st.line_chart(
                daily,
                height=300,
            )

        with right:

            st.markdown(
                "#### Inspection Outcomes"
            )

            outcomes = pd.DataFrame(
                {
                    "Outcome": [
                        "No detection",
                        "Flagged",
                    ],
                    "Inspections": [
                        total - flagged,
                        flagged,
                    ],
                }
            ).set_index("Outcome")

            st.bar_chart(
                outcomes,
                height=300,
            )

        # DEFECT DISTRIBUTION

        left, right = st.columns(
            [3, 2],
            gap="large",
        )

        with left:

            st.markdown(
                "#### Defect Distribution"
            )

            counts = {}

            for raw in history["predictions"]:

                try:
                    preds = json.loads(raw)

                except (
                    TypeError,
                    ValueError,
                ):
                    preds = []

                for pred in preds:

                    name = pred.get(
                        "class",
                        "Unknown",
                    )

                    counts[name] = (
                        counts.get(name, 0) + 1
                    )

            if counts:

                distribution = (
                    pd.Series(
                        counts,
                        name="Detections",
                    )
                    .sort_values(
                        ascending=False
                    )
                )

                st.bar_chart(
                    distribution,
                    horizontal=True,
                    height=340,
                )

            else:

                st.info(
                    "No detected defect categories "
                    "in saved inspections yet."
                )

        with right:

            st.markdown(
                "#### Review Feedback"
            )

            feedback_counts = (
                history["feedback"]
                .fillna("Not reviewed")
                .value_counts()
            )

            st.bar_chart(
                feedback_counts,
                horizontal=True,
                height=340,
            )

        # RECENT INSPECTIONS

        st.markdown(
            "#### Recent Inspections"
        )

        recent = history[
            [
                "inspected_at",
                "filename",
                "status",
                "defect_count",
                "feedback",
            ]
        ].head(8)

        st.dataframe(
            recent,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Charts use locally saved inspection records, "
            "not the training or test dataset."
        )


# ============================================================
# LIVE CAMERA
# ============================================================

with tab_camera:

    st.subheader(
        "Live Camera Inspection"
    )

    st.caption(
        "Laptop camera uses the computer running Streamlit. "
        "Network sources must be reachable video streams."
    )

    source_kind = st.radio(
        "Camera source",
        [
            "Laptop / USB camera",
            "IP camera / network stream",
        ],
        horizontal=True,
    )

    if source_kind == "Laptop / USB camera":

        source = int(
            st.number_input(
                "Camera index",
                0,
                10,
                0,
            )
        )

    else:

        source = st.text_input(
            "Camera stream URL",
            placeholder="rtsp://camera-host/stream",
        )

    if "camera_cap" not in st.session_state:
        st.session_state.camera_cap = None

    if "camera_last_alarm" not in st.session_state:
        st.session_state.camera_last_alarm = 0.0

    start, stop = st.columns(2)

    if start.button(
        "▶ Start camera",
        type="primary",
        use_container_width=True,
    ):

        if (
            isinstance(source, str)
            and not source.strip()
        ):

            st.warning(
                "Enter a camera URL first."
            )

        else:

            old_cap = (
                st.session_state.camera_cap
            )

            if old_cap is not None:
                old_cap.release()

            if isinstance(source, int):

                cap = cv2.VideoCapture(
                    source,
                    cv2.CAP_DSHOW,
                )

            else:

                cap = cv2.VideoCapture(
                    source
                )

            if cap.isOpened():

                st.session_state.camera_cap = cap

                st.session_state.camera_last_alarm = (
                    0.0
                )

                st.success(
                    "Camera connected."
                )

            else:

                cap.release()

                st.session_state.camera_cap = None

                st.error(
                    "Camera not available. "
                    "Check camera permissions, index or URL."
                )

    if stop.button(
        "■ Stop camera",
        use_container_width=True,
    ):

        cap = st.session_state.camera_cap

        if cap is not None:
            cap.release()

        st.session_state.camera_cap = None

        st.info(
            "Camera stopped."
        )

    @st.fragment(run_every="0.5s")
    def camera_view():

        cap = st.session_state.get(
            "camera_cap"
        )

        if cap is None:

            st.info(
                "Camera idle. Click Start camera."
            )
            return

        if not cap.isOpened():

            st.warning(
                "Camera disconnected. Stop and restart."
            )
            return

        ok, frame = cap.read()

        if not ok:

            st.warning(
                "Unable to read camera frame."
            )
            return

        annotated, predictions, latency = inspect(
            frame,
            model,
            conf,
            iou,
            show_classes,
        )

        st.image(
            cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB,
            ),
            use_container_width=True,
        )

        count = len(predictions)

        if count:

            st.error(
                f"🚨 LIVE ALERT: "
                f"{count} defect prediction(s)"
            )

        else:

            st.success(
                "No defect detected in current frame"
            )

        p1, p2, p3 = st.columns(3)

        p1.metric(
            "Frame detections",
            count,
        )

        p2.metric(
            "Inference latency",
            f"{latency:.1f} ms",
        )

        p3.metric(
            "Approx. model FPS",
            (
                f"{1000 / latency:.1f}"
                if latency > 0
                else "N/A"
            ),
        )

        st.caption(
            "Approximate model FPS excludes capture, "
            "rendering and network delays."
        )

        if (
            sound_on
            and count
            and (
                time.monotonic()
                - st.session_state.camera_last_alarm
            ) > 5
        ):

            buzzer()

            st.session_state.camera_last_alarm = (
                time.monotonic()
            )

    camera_view()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "TextiScan Pro · Academic prototype · "
    "A clean result means no predictions above "
    "the chosen threshold, not guaranteed "
    "defect-free material."
)
