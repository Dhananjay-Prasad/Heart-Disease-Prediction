"""
CardioScan — Heart Disease Risk Estimator
A Streamlit front-end for the model trained in heart_disease.ipynb.

Run with:
    streamlit run app.py

Requires heart_disease_model.pkl and heart_disease_scaler.pkl (produced by
cell 28 of the notebook) to be in the same folder as this file.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import base64
from datetime import datetime
from pathlib import Path
import pickle

try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="CardioScan · Heart Disease Risk",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Exact feature order the model was trained on (from notebook cell 29)
# ----------------------------------------------------------------------
FEATURE_COLUMNS = [
    "age", "trestbps", "chol", "thalch", "oldpeak", "ca", "sex_Male",
    "dataset_Hungary", "dataset_Switzerland", "dataset_VA Long Beach",
    "cp_atypical angina", "cp_non-anginal", "cp_typical angina", "fbs_True",
    "restecg_normal", "restecg_st-t abnormality", "exang_True",
    "slope_flat", "slope_upsloping", "thal_normal",
    "thal_reversable defect",
]

FRIENDLY_NAMES = {
    "age": "Age", "trestbps": "Resting blood pressure", "chol": "Cholesterol",
    "thalch": "Max heart rate", "oldpeak": "ST depression (oldpeak)",
    "ca": "Major vessels colored", "sex_Male": "Sex: Male",
    "dataset_Hungary": "Site: Hungary", "dataset_Switzerland": "Site: Switzerland",
    "dataset_VA Long Beach": "Site: VA Long Beach",
    "cp_atypical angina": "Chest pain: atypical angina",
    "cp_non-anginal": "Chest pain: non-anginal", "cp_typical angina": "Chest pain: typical angina",
    "fbs_True": "Fasting blood sugar > 120", "restecg_normal": "ECG: normal",
    "restecg_st-t abnormality": "ECG: ST-T abnormality", "exang_True": "Exercise-induced angina",
    "slope_flat": "ST slope: flat", "slope_upsloping": "ST slope: upsloping",
    "thal_normal": "Thalassemia: normal", "thal_reversable defect": "Thalassemia: reversible defect",
}

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "heart_disease_model.pkl"
SCALER_PATH = BASE_DIR / "heart_disease_scaler.pkl"

# Model loading
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

except FileNotFoundError:
    st.error(
        "Model files not found. Make sure "
        "'heart_disease_model.pkl' and 'heart_disease_scaler.pkl' "
        "are present in the same folder as app.py."
    )
    st.stop()

HISTORY_COLUMNS = [
    "submitted_at",
    "name",
    "age",
    "sex",
    "site",
    "trestbps",
    "chol",
    "thalch",
    "oldpeak",
    "ca",
    "cp",
    "restecg",
    "slope",
    "thal",
    "fbs",
    "exang",
    "pct",
    "risk_level",
]

# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        return None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


model, scaler = load_artifacts()

if "cardioscan_history" not in st.session_state:
    st.session_state["cardioscan_history"] = pd.DataFrame(columns=HISTORY_COLUMNS)


def make_history_pdf(history_df: pd.DataFrame) -> bytes:
    if FPDF is None:
        raise RuntimeError("FPDF is not installed. Install fpdf to enable PDF export.")

    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "CardioScan Prediction History", ln=True, align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica", size=11)

    for _, row in history_df.iterrows():
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, f"{row['submitted_at']} — {row['name']} — {row['risk_level']} ({row['pct']}%)", ln=True)
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(0, 6, (
            f"Age: {row['age']}, Sex: {row['sex']}, Site: {row['site']}, "
            f"Chol: {row['chol']}, Resting BP: {row['trestbps']}, Max HR: {row['thalch']}, "
            f"Oldpeak: {row['oldpeak']}, CA: {row['ca']}, CP: {row['cp']}, "
            f"Rest ECG: {row['restecg']}, Slope: {row['slope']}, Thal: {row['thal']}, "
            f"FBS: {row['fbs']}, Exang: {row['exang']}"
        ))
        pdf.ln(2)

    return pdf.output(dest="S").encode("latin-1")


def make_history_csv(history_df: pd.DataFrame) -> bytes:
    return history_df.to_csv(index=False).encode("utf-8")

# ----------------------------------------------------------------------
# Styling — instrument-panel / ECG-strip aesthetic
# ----------------------------------------------------------------------

def get_background_image_css(image_path: str) -> str:
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"background-image: url('data:image/jpeg;base64,{encoded}');"

background_css = get_background_image_css("cardio_image.jpg")

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {{
    --bg: #EAF0EE;
    --panel: #FFFFFF;
    --grid: #CBDAD4;
    --ink: #101C18;
    --muted: #5B6E67;
    --risk: #C0392B;
    --risk-soft: #F3D9D5;
    --safe: #2F7D6E;
    --safe-soft: #D9EBE6;
    --amber: #B8862B;
    --amber-soft: #F1E3C6;
    --line: #10201C;
}}

html, body, [data-testid="stAppViewContainer"] {{
    background:
        linear-gradient(var(--grid) 1px, transparent 1px) 0 0 / 100% 28px,
        linear-gradient(90deg, var(--grid) 1px, transparent 1px) 0 0 / 28px 100%,
        var(--bg);
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--ink);
}}

[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: var(--panel); }}
.block-container {{ padding-top: 1.6rem; max-width: 1180px; }}

.cs-header {{
    {background_css}
    background-color: rgba(16,28,24,0.85);
    background-size: cover;
    background-position: center;
    border-radius: 14px;
    padding: 26px 32px 18px 32px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}}

/* ---------- Header / signature ECG trace ---------- */
.cs-eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.24em;
    text-transform: uppercase;
    color: #7FBDAE;
    margin-bottom: 6px;
}}
.cs-title {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 700;
    font-size: 32px;
    color: #F4F8F6;
    margin: 0 0 4px 0;
    letter-spacing: -0.01em;
}}
.cs-subtitle {{
    font-size: 14px;
    color: #A9BDB6;
    margin-bottom: 10px;
}}
.ecg-wrap {{ width: 100%; height: 46px; margin-top: 8px; }}
.ecg-path {{
    fill: none;
    stroke: #4FD1B5;
    stroke-width: 2;
    stroke-linecap: round;
    stroke-linejoin: round;
    stroke-dasharray: 620;
    stroke-dashoffset: 620;
    animation: draw 2.6s ease-in-out infinite;
}}
@keyframes draw {{
    0%   {{ stroke-dashoffset: 620; opacity: 0.3; }}
    45%  {{ stroke-dashoffset: 0;   opacity: 1; }}
    55%  {{ stroke-dashoffset: 0;   opacity: 1; }}
    100% {{ stroke-dashoffset: -620; opacity: 0.3; }}
}}
"""

st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Header with animated ECG trace
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="cs-header">
        <div class="cs-eyebrow">Diagnostic Support · Not A Medical Device</div>
        <div class="cs-title">CardioScan</div>
        <div class="cs-subtitle">Heart disease risk estimate from clinical intake values</div>
        <div class="ecg-wrap">
            <svg class="ecg-wrap" viewBox="0 0 620 46" preserveAspectRatio="none">
                <path class="ecg-path" d="M0,23 L110,23 L130,23 L142,6 L154,40 L166,14 L178,23 L210,23
                         L230,23 L242,6 L254,40 L266,14 L278,23 L620,23" />
            </svg>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.warning(
        f"Couldn't find **{MODEL_PATH}** and **{SCALER_PATH}** next to this app. "
        "Run the notebook through cell 28 first to generate them, then place both "
        "files in this app's folder and refresh.",
        icon="⚠️",
    )

# ----------------------------------------------------------------------
# Page navigation
# ----------------------------------------------------------------------
def render_header():
    st.markdown(
        """
        
        """,
        unsafe_allow_html=True,
    )


def render_about_page():
    st.markdown('<div class="cs-panel">', unsafe_allow_html=True)
    st.markdown('<div class="cs-panel-title">About CardioScan</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="line-height: 1.7; color: var(--ink);">
            <p><b>CardioScan</b> is a simple heart-disease risk demo built with Streamlit and a trained machine-learning model.</p>
            <p>Is app ka main purpose hai ki aap patient ke inputs ke basis par ek quick risk estimate dekh sakein, bina kisi complex setup ke.</p>
            <p>Yeh app educational aur demo-oriented hai — isko clinical diagnosis ka replacement nahi maana jana chahiye.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Technologies Used:-")
    st.markdown("- Streamlit: interactive web UI")
    st.markdown("- scikit-learn: trained risk prediction model")
    st.markdown("- pandas, numpy, joblib: data handling aur model loading")
    st.markdown("- fpdf: prediction history export")
    st.markdown("### Next step")
    if st.button("Open analysis dashboard", use_container_width=True):
        st.session_state["current_page"] = "Analysis"
    st.markdown('</div>', unsafe_allow_html=True)


def render_analysis_page():
    if model is None:
        st.warning(
            f"Couldn't find **{MODEL_PATH}** and **{SCALER_PATH}** next to this app. "
            "Run the notebook through cell 28 first to generate them, then place both "
            "files in this app's folder and refresh.",
            icon="⚠️",
        )

    left, right = st.columns([1.35, 1], gap="large")

    with left:
        with st.form("intake_form"):
            st.markdown('<div class="cs-panel">', unsafe_allow_html=True)
            st.markdown('<div class="cs-panel-title">Patient Profile</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            name = c1.text_input("Patient name", value="")
            age = c2.number_input("Age", min_value=18, max_value=100, value=54)
            sex = c3.selectbox("Sex", ["Male", "Female"])
            site = st.selectbox(
                "Study site", ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"],
                help="Included because the model was trained on pooled multi-site data — "
                     "not a clinical risk factor on its own.",
            )
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="cs-panel">', unsafe_allow_html=True)
            st.markdown('<div class="cs-panel-title">Vitals & Labs</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            trestbps = c1.number_input("Resting BP (mm Hg)", min_value=80, max_value=220, value=130)
            chol = c2.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=225)
            thalch = c3.number_input("Max heart rate", min_value=60, max_value=220, value=150)
            fbs = st.checkbox("Fasting blood sugar > 120 mg/dl")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="cs-panel">', unsafe_allow_html=True)
            st.markdown('<div class="cs-panel-title">Cardiac Test Results</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            cp = c1.selectbox(
                "Chest pain type",
                ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
            )
            restecg = c2.selectbox(
                "Resting ECG", ["normal", "st-t abnormality", "lv hypertrophy"],
            )
            c1, c2 = st.columns(2)
            slope = c1.selectbox("ST slope (peak exercise)", ["upsloping", "flat", "downsloping"])
            thal = c2.selectbox("Thalassemia", ["normal", "reversable defect", "fixed defect"])
            c1, c2, c3 = st.columns(3)
            oldpeak = c1.number_input("ST depression (oldpeak)", min_value=-2.0, max_value=7.0, value=1.0, step=0.1)
            ca = c2.number_input("Major vessels colored (0-4)", min_value=0, max_value=4, value=0)
            exang = c3.checkbox("Exercise-induced angina")
            st.markdown('</div>', unsafe_allow_html=True)

            submitted = st.form_submit_button("Run Diagnostic ▸")

    with right:
        if submitted and model is not None:
            row = {col: 0 for col in FEATURE_COLUMNS}
            row["age"] = age
            row["trestbps"] = trestbps
            row["chol"] = chol
            row["thalch"] = thalch
            row["oldpeak"] = oldpeak
            row["ca"] = ca

            if sex == "Male":
                row["sex_Male"] = 1
            if site == "Hungary":
                row["dataset_Hungary"] = 1
            elif site == "Switzerland":
                row["dataset_Switzerland"] = 1
            elif site == "VA Long Beach":
                row["dataset_VA Long Beach"] = 1
            if cp == "atypical angina":
                row["cp_atypical angina"] = 1
            elif cp == "non-anginal":
                row["cp_non-anginal"] = 1
            elif cp == "typical angina":
                row["cp_typical angina"] = 1
            if fbs:
                row["fbs_True"] = 1
            if restecg == "normal":
                row["restecg_normal"] = 1
            elif restecg == "st-t abnormality":
                row["restecg_st-t abnormality"] = 1
            if exang:
                row["exang_True"] = 1
            if slope == "flat":
                row["slope_flat"] = 1
            elif slope == "upsloping":
                row["slope_upsloping"] = 1
            if thal == "normal":
                row["thal_normal"] = 1
            elif thal == "reversable defect":
                row["thal_reversable defect"] = 1

            X_input = pd.DataFrame([row])[FEATURE_COLUMNS]
            X_scaled = scaler.transform(X_input)

            pred = model.predict(X_scaled)[0]
            proba = (
                model.predict_proba(X_scaled)[0][1]
                if hasattr(model, "predict_proba")
                else float(pred)
            )
            pct = round(proba * 100, 1)

            if pct < 30:
                badge_class, badge_text = "badge-safe", "Low likelihood"
                needle_color = "var(--safe)"
                risk_level = "Low"
            elif pct < 60:
                badge_class, badge_text = "badge-amber", "Moderate likelihood"
                needle_color = "var(--amber)"
                risk_level = "Moderate"
            else:
                badge_class, badge_text = "badge-risk", "High likelihood"
                needle_color = "var(--risk)"
                risk_level = "High"

            angle = -90 + (pct / 100) * 180

            st.markdown('<div class="cs-panel">', unsafe_allow_html=True)
            st.markdown('<div class="cs-panel-title">Risk Readout</div>', unsafe_allow_html=True)

            gauge_html = f"""
            <div style="display:flex; flex-direction:column; align-items:center; padding:6px 0 4px 0;">
                <svg width="220" height="130" viewBox="0 0 220 130">
                    <path d="M20,120 A90,90 0 0 1 200,120" fill="none" stroke="#CBDAD4" stroke-width="14" stroke-linecap="round"/>
                    <path d="M20,120 A90,90 0 0 1 200,120" fill="none" stroke="{needle_color}" stroke-width="14"
                          stroke-linecap="round" stroke-dasharray="{(pct/100)*283} 283"/>
                    <line x1="110" y1="120" x2="110" y2="42" stroke="var(--ink)" stroke-width="3"
                          transform="rotate({angle} 110 120)" stroke-linecap="round"/>
                    <circle cx="110" cy="120" r="7" fill="var(--ink)"/>
                </svg>
                <div class="readout-value" style="color:{needle_color};">{pct}%</div>
                <div class="readout-label">Predicted Probability</div>
                <div style="margin-top:10px;"><span class="badge {badge_class}">{badge_text}</span></div>
            </div>
            """
            st.markdown(gauge_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            history_row = {
                "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "name": name.strip() or "Anonymous",
                "age": age,
                "sex": sex,
                "site": site,
                "trestbps": trestbps,
                "chol": chol,
                "thalch": thalch,
                "oldpeak": oldpeak,
                "ca": ca,
                "cp": cp,
                "restecg": restecg,
                "slope": slope,
                "thal": thal,
                "fbs": bool(fbs),
                "exang": bool(exang),
                "pct": pct,
                "risk_level": risk_level,
            }
            st.session_state["cardioscan_history"] = pd.concat(
                [st.session_state["cardioscan_history"], pd.DataFrame([history_row])],
                ignore_index=True,
            )

            if hasattr(model, "feature_importances_"):
                st.markdown('<div class="cs-panel">', unsafe_allow_html=True)
                st.markdown('<div class="cs-panel-title">Model\'s Top Signals</div>', unsafe_allow_html=True)
                importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
                top5 = importances.sort_values(ascending=False).head(5)
                max_imp = top5.max()
                for feat, imp in top5.items():
                    width_pct = round((imp / max_imp) * 100, 1)
                    st.markdown(
                        f"""
                        <div class="factor-row">
                            <div class="factor-name">{FRIENDLY_NAMES.get(feat, feat)}</div>
                            <div class="factor-track">
                                <div class="factor-fill" style="width:{width_pct}%; background:var(--ink);"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.caption("Overall importance in the trained model, not specific to this patient.")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div class="cs-panel" style="text-align:center; padding:48px 22px;">
                    <div style="font-size:40px; margin-bottom:6px;">🫀</div>
                    <div style="color:var(--muted); font-size:13.5px;">
                        Fill in the patient profile and press<br><b>Run Diagnostic</b> to see a risk readout.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if not st.session_state["cardioscan_history"].empty:
        history_df = st.session_state["cardioscan_history"].copy()
        csv_bytes = make_history_csv(history_df)
        pdf_bytes = None
        pdf_message = None
        if FPDF is not None:
            try:
                pdf_bytes = make_history_pdf(history_df)
            except Exception as exc:
                pdf_message = str(exc)

        st.markdown('<div class="cs-panel">', unsafe_allow_html=True)
        st.markdown('<div class="cs-panel-title">Saved Prediction History</div>', unsafe_allow_html=True)
        st.dataframe(history_df, use_container_width=True)
        c1, c2, c3 = st.columns([1, 1, 0.7])
        c1.download_button(
            label="Download CSV",
            data=csv_bytes,
            file_name=f"cardioscan_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
        if pdf_bytes is not None:
            c2.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=f"cardioscan_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
            )
        else:
            c2.write("PDF export is unavailable." if pdf_message is None else pdf_message)
            if FPDF is None:
                c2.caption("Install fpdf with pip to enable PDF export.")
        if c3.button("Clear history"):
            st.session_state["cardioscan_history"] = pd.DataFrame(columns=HISTORY_COLUMNS)
            st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)


render_header()

st.sidebar.markdown("### Navigation")
st.session_state.setdefault("current_page", "About")
current_page = st.sidebar.radio(
    "Go to",
    ["About", "Analysis"],
    index=0 if st.session_state["current_page"] == "About" else 1,
)
st.session_state["current_page"] = current_page

if current_page == "About":
    render_about_page()
else:
    render_analysis_page()

st.markdown(
    """
    <div class="cs-footer">
        CardioScan is a machine-learning demo built on the UCI Heart Disease dataset.
        It is not a diagnostic device and does not replace clinical judgment —
        always consult a qualified clinician about real symptoms or results.
    </div>
    """,
    unsafe_allow_html=True,
)
