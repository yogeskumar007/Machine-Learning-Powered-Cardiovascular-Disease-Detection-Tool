"""
app.py  —  Streamlit GUI for the Cardiovascular Disease Detection Tool
Run with:  streamlit run streamlit_app/app.py
"""

import sys, os, json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ── Path bootstrap ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

MODELS_DIR = ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
SCALER_PATH     = MODELS_DIR / "scaler.pkl"
LE_PATH         = MODELS_DIR / "label_encoder.pkl"
CLASS_MAP_PATH  = MODELS_DIR / "class_map.json"

NUMERIC_COLS = [
    "age", "bmi",
    "systolic_bp", "diastolic_bp", "heart_rate", "oxygen_saturation",
    "physical_activity_level",
]
BINARY_COLS = [
    "gender",
    "chest_pain", "shortness_of_breath", "fatigue", "dizziness",
    "palpitations", "leg_swelling", "persistent_cough", "nausea",
    "excessive_sweating",
    "smoking", "alcohol_consumption", "family_history_cvd",
]

DISEASE_INFO = {
    "Coronary Artery Disease": {
        "icon": "🫀",
        "desc": "Narrowing of the coronary arteries, reducing blood flow to the heart muscle.",
        "advice": "Seek cardiology review. Lifestyle changes (diet, exercise, smoking cessation) and medications (statins, antiplatelets) are first-line."
    },
    "Heart Failure": {
        "icon": "💔",
        "desc": "The heart cannot pump blood efficiently enough to meet the body's needs.",
        "advice": "Specialist assessment required. Medications (ACE inhibitors, beta-blockers, diuretics) and monitoring fluid intake are key."
    },
    "Arrhythmia": {
        "icon": "📈",
        "desc": "Irregular heartbeat — the heart beats too fast, too slow, or irregularly.",
        "advice": "ECG and Holter monitoring recommended. Treatment may include medications, cardioversion, or ablation therapy."
    },
    "Hypertension": {
        "icon": "🩺",
        "desc": "Persistently elevated blood pressure, a major risk factor for stroke and heart disease.",
        "advice": "Lifestyle modifications (low-sodium diet, exercise, weight reduction) and antihypertensive medication as needed."
    },
    "Cardiomyopathy": {
        "icon": "🫁",
        "desc": "Disease of the heart muscle, making it harder for the heart to pump blood.",
        "advice": "Cardiology referral essential. Management depends on subtype (dilated, hypertrophic, restrictive)."
    },
    "Healthy Control": {
        "icon": "✅",
        "desc": "No significant cardiovascular condition detected based on the provided inputs.",
        "advice": "Maintain a healthy lifestyle: regular exercise, balanced diet, no smoking, and annual check-ups."
    },
}


# ── Load artefacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    if not BEST_MODEL_PATH.exists():
        return None, None, None, None
    model  = joblib.load(BEST_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le     = joblib.load(LE_PATH)
    with open(CLASS_MAP_PATH) as fh:
        class_map = json.load(fh)
    return model, scaler, le, class_map


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioSense — CVD Detection",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title   {font-size:2.4rem; font-weight:700; color:#c0392b; margin-bottom:0.2rem;}
    .sub-title    {font-size:1.1rem; color:#555; margin-bottom:1.5rem;}
    .result-box   {background:#fff5f5; border-left:5px solid #c0392b;
                   padding:1.2rem 1.5rem; border-radius:8px; margin-top:1rem;}
    .healthy-box  {background:#f0fff4; border-left:5px solid #27ae60;
                   padding:1.2rem 1.5rem; border-radius:8px; margin-top:1rem;}
    .info-box     {background:#eaf4ff; border-left:5px solid #2980b9;
                   padding:1rem 1.4rem; border-radius:8px; margin-top:0.8rem;}
    .disclaimer   {font-size:0.78rem; color:#888; font-style:italic;}
    div[data-testid="metric-container"] {background:#f9f9f9; border-radius:8px; padding:0.4rem;}
</style>
""", unsafe_allow_html=True)


def build_sidebar() -> dict:
    st.sidebar.markdown("## 🩻 Patient Input")
    st.sidebar.markdown("---")
    inputs = {}

    with st.sidebar.expander("👤 Demographics", expanded=True):
        inputs["age"]    = st.slider("Age (years)", 18, 95, 55)
        inputs["gender"] = 1 if st.radio("Gender", ["Male", "Female"]) == "Male" else 0
        inputs["bmi"]    = st.slider("BMI", 15.0, 55.0, 26.0, step=0.1)

    with st.sidebar.expander("💊 Symptoms", expanded=True):
        sym_labels = {
            "chest_pain":           "Chest Pain",
            "shortness_of_breath":  "Shortness of Breath",
            "fatigue":              "Fatigue",
            "dizziness":            "Dizziness",
            "palpitations":         "Palpitations",
            "leg_swelling":         "Leg Swelling",
            "persistent_cough":     "Persistent Cough",
            "nausea":               "Nausea",
            "excessive_sweating":   "Excessive Sweating",
        }
        for key, label in sym_labels.items():
            inputs[key] = int(st.checkbox(label, value=False))

    with st.sidebar.expander("🩺 Vital Signs", expanded=True):
        inputs["systolic_bp"]       = st.slider("Systolic BP (mmHg)",  80,  220, 120)
        inputs["diastolic_bp"]      = st.slider("Diastolic BP (mmHg)", 40,  140, 80)
        inputs["heart_rate"]        = st.slider("Heart Rate (bpm)",    35,  200, 72)
        inputs["oxygen_saturation"] = st.slider("O₂ Saturation (%)",   80.0, 100.0, 98.0, step=0.5)

    with st.sidebar.expander("🏃 Lifestyle & History", expanded=False):
        inputs["smoking"]               = int(st.checkbox("Current Smoker"))
        inputs["alcohol_consumption"]   = int(st.checkbox("Regular Alcohol Use"))
        inputs["physical_activity_level"] = st.slider("Physical Activity (0=none, 5=intense)", 0.0, 5.0, 2.5, 0.1)
        inputs["family_history_cvd"]    = int(st.checkbox("Family History of CVD"))

    return inputs


def predict(inputs: dict, model, scaler, le, class_map):
    row = {col: [inputs[col]] for col in NUMERIC_COLS + BINARY_COLS}
    X   = pd.DataFrame(row)
    X[NUMERIC_COLS] = scaler.transform(X[NUMERIC_COLS])

    probs      = model.predict_proba(X)[0]
    pred_code  = int(np.argmax(probs))
    pred_label = class_map[str(pred_code)]
    confidence = float(probs[pred_code]) * 100

    class_probs = {class_map[str(i)]: float(p) * 100 for i, p in enumerate(probs)}
    return pred_label, confidence, class_probs


def render_result(pred_label: str, confidence: float, class_probs: dict):
    info = DISEASE_INFO.get(pred_label, {})
    is_healthy = pred_label == "Healthy Control"

    box_class = "healthy-box" if is_healthy else "result-box"
    st.markdown(
        f'<div class="{box_class}">'
        f'<h2 style="margin:0">{info.get("icon","🫀")} {pred_label}</h2>'
        f'<p style="margin:0.4rem 0 0"><b>Confidence:</b> {confidence:.1f}%</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if info:
        st.markdown(
            f'<div class="info-box">'
            f'<b>Description:</b> {info["desc"]}<br><br>'
            f'<b>Suggested Action:</b> {info["advice"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### 📊 Probability Distribution")
    sorted_probs = dict(sorted(class_probs.items(), key=lambda x: x[1], reverse=True))
    fig = go.Figure(go.Bar(
        x=list(sorted_probs.values()),
        y=list(sorted_probs.keys()),
        orientation="h",
        marker=dict(
            color=list(sorted_probs.values()),
            colorscale="RdYlGn",
            cmin=0, cmax=100,
            showscale=True,
            colorbar=dict(title="%"),
        ),
        text=[f"{v:.1f}%" for v in sorted_probs.values()],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis=dict(title="Probability (%)", range=[0, 115]),
        yaxis=dict(autorange="reversed"),
        height=380,
        margin=dict(l=10, r=30, t=10, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<p class="disclaimer">⚠️ This tool is for educational and research purposes only. '
        'It does not constitute medical advice. Always consult a qualified healthcare professional.</p>',
        unsafe_allow_html=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.markdown('<p class="main-title">🫀 CardioSense</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Machine Learning Powered Cardiovascular Disease Detection</p>', unsafe_allow_html=True)

    model, scaler, le, class_map = load_artefacts()

    if model is None:
        st.error("⚠️ No trained model found. Please run `train_models.py` first.")
        st.code("cd project && python src/train_models.py")
        return

    inputs = build_sidebar()

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### 📋 Input Summary")
        summary = {
            "Age":             inputs["age"],
            "Gender":          "Male" if inputs["gender"] else "Female",
            "BMI":             inputs["bmi"],
            "Systolic BP":     f"{inputs['systolic_bp']} mmHg",
            "Diastolic BP":    f"{inputs['diastolic_bp']} mmHg",
            "Heart Rate":      f"{inputs['heart_rate']} bpm",
            "O₂ Saturation":   f"{inputs['oxygen_saturation']}%",
            "Smoker":          "Yes" if inputs["smoking"] else "No",
            "Alcohol Use":     "Yes" if inputs["alcohol_consumption"] else "No",
            "Family Hx CVD":   "Yes" if inputs["family_history_cvd"] else "No",
        }
        for k, v in summary.items():
            st.write(f"**{k}:** {v}")

        active_syms = [
            k.replace("_", " ").title()
            for k in [
                "chest_pain", "shortness_of_breath", "fatigue", "dizziness",
                "palpitations", "leg_swelling", "persistent_cough", "nausea", "excessive_sweating",
            ] if inputs.get(k)
        ]
        if active_syms:
            st.write("**Active Symptoms:**")
            for s in active_syms:
                st.write(f"  • {s}")
        else:
            st.write("**Active Symptoms:** None")

    with col2:
        if st.button("🔍 Run Prediction", type="primary", use_container_width=True):
            with st.spinner("Analysing …"):
                pred_label, confidence, class_probs = predict(inputs, model, scaler, le, class_map)
            render_result(pred_label, confidence, class_probs)
        else:
            st.info("👈 Adjust patient inputs in the sidebar, then click **Run Prediction**.")

    with st.expander("ℹ️ About this tool"):
        st.markdown("""
        **CardioSense** is an academic project developed as part of the AM41PR Dissertation
        at Aston University under the supervision of Dr Mohammed Hadi.

        - **Dataset:** 10,000+ synthetic records across 6 cardiovascular classes
        - **Models trained:** Logistic Regression, Decision Tree, Random Forest, XGBoost,
          LightGBM, SVM, KNN, Naive Bayes
        - **Best model** selected by macro F1-score via 5-fold stratified cross-validation
        - **Features:** Demographics, symptoms, vital signs, lifestyle, and family history
        """)


if __name__ == "__main__":
    main()
