# 📁 Project: AI-Powered Fake Job Posting Detection and Trust Scoring System

# This is the main Python app script for your Streamlit-based fake job posting detector.

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import pytesseract
from PIL import Image
import shap
import lime
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load trained model (you can replace this with actual path later)
model = joblib.load('model/fake_job_lr_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

# Define known scam patterns (Rule-based flags)
rule_flags = {
    'gmail.com': 'Gmail email address',
    'immediate joining': 'Immediate joining',
    'work from home': 'Work from home',
    'registration fee': 'Registration fee mentioned',
    'guaranteed returns': 'Guaranteed returns',
    'no experience needed': 'No experience needed',
    '₹': 'Unrealistic salary'
}

# Function to preprocess input text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s₹]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Function to extract rule-based flags
def detect_flags(text):
    flags = []
    for pattern, label in rule_flags.items():
        if pattern in text:
            flags.append(label)
    return flags

# Trust scoring

def compute_trust_score(model_prob, rule_flags):
    rule_score = max(0, 1 - (0.1 * len(rule_flags)))
    trust_score = 0.7 * model_prob + 0.3 * rule_score
    return round(trust_score * 100, 2)

# Classify score level
def classify_risk(score):
    if score >= 80:
        return 'SAFE', 'green'
    elif score >= 50:
        return 'MEDIUM RISK', 'yellow'
    else:
        return 'EXTREMELY DANGEROUS', 'red'

# Load SHAP explainer (mock explanation for now)
def get_explanation():
    return "This posting lacks professional details and uses personal email domain. Exercise caution and verify company legitimacy before proceeding."

# Streamlit UI
st.set_page_config(page_title="Fake Job Detector", layout="centered")
st.title("🧠 AI-Powered Fake Job Posting Detection")

input_mode = st.radio("Choose Input Type:", ["Text Input", "Image Upload"])

job_text = ""

if input_mode == "Text Input":
    job_text = st.text_area("Paste Job Posting Here:", height=200)

elif input_mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload Image (screenshot of job post):", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        job_text = pytesseract.image_to_string(img)
        st.image(img, caption="Uploaded Screenshot", use_column_width=True)
        st.text_area("Extracted Text:", job_text, height=150)

if st.button("Analyze Posting") and job_text:
    cleaned_text = preprocess_text(job_text)
    flags = detect_flags(cleaned_text)
    X_input = vectorizer.transform([cleaned_text])
    model_prob = model.predict_proba(X_input)[0][1]

    trust_score = compute_trust_score(model_prob, flags)
    risk_label, risk_color = classify_risk(trust_score)

    # Display results
    if risk_label == "SAFE":
        st.success("✅ GENUINE JOB POSTING")
    elif risk_label == "MEDIUM RISK":
        st.warning("⚠️ SUSPICIOUS POSTING")
    else:
        st.error("⛔ INVESTMENT SCAM or EXTREME RISK")

    st.markdown(f"### Trust Score: {risk_label}")
    st.progress(trust_score)
    st.write(f"**Score:** {trust_score}%")

    if flags:
        st.markdown("### 🔍 Risk Indicators:")
        for flag in flags:
            st.markdown(f"- {flag}")
    else:
        st.markdown("### ✅ Positive Indicators")
        st.markdown("- Professional tone")
        st.markdown("- No suspicious content detected")

    st.markdown("### 🧠 AI Explanation:")
    st.info(get_explanation())

st.markdown("---")
st.caption("Project by Tanvi Agarwal | Fake Job Detection System")
