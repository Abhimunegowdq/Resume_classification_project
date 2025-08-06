import streamlit as st
import docx2txt
import fitz  # PyMuPDF
import re
import joblib
import spacy
import numpy as np
import pandas as pd

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load ML tools
model = joblib.load("decision_tree_resume_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ------------- Text Extraction ------------------

def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(uploaded_file):
    return docx2txt.process(uploaded_file)

# ------------- Extract Info ------------------

def extract_email(text):
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else "Not found"

def extract_phone(text):
    match = re.search(r"\+?\d[\d\s().-]{7,}\d", text)
    return match.group(0) if match else "Not found"

def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return "Not found"

def extract_skills(text):
    common_skills = [
        "python", "java", "sql", "excel", "communication", "teamwork", "machine learning",
        "leadership", "problem solving", "c++", "react", "angular", "tableau", "power bi", "html", "css"
    ]
    text_lower = text.lower()
    found_skills = [skill for skill in common_skills if skill in text_lower]
    return ", ".join(sorted(set(found_skills))) if found_skills else "Not found"

def extract_education(text):
    keywords = ["bachelor", "master", "b.tech", "m.tech", "bsc", "msc", "mba", "phd", "degree"]
    lines = text.lower().split("\n")
    education_lines = [line for line in lines if any(keyword in line for keyword in keywords)]
    return ", ".join(education_lines[:2]) if education_lines else "Not found"

def extract_experience(text):
    matches = re.findall(r'(\d+)\+?\s+years?', text.lower())
    return f"{matches[0]} years" if matches else "Not found"

# ------------- Prediction ------------------

def predict_role(text):
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)
    probabilities = model.predict_proba(vector)
    confidence = np.max(probabilities) * 100
    return label_encoder.inverse_transform(prediction)[0], round(confidence, 2)

# ------------- Streamlit App ------------------

st.title("📄 Resume Classifier & Extractor")
st.markdown("Upload a **PDF** or **DOCX** resume to extract info and predict the job role.")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])

if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        resume_text = extract_text_from_pdf(uploaded_file)
    else:
        resume_text = extract_text_from_docx(uploaded_file)

    name = extract_name(resume_text)
    email = extract_email(resume_text)
    phone = extract_phone(resume_text)
    skills = extract_skills(resume_text)
    education = extract_education(resume_text)
    experience = extract_experience(resume_text)
    predicted_role, confidence = predict_role(resume_text)

    st.subheader("📋 Extracted Information")
    st.markdown(f"👤 **Name:** {name}")
    st.markdown(f"📧 **Email:** {email}")
    st.markdown(f"📱 **Phone:** {phone}")
    st.markdown(f"🛠️ **Skills:** {skills}")
    st.markdown(f"🎓 **Education:** {education}")
    st.markdown(f"💼 **Experience:** {experience}")

    st.subheader("📊 Job Role Prediction")
    st.markdown(f"🔮 **Predicted Role:** `{predicted_role}`")
    st.markdown(f"✅ **Confidence:** `{confidence}%`")
