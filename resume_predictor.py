import streamlit as st
import docx2txt
import fitz  # PyMuPDF
import re
import joblib
import pandas as pd
import numpy as np

# ---------------- Load Saved Model and Tools ----------------
model = joblib.load("decision_tree_resume_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ------------------ Helper Functions ------------------------

def extract_text_from_docx(file):
    return docx2txt.process(file)

def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_email(text):
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else "Not found"

def extract_phone(text):
    match = re.search(r'(\+?\d{1,3}[\s.-]?)?(\(?\d{3,5}\)?[\s.-]?)?\d{3,5}[\s.-]?\d{4}', text)
    return match.group(0) if match else "Not found"

def extract_name(text):
    lines = text.strip().split('\n')
    for line in lines:
        clean_line = line.strip()
        if clean_line.lower() in ["curriculum vitae", "resume"]:
            continue  # Skip headers
        if len(clean_line.split()) <= 4 and clean_line.replace(" ", "").isalpha():
            return clean_line.title()
        if re.match(r"^[A-Z][a-z]+ [A-Z][a-z]+", clean_line):
            return clean_line.strip()
    return "Not found"

def extract_skills(text):
    skills = [
        
 "recruitment", "talent acquisition", "human resources", "onboarding", "payroll"
        "excel", "power bi", "tableau", "data analysis","python",
        "tsql", "stored procedure", "ssis","sql","mysql",
        "communication", "leadership", "teamwork", "machine learning",
        "react", "redux", "jsx", "javascript"
        "deep learning", "nlp", "keras", "pytorch", "tensorflow", 
        "pandas", "numpy", "data analysis", "react", "angular", 
        "flask", "django"
    ]
    text = text.lower()
    extracted = [skill for skill in skills if skill in text]
    return ", ".join(extracted) if extracted else "Not found"

def predict_role(text):
    tfidf_text = vectorizer.transform([text])
    prediction = model.predict(tfidf_text)
    job_role = label_encoder.inverse_transform(prediction)[0]
    return job_role

# ------------------ Streamlit UI ------------------------

st.set_page_config(page_title="Resume Classifier", layout="centered")
st.title("📄 Resume Classifier")
st.write("Upload your resume (`.pdf` or `.docx`) and we'll classify the job role.")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])

if uploaded_file is not None:
    # Extract text
    if uploaded_file.name.endswith(".pdf"):
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = extract_text_from_docx(uploaded_file)

    # Extract details
    name = extract_name(text)
    email = extract_email(text)
    phone = extract_phone(text)
    skills = extract_skills(text)
    predicted_role = predict_role(text)

    # Display
    st.subheader("📌 Extracted Information")
    st.write(f"**👤 Name:** {name}")
    st.write(f"**📧 Email:** {email}")
    st.write(f"**📱 Phone:** {phone}")
    st.write(f"**🛠️ Skills:** {skills}")
    st.success(f"🎯 **Predicted Job Role:** {predicted_role}")
