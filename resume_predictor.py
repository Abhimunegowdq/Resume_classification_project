import streamlit as st
import docx2txt
import fitz  # PyMuPDF
import re
import joblib
import pandas as pd
import numpy as np
import spacy
from collections import Counter

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

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
            continue
        if len(clean_line.split()) <= 4 and clean_line.replace(" ", "").isalpha():
            return clean_line.title()
        if re.match(r"^[A-Z][a-z]+ [A-Z][a-z]+", clean_line):
            return clean_line.strip()
    return "Not found"

def extract_skills_dynamic(text):
    doc = nlp(text)
    # Extracting potential skills (noun chunks and proper nouns)
    tokens = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 2]
    # Get top 15 most frequent tokens
    most_common = Counter(tokens).most_common(15)
    skill_list = [word for word, freq in most_common]
    return ", ".join(skill_list).title() if skill_list else "Not found"

def extract_education(text):
    education_keywords = ['education', 'qualifications', 'academic']
    lines = text.lower().split('\n')
    education_lines = []

    for i, line in enumerate(lines):
        if any(keyword in line for keyword in education_keywords):
            for j in range(i + 1, min(i + 10, len(lines))):
                if lines[j].strip() == "":
                    break
                education_lines.append(lines[j].strip())
            break

    return "\n".join(education_lines).strip().title() if education_lines else "Not found"

def extract_experience(text):
    experience_keywords = ['experience', 'employment', 'work history', 'professional summary']
    lines = text.lower().split('\n')
    experience_lines = []

    for i, line in enumerate(lines):
        if any(keyword in line for keyword in experience_keywords):
            for j in range(i + 1, min(i + 15, len(lines))):
                if lines[j].strip() == "":
                    break
                experience_lines.append(lines[j].strip())
            break

    return "\n".join(experience_lines).strip().title() if experience_lines else "Not found"

def predict_role(text):
    tfidf_text = vectorizer.transform([text])
    prediction = model.predict(tfidf_text)
    job_role = label_encoder.inverse_transform(prediction)[0]
    return job_role

# ------------------ Streamlit UI ------------------------

st.set_page_config(page_title="Resume Classifier", layout="centered")
st.title("📄 Resume Classifier")
st.write("Upload your resume (`.pdf` or `.docx`) and we'll extract details and classify the job role.")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".pdf"):
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = extract_text_from_docx(uploaded_file)

    # Extracted Info
    name = extract_name(text)
    email = extract_email(text)
    phone = extract_phone(text)
    skills = extract_skills_dynamic(text)
    education = extract_education(text)
    experience = extract_experience(text)
    predicted_role = predict_role(text)

    # Display
    st.subheader("📌 Extracted Information")
    st.write(f"**👤 Name:** {name}")
    st.write(f"**📧 Email:** {email}")
    st.write(f"**📱 Phone:** {phone}")
    st.write(f"**🎓 Education:**\n{education}")
    st.write(f"**💼 Experience:**\n{experience}")
    st.write(f"**🛠️ Skills (Extracted):** {skills}")
    st.success(f"🎯 **Predicted Job Role:** {predicted_role}")

