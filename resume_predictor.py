import streamlit as st
import docx2txt
import PyPDF2
import re
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
model = joblib.load("decision_tree_resume_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Set Streamlit app title
st.title("📄 Resume Job Role Predictor")
st.markdown("---")

# Helper functions
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_file):
    return docx2txt.process(docx_file)

def extract_email(text):
    match = re.search(r'\S+@\S+', text)
    return match.group(0) if match else "Not found"

def extract_name(text):
    lines = text.strip().split("\n")
    for line in lines:
        if len(line.strip().split()) >= 2 and not any(char in line for char in "@|:•"):
            return line.strip().title()
    return "Not found"

def extract_skills(text):
    keywords = [
        'python', 'java', 'sql', 'excel', 'ms excel', 'tableau', 'power bi',
        'machine learning', 'nlp', 'flask', 'django', 'react', 'angular', 'ssis',
        'keras', 'pytorch', 'html', 'css', 'javascript', 'spark', 'aws', 'azure'
    ]
    found = []
    for word in keywords:
        if re.search(r'\b' + re.escape(word) + r'\b', text.lower()):
            found.append(word.title())
    return ", ".join(sorted(set(found)))[:150] + "..." if found else "Not found"

def extract_experience(text):
    exp_lines = []
    patterns = [
        r'\d+[\s\-]?years?', r'\d+\s?months?', r'experience', r'worked at', r'joined'
    ]
    for line in text.split("\n"):
        if any(re.search(pat, line.lower()) for pat in patterns):
            exp_lines.append(line.strip())
        if len(exp_lines) >= 3:
            break
    return "\n".join(exp_lines) if exp_lines else "Not found"

def extract_education(text):
    edu_keywords = ['bachelor', 'master', 'degree', 'b.tech', 'm.tech', 'mba', 'b.sc', 'm.sc', 'graduation']
    edu_lines = []
    for line in text.split("\n"):
        if any(word in line.lower() for word in edu_keywords):
            edu_lines.append(line.strip())
    return "\n".join(edu_lines) if edu_lines else "Not found"

# File uploader
uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        resume_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        resume_text = extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file format.")
        st.stop()

    # Extract info
    name = extract_name(resume_text)
    email = extract_email(resume_text)
    skills = extract_skills(resume_text)
    experience = extract_experience(resume_text)
    education = extract_education(resume_text)

    # Prediction
    resume_cleaned = re.sub(r'\W+', ' ', resume_text.lower())
    vector_input = vectorizer.transform([resume_cleaned])
    prediction = model.predict(vector_input)[0]
    predicted_role = label_mapping.get(prediction, "Unknown")

    # Display result
    st.markdown("---")
    st.subheader("🔍 Extracted Resume Details")
    st.markdown(f"**👤 Name:** {name}")
    st.markdown(f"**📧 Email:** {email}")
    st.markdown(f"**🛠 Skills:**\n\n{skills}")
    st.markdown(f"**💼 Experience:**\n\n{experience}")
    st.markdown(f"**🎓 Education:**\n\n{education}")
    st.markdown(f"**🧠 Predicted Job Role:** `{predicted_role}`")


model = joblib.load("decision_tree_resume_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")
