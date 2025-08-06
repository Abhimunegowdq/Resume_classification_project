import streamlit as st
import docx2txt
import fitz  # PyMuPDF
import re
import joblib
import numpy as np

# ---------------- Load Saved Model and Tools ----------------
model = joblib.load("decision_tree_resume_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ------------------ Helper Functions ------------------------

def extract_text_from_docx(file):
    return docx2txt.process(file)

def extract_text_from_pdf(file):
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def extract_email(text):
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else "Not found"

def extract_phone(text):
    match = re.search(r'(\+91[\-\s]?)?[0]?[6789]\d{9}', text)
    return match.group(0) if match else "Not found"

def extract_name(text):
    lines = text.strip().split('\n')
    for line in lines:
        clean_line = line.strip()
        if len(clean_line.split()) <= 4 and clean_line.replace(" ", "").isalpha():
            return clean_line.title()
        if re.match(r"^[A-Z][a-z]+ [A-Z][a-z]+", clean_line):
            return clean_line.strip()
    return "Not found"

def extract_skills(text):
    keywords = [
        "python", "java", "c++", "sql", "excel", "power bi", "tableau", 
        "communication", "leadership", "teamwork", "machine learning",
        "deep learning", "nlp", "keras", "pytorch", "tensorflow", 
        "pandas", "numpy", "data analysis", "react", "angular", 
        "flask", "django"
    ]
    skills = [word for word in keywords if word.lower() in text.lower()]
    return list(set(skills)) if skills else ["Not found"]

def predict_job_role(text):
    vectorized_text = vectorizer.transform([text])
    prediction_encoded = model.predict(vectorized_text)[0]
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
    return prediction

# ------------------ Streamlit Interface ---------------------

st.title("📄 Resume Job Role Predictor")

uploaded_file = st.file_uploader("Upload a resume file (.pdf or .docx)", type=["pdf", "docx"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()

    if file_type == "pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    elif file_type == "docx":
        resume_text = extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file format.")
        st.stop()

    # Display resume preview
    with st.expander(" Resume Preview"):
        st.write(resume_text)

    # Extracted Information
    st.subheader(" Extracted Information")
    st.write(" **Name:**", extract_name(resume_text))
    st.write(" **Email:**", extract_email(resume_text))
    st.write(" **Phone:**", extract_phone(resume_text))
    st.write("🛠 **Skills:**", ", ".join(extract_skills(resume_text)))

    # Prediction
    st.subheader(" Predicted Job Role")
    predicted_role = predict_job_role(resume_text)
    st.success(f" The predicted job role is: **{predicted_role}**")


