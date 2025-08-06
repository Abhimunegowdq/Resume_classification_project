import streamlit as st
import docx2txt
import fitz  # PyMuPDF
import re
import joblib

# Load model, vectorizer, label encoder
model = joblib.load("decision_tree_resume_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# --- File Reading ---
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(file):
    return docx2txt.process(file)

# --- Info Extraction ---
def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return match.group(0) if match else "Not found"

def extract_phone(text):
    match = re.search(r'(\+?\d[\d\s\-\(\)]{9,})', text)
    return match.group(0) if match else "Not found"

def extract_name(text):
    match = re.search(r"(?:Name\s*[:\-]?\s*)([A-Z][a-z]+\s[A-Z][a-z]+)", text)
    return match.group(1) if match else "Not found"

def extract_skills(text):
    match = re.search(r"(skills|technical skills)[:\-]?\s*(.+?)(\n\n|\n[A-Z])", text, re.IGNORECASE | re.DOTALL)
    if match:
        raw = match.group(2)
        skills = re.split(r'[\n•,\-\u2022]+', raw)
        skills = [s.strip() for s in skills if len(s.strip()) > 1]
        return ", ".join(skills) if skills else "Not found"
    return "Not found"

def extract_education(text):
    education_keywords = r"(b\.?tech|b\.?e|bachelor|master|mba|b\.?sc|m\.?sc|m\.?tech|ph\.?d|b\.?com|m\.?com)"
    edu_matches = re.findall(rf".*{education_keywords}.*", text, re.IGNORECASE)
    cleaned = [line.strip() for line in edu_matches if len(line.strip()) > 5]
    return cleaned[:3] if cleaned else ["Not found"]

def extract_experience(text):
    exp_patterns = [
        r'\d{4}\s*[-to]{1,3}\s*\d{4}',   # e.g. 2018 - 2022
        r'\d+\+?\s+years?',              # e.g. 5 years, 3+ years
        r'experience\s+in\s+\w+',        # e.g. Experience in Python
    ]
    found = []
    for pat in exp_patterns:
        matches = re.findall(rf".*{pat}.*", text, re.IGNORECASE)
        found.extend(matches)
    cleaned = [line.strip() for line in set(found) if len(line.strip()) > 5]
    return cleaned[:3] if cleaned else ["Not found"]

# --- Streamlit App ---
st.title("📄 Resume Category Predictor")
st.write("Upload a resume to classify job role and extract key information.")

uploaded_file = st.file_uploader("Upload resume file", type=["pdf", "docx"])

if uploaded_file is not None:
    # Read resume text
    if uploaded_file.name.endswith(".pdf"):
        resume_text = extract_text_from_pdf(uploaded_file)
    else:
        resume_text = extract_text_from_docx(uploaded_file)

    # Prediction
    vect = vectorizer.transform([resume_text])
    pred = model.predict(vect)
    category = label_encoder.inverse_transform(pred)[0]

    # Output
    st.subheader("📌 Predicted Job Role")
    st.success(category)

    st.subheader("🔍 Extracted Information")
    st.markdown(f"**👤 Name:** {extract_name(resume_text)}")
    st.markdown(f"**📧 Email:** {extract_email(resume_text)}")
    
    st.markdown(f"**🛠 Skills:** {extract_skills(resume_text)}")

    st.markdown("**🎓 Education:**")
    for edu in extract_education(resume_text):
        st.write("•", edu)

    st.markdown("**💼 Experience:**")
    for exp in extract_experience(resume_text):
        st.write("•", exp)
