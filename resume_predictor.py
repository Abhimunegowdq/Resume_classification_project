import streamlit as st
import re
import docx2txt
import joblib
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model and vectorizers
model = joblib.load("decision_tree_resume_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ----------------- Resume Parsing Functions ---------------- #

def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file):
    return docx2txt.process(file)

def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\S+@\S+', '', text)  # remove emails
    text = re.sub(r'http\S+', '', text)  # remove links
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower()

def extract_email(text):
    match = re.search(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', text)
    return match.group(0) if match else "Not found"

def extract_name(text):
    lines = text.strip().split('\n')
    for line in lines:
        if re.search(r'\bname\b', line.lower()):
            return line.split(':')[-1].strip().title()
    return lines[0].strip().title() if lines else "Not found"

def extract_skills(text):
    skill_keywords = ['python', 'sql', 'excel', 'tableau', 'powerbi', 'oracle', 'pl/sql',
                      'java', 'react', 'html', 'css', 'javascript', 'c++', 'pandas', 'numpy',
                      'scikit-learn', 'tensorflow', 'keras']
    found_skills = []
    for skill in skill_keywords:
        if re.search(r'\b' + re.escape(skill) + r'\b', text.lower()):
            found_skills.append(skill)
    return list(dict.fromkeys(found_skills))[:8]  # remove duplicates, max 8

def extract_experience(text):
    lines = text.split('\n')
    exp_lines = [line for line in lines if 'experience' in line.lower()]
    return exp_lines[:3] if exp_lines else ["Not found"]

def extract_education(text):
    edu_keywords = ['education', 'b.tech', 'm.tech', 'bachelor', 'master', 'degree', 'mba', 'b.sc', 'mca', 'bca']
    lines = text.split('\n')
    edu_lines = [line for line in lines if any(keyword in line.lower() for keyword in edu_keywords)]
    return edu_lines[:3] if edu_lines else ["Not found"]

# ----------------- Streamlit App ---------------- #

st.set_page_config(page_title="Resume Job Role Predictor", layout="centered")
st.title("📄 Resume Job Role Predictor")
st.write("Upload a resume file (.pdf or .docx) to extract details and predict the job role.")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])

if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        raw_text = extract_text_from_pdf(uploaded_file)
    else:
        raw_text = extract_text_from_docx(uploaded_file)

    name = extract_name(raw_text)
    email = extract_email(raw_text)
    skills = extract_skills(raw_text)
    experience = extract_experience(raw_text)
    education = extract_education(raw_text)

    cleaned = clean_text(raw_text)
    vector = vectorizer.transform([cleaned])
    pred_label = model.predict(vector)[0]
    job_role = label_encoder.inverse_transform([pred_label])[0]

    # ----------------- Display Output ---------------- #
    st.markdown(f"👤 **Name:** {name}")
    st.markdown(f"📧 **Email:** {email}")

    st.markdown("🛠 **Skills:**")
    if skills:
        st.markdown(f"- {', '.join(skills[:4])}")
        st.markdown(f"- {', '.join(skills[4:8])}" if len(skills) > 4 else "")
    else:
        st.write("Not found")

    st.markdown("💼 **Experience:**")
    for line in experience:
        st.markdown(f"- {line.strip()}")

    st.markdown("🎓 **Education:**")
    for line in education:
        st.markdown(f"- {line.strip()}")

    st.success(f"🧑‍💼 **Predicted Job Role:** {job_role}")
