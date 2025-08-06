import streamlit as st
import re
import docx2txt
import PyPDF2
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and encoders
model = joblib.load("decision_tree_resume_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Text cleaning
def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,]', '', text)
    return text.strip().lower()

# Extract name
def extract_name(text):
    lines = text.split('\n')
    for line in lines[:5]:
        if len(line.split()) >= 2 and all(w[0].isupper() for w in line.split()[:2]):
            return line.strip()
    return "Not found"

# Extract email
def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return match.group(0) if match else "Not found"

# Extract skills
def extract_skills(text):
    skills_keywords = [
        "python", "java", "sql", "excel", "tableau", "power bi", "ml", "keras", "tensorflow",
        "pandas", "numpy", "matplotlib", "scikit-learn", "flask", "django", "html", "css",
        "javascript", "react", "node", "php", "c++", "c#", "linux", "git", "oracle", "pl/sql",
        "spark", "hadoop", "crm"
    ]
    found = []
    for word in skills_keywords:
        if re.search(r'\b' + re.escape(word) + r'\b', text.lower()):
            found.append(word)
    return list(set(found))

# Extract experience
def extract_experience(text):
    exp_matches = re.findall(r'[\d\+]+ *years? of experience[^.]{0,100}\.', text, re.IGNORECASE)
    if not exp_matches:
        exp_matches = re.findall(r"(worked|experience|developed|responsible)[^.]{0,100}\.", text, re.IGNORECASE)
    return exp_matches[:3] if exp_matches else ["Not found"]

# Extract education
def extract_education(text):
    keywords = ["bachelor", "master", "b.tech", "m.tech", "bsc", "msc", "bca", "mca", "mba", "phd"]
    edu_lines = []
    for line in text.split('\n'):
        for word in keywords:
            if word in line.lower() and line.strip() not in edu_lines:
                edu_lines.append(line.strip())
    return edu_lines[:3] if edu_lines else ["Not found"]

# File reader
def read_file(uploaded_file):
    if uploaded_file.name.endswith('.pdf'):
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    elif uploaded_file.name.endswith('.docx'):
        return docx2txt.process(uploaded_file)
    else:
        return ""

# Streamlit UI
st.set_page_config(page_title="Resume Classifier", layout="centered")
st.title("📄 Resume Classification App")
st.markdown("Upload your resume (PDF or DOCX) to get extracted information and job role prediction.")

uploaded_file = st.file_uploader("Choose your resume file", type=["pdf", "docx"])

if uploaded_file:
    raw_text = read_file(uploaded_file)
    cleaned = clean_text(raw_text)

    name = extract_name(raw_text)
    email = extract_email(raw_text)
    skills = extract_skills(cleaned)
    experience = extract_experience(raw_text)
    education = extract_education(raw_text)

    # Prediction
    vec_text = vectorizer.transform([cleaned])
    prediction_encoded = model.predict(vec_text)[0]
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]

    # Output
    st.markdown(f"👤 **Name:** {name}")
    st.markdown(f"📧 **Email:** {email}")

    st.markdown("🛠 **Skills:**")
    if skills:
        mid = len(skills) // 2
        st.text(", ".join(skills[:mid+1]))
        if skills[mid+1:]:
            st.text(", ".join(skills[mid+1:]))
    else:
        st.text("Not found")

    st.markdown("💼 **Experience:**")
    for exp in experience:
        st.markdown(f"- {exp}")

    st.markdown("🎓 **Education:**")
    for edu in education:
        st.markdown(f"- {edu}")

    st.markdown(f"🧑‍💼 **Predicted Job Role:** {prediction}")
