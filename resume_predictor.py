import streamlit as st
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load models
model = joblib.load("decision_tree_resume_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Text extraction functions
def extract_name(text):
    lines = text.strip().split("\n")
    for line in lines:
        clean_line = line.strip()
        if clean_line and len(clean_line.split()) <= 5:
            if not re.search(r"\b(resume|curriculum|cv|profile|summary)\b", clean_line, re.IGNORECASE):
                if not re.search(r"skills|experience|education|email", clean_line, re.IGNORECASE):
                    return clean_line
    return "Not found"

def extract_email(text):
    match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return match.group(0) if match else "Not found"

def extract_skills(text):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    skill_lines = [line for line in lines if re.search(r"\bskills\b", line, re.IGNORECASE)]
    return "\n".join(skill_lines[:2]) if skill_lines else "Not found"

def extract_experience(text):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    experience_lines = [line for line in lines if re.search(r"\bexperience\b", line, re.IGNORECASE)]
    return "\n".join(experience_lines[:3]) if experience_lines else "Not found"

def extract_education(text):
    edu_lines = [line.strip() for line in text.split("\n") if re.search(r"\b(btech|b\.tech|mtech|m\.tech|master|bachelor|mba|b\.e|m\.e|education)\b", line, re.IGNORECASE)]
    return "\n".join(edu_lines[:2]) if edu_lines else "Not found"

def predict_role(text):
    transformed = vectorizer.transform([text])
    pred = model.predict(transformed)
    return label_encoder.inverse_transform(pred)[0]

# Streamlit UI
st.set_page_config(page_title="Resume Predictor", layout="centered")

st.title("📄 Resume Role Classifier")

uploaded_file = st.file_uploader("Upload your resume (.txt)", type=["txt"])

if uploaded_file:
    resume_text = uploaded_file.read().decode("utf-8")

    name = extract_name(resume_text)
    email = extract_email(resume_text)
    skills = extract_skills(resume_text)
    experience = extract_experience(resume_text)
    education = extract_education(resume_text)
    prediction = predict_role(resume_text)

    st.markdown(f"👤 **Name:** {name}")
    st.markdown(f"📧 **Email:** {email}")
    st.markdown(f"🛠 **Skills:**\n{skills}")
    st.markdown(f"💼 **Experience:**\n{experience}")
    st.markdown(f"🎓 **Education:**\n{education}")
    st.success(f"🔮 **Predicted Job Role:** {prediction}")

