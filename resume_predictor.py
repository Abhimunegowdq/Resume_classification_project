import streamlit as st
import re
import fitz  # PyMuPDF
import docx2txt
import joblib

# Load ML model and vectorizer
model = joblib.load("job_role_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Text extraction
def extract_text(file):
    if file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    elif file.name.endswith(".docx"):
        return docx2txt.process(file)
    return ""

# Clean for ML
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# Extract fields
def extract_name(text):
    lines = text.split("\n")
    for line in lines:
        if line.strip() and len(line.strip().split()) <= 5 and "curriculum" not in line.lower():
            return line.strip()
    return "Not found"

def extract_email(text):
    match = re.search(r"\S+@\S+", text)
    return match.group() if match else "Not found"

def extract_skills(text):
    skills = []
    for line in text.split("\n"):
        if re.search(r"skills|technologies|tools", line, re.IGNORECASE):
            skills.append(line.strip())
    return " ".join(skills[:2]) if skills else "Not found"

def extract_experience(text):
    exp = []
    for line in text.split("\n"):
        if re.search(r"experience|worked|duration|years|months", line, re.IGNORECASE):
            exp.append(line.strip())
    return "\n".join(exp[:5]) if exp else "Not found"

def extract_education(text):
    edu = []
    for line in text.split("\n"):
        if re.search(r"education|bachelor|master|degree|university|college", line, re.IGNORECASE):
            edu.append(line.strip())
    return "\n".join(edu[:3]) if edu else "Not found"

# Streamlit UI
st.set_page_config(page_title="Resume Job Role Predictor", layout="centered")
st.title("📄 Resume Job Role Prediction App")

uploaded = st.file_uploader("Upload your resume (PDF/DOCX)", type=["pdf", "docx"])

if uploaded:
    raw_text = extract_text(uploaded)
    cleaned = clean_text(raw_text)
    tfidf = vectorizer.transform([cleaned])
    prediction = model.predict(tfidf)[0]
    job_role = label_encoder.inverse_transform([prediction])[0]

    st.markdown(f"👤 **Name:** {extract_name(raw_text)}")
    st.markdown(f"📧 **Email:** {extract_email(raw_text)}")
    st.markdown(f"🛠 **Skills:** {extract_skills(raw_text)}")
    st.markdown(f"💼 **Experience:**\n\n{extract_experience(raw_text)}")
    st.markdown(f"🎓 **Education:**\n\n{extract_education(raw_text)}")
    st.success(f"🔮 **Predicted Job Role:** {job_role}")
