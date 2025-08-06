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
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text

def extract_name(text):
    match = re.findall(r"(?i)(name[:\- ]*)([A-Z][a-z]+(?: [A-Z][a-z]+)*)", text)
    return match[0][1] if match else "Not found"

def extract_email(text):
    match = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match[0] if match else "Not found"

def extract_phone(text):
    match = re.findall(r'(\+?\d{1,3})?[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,5}[-.\s]?\d{4}', text)
    return match[0][0] if match else "Not found"

def extract_skills(text):
    skills_list = [
        'python', 'java', 'sql', 'excel', 'tableau', 'power bi', 'pandas', 'numpy',
        'matplotlib', 'seaborn', 'scikit-learn', 'tensorflow', 'keras', 'nlp', 
        'machine learning', 'deep learning', 'html', 'css', 'javascript', 'react',
        'git', 'github', 'communication', 'leadership', 'teamwork'
    ]
    text = text.lower()
    found = [skill for skill in skills_list if skill in text]
    return sorted(set(found))

# -------------------- Streamlit App -------------------------

st.set_page_config(page_title="Resume Job Role Predictor", layout="wide")
st.title("📄 Resume Job Role Predictor (PDF & DOCX)")

uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        text = extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file format.")
        st.stop()

    st.subheader("🔍 Extracted Information")

    name = extract_name(text)
    email = extract_email(text)
    phone = extract_phone(text)
    skills = extract_skills(text)

    st.write(f"**👤 Name:** {name}")
    st.write(f"**📧 Email:** {email}")
    st.write(f"**📱 Phone:** {phone}")
    st.write(f"**🛠️ Skills:** {', '.join(skills) if skills else 'No skills found'}")

    # Predict job role
    vector_input = vectorizer.transform([text])
    prediction = model.predict(vector_input)
    prediction_proba = model.predict_proba(vector_input)

    job_role = label_encoder.inverse_transform(prediction)[0]
    confidence = np.max(prediction_proba) * 100

    st.subheader("🎯 Predicted Job Role")
    st.success(f"{job_role} (Confidence: {confidence:.2f}%)")

    # Optional: Show full probability scores
    proba_df = pd.DataFrame(prediction_proba, columns=label_encoder.classes_)
    with st.expander("🔬 View All Class Probabilities"):
        st.dataframe(proba_df.T.rename(columns={0: "Probability"}))

    st.subheader("📄 Resume Text")
    with st.expander("Click to view full resume text"):
        st.text_area("Resume Content", text, height=300)

