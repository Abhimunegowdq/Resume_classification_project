import streamlit as st
import docx2txt
import PyPDF2
import re
import joblib

# Load model components
model = joblib.load("decision_tree_resume_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("📄 Resume Job Role Predictor")
st.write("Upload a resume (PDF or DOCX) to extract information and predict the job role.")

# ---- RESUME TEXT EXTRACTOR ----
def extract_text(file):
    if file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    elif file.name.endswith(".docx"):
        return docx2txt.process(file)
    else:
        return ""

# ---- CLEAN TEXT ----
def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

# ---- NAME ----
def extract_name(text):
    match = re.search(r"Name\s*[:\-]?\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return "Not found"

# ---- EMAIL ----
def extract_email(text):
    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return match.group(0) if match else "Not found"

# ---- SKILLS ----
def extract_skills(text):
    skills_keywords = ["python", "java", "c++", "sql", "excel", "tableau", "power bi", "html", "css", "javascript", "crm", "oracle", "pl/sql", "ssis", "linux"]
    skills_found = [skill for skill in skills_keywords if skill.lower() in text.lower()]
    return skills_found[:6]  # Top 6 for brevity

# ---- EXPERIENCE ----
def extract_experience(text):
    experience_lines = []
    exp_matches = re.findall(r"(experience|exp\.?|worked|responsible|duration).{0,100}", text, re.IGNORECASE)
    for line in exp_matches:
        if len(experience_lines) >= 3:
            break
        experience_lines.append(line.strip())
    return experience_lines

# ---- EDUCATION ----
def extract_education(text):
    edu_matches = re.findall(r"(Bachelor|Master|B\.Tech|M\.Tech|BSc|MSc|BE|ME|BCA|MCA|BBA|MBA).{0,50}", text, re.IGNORECASE)
    return edu_matches[:3]

# ---- STREAMLIT UI ----
uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "docx"])

if uploaded_file is not None:
    text = extract_text(uploaded_file)
    cleaned = clean_text(text)

    # Extract info
    name = extract_name(cleaned)
    email = extract_email(cleaned)
    skills = extract_skills(cleaned)
    experience = extract_experience(cleaned)
    education = extract_education(cleaned)

    # Predict job role
    features = vectorizer.transform([cleaned])
    predicted_category = model.predict(features)[0]
    predicted_label = label_encoder.inverse_transform([predicted_category])[0]

    # Display Results
    st.markdown(f"👤 **Name:** {name}")
    st.markdown(f"📧 **Email:** {email}")

    st.markdown("🛠 **Skills:**")
    if skills:
        skills_output = ", ".join(skills[:3]) + "\n" + ", ".join(skills[3:]) if len(skills) > 3 else ", ".join(skills)
        st.text(skills_output)
    else:
        st.text("Not found")

    st.markdown("💼 **Experience:**")
    if experience:
        for line in experience:
            st.text(f"• {line}")
    else:
        st.text("Not found")

    st.markdown("🎓 **Education:**")
    if education:
        for degree in education:
            st.text(f"• {degree}")
    else:
        st.text("Not found")

    st.markdown(f"🧑‍💼 **Predicted Job Role:** {predicted_label}")



