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
    # Common invalid headers to skip
    invalid_headers = {'professional summary', 'curriculum vitae', 'resume', 'profile', 'about me'}
    
    # Try: "Name: John Doe"
    name_match = re.search(r'(?:name\s*[:\-]\s*)([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)', text, re.IGNORECASE)
    if name_match:
        name = name_match.group(1).strip()
        if name.lower() not in invalid_headers:
            return name

    # Try: first 10 lines for proper name format
    lines = text.strip().split('\n')
    for line in lines[:10]:
        line_clean = line.strip()
        line_lower = line_clean.lower()

        if line_lower in invalid_headers or len(line_clean.split()) > 6:
            continue

        # Match: John Doe, John A. Smith, etc.
        if re.match(r'^([A-Z][a-z]+\s){1,3}[A-Z][a-z]+\.?$', line_clean):
            return line_clean

        # Match all-uppercase: JOHN DOE → John Doe
        if re.match(r'^([A-Z]{2,}\s?){2,3}$', line_clean):
            return line_clean.title()

    # Fallback: return first capitalized line with 2–3 titlecase words
    for line in lines[:10]:
        words = line.strip().split()
        capitalized = [w for w in words if w.istitle()]
        if 2 <= len(capitalized) <= 4:
            return " ".join(capitalized)

    # Last resort: return first line
    return lines[0].strip().title()
    

def extract_skills(text):
    # Normalize text
    text = text.lower()
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # 1️⃣ Try to extract from skills section
    skills_section = []
    capture = False
    for line in lines:
        if any(header in line for header in ['skills', 'technical skills', 'key skills']):
            capture = True
            continue
        if capture:
            if any(stop in line for stop in ['experience', 'education', 'project', 'summary']):
                break
            skills_section.append(line)
    
    if skills_section:
        # Join all captured lines, split by comma or whitespace
        joined_skills = ' '.join(skills_section)
        found = re.findall(r'\b[a-zA-Z\-\+#\.]{2,}\b', joined_skills)
        return list(set([s.strip() for s in found if len(s.strip()) > 1]))

    # 2️⃣ Fallback to predefined list
    skill_keywords = [
        'python', 'java', 'c++', 'sql', 'javascript', 'html', 'css', 'react', 'node.js',
        'angular', 'c#', 'php', 'mysql', 'mongodb', 'oracle', 'pl/sql',
        'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'keras', 'power bi', 'tableau',
        'aws', 'azure', 'git', 'docker', 'jira', 'crm', 'erp', 'peoplesoft', 'excel'
    ]
    found_skills = []
    for skill in skill_keywords:
        if re.search(r'\b' + re.escape(skill) + r'\b', text):
            found_skills.append(skill)
    return list(set(found_skills)) if found_skills else ["Not found"]







               
   
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
    st.markdown(f" **Name:** {name}")
    

    st.markdown(" **Skills:**")
    if skills:
        st.markdown(f"- {', '.join(skills[:4])}")
        st.markdown(f"- {', '.join(skills[4:8])}" if len(skills) > 4 else "")
    else:
        st.write("Not found")

    st.markdown(" **Experience:**")
    for line in experience:
        st.markdown(f"- {line.strip()}")

    st.markdown(" **Education:**")
    for line in education:
        st.markdown(f"- {line.strip()}")

    st.success(f" **Predicted Job Role:** {job_role}")










