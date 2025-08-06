import streamlit as st
import re
import docx2txt
import PyPDF2
import joblib

# --- Load your trained components ---
model = joblib.load("decision_tree_resume_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# --- Helper functions ---
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    return docx2txt.process(file)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    return text



import re

def extract_name(text):
    lines = text.strip().split('\n')

    # Common headings to skip
    skip_keywords = ['objective', 'summary', 'career', 'resume', 'programming', 'skills', 'experience', 'education', 'languages','PeopleSoft Database Administrator']

    # Check first 10 lines only (to avoid noise)
    for line in lines[:10]:
        line = line.strip()
        if not line:
            continue

        # Skip lines with common non-name headings
        if any(keyword in line.lower() for keyword in skip_keywords):
            continue

        # Check for likely name pattern: starts with letters, contains alphabetic words, not too long
        if re.match(r'^[A-Za-z][A-Za-z.\s]{2,50}$', line):
            return line.strip()

    return "Name not found"


def extract_email(text):
    match = re.search(r'\b[\w.-]+@[\w.-]+\.\w+\b', text)
    return match.group() if match else None

def extract_skills(text):
    skill_keywords = ['python', 'java', 'c++', 'sql', 'javascript', 'html', 'css', 'react', 'node.js',
        'angular', 'c#', 'php', 'mysql', 'mongodb', 'oracle', 'pl/sql',
        'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'keras', 'power bi', 'tableau',
        'aws', 'azure', 'git', 'docker', 'jira', 'crm', 'erp', 'peoplesoft', 'excel']
    text = text.lower()
    return list(set([skill for skill in skill_keywords if skill in text]))

def extract_experience(text):
    lines = text.splitlines()
    exp_lines = [line for line in lines if any(word in line.lower() for word in ["experience", "worked", "project", "intern"])]
    return exp_lines[:3]

def extract_education(text):
    lines = text.splitlines()
    edu_lines = [line for line in lines if any(word in line.lower() for word in ["bachelor", "master", "university", "college", "school", "b.tech", "m.tech", "degree"])]
    return edu_lines[:3]

# --- Streamlit UI ---
st.set_page_config(page_title="Resume Job Role Predictor", layout="centered")
st.title("📄 Resume Job Role Predictor")
st.write("Upload a resume file (.pdf or .docx) to extract details and predict the job role.")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])

if uploaded_file:
    # Step 1: Extract Text
    raw_text = extract_text_from_pdf(uploaded_file) if uploaded_file.name.endswith(".pdf") else extract_text_from_docx(uploaded_file)

    # Step 2: Extract Details
    name = extract_name(raw_text)
    email = extract_email(raw_text)
    skills = extract_skills(raw_text)
    experience = extract_experience(raw_text)
    education = extract_education(raw_text)

    # Step 3: Predict Job Role
    cleaned = clean_text(raw_text)
    vector = vectorizer.transform([cleaned])
    pred_label = model.predict(vector)[0]
    job_role = label_encoder.inverse_transform([pred_label])[0]

    # Step 4: Show Output
    st.markdown(f"**👤 Name:** {name}")
    st.markdown("**🛠 Skills:**")
    if skills:
        st.markdown(f"- {', '.join(skills[:4])}")
        st.markdown(f"- {', '.join(skills[4:8])}" if len(skills) > 4 else "")
    else:
        st.write("Not found")

    st.markdown("**💼 Experience:**")
    for line in experience:
        st.markdown(f"- {line.strip()}")

    st.markdown("**🎓 Education:**")
    for line in education:
        st.markdown(f"- {line.strip()}")

    st.success(f"**🧑‍💼 Predicted Job Role:** {job_role}")

    # Step 5: Download Option
    resume_output = f"""
👤 Name: {name}



🛠 Skills:
{', '.join(skills) if skills else 'Not found'}

💼 Experience:
{chr(10).join([f'- {line.strip()}' for line in experience])}

🎓 Education:
{chr(10).join([f'- {line.strip()}' for line in education])}

🧑‍💼 Predicted Job Role: {job_role}
"""

    st.download_button(
        label="📥 Download Resume Analysis",
        data=resume_output,
        file_name="resume_analysis.txt",
        mime="text/plain"
    )





 













