import streamlit as st
import PyPDF2
import docx
import re
import pickle
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load ML components
model = joblib.load("decision_tree_resume_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Text extraction functions
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text() + '\n'
    return text

def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    return '\n'.join([para.text for para in doc.paragraphs])

# Extraction helpers
def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return match.group(0) if match else None

def extract_name(text):
    lines = text.strip().split('\n')[:15]  # First 15 lines only
    for line in lines:
        doc = nlp(line.strip())
        for ent in doc.ents:
            if ent.label_ == "PERSON" and len(ent.text.split()) <= 3:
                return ent.text
    return "Not found"

def extract_skills(text):
    skills = re.findall(r'\b[a-zA-Z][a-zA-Z0-9\+\#\.]{2,}\b', text)
    return list(set([s.lower() for s in skills if s.isalpha() or re.match(r'^[a-zA-Z0-9\+\#\.]+$', s)]))

def extract_experience(text):
    lines = text.strip().split('\n')
    exp_lines = [line for line in lines if 'experience' in line.lower()]
    return exp_lines[:3] if exp_lines else lines[:3]

def extract_education(text):
    lines = text.strip().split('\n')
    edu_keywords = ['education', 'b.tech', 'm.tech', 'bachelor', 'master', 'degree', 'university']
    return [line for line in lines if any(k in line.lower() for k in edu_keywords)][:3]

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

# Streamlit app
st.set_page_config(page_title="Resume Job Role Predictor", layout="centered")
st.title("📄 Resume Job Role Predictor")
st.write("Upload a resume file (.pdf or .docx) to extract details and predict the job role.")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])

if uploaded_file:
    # Extract text
    raw_text = extract_text_from_pdf(uploaded_file) if uploaded_file.name.endswith('.pdf') else extract_text_from_docx(uploaded_file)

    # Extract components
    name = extract_name(raw_text)
    email = extract_email(raw_text)
    skills = extract_skills(raw_text)
    experience = extract_experience(raw_text)
    education = extract_education(raw_text)

    # Predict job role
    cleaned = clean_text(raw_text)
    vector = vectorizer.transform([cleaned])
    pred_label = model.predict(vector)[0]
    job_role = label_encoder.inverse_transform([pred_label])[0]

    # Display output
    st.markdown(f" **Name:** {name}")
    st.markdown(f" **Email:** {email if email else 'Not found'}")

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

    # Prepare download text
    resume_output = f"""
 Name: {name}
 Email: {email if email else 'Not found'}

 Skills:
{', '.join(skills) if skills else 'Not found'}

 Experience:
{chr(10).join([f'- {line.strip()}' for line in experience])}

 Education:
{chr(10).join([f'- {line.strip()}' for line in education])}

 Predicted Job Role: {job_role}
    """

    st.download_button(
        label=" Download Resume Analysis",
        data=resume_output,
        file_name="resume_analysis.txt",
        mime="text/plain"
    )











