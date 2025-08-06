import streamlit as st
import fitz  # PyMuPDF
import docx2txt
import re

# ----------- Text Extraction ----------
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(file):
    return docx2txt.process(file)

# ----------- Info Extraction ----------
def extract_name(text):
    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and len(line.split()) <= 5 and "curriculum" not in line.lower():
            return line
    return "Not found"

def extract_email(text):
    match = re.search(r'\S+@\S+', text)
    return match.group() if match else "Not found"

def extract_skills(text):
    # Extracts skills section lines
    skill_lines = []
    for line in text.split('\n'):
        if re.search(r'\bskills\b|\btechnolog(?:y|ies)\b|\btools\b|\bexpertise\b', line, re.IGNORECASE):
            skill_lines.append(line.strip())
    if skill_lines:
        # Take first 1-2 lines of skill content
        return ' '.join(skill_lines[:2])
    return "Not found"

def extract_experience(text):
    exp_lines = []
    for line in text.split('\n'):
        if re.search(r'\bexperience\b|\bworked\b|\bproject\b|\bduration\b|\byears? of\b', line, re.IGNORECASE):
            exp_lines.append(line.strip())
    if exp_lines:
        return '\n'.join(exp_lines[:5])
    return "Not found"

# ----------- Streamlit UI ----------
st.set_page_config(page_title="Resume Reader", layout="centered")
st.title("📄 Resume Information Extractor")

uploaded_file = st.file_uploader("Upload a Resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file:
    # Extract text
    if uploaded_file.name.endswith('.pdf'):
        resume_text = extract_text_from_pdf(uploaded_file)
    else:
        resume_text = extract_text_from_docx(uploaded_file)

    # Extract info
    name = extract_name(resume_text)
    email = extract_email(resume_text)
    skills = extract_skills(resume_text)
    experience = extract_experience(resume_text)

    # Display output
    st.markdown(f"👤 **Name:** {name}")
    st.markdown(f"📧 **Email:** {email}")
    st.markdown(f"🛠 **Skills (Top 2 lines):**\n\n{skills}")
    st.markdown(f"💼 **Experience (Top 5 lines):**\n\n{experience}")
