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

def extract_education(text):
    edu_keywords = ['bachelor', 'master', 'b.tech', 'm.tech', 'b.sc', 'm.sc', 'bca', 'mca', 'bcom', 'mcom']
    lines = text.lower().split('\n')
    matches = [line.strip() for line in lines if any(kw in line for kw in edu_keywords)]
    return "\n".join(matches).title() if matches else "Not found"

def extract_experience(text):
    experience_lines = []
    for line in text.split('\n'):
        if re.search(r'\bexperience\b|\bworked\b|\bproject\b|\bduration\b|\byears? of\b', line, re.IGNORECASE):
            experience_lines.append(line.strip())
    return "\n".join(experience_lines).strip() if experience_lines else "Not found"

def extract_skills(text):
    skills_lines = []
    for line in text.split('\n'):
        if re.search(r'\bskills\b|\btechnologies\b|\btools\b|\bproficient\b|\bexpertise\b', line, re.IGNORECASE):
            skills_lines.append(line.strip())
    return "\n".join(skills_lines).strip() if skills_lines else "Not found"

# ----------- Streamlit UI ----------
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
    education = extract_education(resume_text)
    experience = extract_experience(resume_text)
    skills = extract_skills(resume_text)

    # Display results
    st.markdown(f"👤 **Name:** {name}")
    st.markdown(f"📧 **Email:** {email}")
    st.markdown(f"🛠 **Skills:**\n\n{skills}")
    st.markdown(f"🎓 **Education:**\n\n{education}")
    st.markdown(f"💼 **Experience:**\n\n{experience}")
