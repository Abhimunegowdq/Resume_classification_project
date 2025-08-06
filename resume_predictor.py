import streamlit as st
import fitz  # PyMuPDF
import docx2txt
import re
import joblib

# Load ML model and vectorizer
model = joblib.load("decision_tree_resume_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Function to extract text from file
def extract_text(file):
    if file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    elif file.name.endswith(".docx"):
        return docx2txt.process(file)
    return ""

# Clean text for ML prediction
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Extract Name
def extract_name(text):
    lines = text.strip().split("\n")
    for line in lines:
        clean_line = line.strip()
        # Ignore common headings
        if clean_line and len(clean_line.split()) <= 5:
            if not re.search(r"\b(resume|curriculum|cv|profile|bio|summary)\b", clean_line, re.IGNORECASE):
                if not re.search(r"skills|experience|education|email", clean_line, re.IGNORECASE):
                    return clean_line
    return "Not found"


# Extract Email
def extract_email(text):
    match = re.search(r"\S+@\S+", text)
    return match.group() if match else "Not found"

# Extract Skills - limit to 2 lines
def extract_skills(text):
    skills = []
    for line in text.split("\n"):
        if re.search(r"skills|technologies|tools|frameworks", line, re.IGNORECASE):
            skills.append(line.strip())
    return "\n".join(skills[:2]) if skills else "Not found"

# Extract Experience - limit to 3 lines
def extract_experience(text):
    exp = []
    for line in text.split("\n"):
        if re.search(r"experience|worked|years|duration|responsible|developed", line, re.IGNORECASE):
            exp.append(line.strip())
    return "\n".join(exp[:3]) if exp else "Not found"

# Extract Education
def extract_education(text):
    edu = []
    for line in text.split("\n"):
        if re.search(r"education|bachelor|master|degree|university|college", line, re.IGNORECASE):
            edu.append(line.strip())
    return "\n".join(edu[:3]) if edu else "Not found"

# Streamlit UI
st.set_page_config(page_title="Resume Job Role Predictor", layout="centered")
st.title("📄 Resume Job Role Prediction App")

uploaded = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded:
    raw_text = extract_text(uploaded)
    cleaned_text = clean_text(raw_text)

    # Predict job role
    X_input = vectorizer.transform([cleaned_text])
    y_pred = model.predict(X_input)
    predicted_role = label_encoder.inverse_transform(y_pred)[0]

    # Display output
    st.markdown(f"👤 **Name:** {extract_name(raw_text)}")
    st.markdown(f"📧 **Email:** {extract_email(raw_text)}")
    st.markdown(f"🛠 **Skills:**\n\n{extract_skills(raw_text)}")
    st.markdown(f"💼 **Experience:**\n\n{extract_experience(raw_text)}")
    st.markdown(f"🎓 **Education:**\n\n{extract_education(raw_text)}")
    st.success(f"🔮 **Predicted Job Role:** {predicted_role}")

   




