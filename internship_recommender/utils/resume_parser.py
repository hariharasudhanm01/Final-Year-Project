import pdfplumber
import docx

def extract_text_from_pdf(path):
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)

def extract_text_from_docx(path):
    doc = docx.Document(path)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return "\n".join(fullText)

def extract_text_from_file(path):
    """Extract text from resume file (PDF, DOCX, or TXT)"""
    try:
        path_lower = path.lower()
        if path_lower.endswith(".pdf"):
            return extract_text_from_pdf(path)
        elif path_lower.endswith(".docx") or path_lower.endswith(".doc"):
            return extract_text_from_docx(path)
        else:
            # try reading as text
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        print(f"Error extracting text from {path}: {e}")
        return ""
