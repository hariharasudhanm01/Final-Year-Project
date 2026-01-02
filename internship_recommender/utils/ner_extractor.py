import spacy
import re
from pathlib import Path

# load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    # If not present, user must download separately
    nlp = None

# simple skills list for keyword matching (extendable)
BASE_SKILLS = [
    # Core dev/data
    "python", "java", "c++", "c#", "r", "sql", "excel", "tableau", "power bi",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "machine learning", "deep learning", "data visualization", "statistics",
    "matplotlib", "seaborn", "flask", "django", "git", "linux", "ubuntu",
    "rest", "api", "javascript", "react", "vue.js", "angular", "node", "html", "css",
    "typescript", "bootstrap", "sass", "mongodb", "postgresql", "redis",
    "docker", "kubernetes", "aws", "azure", "gcp", "terraform", "ansible",
    "jenkins", "ci/cd", "microservices", "system design", "algorithms",
    "data structures", "jupyter", "apache spark", "hadoop",
    # UI/UX
    "figma", "adobe xd", "sketch", "wireframing", "prototyping",
    "user research", "usability testing", "information architecture",
    "interaction design", "visual design", "typography", "color theory",
    "design systems", "adobe photoshop", "illustrator",
    # Mobile/Game/Blockchain
    "react native", "flutter", "swift", "kotlin", "ios", "android",
    "unity", "unreal engine", "3d modeling", "game design", "blender",
    "solidity", "ethereum", "web3", "smart contracts", "cryptocurrency", "defi",
    # AI/Cybersecurity
    "nlp", "computer vision", "mlops", "network security", "siem",
    "penetration testing", "risk assessment", "incident response", "firewall", "vpn", "encryption",
    # Business/Product
    "product strategy", "analytics", "agile", "scrum", "market research",
    "business intelligence", "requirements gathering", "documentation",
    # Testing/QA
    "selenium", "test automation", "manual testing", "api testing", "jira",
    # Other
    "firebase", "networking", "security", "command line", "linear algebra"
]

def simple_skill_extract(text):
    text_low = text.lower()
    found = set()
    for s in BASE_SKILLS:
        if s in text_low:
            found.add(s.title() if s.islower() else s)
    return sorted(found)

def extract_profile_summary(text, max_sent=3):
    # Improved heuristics: look for sections like PROFILE SUMMARY, OBJECTIVE, ABOUT ME
    text_lower = text.lower()
    summary_keywords = ["profile summary", "objective", "about me", "professional summary", "summary"]

    for keyword in summary_keywords:
        if keyword in text_lower:
            # Find the section after the keyword
            idx = text_lower.find(keyword)
            section_start = idx + len(keyword)
            # Look for the next section or end of text
            next_sections = ["education", "experience", "skills", "projects", "achievements"]
            end_idx = len(text)
            for section in next_sections:
                if section in text_lower[section_start:]:
                    end_idx = text_lower.find(section, section_start)
                    break
            summary_text = text[section_start:end_idx].strip()
            if summary_text:
                # Clean up and return
                summary_text = re.sub(r'[^\w\s.,-]', '', summary_text)  # Remove special chars except common punctuation
                return summary_text[:500]  # Limit length

    # Fallback: take the first non-empty paragraph or first 2-3 sentences
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if paragraphs:
        # Filter out headers and short lines
        valid_paragraphs = [p for p in paragraphs if len(p) > 50 and not p.isupper()]
        if valid_paragraphs:
            return valid_paragraphs[0][:500]

    # Final fallback to first sentences
    sents = re.split(r'(?<=[.!?]) +', text.strip())
    valid_sents = [s for s in sents if len(s) > 10][:max_sent]
    return " ".join(valid_sents)[:500]

def extract_skills_and_summary(text):
    skills = set()
    if nlp:
        doc = nlp(text)
        # Use noun chunks + entities as candidate phrases to match skills
        for ent in doc.ents:
            # common ent labels may not include skills, so use chunk & token matching
            pass
        # simple approach: match existing skill list
        skills.update(simple_skill_extract(text))
    else:
        skills.update(simple_skill_extract(text))

    # Use Ollama for better summary if available
    from .ollama_summarizer import summarizer
    summary = summarizer.summarize_resume(text)

    return sorted(list(skills)), summary
