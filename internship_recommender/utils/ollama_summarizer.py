import requests
import json
import time

class OllamaSummarizer:
    def __init__(self, model="llama3.2", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"

    def is_available(self):
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def summarize_resume(self, resume_text, max_length=500):
        """Generate a professional profile summary using Ollama"""
        if not self.is_available():
            # Fallback to basic extraction if Ollama is not available
            return self._basic_summary(resume_text, max_length)

        prompt = f"""Write a professional summary for this resume. Start directly with the summary in first person and dont give the introductory text like Here is a concise professional summary,"professional summary:", "summary:",
                    "here's a concise professional summary:", "here's a professional summary:",
                    "here is a professional summary for the given resume:",
                    "here is a professional summary:". Keep it concise, under {max_length} characters. Focus on key skills, experience, and goals.give only 3 to 4 sentences.

Resume content:
{resume_text[:1500]}

Summary:"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 250
            }
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                summary = result.get("response", "").strip()
                # Clean up and limit length
                summary = summary.replace("\n", " ").strip()
                # Remove common prefixes
                prefixes_to_remove = [
                    "professional summary:", "summary:",
                    "here's a concise professional summary:", "here's a professional summary:",
                    "here is a professional summary for the given resume:",
                    "here is a professional summary:"
                ]
                for prefix in prefixes_to_remove:
                    if summary.lower().startswith(prefix):
                        summary = summary[len(prefix):].strip()
                        break
                # Ensure we have a meaningful summary, not just repeating input
                if len(summary) < 50 or summary.lower() in resume_text.lower():
                    return self._basic_summary(resume_text, max_length)
                return summary[:max_length] if len(summary) > max_length else summary
            else:
                print(f"Ollama API error: {response.status_code} - {response.text}")
                return self._basic_summary(resume_text, max_length)
        except Exception as e:
            print(f"Ollama summarization failed: {e}")
            return self._basic_summary(resume_text, max_length)

    def _basic_summary(self, resume_text, max_length):
        """Fallback basic summary extraction"""
        # Simple extraction as before
        import re
        text_lower = resume_text.lower()
        summary_keywords = ["profile summary", "objective", "about me", "professional summary", "summary"]

        for keyword in summary_keywords:
            if keyword in text_lower:
                idx = text_lower.find(keyword)
                section_start = idx + len(keyword)
                next_sections = ["education", "experience", "skills", "projects", "achievements"]
                end_idx = len(resume_text)
                for section in next_sections:
                    if section in text_lower[section_start:]:
                        end_idx = text_lower.find(section, section_start)
                        break
                summary_text = resume_text[section_start:end_idx].strip()
                if summary_text:
                    summary_text = re.sub(r'[^\w\s.,-]', '', summary_text)
                    return summary_text[:max_length]

        # Fallback to first paragraph
        paragraphs = [p.strip() for p in resume_text.split("\n\n") if p.strip()]
        if paragraphs:
            valid_paragraphs = [p for p in paragraphs if len(p) > 50 and not p.isupper()]
            if valid_paragraphs:
                return valid_paragraphs[0][:max_length]

        # Final fallback
        sents = re.split(r'(?<=[.!?]) +', resume_text.strip())
        valid_sents = [s for s in sents if len(s) > 10][:3]
        return " ".join(valid_sents)[:max_length]

# Global instance
summarizer = OllamaSummarizer()
