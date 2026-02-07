import os
import google.generativeai as genai
import json
from typing import Dict, Any, Optional
import mimetypes

class GeminiService:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            print("Warning: GOOGLE_API_KEY not found in environment variables.")
        else:
            genai.configure(api_key=self.api_key)
            # Try to load the user-requested model, fallback to 1.5-flash if needed
            try:
                self.model = genai.GenerativeModel('gemini-2.5-flash')
            except Exception:
                self.model = genai.GenerativeModel('gemini-2.5-flash')

    def generate_content(self, prompt: str) -> str:
        """Generic method to generate text content using Gemini."""
        if not self.api_key:
            return "Error: Gemini API Key not configured."
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error creating content: {e}"

    def enhance_resume_content(self, section_name: str, original_content: str, job_description: str, instruction: str) -> str:
        """
        Enhances specific resume content based on job description and instructions.
        """
        prompt = f"""
        You are an expert resume writer.
        Task: Rewrite the following {section_name.upper()} section of a resume.
        
        Instruction: {instruction}
        
        Job Description Context:
        {job_description[:1000] if job_description else "General Optimization for Tech Roles"}
        
        Original Content:
        {original_content}
        
        CRITICAL: Return ONLY the rewritten content. Do not add explanations, markdown, or headers.
        Keep it professional, impactful, and truthful.
        """
        return self.generate_content(prompt)

    def generate_design_feedback(self, resume_text: str) -> Dict[str, Any]:
        """
        Analyzes the resume text structure and generates design feedback and template suggestions.
        """
        prompt = f"""
        Act as a Senior Resume Designer. Analyze the following resume content structure (extracted text) and provide design feedback.
        
        Resume Content Preview:
        {resume_text[:2000]}
        
        Provide a JSON response with the following structure:
        {{
            "design_score": 7,  // 1-10 based on structure clarity imply from text
            "critique": ["Point 1", "Point 2"], // 3 bullet points on formatting/structure
            "recommended_template": "Modern Minimalist", // One of: "Modern Minimalist", "Professional Classic", "Creative Tech", "Academic"
            "template_reason": "Why this template suits the content"
        }}
        
        CRITICAL: Return ONLY the JSON object.
        """
        try:
            response_text = self.generate_content(prompt)
            # Cleanup JSON
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1].strip()
                if response_text.startswith("json"):
                    response_text = response_text[4:].strip()
            
            return json.loads(response_text)
        except Exception:
            return {
                "design_score": 5,
                "critique": ["Ensure consistent spacing", "Use clear section headers", "Quantify achievements where possible"],
                "recommended_template": "Professional Classic",
                "template_reason": "Safe choice for most industries."
            }

    def predict_salary(self, resume_summary: str, skills: str, role: str, experience: int) -> Dict[str, Any]:
        """
        Predicts salary range based on profile using Gemini.
        """
        prompt = f"""
        Act as a compensation expert. Predict the annual salary range (in INR) for a candidate with the following profile:
        
        Role: {role}
        Experience: {experience} years
        Skills: {skills}
        Resume Summary: {resume_summary}
        
        Market Context: Indian Job Market (Tier 1/2 Cities).
        
        Return a JSON object ONLY:
        {{
            "min_salary": 500000,
            "max_salary": 1200000,
            "explanation": "Brief explanation citing key skills affecting value."
        }}
        """
        try:
            response_text = self.generate_content(prompt)
            # Cleanup JSON
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1].strip()
                if response_text.startswith("json"):
                    response_text = response_text[4:].strip()
            
            return json.loads(response_text)
        except Exception:
            # Fallback
            return {"min_salary": 300000, "max_salary": 800000, "explanation": "Could not generate precise estimate."}

    def parse_resume(self, file_path: str) -> Dict[str, Any]:
        """
        Parses a resume file using Gemini to extract structured data and generate suggestions.
        
        Args:
            file_path: Absolute path to the resume file (PDF, Image, etc.)
            
        Returns:
            Dictionary containing extracted details and suggestions.
        """
        if not self.api_key:
            return {"error": "Google API Key not configured."}

        try:
            # Determine mime type
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                # Fallback for common types if mimetypes fails
                if file_path.lower().endswith('.pdf'):
                    mime_type = 'application/pdf'
                elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    mime_type = 'image/jpeg'
                else:
                    return {"error": "Unsupported file type."}

            # Upload file to Gemini (using the Files API if needed, or inline data)
            # For 1.5 Flash, we can pass file data directly for some types, but uploading is safer for PDFs
            # Note: For this implementation we'll use the File API which is standard for docs
            
            # Check if file exists
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}

            # Handle DOCX separately (Gemini API doesn't support DOCX upload directly yet)
            if file_path.lower().endswith('.docx'):
                try:
                    # Circular import workaround or use local import
                    from utils.resume_parser import extract_text_from_file
                    extracted_text = extract_text_from_file(file_path)
                    
                    # Create a prompt with the text content instead of file upload
                    prompt = f"""
                    Analyze this resume text and extract the following details into a valid JSON object. 
                    Also provide professional suggestions for LinkedIn and GitHub.
                    
                    RESUME TEXT:
                    {extracted_text}
                    
                    Required JSON Structure (Extract ALL available details):
                    {{
                        "personal_info": {{
                            "name": "Full Name",
                            "email": "Email Address",
                            "linkedin": "LinkedIn Profile URL (or 'Not Found')",
                            "github": "GitHub Profile URL (or 'Not Found')",
                            "phone": "Phone Number (optional)",
                            "summary": "Professional Summary to appear at top of resume"
                        }},
                        "education": [
                            {{
                                "degree": "Degree Name",
                                "institution": "College/University Name",
                                "year": "Year of Completion/Study",
                                "score": "CGPA/Percentage (optional)"
                            }}
                        ],
                        "skills": {{
                            "technical": ["Skill 1", "Skill 2"],
                            "soft": ["Skill 1", "Skill 2"],
                            "tools": ["Tool 1", "Tool 2"]
                        }},
                        "experience": [
                            {{
                                "role": "Job Title",
                                "company": "Company Name",
                                "duration": "Dates (e.g. Jan 2020 - Present)",
                                "description": ["Bullet point 1", "Bullet point 2"]
                            }}
                        ],
                        "projects": [
                            {{
                                "title": "Project Title",
                                "technologies": "Tech Stack used",
                                "description": ["Bullet point 1", "Bullet point 2"]
                            }}
                        ],
                        "certifications": [
                            "Cert Name - Issuer",
                            "Cert Name 2"
                        ],
                        "achievements": [
                           "Achievement 1",
                           "Achievement 2"
                        ],
                        "professional_info": {{
                            "sector": "Primary Sector",
                            "experience_years": "Total years number"
                        }},
                        "suggestions": {{
                            "linkedin_maintenance": ["Tip 1", "Tip 2"],
                            "github_project_ideas": [
                                {{"title": "Idea 1", "description": "Desc", "tech_stack": ["T1", "T2"]}}
                            ]
                        }}
                    }}
                    
                    IMPORTANT: Return ONLY the JSON object. Do not wrap it in markdown code blocks.
                    If a field is not found, use empty list [] or null.
                    """
                    
                    response = self.model.generate_content(prompt)
                    
                    # Cleanup text to ensure it's valid JSON
                    response_text = response.text.strip()
                    if response_text.startswith("```json"):
                        response_text = response_text[7:]
                    if response_text.startswith("```"):
                        response_text = response_text[3:]
                    if response_text.endswith("```"):
                        response_text = response_text[:-3]
                    
                    parsed_data = json.loads(response_text)
                    return parsed_data

                except Exception as e:
                    print(f"Error processing DOCX in parse_resume: {e}")
                    return {"error": f"Failed to process DOCX: {str(e)}"}

            print(f"Uploading file to Gemini: {file_path}")
            uploaded_file = genai.upload_file(file_path, mime_type=mime_type)
            
            prompt = """
            Analyze this resume and extract the following details into a valid JSON object. 
            Also provide professional suggestions for LinkedIn and GitHub.
            
            Required JSON Structure (Extract ALL available details):
            {
                "personal_info": {
                    "name": "Full Name",
                    "email": "Email Address",
                    "linkedin": "LinkedIn Profile URL (or 'Not Found')",
                    "github": "GitHub Profile URL (or 'Not Found')",
                    "phone": "Phone Number (optional)",
                    "summary": "Professional Summary to appear at top of resume"
                },
                "education": [
                    {
                        "degree": "Degree Name",
                        "institution": "College/University Name",
                        "year": "Year of Completion/Study",
                        "score": "CGPA/Percentage (optional)"
                    }
                ],
                "skills": {
                    "technical": ["Skill 1", "Skill 2"],
                    "soft": ["Skill 1", "Skill 2"],
                    "tools": ["Tool 1", "Tool 2"]
                },
                "experience": [
                    {
                        "role": "Job Title",
                        "company": "Company Name",
                        "duration": "Dates (e.g. Jan 2020 - Present)",
                        "description": ["Bullet point 1", "Bullet point 2"]
                    }
                ],
                "projects": [
                    {
                        "title": "Project Title",
                        "technologies": "Tech Stack used",
                        "description": ["Bullet point 1", "Bullet point 2"]
                    }
                ],
                "certifications": [
                    "Cert Name - Issuer",
                    "Cert Name 2"
                ],
                "achievements": [
                   "Achievement 1",
                   "Achievement 2"
                ],
                "professional_info": {
                    "sector": "Primary Sector",
                    "experience_years": "Total years number"
                },
                "suggestions": {
                    "linkedin_maintenance": ["Tip 1", "Tip 2"],
                    "github_project_ideas": [
                        {"title": "Idea 1", "description": "Desc", "tech_stack": ["T1", "T2"]}
                    ]
                }
            }
            
            IMPORTANT: Return ONLY the JSON object. Do not wrap it in markdown code blocks.
            If a field is not found, use empty list [] or null.
            """

            response = self.model.generate_content([prompt, uploaded_file])
            
            # Cleanup text to ensure it's valid JSON
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            parsed_data = json.loads(response_text)
            return parsed_data

        except Exception as e:
            print(f"Error parsing resume with Gemini: {e}")
            return {"error": str(e)}

# Global instance
gemini_service = GeminiService()
