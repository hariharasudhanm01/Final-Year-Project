"""
HR Chatbot using Ollama for candidate discussions
Provides intelligent responses about candidates using LLM
"""
import requests
import json
import os
from typing import Dict, List, Optional
from .database import db
from .resume_parser import extract_text_from_file
from .candidate_matcher import get_candidate_insights, match_candidates_to_job


class HRChatbot:
    def __init__(self, model="llama3.2", base_url="http://localhost:11434"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.chat_url = f"{base_url}/api/chat"
        # Try to find an available model
        self.model = self._find_available_model(model)
    
    def _find_available_model(self, preferred_model="llama3.2"):
        """Find an available model, preferring the specified one"""
        try:
            available_models = self.get_available_models()
            
            if not available_models:
                return preferred_model  # Return preferred even if not available, error will be handled later
            
            # Check if preferred model exists
            if any(preferred_model in m or m.startswith(preferred_model) for m in available_models):
                return preferred_model
            
            # Try to find a suitable alternative
            # Priority: general llama models (not codellama) > tinyllama > codellama > gemma > others
            # Prefer models that are good for chat/conversation
            
            # First, try to find general llama models (llama3, llama2, etc.)
            for model_name in available_models:
                name_lower = model_name.lower()
                if ('llama3' in name_lower or 'llama2' in name_lower) and 'code' not in name_lower:
                    return model_name.split(':')[0]
            
            # Then try tinyllama (small but good for chat)
            for model_name in available_models:
                if 'tinyllama' in model_name.lower():
                    return model_name.split(':')[0]
            
            # Then any other llama model
            for model_name in available_models:
                if 'llama' in model_name.lower():
                    return model_name.split(':')[0]
            
            # If no llama, try gemma
            for model_name in available_models:
                if 'gemma' in model_name.lower():
                    return model_name.split(':')[0]
            
            # Fallback to first available model
            return available_models[0].split(':')[0]
        except:
            # If we can't check, return preferred model
            return preferred_model

    def is_available(self):
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self):
        """Get list of available Ollama models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [model.get('name', '') for model in data.get('models', [])]
                return models
            return []
        except:
            return []
    
    def check_model_exists(self):
        """Check if the configured model exists"""
        available_models = self.get_available_models()
        # Check for exact match or partial match (e.g., llama3.2, llama3.2:latest)
        model_variants = [self.model, f"{self.model}:latest", f"{self.model}:latest"]
        return any(model in available_models or any(av_model.startswith(self.model) for av_model in available_models) for model in model_variants)

    def _get_candidate_context(self, candidate_id: Optional[int] = None, 
                               job_id: Optional[int] = None) -> str:
        """
        Build context about candidates for the LLM.
        Returns formatted string with candidate information.
        """
        context_parts = []
        
        if candidate_id:
            # Get specific candidate information
            try:
                candidate = db.get_candidate_by_id(candidate_id)
                if candidate:
                    context_parts.append("=== CANDIDATE PROFILE ===")
                    context_parts.append(f"Candidate ID: {candidate_id}")
                    context_parts.append(f"Name: {candidate.get('full_name') or candidate.get('username', 'Unknown')}")
                    context_parts.append(f"Email: {candidate.get('email', 'N/A')}")
                    context_parts.append(f"Degree: {candidate.get('degree', 'N/A')}")
                    context_parts.append(f"Study Year: {candidate.get('study_year', 'N/A')}")
                    context_parts.append(f"Stream: {candidate.get('stream', 'N/A')}")
                    context_parts.append(f"Sector: {candidate.get('sector', 'N/A')}")
                    
                    # Skills - make sure to include all skills
                    if candidate.get('skills'):
                        skills_list = [s.strip() for s in candidate.get('skills', '').split(',') if s.strip()]
                        if skills_list:
                            context_parts.append(f"Skills ({len(skills_list)} total): {', '.join(skills_list)}")
                        else:
                            context_parts.append("Skills: None listed")
                    else:
                        context_parts.append("Skills: None listed")
                else:
                    context_parts.append(f"=== CANDIDATE NOT FOUND ===")
                    context_parts.append(f"Candidate ID {candidate_id} was requested but not found in database.")
            except Exception as e:
                context_parts.append(f"=== ERROR LOADING CANDIDATE ===")
                context_parts.append(f"Error loading candidate {candidate_id}: {str(e)}")
                
                # Resume content
                resume_path = candidate.get('resume_path')
                if resume_path:
                    # Get upload folder path - handle both relative and absolute paths
                    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    upload_folder = os.path.join(current_dir, 'uploads')
                    # Resume path might be just filename or full path
                    if os.path.isabs(resume_path) or '/' in resume_path or '\\' in resume_path:
                        # Extract just filename if full path
                        resume_filename = os.path.basename(resume_path)
                    else:
                        resume_filename = resume_path
                    
                    full_resume_path = os.path.join(upload_folder, resume_filename)
                    if os.path.exists(full_resume_path):
                        try:
                            from .resume_parser import extract_text_from_file
                            resume_text = extract_text_from_file(full_resume_path)
                            if resume_text:
                                # Include first 2000 chars of resume
                                context_parts.append(f"\n=== RESUME CONTENT ===")
                                context_parts.append(resume_text[:2000] + ("..." if len(resume_text) > 2000 else ""))
                        except Exception as e:
                            context_parts.append(f"\nNote: Could not read resume file: {str(e)}")
                
                # Match insights if job_id provided
                if job_id:
                    job_posting = db.get_job_posting_by_id(job_id)
                    if job_posting:
                        insights = get_candidate_insights(candidate, job_posting)
                        context_parts.append("\n=== MATCH ANALYSIS ===")
                        context_parts.append(f"Overall Match Score: {insights['match_score']['overall_score']}%")
                        context_parts.append(f"Skill Match: {insights['match_score']['skill_match_score']}%")
                        context_parts.append(f"Matched Skills: {', '.join(insights['match_score']['matched_skills'][:10])}")
                        if insights['match_score']['missing_skills']:
                            context_parts.append(f"Missing Skills: {', '.join(insights['match_score']['missing_skills'][:10])}")
                        context_parts.append(f"Recommendation: {insights['match_score']['recommendation']}")
                        context_parts.append(f"Analysis: {insights['match_score']['recommendation_reason']}")
        else:
            # Get general candidate statistics
            try:
                all_candidates = db.get_all_candidates(limit=100)
                context_parts.append("=== CANDIDATE DATABASE OVERVIEW ===")
                context_parts.append(f"Total Candidates: {len(all_candidates)}")
                
                if all_candidates:
                    # List some candidate names for reference
                    candidate_names = []
                    for cand in all_candidates[:10]:
                        name = cand.get('full_name') or cand.get('username', 'Unknown')
                        candidate_names.append(name)
                    if candidate_names:
                        context_parts.append(f"Sample Candidates: {', '.join(candidate_names)}")
                    
                    # Skills distribution
                    all_skills = {}
                    degree_dist = {}
                    for cand in all_candidates[:50]:  # Sample first 50
                        if cand.get('skills'):
                            for skill in cand.get('skills', '').split(','):
                                skill = skill.strip().lower()
                                if skill:
                                    all_skills[skill] = all_skills.get(skill, 0) + 1
                        if cand.get('degree'):
                            degree = cand.get('degree')
                            degree_dist[degree] = degree_dist.get(degree, 0) + 1
                    
                    if all_skills:
                        top_skills = sorted(all_skills.items(), key=lambda x: x[1], reverse=True)[:10]
                        context_parts.append(f"\nTop Skills: {', '.join([f'{s}({c})' for s, c in top_skills])}")
                    
                    if degree_dist:
                        context_parts.append(f"\nDegree Distribution: {', '.join([f'{d}({c})' for d, c in degree_dist.items()])}")
                else:
                    context_parts.append("No candidates found in database.")
            except Exception as e:
                context_parts.append(f"Error loading candidate database: {str(e)}")
        
        context_str = "\n".join(context_parts)
        # Ensure we always return something
        if not context_str or context_str.strip() == "":
            context_str = "No candidate information available. You can answer general questions about HR processes."
        
        return context_str

    def _build_system_prompt(self, candidate_id: Optional[int] = None, 
                            job_id: Optional[int] = None) -> str:
        """Build system prompt for the chatbot"""
        context = self._get_candidate_context(candidate_id, job_id)
        
        # Ensure context is not empty
        if not context or context.strip() == "":
            context = "No specific candidate information available. You can answer general questions about the candidate database."
        
        system_prompt = f"""You are an AI assistant helping HR professionals analyze and discuss candidates for job positions. 
You have access to candidate profiles, resumes, and match analysis data.

IMPORTANT: Use the following candidate information to answer questions. Always refer to specific details from this data:

{context}

CRITICAL INSTRUCTIONS:
- Match the level of detail in your response to the question asked
- If asked for just a name, provide ONLY the name(s) - nothing else
- If asked for skills, provide ONLY the skills
- If asked for a summary or detailed analysis, then provide comprehensive information
- Keep responses concise and directly answer what was asked
- Do NOT add extra information unless specifically requested

Your role:
- Answer questions about candidates professionally and accurately
- Provide insights based on candidate profiles, skills, and match scores ONLY when asked
- Suggest which candidates might be good fits for specific roles when asked
- Help compare candidates when asked
- Be concise - answer what is asked, nothing more

Guidelines:
- ALWAYS base your answers on the candidate data provided above
- When asked a simple question (like "what is the name"), give a simple answer (just the name)
- When asked for details or analysis, then provide comprehensive information
- If asked about a specific candidate, refer to their profile details only if relevant to the question
- Be professional and helpful
- If you don't have information in the context, say so clearly

Answer the HR's question directly and concisely. If they ask for just a name, give just the name(s). If they ask for details, provide details."""
        
        return system_prompt

    def _post_process_answer(self, answer: str, question: str) -> str:
        """Post-process answer to ensure it matches the question's simplicity"""
        question_lower = question.lower()
        
        # If asked for just names, extract only names
        if any(word in question_lower for word in ['name', 'who', 'what is the name', 'list names', 'candidate names']):
            # Try to extract just names from the answer
            import re
            # Look for patterns like "1. Name" or "Name:" or just names
            lines = answer.split('\n')
            names_only = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Skip lines that are clearly not names
                if any(word in line.lower() for word in ['skills:', 'degree:', 'email:', 'candidate:', 'overall', 'match score', 'b.tech', 'b.sc', 'mca']):
                    continue
                
                # Remove numbered lists and bullet points
                original_line = line
                line = re.sub(r'^\d+\.\s*', '', line)
                line = re.sub(r'^[-*]\s*', '', line)
                
                # Handle format like "Harish - Name: Harish, Skills: ..."
                if ' - ' in line or ' – ' in line:
                    # Extract name before the dash
                    name_part = line.split(' - ')[0].strip()
                    if ' – ' in name_part:
                        name_part = name_part.split(' – ')[0].strip()
                    # Clean up the name - remove parentheses and numbers
                    name = re.sub(r'\(.*?\)', '', name_part).strip()
                    name = re.sub(r'\d+', '', name).strip()  # Remove numbers
                    if name and len(name) < 30 and name[0].isalpha():
                        if not any(word in name.lower() for word in ['skills', 'degree', 'email', 'candidate']):
                            names_only.append(name)
                    continue
                
                # Handle format like "Name: John"
                if ':' in line:
                    parts = line.split(':')
                    if len(parts) > 1:
                        name_part = parts[0].strip().lower()
                        # If it's "Name:" or similar, get the value
                        if 'name' in name_part:
                            name = parts[1].split(',')[0].strip()
                            name = re.sub(r'\(.*?\)', '', name).strip()  # Remove parentheses
                            if name and len(name) < 30:
                                names_only.append(name)
                            continue
                
                # Simple name format - extract first word(s) that look like a name
                words = line.split()
                if words:
                    # Take first 1-2 words as potential name
                    potential_name = ' '.join(words[:2]).strip()
                    # Remove common prefixes/suffixes and numbers
                    potential_name = re.sub(r'\(.*?\)', '', potential_name).strip()
                    potential_name = re.sub(r'\d+', '', potential_name).strip()
                    # Check if it looks like a name (not too long, starts with letter, no common words)
                    if (len(potential_name) < 30 and 
                        len(potential_name) > 1 and
                        potential_name[0].isalpha() and
                        not any(word in potential_name.lower() for word in ['skills', 'degree', 'email', 'candidate', 'name:', 'b.tech', 'b.sc', 'mca'])):
                        names_only.append(potential_name)
            
            # If we found names, return them as a simple list
            if names_only:
                # Remove duplicates while preserving order
                seen = set()
                unique_names = []
                for name in names_only:
                    if name.lower() not in seen:
                        seen.add(name.lower())
                        unique_names.append(name)
                if unique_names:
                    return ', '.join(unique_names)
        
        return answer

    def chat(self, question: str, candidate_id: Optional[int] = None, 
             job_id: Optional[int] = None, conversation_history: List[Dict] = None) -> Dict:
        """
        Chat with the HR chatbot about candidates.
        
        Args:
            question: HR's question
            candidate_id: Optional specific candidate ID to focus on
            job_id: Optional job posting ID for match analysis
            conversation_history: Previous conversation messages
        
        Returns:
            Dict with 'response', 'status', and 'error' (if any)
        """
        if not self.is_available():
            return {
                'response': 'Ollama server is not available. Please ensure Ollama is running on localhost:11434',
                'status': 'error',
                'error': 'Ollama not available'
            }
        
        # Build system prompt with context
        system_prompt = self._build_system_prompt(candidate_id, job_id)
        
        # Debug: Print context info (can be removed in production)
        context_info = self._get_candidate_context(candidate_id, job_id)
        if context_info:
            print(f"[Chatbot Debug] Context length: {len(context_info)} chars")
            print(f"[Chatbot Debug] Candidate ID: {candidate_id}, Job ID: {job_id}")
            print(f"[Chatbot Debug] Context preview: {context_info[:200]}...")
        
        # Prepare messages for chat API
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history[-5:]:  # Keep last 5 messages for context
                if msg.get('role') in ['user', 'assistant']:
                    messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })
        
        # Add current question with emphasis on simplicity
        user_question = question
        # Add instruction for simple questions
        if any(word in question.lower() for word in ['name', 'who', 'what is the name', 'list names']):
            user_question = f"{question}\n\nIMPORTANT: Answer ONLY what was asked. If asked for a name, provide ONLY the name(s), nothing else."
        
        messages.append({
            "role": "user",
            "content": user_question
        })
        
        # Call Ollama chat API
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1000
            }
        }
        
        try:
            response = requests.post(self.chat_url, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                assistant_message = result.get("message", {})
                answer = assistant_message.get("content", "").strip()
                
                # Post-process answer to ensure it's concise for simple questions
                answer = self._post_process_answer(answer, question)
                
                return {
                    'response': answer,
                    'status': 'success',
                    'model': self.model
                }
            elif response.status_code == 404:
                # Model not found - check available models
                available_models = self.get_available_models()
                if available_models:
                    error_msg = f"Model '{self.model}' not found. Available models: {', '.join(available_models[:5])}"
                    if len(available_models) > 5:
                        error_msg += f" (and {len(available_models) - 5} more)"
                else:
                    error_msg = f"Model '{self.model}' not found. Please install it using: ollama pull {self.model}"
                
                # Try fallback with better error message
                try:
                    return self._fallback_generate(question, system_prompt)
                except:
                    return {
                        'response': f"Error: {error_msg}. Please install the model using 'ollama pull {self.model}'",
                        'status': 'error',
                        'error': f'Model not found: {self.model}'
                    }
            else:
                # Other error - try fallback
                error_text = response.text[:200] if hasattr(response, 'text') else f"HTTP {response.status_code}"
                try:
                    return self._fallback_generate(question, system_prompt)
                except Exception as fallback_error:
                    return {
                        'response': f'Ollama API error ({response.status_code}): {error_text}. Please check if Ollama is running and the model "{self.model}" is installed.',
                        'status': 'error',
                        'error': f'API error {response.status_code}'
                    }
        except Exception as e:
            print(f"Chatbot error: {e}")
            # Try fallback
            try:
                return self._fallback_generate(question, system_prompt)
            except Exception as e2:
                return {
                    'response': f'Sorry, I encountered an error: {str(e2)}',
                    'status': 'error',
                    'error': str(e2)
                }

    def _fallback_generate(self, question: str, context: str) -> Dict:
        """Fallback to generate API if chat API is not available"""
        prompt = f"""{context}

HR Question: {question}

Answer:"""
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1000
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "").strip()
                return {
                    'response': answer,
                    'status': 'success',
                    'model': self.model
                }
            elif response.status_code == 404:
                # Model not found
                available_models = self.get_available_models()
                if available_models:
                    error_msg = f"Model '{self.model}' not found. Available models: {', '.join(available_models[:5])}"
                    if len(available_models) > 5:
                        error_msg += f" (and {len(available_models) - 5} more)"
                else:
                    error_msg = f"Model '{self.model}' not found. Please install it using: ollama pull {self.model}"
                
                return {
                    'response': f"Error: {error_msg}",
                    'status': 'error',
                    'error': f'Model not found: {self.model}'
                }
            else:
                error_text = response.text[:200] if hasattr(response, 'text') else f"HTTP {response.status_code}"
                return {
                    'response': f'Ollama API error ({response.status_code}): {error_text}. Please check if Ollama is running and the model "{self.model}" is installed.',
                    'status': 'error',
                    'error': f'API error {response.status_code}'
                }
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            return {
                'response': 'Cannot connect to Ollama server. Please ensure Ollama is running on localhost:11434. Start it with: ollama serve',
                'status': 'error',
                'error': 'Connection error'
            }
        except Exception as e:
            return {
                'response': f'Error connecting to Ollama: {str(e)}. Please ensure Ollama is running and the model "{self.model}" is installed.',
                'status': 'error',
                'error': str(e)
            }

    def get_suggested_questions(self, candidate_id: Optional[int] = None, 
                               job_id: Optional[int] = None) -> List[str]:
        """Get suggested questions based on context"""
        if candidate_id:
            return [
                "What are this candidate's key strengths?",
                "What skills does this candidate have?",
                "How well does this candidate match the job requirements?",
                "What are the candidate's weaknesses or gaps?",
                "Should I interview this candidate?",
                "What questions should I ask in the interview?",
                "Summarize this candidate's profile"
            ]
        elif job_id:
            return [
                "Who are the top candidates for this job?",
                "What skills are most common among candidates?",
                "Which candidates have the highest match scores?",
                "What are the main skill gaps among candidates?",
                "Compare the top 3 candidates"
            ]
        else:
            return [
                "How many candidates do we have?",
                "What are the most common skills?",
                "What degrees do candidates have?",
                "Show me candidates with Python skills",
                "Who are the best candidates overall?"
            ]


# Global instance
hr_chatbot = HRChatbot()

