"""
RAG-based ATS Education Chatbot
Provides context-aware explanations about ATS systems using Retrieval-Augmented Generation
"""
import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import requests

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class KnowledgeChunk:
    """Represents a chunk of knowledge for RAG"""
    id: str
    content: str
    content_type: str  # 'ats_guide', 'skill_info', 'best_practice', etc.
    metadata: Dict
    embedding: Optional[np.ndarray] = None


class RAGATSEducator:
    """
    RAG-based ATS education system
    Uses vector search to retrieve relevant knowledge and LLM to generate explanations
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", ollama_base_url: str = "http://localhost:11434"):
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.ollama_base_url = ollama_base_url
        self.ollama_model = "tinyllama"
        self.knowledge_base: List[KnowledgeChunk] = []
        
        self._load_embedding_model()
        self._load_knowledge_base()
    
    def _load_embedding_model(self):
        """Load embedding model"""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                print(f"Loaded RAG embedding model: {self.embedding_model_name}")
            except Exception as e:
                print(f"Warning: Could not load embedding model: {e}")
        else:
            print("Warning: sentence-transformers not available for RAG")
    
    def _load_knowledge_base(self):
        """Load ATS knowledge base"""
        knowledge_chunks = [
            KnowledgeChunk(
                id="ats_001",
                content_type="ats_guide",
                content="""
                How ATS (Applicant Tracking Systems) Work:
                
                1. Resume Parsing: ATS systems parse resumes to extract structured data including:
                   - Contact information
                   - Skills and technologies
                   - Work experience and years
                   - Education details
                   - Certifications
                
                2. Keyword Matching: ATS scans for specific keywords mentioned in the job description.
                   Missing keywords can cause automatic rejection.
                
                3. Scoring Algorithm: ATS assigns scores based on:
                   - Skill match percentage
                   - Experience level alignment
                   - Education requirements
                   - Keyword density
                
                4. Ranking: Resumes are ranked by score, and only top candidates proceed to human review.
                
                5. Filtering: ATS can automatically filter out resumes that don't meet minimum requirements.
                """,
                metadata={"topic": "ats_basics", "importance": "high"}
            ),
            KnowledgeChunk(
                id="ats_002",
                content_type="ats_guide",
                content="""
                Why Resumes Get Rejected by ATS:
                
                1. Missing Required Skills: If a job requires "Python" and your resume doesn't mention it, 
                   the ATS will likely reject your application.
                
                2. Experience Gap: If the job requires 3+ years of experience but your resume shows only 1 year,
                   the ATS may filter you out.
                
                3. Keyword Mismatch: Using "JavaScript" when the job description uses "JS" or "ECMAScript" 
                   can cause missed matches.
                
                4. Formatting Issues: Complex formatting, images, or tables can break ATS parsing.
                
                5. Low Match Score: Even if you have some skills, if the overall match score is below threshold,
                   the ATS will reject.
                """,
                metadata={"topic": "rejection_reasons", "importance": "high"}
            ),
            KnowledgeChunk(
                id="ats_003",
                content_type="best_practice",
                content="""
                How to Improve ATS Compatibility:
                
                1. Use Standard Section Headers: "Experience", "Education", "Skills" - avoid creative names.
                
                2. Include Keywords: Mirror the language used in the job description.
                
                3. Quantify Achievements: Use numbers and metrics (e.g., "Improved performance by 30%").
                
                4. Use Standard File Formats: PDF is preferred, but ensure it's text-searchable.
                
                5. Avoid Graphics and Images: ATS cannot read text in images.
                
                6. Include Skills Section: Have a dedicated skills section with relevant technologies.
                
                7. Match Job Description: Tailor your resume for each application by including relevant keywords.
                """,
                metadata={"topic": "optimization", "importance": "high"}
            ),
            KnowledgeChunk(
                id="ats_004",
                content_type="ats_guide",
                content="""
                ATS Scoring Explained:
                
                ATS systems typically use weighted scoring:
                - Skill Match: 40-60% of total score
                  * Required skills: Higher weight
                  * Preferred skills: Lower weight
                  * Exact matches score higher than semantic matches
                
                - Experience Match: 20-30% of total score
                  * Years of experience alignment
                  * Relevant industry experience
                
                - Education Match: 10-20% of total score
                  * Degree level match
                  * Field of study relevance
                
                - Additional Factors: 5-10%
                  * Certifications
                  * Projects
                  * Keywords density
                
                Scores are typically on a 0-100 scale, with 70+ considered good matches.
                """,
                metadata={"topic": "scoring", "importance": "high"}
            ),
            KnowledgeChunk(
                id="ats_005",
                content_type="skill_info",
                content="""
                Skill Matching in ATS:
                
                ATS systems use multiple methods to match skills:
                
                1. Exact Matching: "Python" matches "Python" exactly.
                
                2. Semantic Matching: "Python" may match "Python Programming" or related terms.
                
                3. Skill Graphs: Some ATS use knowledge graphs to understand skill relationships.
                   For example, knowing "TensorFlow" implies knowledge of "Machine Learning".
                
                4. Weighted Importance: Required skills have higher weight than preferred skills.
                
                5. Context Awareness: Skills mentioned in experience section may be weighted differently
                   than skills in a dedicated skills section.
                """,
                metadata={"topic": "skill_matching", "importance": "medium"}
            ),
            KnowledgeChunk(
                id="ats_006",
                content_type="best_practice",
                content="""
                Resume Optimization Tips:
                
                1. Skills Section: List 10-15 relevant skills, prioritizing those in the job description.
                
                2. Experience Descriptions: Use action verbs and include technologies used.
                   Example: "Developed REST APIs using Python and Flask"
                
                3. Keyword Optimization: Include variations of keywords (e.g., "ML", "Machine Learning", "ML/AI").
                
                4. Quantify Results: "Increased user engagement by 25%" is better than "Improved user engagement".
                
                5. Tailor for Each Job: Customize your resume for each application to maximize keyword matches.
                
                6. Use Standard Fonts: Arial, Calibri, Times New Roman are ATS-friendly.
                
                7. Save as PDF: Ensure PDF is text-searchable, not just an image.
                """,
                metadata={"topic": "optimization", "importance": "high"}
            ),
            KnowledgeChunk(
                id="ats_007",
                content_type="ats_guide",
                content="""
                Understanding ATS Rejection Reasons:
                
                Common rejection reasons include:
                
                1. "Missing Critical Skill": A required skill is not found in your resume.
                   Solution: Add the skill if you have it, or learn it if you don't.
                
                2. "Experience Gap": Your years of experience don't meet the requirement.
                   Solution: Highlight relevant projects or internships that demonstrate experience.
                
                3. "Low Match Score": Overall compatibility is below the threshold.
                   Solution: Improve skill matches, add relevant keywords, tailor your resume.
                
                4. "Education Mismatch": Your degree doesn't match requirements.
                   Solution: Emphasize relevant coursework or certifications.
                
                5. "Keyword Density": Not enough relevant keywords in your resume.
                   Solution: Naturally incorporate job description keywords throughout your resume.
                """,
                metadata={"topic": "rejection_reasons", "importance": "high"}
            ),
            KnowledgeChunk(
                id="ats_008",
                content_type="skill_info",
                content="""
                Skill Importance and Weighting:
                
                In ATS systems, skills are weighted based on:
                
                1. Required vs Preferred: Required skills have 2-3x higher weight.
                
                2. Frequency in Job Description: Skills mentioned multiple times are more important.
                
                3. Context: Skills in "Requirements" section are more important than in "Nice to Have".
                
                4. Industry Standards: Some skills are universally important for certain roles
                   (e.g., Git for developers, SQL for data roles).
                
                5. Skill Relationships: Related skills can boost each other's importance.
                   For example, "React" and "JavaScript" together are stronger than either alone.
                """,
                metadata={"topic": "skill_importance", "importance": "medium"}
            ),
            # --- Application Specific Knowledge ---
            KnowledgeChunk(
                id="app_001",
                content_type="app_guide",
                content="""
                How the TalentMatch Candidate Portal Works:
                
                The Candidate Portal is your central hub for career advancement. Here you can:
                1. Dashboard: See a summary of your profile, recent activities, and quick actions.
                2. Profile: Manage your personal details, education, and skills.
                3. Upload Resume: The starting point for getting recommendations.
                4. AI Enhancer: Optimize your resume using our LLM-powered tool.
                5. Jobs/Internships: View recommendation listings matched to your profile.
                6. Learning Path: See course recommendations to close your skill gaps.
                """,
                metadata={"topic": "app_overview", "importance": "high"}
            ),
            KnowledgeChunk(
                id="app_002",
                content_type="app_guide",
                content="""
                How Resume Uploading Works in TalentMatch:
                
                1. Navigate to 'Upload Resume' from the sidebar or dashboard.
                2. Select your resume file (PDF or DOCX format recommended).
                3. Specify your target Role (e.g., "Data Scientist") and preferred Location.
                4. The system automatically extracts:
                   - Your contact info
                   - Skills & Technologies
                   - Education details
                5. This data is used to populate your profile and generate immediate recommendations.
                """,
                metadata={"topic": "app_upload", "importance": "high"}
            ),
            KnowledgeChunk(
                id="app_003",
                content_type="app_guide",
                content="""
                How the Analysis Engine Works:
                
                After you upload a resume or update your profile, our system performs a 3-step analysis:
                
                1. Skill Gap Analysis: We compare your skills against the requirements for your target role.
                   - "Have": Skills you possess that are relevant.
                   - "Missing": Critical skills you need to learn.
                
                2. Salary Prediction: We estimate your potential salary range based on your skills, role, and market trends.
                   - We use an Explainable AI (XAI) model to show exactly which skills contribute to your salary.
                
                3. Market Match: We check how well you fit current job market demands.
                """,
                metadata={"topic": "app_analysis", "importance": "high"}
            ),
            KnowledgeChunk(
                id="app_004",
                content_type="app_guide",
                content="""
                How the AI Resume Enhancer Works:
                
                The Enhancer uses advanced AI (Ollama LLM) to rewrite your resume for better ATS performance:
                
                1. Analysis: It identifies weak bullet points and missing keywords.
                2. Optimization: It rewrites your experience descriptions to:
                   - Use strong action verbs
                   - Quantify achievements (add numbers/metrics)
                   - Incorporate relevant keywords naturally
                3. Formatting: It attempts to preserve your original layout while improving the content.
                4. Download: You get a downloadable version of your optimized resume.
                
                Note: This helps increase your "ATS Score" and chances of passing automated filters.
                """,
                metadata={"topic": "app_enhancer", "importance": "high"}
            ),
            KnowledgeChunk(
                id="app_005",
                content_type="app_guide",
                content="""
                How Recommendations Work:
                
                TalentMatch provides two types of recommendations:
                
                1. Internship/Job Recommendations:
                   - We Scrape real-time listings from the web (via DuckDuckGo) matching your role/location.
                   - We also match you against our internal database of HR postings.
                   - Matches are ranked by "Match Score" based on skill overlap.
                
                2. Course Recommendations (Learning Path):
                   - Based on your "Missing Skills", we suggest specific courses (Coursera, Udemy, etc.).
                   - Following this path helps you bridge the gap to your target role.
                """,
                metadata={"topic": "app_recommendations", "importance": "high"}
            )
        ]
        
        # Compute embeddings if model is available
        if self.embedding_model:
            for chunk in knowledge_chunks:
                try:
                    chunk.embedding = self.embedding_model.encode(
                        chunk.content,
                        convert_to_numpy=True
                    )
                except Exception as e:
                    print(f"Error encoding chunk {chunk.id}: {e}")
        
        self.knowledge_base = knowledge_chunks
        print(f"Loaded {len(self.knowledge_base)} knowledge chunks")
    
    def retrieve_relevant_knowledge(
        self,
        query: str,
        top_k: int = 3,
        content_types: Optional[List[str]] = None
    ) -> List[Tuple[KnowledgeChunk, float]]:
        """
        Retrieve relevant knowledge chunks using semantic search
        
        Returns:
            List of (chunk, similarity_score) tuples
        """
        if not self.embedding_model:
            # Fallback: simple keyword matching
            results = []
            query_lower = query.lower()
            for chunk in self.knowledge_base:
                if content_types and chunk.content_type not in content_types:
                    continue
                score = sum(1 for word in query_lower.split() if word in chunk.content.lower())
                if score > 0:
                    results.append((chunk, score))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
        
        try:
            # Compute query embedding
            query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            
            # Compute similarities
            results = []
            for chunk in self.knowledge_base:
                if content_types and chunk.content_type not in content_types:
                    continue
                
                if chunk.embedding is not None:
                    # Cosine similarity
                    similarity = np.dot(query_embedding, chunk.embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(chunk.embedding)
                    )
                    results.append((chunk, float(similarity)))
            
            # Sort by similarity
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
        
        except Exception as e:
            print(f"Error in retrieval: {e}")
            return []
    
    def generate_explanation(
        self,
        question: str,
        context: Optional[Dict] = None,
        use_llm: bool = True
    ) -> Dict:
        """
        Generate explanation using RAG
        
        Args:
            question: User's question
            context: Optional context (e.g., resume_id, job_id, match_results)
            use_llm: Whether to use LLM for generation
        
        Returns:
            Dict with answer, sources, and confidence
        """
        # Retrieve relevant knowledge
        relevant_chunks = self.retrieve_relevant_knowledge(question, top_k=3)
        
        if not relevant_chunks:
            return {
                'answer': "I don't have specific information about that topic. Please try rephrasing your question.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Build context from retrieved chunks
        retrieved_context = "\n\n".join([
            f"[Source {i+1} - {chunk.content_type}]:\n{chunk.content}"
            for i, (chunk, score) in enumerate(relevant_chunks)
        ])
        
        # Add user-specific context if available
        user_context = ""
        if context:
            if context.get('resume_id'):
                user_context += f"\nUser's Resume ID: {context['resume_id']}\n"
            if context.get('job_id'):
                user_context += f"Job ID: {context['job_id']}\n"
            if context.get('ats_score'):
                user_context += f"Current ATS Score: {context['ats_score']}\n"
            if context.get('missing_skills'):
                user_context += f"Missing Skills: {', '.join(context['missing_skills'][:5])}\n"
        
        # Generate answer
        if use_llm:
            answer = self._generate_with_llm(question, retrieved_context, user_context)
        else:
            # Simple template-based answer
            answer = self._generate_template_answer(question, relevant_chunks[0][0])
        
        return {
            'answer': answer,
            'sources': [
                {
                    'id': chunk.id,
                    'type': chunk.content_type,
                    'content': chunk.content[:200] + "...",
                    'relevance_score': float(score)
                }
                for chunk, score in relevant_chunks
            ],
            'confidence': float(relevant_chunks[0][1]) if relevant_chunks else 0.0
        }
    
    def _generate_with_llm(
        self,
        question: str,
        retrieved_context: str,
        user_context: str
    ) -> str:
        """Generate answer using LLM (Ollama)"""
        prompt = f"""### System:
You are an expert ATS (Applicant Tracking System) consultant and TalentMatch Platform Expert.
Answer the user's question clearly and concisely based on the provided Context.
- Do NOT start with phrases like "To answer this question", "Based on the context", "Here is", or "Certainly".
- Do NOT announce what you are going to do.
- Start your answer IMMEDIATELY.
- Use bullet points for readability.
- Speak directly to the user as "you".

### Context:
{retrieved_context}

{user_context if user_context else ""}

### User Question:
{question}

### Answer:"""
        
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 500
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                return self._generate_template_answer(question, None)
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return self._generate_template_answer(question, None)
    
    def _generate_template_answer(self, question: str, chunk: Optional[KnowledgeChunk]) -> str:
        """Generate template-based answer as fallback"""
        if not chunk:
            return "I'm sorry, I don't have enough information to answer that question. Please try rephrasing."
        
        question_lower = question.lower()
        
        if "why" in question_lower and "reject" in question_lower:
            return f"Based on ATS analysis, resumes are typically rejected for: missing required skills, experience gaps, low match scores, or formatting issues. {chunk.content[:300]}"
        elif "how" in question_lower and "improve" in question_lower:
            return f"To improve your ATS compatibility: {chunk.content[:300]}"
        elif "score" in question_lower:
            return f"ATS scoring works by: {chunk.content[:300]}"
        else:
            return chunk.content[:500]
    
    def explain_ats_concept(self, concept: str) -> Dict:
        """Explain a specific ATS concept"""
        return self.generate_explanation(f"Explain {concept} in ATS systems")
    
    def explain_rejection(self, context: Dict) -> Dict:
        """Explain why a resume was rejected"""
        question = "Why was my resume rejected by the ATS?"
        return self.generate_explanation(question, context=context)
    
    def suggest_improvements(self, context: Dict) -> Dict:
        """Suggest improvements based on ATS analysis"""
        question = "How can I improve my ATS score and resume compatibility?"
        return self.generate_explanation(question, context=context)


# Global instance
rag_educator = RAGATSEducator()

