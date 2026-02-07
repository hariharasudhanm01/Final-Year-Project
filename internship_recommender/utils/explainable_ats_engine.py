"""
Advanced Explainable AI-Powered ATS Engine
Provides transparent, interpretable resume-to-job matching with detailed explanations
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


@dataclass
class SkillMatch:
    """Represents a skill match with explanation"""
    skill: str
    match_type: str  # 'exact', 'semantic', 'related', 'missing', 'weak'
    confidence: float  # 0-1
    importance: float  # 0-1, how important for the role
    explanation: str
    expected_score_impact: float  # Points added/subtracted


@dataclass
class ATSExplanation:
    """Complete ATS match explanation"""
    overall_score: float
    score_breakdown: Dict[str, float]
    missing_skills: List[SkillMatch]
    weak_skills: List[SkillMatch]
    irrelevant_skills: List[SkillMatch]
    strong_matches: List[SkillMatch]
    feature_importance: Dict[str, float]
    rejection_reasons: List[str]
    acceptance_factors: List[str]
    improvement_suggestions: List[Dict[str, any]]


class ExplainableATSEngine:
    """
    Advanced ATS engine with full explainability
    Uses semantic matching, feature importance, and natural language explanations
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize the ATS engine"""
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self._load_embedding_model()
        
        # Enhanced Industry Standard Weights
        self.weights = {
            'skills': 0.35,        # 35% - Hard & Soft Skills (Core)
            'experience': 0.20,    # 20% - Work History & Years
            'education': 0.15,     # 15% - Degree & Field
            'formatting': 0.15,    # 15% - Structure, Sections, Readability
            'content': 0.15        # 15% - Impact, Metrics, Contact Info
        }
        
    def _load_embedding_model(self):
        """Load sentence transformer model for embeddings"""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                print(f"Loaded embedding model: {self.embedding_model_name}")
            except Exception as e:
                print(f"Warning: Could not load embedding model: {e}")
                self.embedding_model = None
        else:
            print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")
    
    def compute_skill_embeddings(self, skills: List[str]) -> np.ndarray:
        """Compute embeddings for a list of skills"""
        if not self.embedding_model or not skills:
            return np.array([])
        
        try:
            embeddings = self.embedding_model.encode(skills, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            print(f"Error computing embeddings: {e}")
            return np.array([])
    
    def semantic_skill_match(self, resume_skill: str, job_skill: str) -> float:
        """Compute semantic similarity between two skills"""
        if not self.embedding_model:
            # Fallback to simple string matching
            return 1.0 if resume_skill.lower() == job_skill.lower() else 0.0
        
        try:
            embeddings = self.embedding_model.encode(
                [resume_skill, job_skill],
                convert_to_numpy=True
            )
            # Cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            print(f"Error in semantic matching: {e}")
            return 0.0
    
    def analyze_resume_vs_jd(
        self,
        resume_skills: List[str],
        job_required_skills: List[str],
        job_preferred_skills: List[str] = None,
        resume_experience_years: float = 0.0,
        job_required_experience: float = 0.0,
        resume_education: str = None,
        job_required_education: str = None,
        resume_text: str = None,
        job_description: str = None
    ) -> ATSExplanation:
        """
        Comprehensive resume vs job description analysis with full explainability
        
        Returns:
            ATSExplanation with detailed breakdown and explanations
        """
        job_preferred_skills = job_preferred_skills or []
        
        # Normalize skills
        resume_skills_lower = [s.lower().strip() for s in resume_skills if s]
        job_required_lower = [s.lower().strip() for s in job_required_skills if s]
        job_preferred_lower = [s.lower().strip() for s in job_preferred_skills if s]
        
        # 1. Skill Analysis
        skill_matches = self._analyze_skill_matches(
            resume_skills_lower,
            job_required_lower,
            job_preferred_lower
        )
        skill_match_score = self._calculate_skill_score(skill_matches)
        
        # 2. Experience Analysis
        experience_match = self._analyze_experience_match(
            resume_experience_years,
            job_required_experience
        )
        
        # 3. Education Analysis
        education_match = self._analyze_education_match(
            resume_education,
            job_required_education
        )
        
        # 4. Formatting Analysis
        formatting_match = self._analyze_formatting(resume_text)
        
        # 5. Content Quality Analysis
        content_match = self._analyze_content_quality(resume_text, job_description)
        
        # Calculate overall weighted score
        overall_score = (
            skill_match_score * self.weights['skills'] +
            experience_match['score'] * self.weights['experience'] +
            education_match['score'] * self.weights['education'] +
            formatting_match['score'] * self.weights['formatting'] +
            content_match['score'] * self.weights['content']
        )
        
        # Compute feature importance
        feature_importance = self._compute_feature_importance(
            skill_matches,
            experience_match,
            education_match
        )
        
        # Generate explanations
        rejection_reasons = self._generate_rejection_reasons(
            skill_matches,
            experience_match,
            education_match
        )
        
        acceptance_factors = self._generate_acceptance_factors(
            skill_matches,
            experience_match,
            education_match
        )
        
        improvement_suggestions = self._generate_improvement_suggestions(
            skill_matches,
            experience_match,
            education_match,
            overall_score
        )
        
        # Append structure/content suggestions
        improvement_suggestions.extend(formatting_match.get('suggestions', []))
        improvement_suggestions.extend(content_match.get('suggestions', []))
        
        # Categorize skills
        missing_skills = [m for m in skill_matches if m.match_type == 'missing']
        weak_skills = [m for m in skill_matches if m.match_type == 'weak']
        irrelevant_skills = [m for m in skill_matches if m.match_type == 'irrelevant']
        strong_matches = [m for m in skill_matches if m.match_type in ['exact', 'semantic']]
        
        return ATSExplanation(
            overall_score=round(overall_score, 1),
            score_breakdown={
                'Skill Match': round(skill_match_score, 1),
                'Experience': round(experience_match['score'], 1),
                'Education': round(education_match['score'], 1),
                'Formatting': round(formatting_match['score'], 1),
                'Content Quality': round(content_match['score'], 1)
            },
            missing_skills=missing_skills,
            weak_skills=weak_skills,
            irrelevant_skills=irrelevant_skills,
            strong_matches=strong_matches,
            feature_importance=feature_importance,
            rejection_reasons=rejection_reasons,
            acceptance_factors=acceptance_factors,
            improvement_suggestions=improvement_suggestions
        )

    def _analyze_formatting(self, resume_text: str) -> Dict:
        """Analyze resume formatting and structure"""
        if not resume_text:
            return {'score': 50.0, 'suggestions': []}
            
        score = 100.0
        suggestions = []
        text_upper = resume_text.upper()
        
        # Check for standard sections
        required_sections = ['EXPERIENCE', 'EDUCATION', 'SKILLS']
        found_sections = [sec for sec in required_sections if sec in text_upper]
        
        missing_sections = set(required_sections) - set(found_sections)
        if missing_sections:
            score -= 20 * len(missing_sections)
            for sec in missing_sections:
                suggestions.append({
                    'type': 'formatting',
                    'priority': 'high',
                    'action': f"Add section: {sec.title()}",
                    'reason': "Standard section missing, confusing ATS parsers.",
                    'expected_score_increase': 5
                })
        
        # Check length (approx word count)
        word_count = len(resume_text.split())
        if word_count < 200:
            score -= 20
            suggestions.append({
                'type': 'formatting',
                'priority': 'high',
                'action': "Increase content length",
                'reason': "Resume is too short (< 200 words), lacking detail.",
                'expected_score_increase': 5
            })
        elif word_count > 2000:
             score -= 10
             suggestions.append({
                'type': 'formatting',
                'priority': 'medium',
                'action': "Condense content",
                'reason': "Resume is too long (> 2000 words), typically 1-2 pages is best.",
                'expected_score_increase': 3
            })
            
        return {'score': max(0, score), 'suggestions': suggestions}

    def _analyze_content_quality(self, resume_text: str, jd_text: str) -> Dict:
        """Analyze content impact, metrics, and contact info"""
        if not resume_text:
             return {'score': 50.0, 'suggestions': []}
             
        score = 100.0
        suggestions = []
        text_lower = resume_text.lower()
        
        # 1. Contact Info Check
        if '@' not in text_lower:
            score -= 30
            suggestions.append({
                'type': 'content',
                'priority': 'critical',
                'action': "Add Email Address",
                'reason': "No email found. Recruiters cannot contact you.",
                'expected_score_increase': 10
            })
            
        # 2. Measurable Results (simple digit check in context)
        # We look for digits followed by % or words like 'increased', 'reduced' near numbers
        import re
        metrics_count = len(re.findall(r'\d+%|\$\d+|\d+ [Ii]ncrease', resume_text))
        if metrics_count < 3:
            score -= 20
            suggestions.append({
                'type': 'content',
                'priority': 'medium',
                'action': "Add Quantifiable Metrics",
                'reason': "Resume lacks measurable achievements (e.g., 'Increased sales by 20%').",
                'expected_score_increase': 8
            })
            
        # 3. Action Verbs (simple list)
        action_verbs = ['led', 'managed', 'developed', 'created', 'designed', 'implemented', 'optimized', 'achieved']
        found_verbs = [v for v in action_verbs if v in text_lower]
        if len(found_verbs) < 3:
            score -= 10
            suggestions.append({
                 'type': 'content',
                 'priority': 'medium',
                 'action': "Use Strong Action Verbs",
                 'reason': "Use words like 'Led', 'Developed', 'Optimized' to start bullet points.",
                 'expected_score_increase': 5
            })
            
        return {'score': max(0, score), 'suggestions': suggestions}
    
    def _analyze_skill_matches(
        self,
        resume_skills: List[str],
        job_required: List[str],
        job_preferred: List[str]
    ) -> List[SkillMatch]:
        """Analyze all skill matches with detailed explanations"""
        matches = []
        matched_resume_skills = set()
        
        # Check required skills
        for job_skill in job_required:
            best_match = None
            best_score = 0.0
            match_type = 'missing'
            
            for resume_skill in resume_skills:
                if resume_skill in matched_resume_skills:
                    continue
                
                # Exact match
                if resume_skill == job_skill:
                    best_match = resume_skill
                    best_score = 1.0
                    match_type = 'exact'
                    break
                
                # Semantic match
                semantic_score = self.semantic_skill_match(resume_skill, job_skill)
                if semantic_score > best_score:
                    best_score = semantic_score
                    best_match = resume_skill
                    if semantic_score > 0.85: # Stricter threshold
                        match_type = 'semantic'
                    elif semantic_score > 0.65:
                        match_type = 'weak'
            
            if best_match:
                matched_resume_skills.add(best_match)
                importance = 1.0  # Required skills are always important
                explanation = self._generate_skill_explanation(
                    job_skill, best_match, match_type, best_score, True
                )
                expected_impact = best_score * 10  # Max 10 points per required skill
            else:
                importance = 1.0
                explanation = f"Missing required skill: {job_skill}. This skill is mandatory for the role."
                expected_impact = -15  # Penalty for missing required skill
            
            matches.append(SkillMatch(
                skill=job_skill,
                match_type=match_type,
                confidence=best_score,
                importance=importance,
                explanation=explanation,
                expected_score_impact=expected_impact
            ))
        
        # Check preferred skills
        for job_skill in job_preferred:
            if job_skill in [m.skill for m in matches]:
                continue  # Already checked as required
            
            best_match = None
            best_score = 0.0
            match_type = 'missing'
            
            for resume_skill in resume_skills:
                if resume_skill in matched_resume_skills:
                    continue
                
                semantic_score = self.semantic_skill_match(resume_skill, job_skill)
                if semantic_score > best_score:
                    best_score = semantic_score
                    best_match = resume_skill
                    if semantic_score > 0.85:
                        match_type = 'semantic'
                    elif semantic_score > 0.65:
                        match_type = 'weak'
            
            if best_match:
                matched_resume_skills.add(best_match)
                importance = 0.6  # Preferred skills are less important
                explanation = self._generate_skill_explanation(
                    job_skill, best_match, match_type, best_score, False
                )
                expected_impact = best_score * 5  # Max 5 points per preferred skill
            else:
                importance = 0.6
                explanation = f"Missing preferred skill: {job_skill}. This would strengthen your application."
                expected_impact = -5  # Small penalty for missing preferred skill
            
            matches.append(SkillMatch(
                skill=job_skill,
                match_type=match_type,
                confidence=best_score,
                importance=importance,
                explanation=explanation,
                expected_score_impact=expected_impact
            ))
        
        # Check for irrelevant skills (in resume but not in job)
        for resume_skill in resume_skills:
            if resume_skill not in matched_resume_skills:
                # Check if it's truly irrelevant or just not mentioned
                is_irrelevant = True
                for job_skill in job_required + job_preferred:
                    if self.semantic_skill_match(resume_skill, job_skill) > 0.5:
                        is_irrelevant = False
                        break
                
                if is_irrelevant:
                    matches.append(SkillMatch(
                        skill=resume_skill,
                        match_type='irrelevant',
                        confidence=0.0,
                        importance=0.0,
                        explanation=f"Skill '{resume_skill}' is not mentioned in the job description and may not be relevant for this role.",
                        expected_score_impact=0.0
                    ))
        
        return matches
    
    def _calculate_skill_score(self, skill_matches: List[SkillMatch]) -> float:
        """Calculate overall skill match score"""
        if not skill_matches:
            return 0.0
        
        total_points = 0.0
        max_points = 0.0
        
        for match in skill_matches:
            if match.match_type == 'missing':
                max_points += 15  # Penalty for missing
            elif match.match_type in ['exact', 'semantic']:
                max_points += 10 * match.importance
                total_points += 10 * match.importance * match.confidence # Weighted by confidence
            elif match.match_type == 'weak':
                max_points += 10 * match.importance
                total_points += 5 * match.importance # Half points for weak
        
        if max_points == 0:
            return 0.0
        
        # Normalize to 0-100
        score = (total_points / max_points) * 100
        return max(0, min(100, score))
    
    def _analyze_experience_match(
        self,
        resume_years: float,
        required_years: float
    ) -> Dict:
        """Analyze experience match"""
        if required_years == 0:
            return {
                'score': 100.0, 
                'match': True, 
                'gap': 0.0,
                'resume_years': resume_years,
                'required_years': required_years
            }
        
        gap = required_years - resume_years
        
        if gap <= 0:
            score = 100.0
            match = True
        elif gap <= 1:
            score = 80.0
            match = True
        elif gap <= 2:
            score = 60.0
            match = False
        else:
            score = 40.0
            match = False
        
        return {
            'score': score,
            'match': match,
            'gap': gap,
            'resume_years': resume_years,
            'required_years': required_years
        }
    
    def _analyze_education_match(
        self,
        resume_education: Optional[str],
        required_education: Optional[str]
    ) -> Dict:
        """Analyze education match"""
        if not required_education:
            return {'score': 100.0, 'match': True}
        
        if not resume_education:
            return {'score': 50.0, 'match': False}
        
        # Simple matching (can be enhanced)
        resume_lower = resume_education.lower()
        required_lower = required_education.lower()
        
        if required_lower in resume_lower or resume_lower in required_lower:
            return {'score': 100.0, 'match': True}
        
        # Check for degree level
        degree_levels = {
            'phd': 4,
            'master': 3,
            'bachelor': 2,
            'diploma': 1
        }
        
        resume_level = 0
        required_level = 0
        
        for level, value in degree_levels.items():
            if level in resume_lower:
                resume_level = value
            if level in required_lower:
                required_level = value
        
        if resume_level >= required_level:
            return {'score': 100.0, 'match': True}
        else:
            return {'score': 60.0, 'match': False}
    
    def _compute_feature_importance(
        self,
        skill_matches: List[SkillMatch],
        experience_match: Dict,
        education_match: Dict
    ) -> Dict[str, float]:
        """Compute feature importance scores"""
        importance = {}
        
        # Skill importance
        for match in skill_matches:
            if match.match_type != 'irrelevant':
                importance[match.skill] = match.importance * abs(match.expected_score_impact) / 15
        
        # Experience importance
        if experience_match['gap'] > 0:
            importance['experience_years'] = self.weights['experience']
        
        # Education importance
        if not education_match['match']:
            importance['education'] = self.weights['education']
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}
        
        return importance
    
    def _generate_skill_explanation(
        self,
        job_skill: str,
        resume_skill: str,
        match_type: str,
        confidence: float,
        is_required: bool
    ) -> str:
        """Generate natural language explanation for skill match"""
        if match_type == 'exact':
            return f"Perfect match: '{resume_skill}' exactly matches required skill '{job_skill}'."
        elif match_type == 'semantic':
            return f"Strong match: '{resume_skill}' is semantically similar to '{job_skill}' (confidence: {confidence:.2%})."
        elif match_type == 'weak':
            return f"Weak match: '{resume_skill}' is somewhat related to '{job_skill}' (confidence: {confidence:.2%}). Consider strengthening this skill."
        else:
            return f"Missing {'required' if is_required else 'preferred'} skill: '{job_skill}'."
    
    def _generate_rejection_reasons(
        self,
        skill_matches: List[SkillMatch],
        experience_match: Dict,
        education_match: Dict
    ) -> List[str]:
        """Generate reasons why resume might be rejected"""
        reasons = []
        
        # Missing required skills
        missing_required = [m for m in skill_matches if m.match_type == 'missing' and m.importance == 1.0]
        if missing_required:
            skills_list = ', '.join([m.skill for m in missing_required[:3]])
            reasons.append(f"Missing critical required skills: {skills_list}")
        
        # Experience gap
        if experience_match['gap'] > 2:
            reasons.append(f"Experience gap: Job requires {experience_match['required_years']} years, resume shows {experience_match['resume_years']} years")
        
        # Education mismatch
        if not education_match['match']:
            reasons.append("Education requirements not fully met")
        
        # Too many weak matches
        weak_count = len([m for m in skill_matches if m.match_type == 'weak'])
        if weak_count > len(skill_matches) * 0.5:
            reasons.append(f"Many skills show weak matches ({weak_count} skills), indicating skill gap")
        
        return reasons
    
    def _generate_acceptance_factors(
        self,
        skill_matches: List[SkillMatch],
        experience_match: Dict,
        education_match: Dict
    ) -> List[str]:
        """Generate factors that support acceptance"""
        factors = []
        
        # Strong skill matches
        strong_matches = [m for m in skill_matches if m.match_type in ['exact', 'semantic']]
        if strong_matches:
            skills_list = ', '.join([m.skill for m in strong_matches[:3]])
            factors.append(f"Strong matches in key skills: {skills_list}")
        
        # Experience match
        if experience_match['match']:
            factors.append(f"Experience level meets requirements ({experience_match['resume_years']} years)")
        
        # Education match
        if education_match['match']:
            factors.append("Education requirements met")
        
        return factors
    
    def _generate_improvement_suggestions(
        self,
        skill_matches: List[SkillMatch],
        experience_match: Dict,
        education_match: Dict,
        current_score: float
    ) -> List[Dict[str, any]]:
        """Generate actionable improvement suggestions"""
        suggestions = []
        
        # Skill improvements
        missing_required = [m for m in skill_matches if m.match_type == 'missing' and m.importance == 1.0]
        for match in missing_required[:5]:
            suggestions.append({
                'type': 'skill',
                'priority': 'high',
                'action': f"Learn/Add skill: {match.skill}",
                'reason': match.explanation,
                'expected_score_increase': abs(match.expected_score_impact),
                'expected_new_score': min(100, current_score + abs(match.expected_score_impact))
            })
        
        # Experience improvement
        if experience_match['gap'] > 0:
            suggestions.append({
                'type': 'experience',
                'priority': 'medium',
                'action': f"Gain {experience_match['gap']:.1f} more years of experience",
                'reason': f"Job requires {experience_match['required_years']} years, you have {experience_match['resume_years']} years",
                'expected_score_increase': 10,
                'expected_new_score': min(100, current_score + 10)
            })
        
        return suggestions
    
    def explain_score(self, explanation: ATSExplanation) -> Dict:
        """Convert explanation to JSON-serializable format"""
        return {
            'overall_score': explanation.overall_score,
            'score_breakdown': explanation.score_breakdown,
            'missing_skills': [
                {
                    'skill': m.skill,
                    'importance': m.importance,
                    'explanation': m.explanation,
                    'expected_impact': m.expected_score_impact
                }
                for m in explanation.missing_skills
            ],
            'weak_skills': [
                {
                    'skill': m.skill,
                    'confidence': m.confidence,
                    'explanation': m.explanation,
                    'expected_impact': m.expected_score_impact
                }
                for m in explanation.weak_skills
            ],
            'strong_matches': [
                {
                    'skill': m.skill,
                    'match_type': m.match_type,
                    'confidence': m.confidence,
                    'explanation': m.explanation
                }
                for m in explanation.strong_matches
            ],
            'feature_importance': explanation.feature_importance,
            'rejection_reasons': explanation.rejection_reasons,
            'acceptance_factors': explanation.acceptance_factors,
            'improvement_suggestions': explanation.improvement_suggestions
        }


# Global instance
ats_engine = ExplainableATSEngine()

