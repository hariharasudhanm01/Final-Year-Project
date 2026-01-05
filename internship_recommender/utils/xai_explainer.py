"""
Explainable AI (XAI) Module for Internship Recommender System
Provides explanations for ML predictions and recommendations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import joblib
from pathlib import Path

# For SHAP explanations (optional, install with: pip install shap)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Note: SHAP not installed. Install with 'pip install shap' for advanced explanations.")


class SalaryExplainer:
    """Explain salary predictions using feature importance and SHAP values."""
    
    def __init__(self, model_path: Path):
        """Initialize explainer with trained model."""
        self.model_path = model_path
        self.model_data = None
        self.model = None
        self.feature_columns = None
        self.label_encoder = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained salary prediction model."""
        if self.model_path.exists():
            self.model_data = joblib.load(self.model_path)
            self.model = self.model_data['model']
            self.feature_columns = self.model_data['feature_columns']
            self.label_encoder = self.model_data.get('label_encoder')
    
    def explain_prediction(self, skills: List[str], role: str, experience_years: int) -> Dict[str, Any]:
        """
        Generate explanation for salary prediction.
        
        Returns:
            Dictionary with prediction, feature contributions, and explanations
        """
        if not self.model:
            return {"error": "Model not loaded"}
        
        # Prepare features
        skills_count = len(skills) if skills else 0
        try:
            role_encoded = self.label_encoder.transform([role.lower()])[0]
        except (ValueError, AttributeError):
            role_encoded = 0
        
        features = np.array([[
            skills_count,
            experience_years,
            role_encoded,
            experience_years ** 2,
            skills_count * experience_years
        ]])
        
        # Get prediction
        pred = self.model.predict(features)[0]
        
        # Feature importance (global)
        feature_importance = self._get_feature_importance()
        
        # Feature contributions (local explanation)
        contributions = self._get_feature_contributions(
            features[0], pred, skills_count, experience_years, role
        )
        
        # Generate natural language explanation
        explanation = self._generate_explanation(
            pred, skills_count, experience_years, role, contributions, feature_importance
        )
        
        return {
            "predicted_salary": float(pred),
            "features": {
                "skills_count": int(skills_count),
                "experience_years": int(experience_years),
                "role": role,
                "experience_squared": int(experience_years ** 2),
                "skills_experience_interaction": int(skills_count * experience_years)
            },
            "feature_importance": feature_importance,
            "feature_contributions": contributions,
            "explanation": explanation,
            "confidence_factors": {
                "experience_impact": self._calculate_experience_impact(experience_years),
                "skills_impact": self._calculate_skills_impact(skills_count),
                "role_impact": self._calculate_role_impact(role, role_encoded)
            }
        }
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get global feature importance from the model."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return dict(zip(self.feature_columns, importances.tolist()))
        return {}
    
    def _get_feature_contributions(self, features: np.ndarray, prediction: float,
                                  skills_count: int, experience: int, role: str) -> Dict[str, Any]:
        """
        Calculate how each feature contributes to the prediction.
        Uses TreeSHAP if available, otherwise uses approximation.
        """
        contributions = {}
        
        if SHAP_AVAILABLE and hasattr(self.model, 'predict'):
            try:
                # Create SHAP explainer
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(features.reshape(1, -1))
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                
                for i, feature_name in enumerate(self.feature_columns):
                    contributions[feature_name] = {
                        "value": float(features[i]),
                        "shap_value": float(shap_values[0][i]),
                        "contribution_pct": float((shap_values[0][i] / prediction) * 100) if prediction != 0 else 0
                    }
            except Exception as e:
                print(f"SHAP explanation failed: {e}, using approximation")
                contributions = self._approximate_contributions(features, prediction, skills_count, experience, role)
        else:
            contributions = self._approximate_contributions(features, prediction, skills_count, experience, role)
        
        return contributions
    
    def _approximate_contributions(self, features: np.ndarray, prediction: float,
                                  skills_count: int, experience: int, role: str) -> Dict[str, Any]:
        """Approximate feature contributions using feature importance and values."""
        contributions = {}
        importance = self._get_feature_importance()
        
        # Base contribution = feature_value * importance
        base_avg = prediction / len(self.feature_columns) if len(self.feature_columns) > 0 else 0
        
        for i, feature_name in enumerate(self.feature_columns):
            feature_value = float(features[i])
            imp = importance.get(feature_name, 0.2)  # Default importance
            
            # Estimate contribution
            contribution = base_avg * imp * (feature_value / (np.mean(features) + 1e-6))
            
            contributions[feature_name] = {
                "value": feature_value,
                "estimated_contribution": float(contribution),
                "contribution_pct": float((contribution / prediction) * 100) if prediction != 0 else 0,
                "importance": float(imp)
            }
        
        return contributions
    
    def _calculate_experience_impact(self, experience: int) -> Dict[str, Any]:
        """Calculate how experience affects salary."""
        if experience == 0:
            return {
                "level": "Entry Level",
                "impact": "Starting salary range",
                "growth_potential": "High - expect 20-30% increase per year initially"
            }
        elif experience <= 2:
            return {
                "level": "Junior",
                "impact": "Moderate - 1.2x to 1.5x entry level",
                "growth_potential": "Good - expect 15-25% increase per year"
            }
        elif experience <= 5:
            return {
                "level": "Mid-Level",
                "impact": "Significant - 1.5x to 2.5x entry level",
                "growth_potential": "Moderate - expect 10-20% increase per year"
            }
        else:
            return {
                "level": "Senior",
                "impact": "High - 2.5x+ entry level",
                "growth_potential": "Stable - expect 5-15% increase per year"
            }
    
    def _calculate_skills_impact(self, skills_count: int) -> Dict[str, Any]:
        """Calculate how number of skills affects salary."""
        if skills_count <= 3:
            return {
                "assessment": "Basic skill set",
                "impact": "Lower salary range",
                "recommendation": "Consider learning 2-3 more relevant skills to increase market value"
            }
        elif skills_count <= 5:
            return {
                "assessment": "Good skill set",
                "impact": "Average salary range",
                "recommendation": "You have a solid foundation. Adding specialized skills can boost salary"
            }
        elif skills_count <= 7:
            return {
                "assessment": "Strong skill set",
                "impact": "Above average salary range",
                "recommendation": "Excellent skill diversity. Focus on depth in key areas"
            }
        else:
            return {
                "assessment": "Expert skill set",
                "impact": "Premium salary range",
                "recommendation": "You have extensive skills. Consider specializing for senior roles"
            }
    
    def _calculate_role_impact(self, role: str, role_encoded: int) -> Dict[str, Any]:
        """Calculate how role affects salary."""
        high_demand_roles = ["data scientist", "machine learning engineer", "devops engineer"]
        role_lower = role.lower()
        
        if any(hr in role_lower for hr in high_demand_roles):
            return {
                "demand": "High",
                "impact": "Premium salary range due to high market demand",
                "market_trend": "Growing field with strong compensation"
            }
        else:
            return {
                "demand": "Moderate to High",
                "impact": "Competitive salary range",
                "market_trend": "Stable field with good opportunities"
            }
    
    def _generate_explanation(self, prediction: float, skills_count: int, 
                             experience: int, role: str, contributions: Dict, 
                             importance: Dict) -> str:
        """Generate natural language explanation of the prediction."""
        pred_lpa = prediction / 100000
        
        # Get top contributing features
        sorted_contribs = sorted(
            contributions.items(),
            key=lambda x: abs(x[1].get('shap_value', x[1].get('estimated_contribution', 0))),
            reverse=True
        )
        
        top_feature = sorted_contribs[0][0] if sorted_contribs else "experience"
        
        explanation_parts = [
            f"Your predicted salary is ₹{pred_lpa:.1f} LPA based on:",
            f"• {experience} years of experience ({self._calculate_experience_impact(experience)['level']})",
            f"• {skills_count} relevant skills ({self._calculate_skills_impact(skills_count)['assessment']})",
            f"• Role: {role.title()}",
        ]
        
        if top_feature == "experience":
            explanation_parts.append(f"• Experience is the strongest factor, contributing significantly to your salary estimate")
        elif top_feature == "skills_count":
            explanation_parts.append(f"• Your skill diversity is a key factor in the salary prediction")
        elif "interaction" in top_feature:
            explanation_parts.append(f"• The combination of your experience and skills creates a strong profile")
        
        return "\n".join(explanation_parts)


class RecommendationExplainer:
    """Explain internship recommendations."""
    
    def explain_recommendation(self, internship: Dict, user_skills: List[str], 
                              role: str, location: str, score: int) -> Dict[str, Any]:
        """
        Explain why an internship was recommended.
        
        Args:
            internship: Internship details
            user_skills: User's skills
            role: Target role
            location: Preferred location
            score: Recommendation score
        
        Returns:
            Explanation dictionary
        """
        title = internship.get('title', '').lower()
        description = internship.get('snippet', internship.get('description', '')).lower()
        job_location = internship.get('location', '').lower()
        
        # Analyze matches
        role_match = role.lower() in title or role.lower() in description
        location_match = location.lower() in job_location if location else False
        
        # Find skill matches
        matched_skills = []
        for skill in user_skills:
            skill_lower = skill.lower()
            if skill_lower in title or skill_lower in description:
                matched_skills.append(skill)
        
        # Calculate score breakdown
        score_breakdown = {
            "role_match": 4 if role_match else 0,
            "location_match": 3 if location_match else 0,
            "skill_matches": len(matched_skills),
            "total_score": score
        }
        
        # Generate explanation
        explanation_parts = []
        
        if role_match:
            explanation_parts.append(f"✓ Strong match: This internship is for a {role} role")
        else:
            explanation_parts.append(f"⚠ Partial match: Role alignment could be better")
        
        if location_match:
            explanation_parts.append(f"✓ Location match: Position is in your preferred location")
        elif location:
            explanation_parts.append(f"⚠ Different location: May require relocation")
        
        if matched_skills:
            explanation_parts.append(f"✓ Skill match: You have {len(matched_skills)} relevant skills: {', '.join(matched_skills[:5])}")
        else:
            explanation_parts.append(f"⚠ Limited skill match: Consider learning more relevant skills")
        
        return {
            "internship_title": internship.get('title', ''),
            "score": score,
            "score_breakdown": score_breakdown,
            "matched_skills": matched_skills,
            "role_match": role_match,
            "location_match": location_match,
            "explanation": "\n".join(explanation_parts),
            "recommendation_strength": self._get_recommendation_strength(score)
        }
    
    def _get_recommendation_strength(self, score: int) -> str:
        """Determine recommendation strength based on score."""
        if score >= 8:
            return "Excellent Match"
        elif score >= 5:
            return "Good Match"
        elif score >= 3:
            return "Moderate Match"
        else:
            return "Weak Match"


class SkillGapExplainer:
    """Explain skill gap analysis and learning path recommendations."""
    
    def explain_skill_gap(self, have_skills: List[str], missing_skills: List[str],
                         ranked_missing: List[str], role: str) -> Dict[str, Any]:
        """
        Explain the skill gap analysis.
        
        Returns:
            Explanation dictionary
        """
        coverage = (len(have_skills) / (len(have_skills) + len(missing_skills)) * 100) if (have_skills or missing_skills) else 0
        
        # Analyze skill categories
        skill_categories = self._categorize_skills(have_skills, missing_skills)
        
        # Priority explanation
        priority_explanation = self._explain_priorities(ranked_missing[:5] if ranked_missing else [])
        
        # Build explanation text
        status_msg = '✓ Strong foundation: You have most required skills!' if coverage >= 70 else \
                      '⚠ Some gaps: Focus on learning missing skills to improve your profile' if coverage >= 40 else \
                      '⚠ Significant gaps: Consider a structured learning path'
        
        priorities_text = '\n'.join(f'  {i+1}. {skill}' for i, skill in enumerate(ranked_missing[:5])) if ranked_missing else '  All required skills covered!'
        
        explanation = f"""You have {len(have_skills)} out of {len(have_skills) + len(missing_skills)} required skills ({coverage:.1f}% coverage).

{status_msg}

Top priority skills to learn:
{priorities_text}"""
        
        return {
            "coverage_percentage": round(coverage, 1),
            "have_count": len(have_skills),
            "missing_count": len(missing_skills),
            "skill_categories": skill_categories,
            "top_priorities": ranked_missing[:5] if ranked_missing else [],
            "priority_explanation": priority_explanation,
            "explanation": explanation.strip(),
            "readiness_level": self._get_readiness_level(coverage)
        }
    
    def _categorize_skills(self, have: List[str], missing: List[str]) -> Dict[str, Any]:
        """Categorize skills into different types."""
        categories = {
            "programming": {"have": [], "missing": []},
            "frameworks": {"have": [], "missing": []},
            "tools": {"have": [], "missing": []},
            "concepts": {"have": [], "missing": []}
        }
        
        # Simple categorization (can be enhanced)
        for skill in have + missing:
            skill_lower = skill.lower()
            if any(term in skill_lower for term in ["python", "java", "javascript", "sql", "c++"]):
                cat = "programming"
            elif any(term in skill_lower for term in ["react", "django", "flask", "angular", "vue"]):
                cat = "frameworks"
            elif any(term in skill_lower for term in ["git", "docker", "aws", "kubernetes", "jenkins"]):
                cat = "tools"
            else:
                cat = "concepts"
            
            if skill in have:
                categories[cat]["have"].append(skill)
            else:
                categories[cat]["missing"].append(skill)
        
        return categories
    
    def _explain_priorities(self, priorities: List[str]) -> str:
        """Explain why certain skills are prioritized."""
        if not priorities:
            return "All required skills are covered!"
        
        explanations = []
        for i, skill in enumerate(priorities[:3], 1):
            if i == 1:
                explanations.append(f"{skill} is the highest priority - it's fundamental for this role")
            elif i == 2:
                explanations.append(f"{skill} is important - it complements your existing skills")
            else:
                explanations.append(f"{skill} will strengthen your profile")
        
        return "\n".join(explanations)
    
    def _get_readiness_level(self, coverage: float) -> str:
        """Determine readiness level based on coverage."""
        if coverage >= 80:
            return "Highly Ready"
        elif coverage >= 60:
            return "Ready"
        elif coverage >= 40:
            return "Partially Ready"
        else:
            return "Needs Development"


# Global explainer instances
_salary_explainer = None
_recommendation_explainer = None
_skill_gap_explainer = None

def get_salary_explainer(model_path: Path = None) -> SalaryExplainer:
    """Get or create salary explainer instance."""
    global _salary_explainer
    if _salary_explainer is None:
        if model_path is None:
            model_path = Path(__file__).resolve().parents[1] / "models" / "salary_model.pkl"
        _salary_explainer = SalaryExplainer(model_path)
    return _salary_explainer

def get_recommendation_explainer() -> RecommendationExplainer:
    """Get or create recommendation explainer instance."""
    global _recommendation_explainer
    if _recommendation_explainer is None:
        _recommendation_explainer = RecommendationExplainer()
    return _recommendation_explainer

def get_skill_gap_explainer() -> SkillGapExplainer:
    """Get or create skill gap explainer instance."""
    global _skill_gap_explainer
    if _skill_gap_explainer is None:
        _skill_gap_explainer = SkillGapExplainer()
    return _skill_gap_explainer

