# Explainable AI (XAI) Integration Guide

## Overview

This project now includes **Explainable AI (XAI)** capabilities to provide transparent explanations for all AI/ML predictions and recommendations. Users can understand **why** the system makes certain predictions, not just **what** it predicts.

## XAI Components

### 1. **Salary Prediction Explanations** (`SalaryExplainer`)

Explains how salary predictions are calculated:

- **Feature Importance**: Shows which factors (experience, skills, role) matter most
- **Feature Contributions**: Quantifies how each feature affects the prediction
- **SHAP Values**: Uses TreeSHAP (if available) for precise local explanations
- **Natural Language Explanations**: Human-readable explanations

**Example Explanation:**
```
Your predicted salary is ₹4.5 LPA based on:
• 2 years of experience (Mid-Level)
• 5 relevant skills (Good skill set)
• Role: Python Developer
• Experience is the strongest factor, contributing significantly to your salary estimate
```

### 2. **Internship Recommendation Explanations** (`RecommendationExplainer`)

Explains why specific internships are recommended:

- **Score Breakdown**: Shows role match, location match, and skill matches
- **Matched Skills**: Lists which user skills align with the internship
- **Recommendation Strength**: Categorizes match quality (Excellent/Good/Moderate/Weak)

**Example Explanation:**
```
✓ Strong match: This internship is for a Python Developer role
✓ Location match: Position is in your preferred location
✓ Skill match: You have 4 relevant skills: Python, Django, SQL, Git
Recommendation Strength: Excellent Match
```

### 3. **Skill Gap Analysis Explanations** (`SkillGapExplainer`)

Explains skill gap analysis and learning priorities:

- **Coverage Percentage**: Shows how many required skills you have
- **Skill Categories**: Groups skills into programming, frameworks, tools, concepts
- **Priority Explanation**: Explains why certain skills are prioritized
- **Readiness Level**: Assesses overall readiness (Highly Ready/Ready/Partially Ready/Needs Development)

**Example Explanation:**
```
You have 6 out of 10 required skills (60.0% coverage).

⚠ Some gaps: Focus on learning missing skills to improve your profile

Top priority skills to learn:
  1. React - is the highest priority - it's fundamental for this role
  2. TypeScript - is important - it complements your existing skills
  3. Docker - will strengthen your profile
```

## API Endpoints

### 1. `/api/explain_salary` (POST)
Get detailed salary prediction explanation.

**Request:**
```json
{
  "skills": ["Python", "Django", "SQL"],
  "role": "python developer",
  "experience": 2
}
```

**Response:**
```json
{
  "predicted_salary": 450000,
  "features": {...},
  "feature_importance": {...},
  "feature_contributions": {...},
  "explanation": "Your predicted salary is ₹4.5 LPA...",
  "confidence_factors": {...}
}
```

### 2. `/api/explain_recommendation` (POST)
Explain why an internship was recommended.

**Request:**
```json
{
  "internship": {"title": "...", "snippet": "..."},
  "skills": ["Python", "Django"],
  "role": "python developer",
  "location": "Bangalore",
  "score": 8
}
```

### 3. `/api/explain_skill_gap` (POST)
Explain skill gap analysis.

**Request:**
```json
{
  "have_skills": ["Python", "SQL"],
  "missing_skills": ["React", "Docker"],
  "ranked_missing": ["React", "Docker"],
  "role": "full stack developer"
}
```

### 4. Enhanced `/api/predict_salary` (POST)
Now supports `explain` parameter to include explanations.

**Request:**
```json
{
  "skills": ["Python", "Django"],
  "role": "python developer",
  "experience": 2,
  "explain": true
}
```

## Integration Points

### In Templates

Explanations are automatically passed to `results.html`:

```python
salary_explanation = salary_explainer.explain_prediction(skills, role, 0)
skill_gap_explanation = skill_gap_explainer.explain_skill_gap(have, missing, ranked_missing, role)
```

### In Frontend

Use JavaScript to fetch and display explanations:

```javascript
// Get salary explanation
const response = await fetch('/api/explain_salary', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    skills: skills,
    role: role,
    experience: experience
  })
});
const explanation = await response.json();
```

## Technical Details

### SHAP Integration

The system uses **SHAP (SHapley Additive exPlanations)** for precise feature contribution analysis:

- **TreeSHAP**: Used for tree-based models (RandomForest, GradientBoosting)
- **Fallback**: If SHAP unavailable, uses feature importance approximation
- **Installation**: `pip install shap` (optional but recommended)

### Feature Importance

The explainer calculates:
- **Global Importance**: Which features matter most overall
- **Local Contributions**: How each feature affects this specific prediction
- **Interaction Effects**: How features combine (e.g., skills × experience)

## Usage Examples

### 1. Display Salary Explanation in UI

```html
{% if salary_explanation %}
<div class="explanation-card">
  <h4>Why this salary range?</h4>
  <p>{{ salary_explanation.explanation }}</p>
  
  <h5>Key Factors:</h5>
  <ul>
    <li>Experience Impact: {{ salary_explanation.confidence_factors.experience_impact.level }}</li>
    <li>Skills Impact: {{ salary_explanation.confidence_factors.skills_impact.assessment }}</li>
  </ul>
</div>
{% endif %}
```

### 2. Show Recommendation Reasons

```javascript
async function explainRecommendation(internship) {
  const response = await fetch('/api/explain_recommendation', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      internship: internship,
      skills: userSkills,
      role: role,
      location: location,
      score: internship.score
    })
  });
  return await response.json();
}
```

## Benefits of XAI

1. **Transparency**: Users understand how predictions are made
2. **Trust**: Builds confidence in AI recommendations
3. **Actionable Insights**: Users know what to improve
4. **Debugging**: Helps identify model issues
5. **Compliance**: Meets explainability requirements for AI systems

## Future Enhancements

- [ ] Visual explanations (feature importance charts)
- [ ] Counterfactual explanations ("What if you had 1 more year of experience?")
- [ ] Explanation quality metrics
- [ ] User feedback on explanation usefulness
- [ ] A/B testing different explanation formats

## Installation

Install optional SHAP library for advanced explanations:

```bash
pip install shap
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

## References

- [SHAP Documentation](https://shap.readthedocs.io/)
- [Explainable AI Principles](https://www.nist.gov/itl/ai-risk-management-framework)
- [XAI Research Papers](https://arxiv.org/search/?query=explainable+ai)

