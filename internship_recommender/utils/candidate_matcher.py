"""
AI-powered candidate matching system for HR
Uses skill analysis, semantic matching, and scoring algorithms
"""
from .recommender import analyze_skill_gap, find_best_role
from .ner_extractor import extract_skills_and_summary
from .salary_predictor import predict_salary
import json
from typing import List, Dict, Tuple
import difflib


try:
    from .skill_graph import rank_missing_skills, G as SKILL_GRAPH
    SKILL_GRAPH_AVAILABLE = True
except:
    SKILL_GRAPH = None
    SKILL_GRAPH_AVAILABLE = False

def _semantic_skill_match(candidate_skill: str, job_skill: str) -> float:
    """Calculate semantic similarity between two skills using fuzzy matching"""
    candidate_skill = candidate_skill.lower().strip()
    job_skill = job_skill.lower().strip()
    
    # Exact match
    if candidate_skill == job_skill:
        return 1.0
    
    # Substring match
    if candidate_skill in job_skill or job_skill in candidate_skill:
        return 0.8
    
    # Fuzzy string matching
    similarity = difflib.SequenceMatcher(None, candidate_skill, job_skill).ratio()
    if similarity > 0.7:
        return similarity
    
    # Check skill graph relationships
    if SKILL_GRAPH_AVAILABLE and SKILL_GRAPH:
        try:
            if SKILL_GRAPH.has_node(candidate_skill) and SKILL_GRAPH.has_node(job_skill):
                # Check if skills are connected in the graph
                if SKILL_GRAPH.has_edge(candidate_skill, job_skill) or SKILL_GRAPH.has_edge(job_skill, candidate_skill):
                    return 0.6
                # Check for common neighbors (related skills)
                candidate_neighbors = set(SKILL_GRAPH.neighbors(candidate_skill))
                job_neighbors = set(SKILL_GRAPH.neighbors(job_skill))
                if candidate_neighbors & job_neighbors:  # Intersection
                    return 0.5
        except:
            pass
    
    return 0.0

def calculate_match_score(candidate_skills: List[str], job_skills: List[str], 
                         job_role: str = None, candidate_profile: Dict = None) -> Dict:
    """
    Calculate comprehensive match score between candidate and job posting using advanced techniques.
    Returns: {
        'overall_score': float (0-100),
        'skill_match_score': float,
        'skill_coverage': float,
        'semantic_match_score': float,
        'missing_skills': List[str],
        'matched_skills': List[str],
        'recommendation': str,
        'analysis_details': Dict
    }
    """
    if not candidate_skills:
        candidate_skills = []
    if not job_skills:
        job_skills = []
    
    # Normalize skills to lowercase for comparison
    candidate_skills_lower = [s.lower().strip() for s in candidate_skills if s]
    job_skills_lower = [s.lower().strip() for s in job_skills if s]
    
    # Advanced matching: Use semantic similarity for better matching
    matched_skills = []
    missing_skills = []
    skill_match_details = {}
    
    for job_skill in job_skills_lower:
        best_match = None
        best_score = 0.0
        
        for candidate_skill in candidate_skills_lower:
            similarity = _semantic_skill_match(candidate_skill, job_skill)
            if similarity > best_score:
                best_score = similarity
                best_match = candidate_skill
        
        if best_score >= 0.6:  # Threshold for considering it a match
            matched_skills.append(job_skill)
            skill_match_details[job_skill] = {
                'matched_with': best_match,
                'similarity': best_score
            }
        else:
            missing_skills.append(job_skill)
    
    # Calculate semantic match score (weighted by similarity)
    semantic_match_score = 0.0
    if job_skills_lower:
        total_similarity = sum(skill_match_details.get(skill, {}).get('similarity', 0) for skill in matched_skills)
        semantic_match_score = (total_similarity / len(job_skills_lower)) * 100
    
    # Skill match score (percentage of required skills matched)
    skill_match_score = (len(matched_skills) / len(job_skills_lower) * 100) if job_skills_lower else 0
    
    # Skill coverage (how many of candidate's skills are relevant)
    skill_coverage = (len(matched_skills) / len(candidate_skills_lower) * 100) if candidate_skills_lower else 0
    
    # Overall score calculation with weighted factors
    # 50% semantic match, 30% skill match, 15% coverage, 5% bonus for role alignment
    overall_score = (semantic_match_score * 0.5) + (skill_match_score * 0.3) + (skill_coverage * 0.15)
    
    # Role alignment bonus using skill gap analysis
    role_bonus = 0
    if job_role and candidate_profile:
        role = find_best_role(job_role)
        if role:
            have, missing, ranked_missing = analyze_skill_gap(candidate_skills, role)
            if len(missing) == 0:  # Perfect match
                role_bonus = 10
            elif len(missing) <= 2:  # Good match
                role_bonus = 5
            elif len(missing) <= 4:  # Moderate match
                role_bonus = 2
    
    overall_score += role_bonus
    overall_score = min(100, overall_score)  # Cap at 100
    
    # Generate recommendation
    if overall_score >= 85:
        recommendation = "Excellent Match - Highly Recommended"
    elif overall_score >= 70:
        recommendation = "Good Match - Recommended"
    elif overall_score >= 50:
        recommendation = "Moderate Match - Consider with Training"
    else:
        recommendation = "Weak Match - Not Recommended"
    
    # Get canonical skill names for display
    from .recommender import CANONICAL_SKILL_MAP
    matched_pretty = [CANONICAL_SKILL_MAP.get(s, s.title()) for s in matched_skills]
    missing_pretty = [CANONICAL_SKILL_MAP.get(s, s.title()) for s in missing_skills]
    
    # Generate recommendation with detailed analysis
    if overall_score >= 85:
        recommendation = "Excellent Match - Highly Recommended"
        recommendation_reason = "Candidate has strong alignment with job requirements"
    elif overall_score >= 70:
        recommendation = "Good Match - Recommended"
        recommendation_reason = "Candidate meets most requirements, minor gaps can be addressed"
    elif overall_score >= 50:
        recommendation = "Moderate Match - Consider with Training"
        recommendation_reason = "Candidate has potential but requires skill development"
    else:
        recommendation = "Weak Match - Not Recommended"
        recommendation_reason = "Significant skill gaps, not suitable without extensive training"
    
    return {
        'overall_score': round(overall_score, 2),
        'skill_match_score': round(skill_match_score, 2),
        'skill_coverage': round(skill_coverage, 2),
        'semantic_match_score': round(semantic_match_score, 2),
        'missing_skills': missing_pretty,
        'matched_skills': matched_pretty,
        'recommendation': recommendation,
        'recommendation_reason': recommendation_reason,
        'matched_count': len(matched_skills),
        'missing_count': len(missing_skills),
        'total_required': len(job_skills_lower),
        'role_bonus': role_bonus,
        'analysis_details': {
            'skill_match_details': skill_match_details,
            'semantic_matching_used': True,
            'advanced_analysis': True
        }
    }

def match_candidates_to_job(candidates: List[Dict], job_posting: Dict) -> List[Dict]:
    """
    Match multiple candidates to a job posting and return sorted by match score.
    """
    job_skills_str = job_posting.get('required_skills', '')
    job_skills = [s.strip() for s in job_skills_str.split(',') if s.strip()] if job_skills_str else []
    
    # Extract skills from job description if needed
    if not job_skills and job_posting.get('description'):
        extracted_skills, _ = extract_skills_and_summary(job_posting['description'])
        job_skills = extracted_skills
    
    job_role = job_posting.get('title', '')
    
    matched_candidates = []
    
    for candidate in candidates:
        candidate_skills_str = candidate.get('skills', '')
        candidate_skills = [s.strip() for s in candidate_skills_str.split(',') if s.strip()] if candidate_skills_str else []
        
        match_result = calculate_match_score(
            candidate_skills, 
            job_skills, 
            job_role, 
            candidate
        )
        
        # Add candidate info to match result
        match_result['candidate'] = candidate
        match_result['candidate_id'] = candidate.get('id')
        match_result['candidate_name'] = candidate.get('full_name') or candidate.get('username', 'Unknown')
        match_result['candidate_email'] = candidate.get('email', '')
        
        matched_candidates.append(match_result)
    
    # Sort by overall score (descending)
    matched_candidates.sort(key=lambda x: x['overall_score'], reverse=True)
    
    return matched_candidates

def search_and_match_candidates(job_posting: Dict, filters: Dict = None) -> List[Dict]:
    """
    Search candidates based on filters and match them to job posting.
    This is a convenience function that combines search and matching.
    """
    from .database import db
    
    # Extract search filters
    skills_filter = filters.get('skills') if filters else None
    location_filter = filters.get('location') if filters else None
    degree_filter = filters.get('degree') if filters else None
    stream_filter = filters.get('stream') if filters else None
    
    # Search candidates
    candidates = db.search_candidates(
        skills=skills_filter,
        location=location_filter,
        degree=degree_filter,
        stream=stream_filter,
        limit=100
    )
    
    # Match candidates to job
    matched = match_candidates_to_job(candidates, job_posting)
    
    return matched

def get_candidate_insights(candidate: Dict, job_posting: Dict) -> Dict:
    """
    Generate detailed insights about a candidate for a specific job.
    """
    candidate_skills_str = candidate.get('skills', '')
    candidate_skills = [s.strip() for s in candidate_skills_str.split(',') if s.strip()] if candidate_skills_str else []
    
    job_skills_str = job_posting.get('required_skills', '')
    job_skills = [s.strip() for s in job_skills_str.split(',') if s.strip()] if job_skills_str else []
    
    if not job_skills and job_posting.get('description'):
        extracted_skills, _ = extract_skills_and_summary(job_posting['description'])
        job_skills = extracted_skills
    
    match_result = calculate_match_score(candidate_skills, job_skills, job_posting.get('title'), candidate)
    
    # Estimate salary fit
    estimated_salary = None
    if candidate_skills:
        try:
            sal_low, sal_high = predict_salary(candidate_skills, job_posting.get('title', ''), experience_years=0)
            estimated_salary = {'low': sal_low, 'high': sal_high}
        except:
            pass
    
    # Skill gap analysis
    have, missing, ranked_missing = analyze_skill_gap(candidate_skills, job_posting.get('title', ''))
    
    return {
        'match_score': match_result,
        'estimated_salary': estimated_salary,
        'skill_gap': {
            'have': have,
            'missing': missing,
            'ranked_missing': ranked_missing
        },
        'candidate_profile': candidate
    }

