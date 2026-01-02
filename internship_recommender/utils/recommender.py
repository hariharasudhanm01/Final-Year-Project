import json
import pandas as pd
from pathlib import Path
import difflib
from .skill_graph import rank_missing_skills

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SKILLS_FILE = DATA_DIR / "skills.json"
RESOURCES_FILE = DATA_DIR / "resources.json"
INTERNS_CSV = DATA_DIR / "internships.csv"

with open(SKILLS_FILE, "r", encoding="utf-8") as f:
    ROLE_SKILLS = json.load(f)

with open(RESOURCES_FILE, "r", encoding="utf-8") as f:
    RESOURCES = json.load(f)

intern_df = pd.read_csv(INTERNS_CSV)

# Cache for internship skill extraction to improve efficiency
skill_cache = {}

# Build a canonical map for skill casing so acronyms (e.g., SQL) are preserved
# and names match the skill graph nodes.
CANONICAL_SKILL_MAP = {}
try:
    # Known skills from roles
    for role_values in ROLE_SKILLS.values():
        for skill in role_values:
            CANONICAL_SKILL_MAP.setdefault(skill.lower(), skill)
    # Known skills from graph
    from .skill_graph import G as _SKILL_GRAPH
    for node in _SKILL_GRAPH.nodes:
        CANONICAL_SKILL_MAP.setdefault(str(node).lower(), str(node))
    # Known skills from base extractor list (optional)
    try:
        from .ner_extractor import BASE_SKILLS as _BASE_SKILLS
        for base_skill in _BASE_SKILLS:
            # Prefer already-known canonical casing if present
            CANONICAL_SKILL_MAP.setdefault(base_skill.lower(), base_skill.title() if base_skill.islower() else base_skill)
    except Exception:
        pass
except Exception:
    # In case of any import-time issues, keep map minimal
    CANONICAL_SKILL_MAP = CANONICAL_SKILL_MAP or {}


def _canonicalize_pretty(skill: str) -> str:
    """
    Return the preferred display casing for a skill using a canonical map.
    Falls back to title-case when unknown.
    """
    key = (skill or "").strip().lower()
    if not key:
        return ""
    return CANONICAL_SKILL_MAP.get(key, skill.title())

def find_best_role(role_input):
    """
    Return the best matching role key from ROLE_SKILLS (case-insensitive, substring, fuzzy).
    If nothing matches, return None.
    """
    if not role_input:
        return None
    role_input = role_input.strip().lower()

    # exact / case insensitive
    for k in ROLE_SKILLS.keys():
        if k.lower() == role_input:
            return k

    # substring (user typed 'python dev' or 'python developer' vs 'Python Developer')
    for k in ROLE_SKILLS.keys():
        if role_input in k.lower() or k.lower() in role_input:
            return k

    # alias-based mapping to improve matching across common variants
    # Check for exact alias matches first (most specific)
    ROLE_ALIASES = {
        "software developer": "Software Engineer",
        "software dev": "Software Engineer", 
        "software eng": "Software Engineer",
        "python dev": "Python Developer",
        "python developer": "Python Developer",
        "frontend dev": "Frontend Developer",
        "frontend developer": "Frontend Developer",
        "front end": "Frontend Developer",
        "backend dev": "Backend Developer",
        "backend developer": "Backend Developer",
        "full stack dev": "Full Stack Developer",
        "full stack developer": "Full Stack Developer",
        "mobile dev": "Mobile App Developer",
        "mobile developer": "Mobile App Developer",
        "game dev": "Game Developer",
        "game developer": "Game Developer",
        "blockchain dev": "Blockchain Developer",
        "blockchain developer": "Blockchain Developer",
        "ai dev": "AI Engineer",
        "ai developer": "AI Engineer",
        "ai engineer": "AI Engineer",
        "ml engineer": "Machine Learning Engineer",
        "ml dev": "Machine Learning Engineer",
        "machine learning engineer": "Machine Learning Engineer",
        "devops dev": "DevOps Engineer",
        "devops engineer": "DevOps Engineer",
        "cloud dev": "Cloud Engineer",
        "cloud engineer": "Cloud Engineer",
        "qa dev": "QA Engineer",
        "qa engineer": "QA Engineer",
        "test engineer": "QA Engineer",
        "ui designer": "UI Designer",
        "ux designer": "UX Designer",
        "ui ux": "UI UX Designer",
        "ui/ux": "UI UX Designer",
        "data science": "Data Scientist",
        "data scientist": "Data Scientist",
        "data analyst": "Data Analyst"
    }
    
    # Check for exact alias matches first
    for alias, target in ROLE_ALIASES.items():
        if role_input == alias:
            return target
    
    # Then check for partial alias matches
    for alias, target in ROLE_ALIASES.items():
        if alias in role_input:
            return target

    # fuzzy match using difflib
    candidates = [k.lower() for k in ROLE_SKILLS.keys()]
    matches = difflib.get_close_matches(role_input, candidates, n=1, cutoff=0.6)
    if matches:
        # find original key whose lower() == matches[0]
        lower_to_key = {k.lower(): k for k in ROLE_SKILLS.keys()}
        return lower_to_key[matches[0]]

    return None

def analyze_skill_gap(extracted_skills, role_input):
    """
    Returns: (have_pretty, missing_pretty, ranked_missing)
    - have_pretty: list of skills from role that user already has
    - missing_pretty: list of required skills user misses
    - ranked_missing: missing skills ordered by importance via skill graph
    """
    try:
        # Handle None or empty inputs
        if not extracted_skills or not role_input:
            return [], [], []

        # normalize extracted skills set
        extracted = {s.lower().strip() for s in extracted_skills if isinstance(s, str)}

        # find best role key
        role_key = find_best_role(role_input)
        if role_key is None:
            # Extract skills from role_input for unknown roles
            from .ner_extractor import extract_skills_and_summary
            req_skills, _ = extract_skills_and_summary(role_input)
            req_set = {r.lower().strip() for r in req_skills}
        else:
            req = ROLE_SKILLS.get(role_key, [])
            req_set = {r.lower().strip() for r in req}

        # compute intersection relative to required skills (so have = required ∩ extracted)
        have = [r for r in req_set if r in extracted]
        missing = [r for r in req_set if r not in extracted]

        have_pretty = [_canonicalize_pretty(h) for h in have]
        missing_pretty = [_canonicalize_pretty(m) for m in missing]

        # ranked missing uses title-cased have and missing as skill_graph expects readable names
        # Handle empty missing skills
        if not missing_pretty:
            ranked_missing = []
        else:
            ranked_missing = rank_missing_skills(have_pretty, missing_pretty)

        return have_pretty, missing_pretty, ranked_missing
    except Exception as e:
        # Log the error and return empty lists to prevent app crash
        print(f"Error in analyze_skill_gap: {e}")
        return [], [], []

def recommend_internships_from_profile(skills, role_input, location, top_k=5):
    """
    Fallback local CSV-based recommender.
    Score by: role match (strong), location match, plus skill matches.
    """
    df = intern_df.copy()
    df["title"] = df["title"].astype(str)
    df["description"] = df["description"].astype(str)
    df["location"] = df["location"].astype(str)

    # choose role keyword: use fuzzy-matched role if available
    best_role = find_best_role(role_input) or role_input
    role_kw = best_role.lower()
    loc_kw = (location or "").lower()

    skills_l = [s.lower() for s in skills if isinstance(s, str)]

    def row_score(r):
        score = 0
        title = r["title"].lower()
        desc = r["description"].lower()
        loc = r["location"].lower()
        # strong weight for explicit role keyword in title/description
        if role_kw and role_kw in title:
            score += 4
        if role_kw and role_kw in desc:
            score += 2
        # location match
        if loc_kw and loc_kw in loc:
            score += 3
        # skills matches (each skill match adds 1)
        for s in skills_l:
            if s and (s in title or s in desc):
                score += 1
        return score

    df["score"] = df.apply(row_score, axis=1)
    df = df[df["score"] > 0].sort_values("score", ascending=False)

    results = []
    for _, row in df.head(top_k).iterrows():
        results.append({
            "title": row["title"],
            "company": row.get("company", "") or "",
            "location": row.get("location", "") or "",
            "link": row.get("link", "") or "",
            "snippet": (row.get("description", "") or "")[:300]
        })

    return results

def analyze_internship_skill_gap(user_skills, internship_description):
    """
    Extract skills from internship description and compute gap with user skills.
    Returns: (have, missing, ranked_missing)
    """
    from .ner_extractor import extract_skills_and_summary
    try:
        # Extract skills from description with caching
        cache_key = internship_description.strip()
        if cache_key in skill_cache:
            desc_skills = skill_cache[cache_key]
        else:
            desc_skills, _ = extract_skills_and_summary(internship_description)
            skill_cache[cache_key] = desc_skills
        if not desc_skills:
            return [], [], []
        
        # Normalize
        user_skills_set = {s.lower().strip() for s in user_skills if isinstance(s, str)}
        desc_skills_set = {s.lower().strip() for s in desc_skills}
        
        # Compute gap
        have = [s for s in desc_skills_set if s in user_skills_set]
        missing = [s for s in desc_skills_set if s not in user_skills_set]
        
        have_pretty = [_canonicalize_pretty(h) for h in have]
        missing_pretty = [_canonicalize_pretty(m) for m in missing]
        
        # Rank missing
        if not missing_pretty:
            ranked_missing = []
        else:
            ranked_missing = rank_missing_skills(have_pretty, missing_pretty)
        
        return have_pretty, missing_pretty, ranked_missing
    except Exception as e:
        print(f"Error in analyze_internship_skill_gap: {e}")
        return [], [], []
