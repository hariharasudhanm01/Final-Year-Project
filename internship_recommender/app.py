import os
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, session, send_from_directory
from datetime import datetime
from utils.resume_parser import extract_text_from_file
from utils.ner_extractor import extract_skills_and_summary
from utils.ollama_summarizer import summarizer
from utils.recommender import analyze_skill_gap, recommend_internships_from_profile, find_best_role
from utils.salary_predictor import predict_salary, ensure_trained_model
from utils.scraper import ddg_search_internships
from utils.course_recommender import create_learning_path, get_enhanced_course_recommendations
from utils.database import db
from utils.xai_explainer import (
    get_salary_explainer, get_recommendation_explainer, get_skill_gap_explainer
)
from utils.candidate_matcher import (
    match_candidates_to_job, search_and_match_candidates, get_candidate_insights
)
from utils.hr_chatbot import hr_chatbot
from utils.explainable_ats_engine import ats_engine, ExplainableATSEngine
from utils.rag_ats_educator import rag_educator
from utils.rag_ats_educator import rag_educator
from utils.resume_editor import resume_editor
from utils.gemini_service import gemini_service
import json

def format_salary_lpa(rupees):
    """Convert rupees to LPA (Lakhs Per Annum) format with 1 decimal place."""
    if rupees is None:
        return "0.0"
    lakhs = rupees / 100000
    return f"{lakhs:.1f}"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("models", exist_ok=True)

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'replace-with-secure-key-in-production')
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Register template filter for LPA formatting
@app.template_filter('lpa')
def lpa_filter(rupees):
    """Jinja2 filter to format salary in LPA."""
    return format_salary_lpa(rupees)

@app.template_filter('basename')
def basename_filter(path):
    """Jinja2 filter to get filename from path."""
    return os.path.basename(path) if path else ""

@app.template_filter('format_date')
def format_date_filter(value, format='%Y-%m-%d'):
    """Format a date string or datetime object."""
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            # Try parsing typical SQL timestamp formats
            if "T" in value:
                dt = datetime.fromisoformat(value)
            else:
                dt = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
            return dt.strftime(format)
        except (ValueError, TypeError):
            # If parsing fails, return original string or a fallback substring
            return value.split(' ')[0] if ' ' in value else value
    return value.strftime(format)

# Ensure salary model exists / train lightweight sample
ensure_trained_model()

def login_required(f):
    """Decorator to require login for protected routes"""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def get_current_user():
    """Get current user from session"""
    if 'user_id' in session:
        return db.get_user_by_session(session.get('session_token'))
    return None

def hr_required(f):
    """Decorator to require HR role for protected routes"""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if not user:
            flash('Please login to access this page')
            return redirect(url_for('login'))
        if user.get('user_role') != 'hr':
            flash('Access denied. HR role required.')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/", methods=["GET"])
def index():
    user = get_current_user()
    if user:
        return redirect(url_for('dashboard'))
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")
    
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    
    if not username or not password:
        flash("Please fill in all fields")
        return redirect(url_for('login'))
    
    user = db.authenticate_user(username, password)
    if user:
        session_token = db.create_session(user['id'])
        if session_token:
            session['user_id'] = user['id']
            session['session_token'] = session_token
            flash("Login successful!")
            return redirect(url_for('dashboard'))
    
    flash("Invalid username or password")
    return redirect(url_for('login'))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")
    
    username = request.form.get("username", "").strip()
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "").strip()
    confirm_password = request.form.get("confirm_password", "").strip()
    user_role = request.form.get("user_role", "student").strip()
    
    if not all([username, email, password, confirm_password]):
        flash("Please fill in all fields")
        return redirect(url_for('register'))
    
    if password != confirm_password:
        flash("Passwords do not match")
        return redirect(url_for('register'))
    
    if len(password) < 6:
        flash("Password must be at least 6 characters long")
        return redirect(url_for('register'))
    
    if user_role not in ['student', 'hr']:
        user_role = 'student'
    
    success = db.create_user(username, email, password, full_name=None, user_role=user_role)
    if success:
        flash("Account created successfully! Please login.")
        return redirect(url_for('login'))
    else:
        flash("Username or email already exists")
        return redirect(url_for('register'))

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully")
    return redirect(url_for('index'))

@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user = get_current_user()
    if request.method == "GET":
        profile = db.get_user_profile(user['id'])
        return render_template("profile.html", user=user, profile=profile)
    
    # POST: handle form submission
    degree = request.form.get("degree", "").strip()
    study_year = request.form.get("study_year", "").strip()
    sector = request.form.get("sector", "").strip()
    stream = request.form.get("stream", "").strip()
    skills = request.form.get("skills", "").strip()
    
    profile_data = {
        'degree': degree if degree else None,
        'study_year': int(study_year) if study_year else None,
        'sector': sector if sector else None,
        'stream': stream if stream else None,
        'skills': skills if skills else None
    }
    
    success = db.update_user_profile(user['id'], profile_data)
    if success:
        flash("Profile updated successfully!")
    else:
        flash("Error updating profile")
    
    return redirect(url_for('dashboard'))

@app.route("/dashboard")
@login_required
def dashboard():
    user = get_current_user()
    
    # Redirect HR users to HR dashboard
    if user.get('user_role') == 'hr':
        return redirect(url_for('hr_dashboard'))
    
    profile = db.get_user_profile(user['id'])
    history = db.get_recommendation_history(user['id'], limit=5)
    enhancement_history = db.get_modification_history(user['id'], limit=5)
    
    return render_template("dashboard.html", user=user, profile=profile, history=history, enhancement_history=enhancement_history)

@app.route("/apply", methods=["GET", "POST"])
@login_required
def apply():
    if request.method == "GET":
        return render_template("apply.html")
    
    user = get_current_user()
    # POST: handle form submission
    full_name = request.form.get("full_name", "").strip()
    email = request.form.get("email", "").strip()
    role = request.form.get("role", "").strip()
    location = request.form.get("location", "").strip()
    degree = request.form.get("degree", "")
    year = request.form.get("year", "")
    sector = request.form.get("sector", "")
    stream = request.form.get("stream", "")
    # Multiple checkboxes with same name "skills"
    form_skills = request.form.getlist("skills")

    # Update user profile in database
    profile_data = {
        'degree': degree,
        'study_year': year if year else None,
        'sector': sector,
        'stream': stream,
        'skills': ','.join(form_skills)
    }
    db.update_user_profile(user['id'], profile_data)

    # Treat form skills as extracted skills and generate a short summary
    skills = form_skills
    summary = f"{full_name} ({email}) — {degree}, Year: {year}, Sector: {sector}, Stream: {stream}."

    # Skill gap
    have, missing, ranked_missing = analyze_skill_gap(skills, role)
    learning_path = create_learning_path(ranked_missing, have)

    # Internships (scrape → fallback)
    internships = ddg_search_internships(role, location, top_k=5)
    if not internships:
        internships = recommend_internships_from_profile(skills, role, location, top_k=5)

    # Fallback location for scraped internships if location is blank
    for job in internships:
        if not job.get("location"):
            job["location"] = location

    # Overall salary
    sal_low, sal_high = predict_salary(skills, role, experience_years=0)
    
    # ✅ Generate XAI explanations
    salary_explanation = None
    skill_gap_explanation = None
    try:
        salary_explainer = get_salary_explainer()
        salary_explanation = salary_explainer.explain_prediction(skills, role, 0)
        
        skill_gap_explainer = get_skill_gap_explainer()
        skill_gap_explanation = skill_gap_explainer.explain_skill_gap(have, missing, ranked_missing, role)
    except Exception as e:
        print(f"Error generating explanations: {e}")

    # Save recommendation history
    db.save_recommendation_history(user['id'], role, location, skills, missing, (sal_low, sal_high))

    # Initial per-job salary estimates (hidden in UI but kept for future use)
    for job in internships:
        job_role = find_best_role(job.get("title", "")) or role
        jlow, jhigh = predict_salary(skills, job_role, experience_years=0)
        job["salary_low"] = jlow
        job["salary_high"] = jhigh

    return render_template(
        "results.html",
        role=role,
        location=location,
        skills=skills,
        summary=summary,
        have=have,
        missing=missing,
        ranked_missing=ranked_missing,
        learning_path=learning_path,
        internships=internships,
        salary_range=(sal_low, sal_high),
        salary_explanation=salary_explanation,
        skill_gap_explanation=skill_gap_explanation,
    )

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "GET":
        return render_template("upload.html")

    if "resume" not in request.files:
        flash("No resume file provided")
        return redirect(url_for("upload"))

    user = get_current_user()
    file = request.files["resume"]
    role = request.form.get("role", "").strip()
    location = request.form.get("location", "").strip()

    if file.filename == "":
        flash("No selected file")
        return redirect(url_for("upload"))

    # Save file with user-specific naming
    filename = f"{user['id']}_{file.filename}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    # ✅ Use Gemini for parsing
    print(f"Parsing resume (upload) with Gemini: {save_path}")
    gemini_result = gemini_service.parse_resume(save_path)
    
    skills = []
    summary = ""

    if gemini_result and not gemini_result.get("error"):
        # Extract data from Gemini result
        personal = gemini_result.get("personal_info", {})
        edu = gemini_result.get("education", {})
        prof = gemini_result.get("professional_info", {})
        
        skills = prof.get("skills", [])
        if isinstance(skills, str):
             skills = [s.strip() for s in skills.split(',')]
        
        summary = f"{personal.get('name')} | {edu.get('degree')} - {edu.get('stream')} | {prof.get('sector')}"
        
        # Update profile with Gemini data
        profile_data = {
            'resume_path': filename,
            'degree': edu.get('degree'),
            'study_year': edu.get('study_year'),
            'sector': prof.get('sector'),
            'stream': edu.get('stream'),
            'skills': ','.join(skills) if skills else None
        }
        db.update_user_profile(user['id'], profile_data)
        flash(f"Resume parsed by Gemini! Extracted {len(skills)} skills.")
    else:
        # Fallback to legacy extraction if Gemini fails
        print("Gemini parsing failed, falling back to legacy parser.")
        text = extract_text_from_file(save_path)
        if not text or len(text.strip()) < 10:
             flash("Warning: Could not extract text from resume.")
             return redirect(url_for("upload"))
        skills, summary = extract_skills_and_summary(text)
        
        profile_data = {
            'resume_path': filename,
            'skills': ','.join(skills) if skills else None
        }
        db.update_user_profile(user['id'], profile_data)

    # Store only the filename (not full path) for easier retrieval
    resume_filename = filename
    
    # Check if Ollama is available and show status
    ollama_available = summarizer.is_available()
    if not ollama_available:
        flash("Note: Ollama AI summarization not available. Using basic text extraction. For better summaries, install Ollama and run 'ollama serve'.")

    # ✅ Skill gap analysis
    have, missing, ranked_missing = analyze_skill_gap(skills, role)

    # ✅ Create learning path with course recommendations
    learning_path = create_learning_path(ranked_missing, have)

    # ✅ Internship scraping (DuckDuckGo → fallback to CSV)
    internships = ddg_search_internships(role, location, top_k=5)
    if not internships:
        internships = recommend_internships_from_profile(skills, role, location, top_k=5)

    # Fallback location for scraped internships if location is blank
    for job in internships:
        if not job.get("location"):
            job["location"] = location

    # ✅ Salary prediction (Gemini Only)
    # sal_low, sal_high = predict_salary(skills, role, experience_years=0)
    
    # Use Gemini for salary prediction
    gemini_salary = gemini_service.predict_salary(
        resume_summary=summary, 
        skills=', '.join(skills), 
        role=role, 
        experience=0 # Default to 0 for upload
    )
    sal_low = gemini_salary.get('min_salary', 300000)
    sal_high = gemini_salary.get('max_salary', 800000)
    salary_explanation = gemini_salary.get('explanation', "Estimated by Gemini AI based on your profile.")

    # ✅ Generate XAI explanations (Skill Gap Only - Salary handled by Gemini)
    skill_gap_explanation = None
    try:
        skill_gap_explainer = get_skill_gap_explainer()
        skill_gap_explanation = skill_gap_explainer.explain_skill_gap(have, missing, ranked_missing, role)
    except Exception as e:
        print(f"Error generating explanations: {e}")

    # Save recommendation history
    db.save_recommendation_history(user['id'], role, location, skills, missing, (sal_low, sal_high))

    # ✅ Compute initial per-job salary estimates (experience default 0)
    for job in internships:
        # Map job title to known role if possible
        job_role = find_best_role(job.get("title", "")) or role
        jlow, jhigh = predict_salary(skills, job_role, experience_years=0)
        job["salary_low"] = jlow
        job["salary_high"] = jhigh

    return render_template(
        "results.html",
        role=role,
        location=location,
        skills=skills,
        summary=summary,
        have=have,
        missing=missing,
        ranked_missing=ranked_missing,
        learning_path=learning_path,
        internships=internships,
        salary_range=(sal_low, sal_high),
        salary_explanation=salary_explanation,
        skill_gap_explanation=skill_gap_explanation,
    )

@app.route("/profile-builder", methods=["GET", "POST"])
@login_required
def profile_builder():
    """
    AI-powered Profile Builder using Gemini for OCR and suggestions.
    """
    if request.method == "GET":
        return render_template("profile_builder.html")
    
    if "resume" not in request.files:
        flash("No resume file provided")
        return redirect(url_for("profile_builder"))

    user = get_current_user()
    file = request.files["resume"]
    
    if file.filename == "":
        flash("No selected file")
        return redirect(url_for("profile_builder"))

    # Save file
    filename = f"gemini_{user['id']}_{file.filename}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)
    
    # Parse with Gemini
    print(f"Parsing resume with Gemini: {save_path}")
    result = gemini_service.parse_resume(save_path)
    
    if result.get("error"):
        flash(f"Error parsing resume: {result.get('error')}")
        return redirect(url_for("profile_builder"))
    
    # ✅ Update User Profile Data based on Gemini result
    try:
        edu = result.get('education', {})
        prof = result.get('professional_info', {})
        skills_list = prof.get('skills', [])
        if isinstance(skills_list, str):
            skills_list = [s.strip() for s in skills_list.split(',')]
            
        profile_update = {
            'resume_path': filename,
            'degree': edu.get('degree'),
            'stream': edu.get('stream'),
            'study_year': edu.get('study_year'),
            'sector': prof.get('sector'),
            'skills': ','.join(skills_list) if skills_list else None
        }
        # Update DB
        db.update_user_profile(user['id'], profile_update)
        flash("Profile automatically updated with extracted details!")
    except Exception as e:
        print(f"Error updating profile from builder: {e}")

    return render_template("profile_builder.html", result=result)

@app.route("/recommendations")
@login_required
def recommendations():
    user = get_current_user()
    
    # 1. Try to get latest recommendation context (Role, Location, Skills)
    # Priority: Recommendation History > Profile
    rec_history = db.get_recommendation_history(user['id'], limit=1)
    profile = db.get_user_profile(user['id'])
    
    role = None
    location = None
    skills = []
    
    if rec_history:
        last_rec = rec_history[0]
        role = last_rec.get('role')
        location = last_rec.get('location')
        # DB stores skills as "skill1,skill2" string or similar in 'skills_used' ??
        # Let's check db schema in database.py: 'skills_used TEXT'
        # In apply route: db.save_recommendation_history(..., skills, ...)
        # In save_recommendation_history: ','.join(skills_used)
        if last_rec.get('skills_used'):
            skills = last_rec.get('skills_used').split(',')
    
    if not role and profile:
        # Fallback to Profile if history is empty
        # Profile has degree, sector, stream, skills (string)
        # But NOT explicit "Target Role". 
        # We might have to guess or ask user.
        # For now, let's try to infer or just redirect if missing.
        role = profile.get('stream') # Weak fallback
        if profile.get('skills'):
            skills = profile.get('skills').split(',')
    
    if not role:
        flash("We need to know your Target Role to provide recommendations. Please upload a resume or fill the form.")
        return redirect(url_for('upload'))
    
    # Clean up data
    role = role.strip()
    location = location.strip() if location else ""
    skills = [s.strip() for s in skills if s.strip()]
    
    # 2. Run Live Analysis (Getting fresh jobs is better than stale history)
    have, missing, ranked_missing = analyze_skill_gap(skills, role)
    learning_path = create_learning_path(ranked_missing, have)
    
    # Search jobs (fresh)
    internships = ddg_search_internships(role, location, top_k=5)
    if not internships:
        internships = recommend_internships_from_profile(skills, role, location, top_k=5)
        
    for job in internships:
        if not job.get("location"):
            job["location"] = location
            
        # Add salary estimate to job
        job_role = find_best_role(job.get("title", "")) or role
        jlow, jhigh = predict_salary(skills, job_role, experience_years=0)
        job["salary_low"] = jlow
        job["salary_high"] = jhigh

    # Overall salary
    sal_low, sal_high = predict_salary(skills, role, experience_years=0)
    
    return render_template(
        "recommendations.html",
        user=user,
        role=role,
        internships=internships,
        salary_range=(sal_low, sal_high),
        have=have,
        missing=missing,
        ranked_missing=ranked_missing,
        learning_path=learning_path
    )


@app.route("/api/predict_salary", methods=["POST"])
def api_predict_salary():
    data = request.get_json(force=True) or {}
    skills = data.get("skills", [])
    role = data.get("role", "")
    experience = int(data.get("experience", 0) or 0)
    include_explanation = data.get("explain", False)
    
    low, high = predict_salary(skills, role, experience_years=experience)
    
    response = {
        "low": low, 
        "high": high,
        "low_lpa": format_salary_lpa(low),
        "high_lpa": format_salary_lpa(high)
    }
    
    # Add XAI explanation if requested
    if include_explanation:
        try:
            explainer = get_salary_explainer()
            explanation = explainer.explain_prediction(skills, role, experience)
            response["explanation"] = explanation
        except Exception as e:
            print(f"Error generating explanation: {e}")
            response["explanation"] = None
    
    return jsonify(response)

@app.route("/api/ats-educator/chat", methods=["POST"])
def api_ats_educator_chat():
    """API endpoint for RAG ATS Educator"""
    data = request.get_json(force=True) or {}
    question = data.get("question", "").strip()
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    user = get_current_user()
    context = {}
    
    # Enrich context with user data if available
    if user:
        try:
            # Get latest enhancement history to provide context
            enhancement_history = db.get_modification_history(user['id'], limit=1)
            if enhancement_history:
                latest = enhancement_history[0]
                context['ats_score'] = latest.get('ats_score_before')
                # Add more context as needed
        except Exception as e:
            print(f"Error getting user context: {e}")
            
    # Generate response
    response = rag_educator.generate_explanation(question, context=context)
    
    return jsonify(response)

@app.route("/api/course_recommendations", methods=["POST"])
def api_course_recommendations():
    data = request.get_json(force=True) or {}
    skill = data.get("skill", "")
    if not skill:
        return jsonify({"error": "Skill parameter required"}), 400
    
    courses = get_enhanced_course_recommendations(skill)
    return jsonify(courses)

@app.route("/api/explain_salary", methods=["POST"])
def api_explain_salary():
    """API endpoint to get detailed salary prediction explanation."""
    data = request.get_json(force=True) or {}
    skills = data.get("skills", [])
    role = data.get("role", "")
    experience = int(data.get("experience", 0) or 0)
    
    try:
        explainer = get_salary_explainer()
        explanation = explainer.explain_prediction(skills, role, experience)
        return jsonify(explanation)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/explain_recommendation", methods=["POST"])
def api_explain_recommendation():
    """API endpoint to explain why an internship was recommended."""
    data = request.get_json(force=True) or {}
    internship = data.get("internship", {})
    user_skills = data.get("skills", [])
    role = data.get("role", "")
    location = data.get("location", "")
    score = data.get("score", 0)
    
    try:
        explainer = get_recommendation_explainer()
        explanation = explainer.explain_recommendation(internship, user_skills, role, location, score)
        return jsonify(explanation)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/explain_skill_gap", methods=["POST"])
def api_explain_skill_gap():
    """API endpoint to explain skill gap analysis."""
    data = request.get_json(force=True) or {}
    have_skills = data.get("have_skills", [])
    missing_skills = data.get("missing_skills", [])
    ranked_missing = data.get("ranked_missing", [])
    role = data.get("role", "")
    
    try:
        explainer = get_skill_gap_explainer()
        explanation = explainer.explain_skill_gap(have_skills, missing_skills, ranked_missing, role)
        return jsonify(explanation)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==================== HR ROUTES ====================

@app.route("/hr/dashboard")
@login_required
@hr_required
def hr_dashboard():
    """HR Dashboard - Overview of candidates and job postings with analysis"""
    user = get_current_user()
    
    # Get all candidates count
    all_candidates = db.get_all_candidates(limit=1000)
    total_candidates = len(all_candidates)
    
    # Get recent candidates for preview
    recent_candidates = db.get_all_candidates(limit=10)
    
    # Get job postings
    job_postings = db.get_job_postings(hr_user_id=user['id'])
    
    # Get analyzed matches for the most recent job posting (if exists)
    analyzed_results = None
    analysis_stats = None
    selected_job_id = request.args.get('job_id', type=int)
    selected_job = None
    
    if selected_job_id:
        selected_job = db.get_job_posting_by_id(selected_job_id)
        if selected_job and selected_job.get('hr_user_id') == user['id']:
            from utils.candidate_matcher import search_and_match_candidates
            analyzed_results = search_and_match_candidates(selected_job, {})
            # Calculate statistics
            if analyzed_results:
                excellent = len([r for r in analyzed_results if r['overall_score'] >= 85])
                good = len([r for r in analyzed_results if 70 <= r['overall_score'] < 85])
                moderate = len([r for r in analyzed_results if 50 <= r['overall_score'] < 70])
                weak = len([r for r in analyzed_results if r['overall_score'] < 50])
                avg_score = sum(r['overall_score'] for r in analyzed_results) / len(analyzed_results) if analyzed_results else 0
                
                analysis_stats = {
                    'total_candidates': len(analyzed_results),
                    'excellent_matches': excellent,
                    'good_matches': good,
                    'moderate_matches': moderate,
                    'weak_matches': weak,
                    'average_score': round(avg_score, 2)
                }
    elif job_postings:
        # Auto-analyze the most recent job posting
        selected_job = job_postings[0]
        from utils.candidate_matcher import search_and_match_candidates
        analyzed_results = search_and_match_candidates(selected_job, {})
        # Calculate statistics
        if analyzed_results:
            excellent = len([r for r in analyzed_results if r['overall_score'] >= 85])
            good = len([r for r in analyzed_results if 70 <= r['overall_score'] < 85])
            moderate = len([r for r in analyzed_results if 50 <= r['overall_score'] < 70])
            weak = len([r for r in analyzed_results if r['overall_score'] < 50])
            avg_score = sum(r['overall_score'] for r in analyzed_results) / len(analyzed_results) if analyzed_results else 0
            
            analysis_stats = {
                'total_candidates': len(analyzed_results),
                'excellent_matches': excellent,
                'good_matches': good,
                'moderate_matches': moderate,
                'weak_matches': weak,
                'average_score': round(avg_score, 2)
            }
    
    return render_template("hr/dashboard.html", 
                         user=user, 
                         candidates=recent_candidates,
                         total_candidates=total_candidates,
                         job_postings=job_postings,
                         analyzed_results=analyzed_results,
                         selected_job_id=selected_job_id,
                         selected_job=selected_job,
                         analysis_stats=analysis_stats)

@app.route("/hr/candidates")
@login_required
@hr_required
def hr_candidates():
    """HR Candidate Search and Browse"""
    user = get_current_user()
    
    # Get search filters
    skills = request.args.get("skills", "").strip()
    location = request.args.get("location", "").strip()
    degree = request.args.get("degree", "").strip()
    stream = request.args.get("stream", "").strip()
    
    # Search candidates
    if any([skills, location, degree, stream]):
        candidates = db.search_candidates(
            skills=skills if skills else None,
            location=location if location else None,
            degree=degree if degree else None,
            stream=stream if stream else None,
            limit=100
        )
    else:
        candidates = db.get_all_candidates(limit=100)
    
    return render_template("hr/candidates.html", user=user, candidates=candidates,
                         search_skills=skills, search_location=location, 
                         search_degree=degree, search_stream=stream)

@app.route("/hr/candidate/<int:candidate_id>")
@login_required
@hr_required
def hr_candidate_detail(candidate_id):
    """View detailed candidate profile"""
    user = get_current_user()
    candidate = db.get_candidate_by_id(candidate_id)
    
    if not candidate:
        flash("Candidate not found")
        return redirect(url_for('hr_candidates'))
    
    # Get candidate's recommendation history
    history = db.get_recommendation_history(candidate_id, limit=10)
    
    return render_template("hr/candidate_detail.html", user=user, candidate=candidate, history=history)

@app.route("/hr/jobs", methods=["GET", "POST"])
@login_required
@hr_required
def hr_jobs():
    """Manage job postings"""
    user = get_current_user()
    
    if request.method == "POST":
        # Create new job posting
        title = request.form.get("title", "").strip()
        description = request.form.get("description", "").strip()
        required_skills = request.form.get("required_skills", "").strip()
        location = request.form.get("location", "").strip()
        salary_low = request.form.get("salary_low", "").strip()
        salary_high = request.form.get("salary_high", "").strip()
        
        if not title:
            flash("Job title is required")
            return redirect(url_for('hr_jobs'))
        
        salary_low = int(salary_low) if salary_low and salary_low.isdigit() else None
        salary_high = int(salary_high) if salary_high and salary_high.isdigit() else None
        
        job_id = db.create_job_posting(
            user['id'], title, description, required_skills, 
            location, salary_low, salary_high
        )
        
        if job_id:
            flash("Job posting created successfully!")
        else:
            flash("Error creating job posting")
        
        return redirect(url_for('hr_jobs'))
    
    # GET: Show all job postings
    job_postings = db.get_job_postings(hr_user_id=user['id'])
    return render_template("hr/jobs.html", user=user, job_postings=job_postings)

@app.route("/hr/job/<int:job_id>/match")
@login_required
@hr_required
def hr_job_match(job_id):
    """Match candidates to a specific job posting"""
    user = get_current_user()
    job_posting = db.get_job_posting_by_id(job_id)
    
    if not job_posting or job_posting.get('hr_user_id') != user['id']:
        flash("Job posting not found")
        return redirect(url_for('hr_jobs'))
    
    # Get search filters
    skills = request.args.get("skills", "").strip()
    location = request.args.get("location", "").strip()
    degree = request.args.get("degree", "").strip()
    stream = request.args.get("stream", "").strip()
    
    filters = {
        'skills': skills if skills else None,
        'location': location if location else None,
        'degree': degree if degree else None,
        'stream': stream if stream else None
    }
    
    # Search and match candidates
    matched_candidates = search_and_match_candidates(job_posting, filters)
    
    return render_template("hr/job_match.html", user=user, job_posting=job_posting,
                         matched_candidates=matched_candidates, filters=filters)

@app.route("/hr/job/<int:job_id>/candidate/<int:candidate_id>")
@login_required
@hr_required
def hr_job_candidate_insights(job_id, candidate_id):
    """Get detailed insights for a candidate matched to a job"""
    user = get_current_user()
    job_posting = db.get_job_posting_by_id(job_id)
    candidate = db.get_candidate_by_id(candidate_id)
    
    if not job_posting or job_posting.get('hr_user_id') != user['id']:
        flash("Job posting not found")
        return redirect(url_for('hr_jobs'))
    
    if not candidate:
        flash("Candidate not found")
        return redirect(url_for('hr_job_match', job_id=job_id))
    
    # Get detailed insights
    insights = get_candidate_insights(candidate, job_posting)
    
    return render_template("hr/candidate_insights.html", user=user, 
                         job_posting=job_posting, candidate=candidate, insights=insights)

@app.route("/api/hr/match_candidates", methods=["POST"])
@login_required
@hr_required
def api_match_candidates():
    """API endpoint for candidate matching"""
    data = request.get_json(force=True) or {}
    job_id = data.get("job_id")
    
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    
    job_posting = db.get_job_posting_by_id(job_id)
    if not job_posting:
        return jsonify({"error": "Job posting not found"}), 404
    
    filters = data.get("filters", {})
    matched = search_and_match_candidates(job_posting, filters)
    
    return jsonify({"matched_candidates": matched})

@app.route("/api/hr/chatbot", methods=["POST"])
@login_required
@hr_required
def api_hr_chatbot():
    """API endpoint for HR chatbot"""
    data = request.get_json(force=True) or {}
    question = data.get("question", "").strip()
    candidate_id = data.get("candidate_id")
    job_id = data.get("job_id")
    conversation_history = data.get("history", [])
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    # Validate candidate_id and job_id if provided
    if candidate_id:
        candidate = db.get_candidate_by_id(candidate_id)
        if not candidate:
            return jsonify({"error": "Candidate not found"}), 404
    
    if job_id:
        job_posting = db.get_job_posting_by_id(job_id)
        if not job_posting:
            return jsonify({"error": "Job posting not found"}), 404
    
    # Get chatbot response
    result = hr_chatbot.chat(
        question=question,
        candidate_id=candidate_id,
        job_id=job_id,
        conversation_history=conversation_history
    )
    
    return jsonify(result)

@app.route("/api/hr/chatbot/suggestions", methods=["GET"])
@login_required
@hr_required
def api_hr_chatbot_suggestions():
    """Get suggested questions for chatbot"""
    candidate_id = request.args.get("candidate_id", type=int)
    job_id = request.args.get("job_id", type=int)
    
    suggestions = hr_chatbot.get_suggested_questions(
        candidate_id=candidate_id,
        job_id=job_id
    )
    
    return jsonify({"suggestions": suggestions})

@app.route("/api/hr/chatbot/debug", methods=["GET"])
@login_required
@hr_required
def api_hr_chatbot_debug():
    """Debug endpoint to check chatbot context"""
    candidate_id = request.args.get("candidate_id", type=int)
    job_id = request.args.get("job_id", type=int)
    
    context = hr_chatbot._get_candidate_context(candidate_id, job_id)
    system_prompt = hr_chatbot._build_system_prompt(candidate_id, job_id)
    
    return jsonify({
        "candidate_id": candidate_id,
        "job_id": job_id,
        "context_length": len(context),
        "context_preview": context[:500] if context else "No context",
        "system_prompt_length": len(system_prompt),
        "system_prompt_preview": system_prompt[:500] if system_prompt else "No prompt"
    })

# ==================== RESUME ENHANCEMENT ROUTES ====================

@app.route("/enhance-resume", methods=["GET", "POST"])
@login_required
def enhance_resume():
    """AI-powered resume enhancement based on job description"""
    user = get_current_user()
    
    if request.method == "GET":
        return render_template("enhance_resume.html", user=user)
    
    # POST: Process resume enhancement
    if "resume" not in request.files:
        flash("Please upload a resume file")
        return redirect(url_for("enhance_resume"))
    
    file = request.files["resume"]
    job_description = request.form.get("job_description", "").strip()
    
    if not job_description:
        flash("Please provide a job description")
        return redirect(url_for("enhance_resume"))
    
    if file.filename == "":
        flash("No file selected")
        return redirect(url_for("enhance_resume"))
    
    # Check file type
    if not file.filename.lower().endswith(('.docx',)):
        flash("Currently only DOCX files are supported for enhancement")
        return redirect(url_for("enhance_resume"))
    
    # Initialize resume modifications table
    db.create_resume_modifications_table()
    
    # Save original file
    filename = f"{user['id']}_{file.filename}"
    original_path = os.path.join(app.config["UPLOAD_FOLDER"], f"original_{filename}")
    file.save(original_path)
    print(f"[DEBUG] File saved to: {original_path}")
    
    try:
        # 1. Extract text from resume
        print("[DEBUG] Step 1: Extracting text from resume...")
        text = extract_text_from_file(original_path)
        print(f"[DEBUG] Extracted {len(text)} characters from resume")
        
        print("[DEBUG] Step 2: Extracting skills from resume...")
        skills, summary = extract_skills_and_summary(text)
        print(f"[DEBUG] Found {len(skills)} skills in resume")
        
        # 2. Extract skills from JD
        print("[DEBUG] Step 3: Extracting skills from JD...")
        jd_skills, _ = extract_skills_and_summary(job_description)
        print(f"[DEBUG] Found {len(jd_skills)} skills in JD")
        
        # 3. Run ATS analysis (before modification)
        print("[DEBUG] Step 4: Running ATS analysis...")
        ats_result = ats_engine.analyze_resume_vs_jd(
            resume_skills=skills,
            job_required_skills=jd_skills,
            job_preferred_skills=[],
            resume_text=text,
            job_description=job_description
        )
        
        ats_score_before = ats_result.overall_score
        print(f"[DEBUG] ATS Score: {ats_score_before}%")
        print(f"[DEBUG] Missing skills: {[m.skill for m in ats_result.missing_skills][:5]}")
        print(f"[DEBUG] Weak skills: {[w.skill for w in ats_result.weak_skills][:5]}")
        
        # 4. Generate modification suggestions (Gemini)
        print("[DEBUG] Step 5: Generating AI suggestions with Gemini...")
        
        suggestions = []
        
        # Skill Enhancement
        if ats_result.missing_skills:
            instruction = f"Add these missing skills naturally: {', '.join([m.skill for m in ats_result.missing_skills][:5])}. Keep original format."
            # Find relevant section content in 'text' (simple heuristic for now, or use gemini to rewrite whole resume which is safer?)
            # For this exercise, we'll try to use resume_editor.extract_sections to send specific chunks
            sections = resume_editor.extract_sections(text)
            skills_section = next((s for s in sections if 'SKILL' in s.name.upper()), None)
            
            if skills_section:
                rewritten = gemini_service.enhance_resume_content("Skills", skills_section.content, job_description, instruction)
                suggestions.append({
                    'section': "Skills",
                    'original_text': skills_section.content,
                    'suggested_text': rewritten,
                    'reason': "Added missing skills for ATS compliance.",
                    'impact_score': len(ats_result.missing_skills) * 5
                })

        # Experience Enhancement
        exp_section = next((s for s in resume_editor.extract_sections(text) if 'EXPERIENCE' in s.name.upper()), None)
        if exp_section:
            instruction = f"Highlight experiences relevant to: {[m.skill for m in ats_result.missing_skills][:3]}. Use strong action verbs."
            rewritten_exp = gemini_service.enhance_resume_content("Experience", exp_section.content, job_description, instruction)
            suggestions.append({
                'section': "Experience",
                'original_text': exp_section.content,
                'suggested_text': rewritten_exp,
                'reason': "Optimized experience descriptions for role alignment.",
                'impact_score': 15
            })
            
        # Summary Enhancement
        sum_section = next((s for s in resume_editor.extract_sections(text) if 'SUMMARY' in s.name.upper() or 'PROFILE' in s.name.upper()), None)
        if sum_section:
             instruction = "Rewrite summary to strongly align with the Job Description."
             rewritten_sum = gemini_service.enhance_resume_content("Summary", sum_section.content, job_description, instruction)
             suggestions.append({
                'section': "Summary",
                'original_text': sum_section.content,
                'suggested_text': rewritten_sum,
                'reason': "Aligned professional summary with target role.",
                'impact_score': 10
            })

        print(f"[DEBUG] Generated {len(suggestions)} suggestions")
        
        # Store suggestions in session (even if empty, so user can edit)
        print("[DEBUG] Step 6: Storing suggestions in session...")
        
        # If no suggestions, create a dummy one or just pass empty list
        if not suggestions:
            print("[DEBUG] No suggestions generated - creating placeholder")
            pass # suggestions is []

        session['enhancement_data'] = {
            'original_path': original_path,
            'filename': filename,
            'suggestions': [
                {
                    'section': s['section'],
                    'original': s['original_text'],
                    'suggested': s['suggested_text'],
                    'reason': s['reason'],
                    'impact': s['impact_score']
                }
                for s in suggestions
            ],
            'ats_score_before': ats_score_before,
            'job_description': job_description[:1000]
        }
        
        print("[DEBUG] Step 7: Rendering review template...")
        return render_template(
            "review_enhancements.html",
            user=user,
            suggestions=suggestions,
            ats_score_before=ats_score_before,
            job_description=job_description
        )
    
    except Exception as e:
        import traceback
        print(f"\n[ERROR] ========== ERROR IN RESUME ENHANCEMENT ==========")
        print(f"[ERROR] Error type: {type(e).__name__}")
        print(f"[ERROR] Error message: {e}")
        print(f"[ERROR] Full traceback:")
        print(traceback.format_exc())
        print(f"[ERROR] ================================================\n")
        flash(f"Error processing resume: {str(e)}")
        return redirect(url_for("enhance_resume"))


@app.route("/apply-enhancements", methods=["POST"])
@login_required
def apply_enhancements():
    """Apply approved resume enhancements"""
    user = get_current_user()
    
    enhancement_data = session.get('enhancement_data')
    if not enhancement_data:
        flash("No enhancement data found. Please start again.")
        return redirect(url_for("enhance_resume"))
    
    # Get all indices and approved ones
    all_indices = request.form.getlist('suggestion_indices')
    approved_indices = request.form.getlist('approved_indices')
    download_format = request.form.get('download_format', 'docx')
    
    try:
        # Reconstruct suggestions from session/file
        text = extract_text_from_file(enhancement_data['original_path'])
        jd_text = enhancement_data['job_description']
        
        # We need to rebuild the full suggestion objects as they were in the session
        # But we also need to update them with any USER EDITS from the form
        
        # Load original suggestions (lightweight version from session)
        session_suggestions = enhancement_data['suggestions']
        
        # Re-run generation or just use session data?
        # Ideally we trust the session data for the structure, but we need the full objects for the editor
        # For simplicity, we'll recreate modification objects from the session and form data
        
        from utils.resume_editor import ModificationSuggestion
        
        final_suggestions_to_apply = []
        
        for idx in all_indices:
            if idx in approved_indices:
                # This suggestion is approved
                i = int(idx)
                if i < len(session_suggestions):
                    s_data = session_suggestions[i]
                    
                    # Get user-edited text
                    user_edited_text = request.form.get(f"suggestion_text_{idx}", "").strip()
                    
                    # Create object
                    suggestion = ModificationSuggestion(
                        section=s_data['section'],
                        original_text=s_data['original'], # Note: this might be truncated in session, potentially risky if we need full match
                        # But resume_editor uses fuzzy finding or naive replacement. 
                        # Let's hope the replace logic finds the section header and replaces content
                        suggested_text=user_edited_text if user_edited_text else s_data['suggested'],
                        reason=s_data['reason'],
                        impact_score=s_data['impact']
                    )
                    final_suggestions_to_apply.append(suggestion)
        
        # Apply modifications
        modified_filename = f"enhanced_{enhancement_data['filename']}"
        modified_path = os.path.join(app.config["UPLOAD_FOLDER"], modified_filename)
        
        # Note: apply_modifications_to_docx needs to work well. 
        # The session stored 'original_path' is still valid.
        success = resume_editor.apply_modifications_to_docx(
            enhancement_data['original_path'],
            final_suggestions_to_apply,
            modified_path
        )
        
        if not success:
            flash("Error applying modifications. Make sure python-docx is installed.")
            return redirect(url_for("enhance_resume"))
        
        # DB Save
        db.save_resume_modification(
            user_id=user['id'],
            original_path=enhancement_data['original_path'],
            modified_path=modified_path,
            modifications=[
                {'section': s.section, 'reason': s.reason, 'impact': s.impact_score}
                for s in final_suggestions_to_apply
            ],
            score_before=enhancement_data['ats_score_before'],
            score_after=None, 
            job_desc=jd_text
        )
        
        # Handle PDF Conversion
        final_file_path = modified_path
        final_filename = modified_filename
        mimetype = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        
        if download_format == 'pdf':
            try:
                # Try converting to PDF
                pdf_filename = modified_filename.replace('.docx', '.pdf')
                pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf_filename)
                
                # Check for docx2pdf
                from docx2pdf import convert
                convert(modified_path, pdf_path)
                
                final_file_path = pdf_path
                final_filename = pdf_filename
                mimetype = "application/pdf"
            except ImportError:
                flash("PDF conversion requires 'docx2pdf' library. Downloading as DOCX instead.")
            except Exception as e:
                print(f"PDF Conversion failed: {e}")
                flash("PDF conversion failed (requires Microsoft Word installed). Downloading as DOCX instead.")
        
        # Clear session
        session.pop('enhancement_data', None)
        
        flash("✅ Resume enhanced successfully!")
        return send_from_directory(
            app.config["UPLOAD_FOLDER"],
            final_filename,
            as_attachment=True,
            download_name=f"enhanced_{user['username']}_resume.{'pdf' if final_filename.endswith('.pdf') else 'docx'}",
            mimetype=mimetype
        )
    
    except Exception as e:
        import traceback
        print(f"Error applying enhancements: {e}")
        print(traceback.format_exc())
        flash(f"Error applying enhancements: {str(e)}")
        return redirect(url_for("enhance_resume"))


@app.route("/promote-to-hr", methods=["GET", "POST"])
def promote_to_hr():
    """Simple route to promote a user to HR role (for testing/setup)"""
    messages = []
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        if not username:
            messages.append("Username is required")
        else:
            user = db.get_user_by_username(username)
            if not user:
                messages.append(f"User '{username}' not found")
            else:
                success = db.update_user_role(user['id'], 'hr')
                if success:
                    messages.append(f"✓ User '{username}' has been promoted to HR role! Please log out and log back in.")
                else:
                    messages.append(f"✗ Error promoting user '{username}'")
    
    # Get list of all users for reference
    all_users = []
    try:
        cursor = db._get_cursor(dictionary=True)
        cursor.execute("SELECT id, username, email, user_role FROM users ORDER BY id")
        all_users = cursor.fetchall()
        cursor.close()
    except:
        pass
    
    messages_html = ""
    if messages:
        messages_html = "<div style='padding: 10px; margin: 10px 0; background: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px;'>" + "<br>".join(messages) + "</div>"
    
    users_list = ""
    if all_users:
        users_list = "<h3>All Users:</h3><table style='width: 100%; border-collapse: collapse; margin-top: 10px;'><tr style='background: #f0f0f0;'><th style='padding: 8px; text-align: left; border: 1px solid #ddd;'>ID</th><th style='padding: 8px; text-align: left; border: 1px solid #ddd;'>Username</th><th style='padding: 8px; text-align: left; border: 1px solid #ddd;'>Email</th><th style='padding: 8px; text-align: left; border: 1px solid #ddd;'>Role</th></tr>"
        for u in all_users:
            role_color = "#28a745" if u.get('user_role') == 'hr' else "#6c757d"
            users_list += f"<tr><td style='padding: 8px; border: 1px solid #ddd;'>{u.get('id')}</td><td style='padding: 8px; border: 1px solid #ddd;'>{u.get('username')}</td><td style='padding: 8px; border: 1px solid #ddd;'>{u.get('email')}</td><td style='padding: 8px; border: 1px solid #ddd;'><span style='color: {role_color}; font-weight: bold;'>{u.get('user_role', 'student')}</span></td></tr>"
        users_list += "</table>"
    
    return f"""
    <html>
    <head><title>Promote to HR</title></head>
    <body style="font-family: Arial; max-width: 700px; margin: 50px auto; padding: 20px;">
        <h2>Promote User to HR</h2>
        {messages_html}
        <form method="POST" style="margin: 20px 0;">
            <p>
                <label><strong>Username or Email:</strong></label><br>
                <input type="text" name="username" required style="width: 100%; padding: 8px; margin-top: 5px; border: 1px solid #ddd; border-radius: 4px;">
            </p>
            <button type="submit" style="background: #ff7b00; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px;">
                Promote to HR
            </button>
        </form>
        {users_list}
        <p style="margin-top: 20px; color: #666; font-size: 0.9em;">
            <strong>Note:</strong> After promoting, the user must log out and log back in to access the HR dashboard.
        </p>
        <p><a href="/login" style="color: #ff7b00;">Go to Login</a> | <a href="/" style="color: #ff7b00;">Home</a></p>
    </body>
    </html>
    """

@app.route("/uploads/<filename>")
@login_required
def serve_resume(filename):
    """Serve uploaded resume files"""
    from flask import send_from_directory, abort
    import os
    
    # Security: ensure filename doesn't contain path traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        abort(404)
    
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        abort(404)
    
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=True)

@app.route("/api/rewrite-text", methods=["POST"])
@login_required
def rewrite_text_api():
    """API endpoint to rewrite resume text"""
    data = request.json
    text = data.get('text')
    style = data.get('style', 'professional')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
        
    variations = resume_editor.rewrite_text_segment(text, style)
    return jsonify({"variations": variations})

@app.route("/reset-profile", methods=["POST"])
@login_required
def reset_profile():
    """Reset user profile data including skills, extracted info, recommendations, and enhancements"""
    user = get_current_user()
    
    try:
        if db.reset_user_data(user['id']):
            flash("Profile successfully reset! All data has been cleared. Please upload a resume to start fresh.")
        else:
            flash("Error resetting profile. Please try again.")
    except Exception as e:
        print(f"Error in reset_profile: {e}")
        flash("An unexpected error occurred during reset.")
        
    return redirect(url_for('dashboard'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
