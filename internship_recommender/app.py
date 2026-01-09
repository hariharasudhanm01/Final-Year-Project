import os
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, session
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
    
    return render_template("dashboard.html", user=user, profile=profile, history=history)

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

    # ✅ Extract text from resume
    text = extract_text_from_file(save_path)
    
    # Debug: Check if text extraction worked
    if not text or len(text.strip()) < 10:
        flash("Warning: Could not extract text from resume. Please ensure the file is a valid PDF or DOCX.")
        # Don't proceed with empty text
        return redirect(url_for("upload"))
    
    print(f"[DEBUG] Extracted {len(text)} characters from resume")

    # ✅ Extract skills & profile summary
    skills, summary = extract_skills_and_summary(text)
    
    print(f"[DEBUG] Extracted {len(skills)} skills: {skills}")
    
    # Debug: Check if skills were extracted
    if not skills or len(skills) == 0:
        flash("Warning: No skills found in resume. Please ensure your resume contains skill keywords.")
        # Still proceed but with empty skills list - this will clear existing skills
        skills = []

    # Store only the filename (not full path) for easier retrieval
    resume_filename = filename
    
    # Update user profile with resume path and extracted data
    # IMPORTANT: Always update skills from resume, even if empty (to clear old/default skills)
    profile_data = {
        'resume_path': resume_filename,  # Store only filename, not full path
        'skills': ','.join(skills) if skills else None  # Clear skills if none found
    }
    db.update_user_profile(user['id'], profile_data)
    
    # Flash success message with extracted skills count
    if skills:
        flash(f"Resume uploaded successfully! Extracted {len(skills)} skills from your resume: {', '.join(skills[:5])}{'...' if len(skills) > 5 else ''}")
    else:
        flash("Resume uploaded, but no skills were detected. You can add skills manually in your profile.")

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

    # ✅ Salary prediction (overall)
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
