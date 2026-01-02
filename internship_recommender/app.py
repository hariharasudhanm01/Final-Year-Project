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

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("models", exist_ok=True)

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'replace-with-secure-key-in-production')
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

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
    full_name = request.form.get("full_name", "").strip()
    
    if not all([username, email, password, confirm_password]):
        flash("Please fill in all fields")
        return redirect(url_for('register'))
    
    if password != confirm_password:
        flash("Passwords do not match")
        return redirect(url_for('register'))
    
    if len(password) < 6:
        flash("Password must be at least 6 characters long")
        return redirect(url_for('register'))
    
    success = db.create_user(username, email, password, full_name)
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

    # ✅ Extract skills & profile summary
    skills, summary = extract_skills_and_summary(text)

    # Update user profile with resume path and extracted data
    profile_data = {
        'resume_path': save_path,
        'skills': ','.join(skills) if skills else None
    }
    db.update_user_profile(user['id'], profile_data)

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
    )

@app.route("/api/predict_salary", methods=["POST"])
def api_predict_salary():
    data = request.get_json(force=True) or {}
    skills = data.get("skills", [])
    role = data.get("role", "")
    experience = int(data.get("experience", 0) or 0)
    low, high = predict_salary(skills, role, experience_years=experience)
    return jsonify({"low": low, "high": high})

@app.route("/api/course_recommendations", methods=["POST"])
def api_course_recommendations():
    data = request.get_json(force=True) or {}
    skill = data.get("skill", "")
    if not skill:
        return jsonify({"error": "Skill parameter required"}), 400
    
    courses = get_enhanced_course_recommendations(skill)
    return jsonify(courses)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
