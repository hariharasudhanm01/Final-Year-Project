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
    
    # Save original file
    filename = f"{user['id']}_{file.filename}"
    original_path = os.path.join(app.config["UPLOAD_FOLDER"], f"original_{filename}")
    file.save(original_path)
    
    try:
        # 1. Extract text from resume
        text = extract_text_from_file(original_path)
        skills, summary = extract_skills_and_summary(text)
        
        # 2. Extract skills from JD
        jd_skills, _ = extract_skills_and_summary(job_description)
        
        # 3. Run ATS analysis (before modification)
        ats_result = ats_engine.analyze_resume_vs_jd(
            resume_skills=skills,
            job_required_skills=jd_skills,
            job_preferred_skills=[],
            resume_text=text,
            job_description=job_description
        )
        
        ats_score_before = ats_result.overall_score
        
        # 4. Generate modification suggestions
        suggestions = resume_editor.suggest_resume_modifications(
            resume_text=text,
            jd_text=job_description,
            missing_skills=[m.skill for m in ats_result.missing_skills],
            weak_skills=[w.skill for w in ats_result.weak_skills],
            ats_score=ats_score_before
        )
        
        if not suggestions:
            flash("No modifications needed - your resume is well-matched!")
            return redirect(url_for("enhance_resume"))
        
        # Store suggestions in session for approval
        session['enhancement_data'] = {
            'original_path': original_path,
            'filename': filename,
            'suggestions': [
                {
                    'section': s.section,
                    'original': s.original_text,
                    'suggested': s.suggested_text,
                    'reason': s.reason,
                    'impact': s.impact_score
                }
                for s in suggestions
            ],
            'ats_score_before': ats_score_before,
            'job_description': job_description
        }
        
        return render_template(
            "review_enhancements.html",
            user=user,
            suggestions=suggestions,
            ats_score_before=ats_score_before,
            job_description=job_description
        )
    
    except Exception as e:
        print(f"Error in resume enhancement: {e}")
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
    
    # Get approved suggestions (user can select which ones to apply)
    approved_indices = request.form.getlist('approved_suggestions')
    
    try:
        # Reconstruct suggestion objects
        from utils.resume_editor import ModificationSuggestion
        all_suggestions = [
            ModificationSuggestion(
                section=s['section'],
                original_text=s['original'],
                suggested_text=s['suggested'],
                reason=s['reason'],
                impact_score=s['impact']
            )
            for s in enhancement_data['suggestions']
        ]
        
        # Filter approved suggestions
        approved_suggestions = [
            all_suggestions[int(i)] for i in approved_indices
        ] if approved_indices else all_suggestions
        
        # Apply modifications
        modified_filename = f"enhanced_{enhancement_data['filename']}"
        modified_path = os.path.join(app.config["UPLOAD_FOLDER"], modified_filename)
        
        success, error_msg = resume_editor.apply_modifications_to_docx(
            enhancement_data['original_path'],
            approved_suggestions,
            modified_path
        )
        
        if not success:
            flash(f"Error applying modifications: {error_msg}")
            return redirect(url_for("enhance_resume"))
        
        # Save to database
        db.create_resume_modifications_table()
        db.save_resume_modification(
            user_id=user['id'],
            original_path=enhancement_data['original_path'],
            modified_path=modified_path,
            modifications=[s.__dict__ for s in approved_suggestions],
            score_before=enhancement_data['ats_score_before'],
            score_after=None,  # TODO: Re-run ATS on modified resume
            job_desc=enhancement_data['job_description']
        )
        
        # Clear session
        session.pop('enhancement_data', None)
        
        flash("Resume enhanced successfully! Download your improved resume below.")
        return send_from_directory(
            app.config["UPLOAD_FOLDER"],
            modified_filename,
            as_attachment=True,
            download_name=f"enhanced_{user['username']}_resume.docx"
        )
    
    except Exception as e:
        print(f"Error applying enhancements: {e}")
        flash(f"Error applying enhancements: {str(e)}")
        return redirect(url_for("enhance_resume"))
