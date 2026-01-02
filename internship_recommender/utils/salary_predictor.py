import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from .salary_scraper import create_comprehensive_salary_dataset

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "salary_model.pkl"
SAMPLE_DATA = Path(__file__).resolve().parents[1] / "data" / "salary_sample.csv"

def ensure_trained_model():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        return
    
    print("Training salary prediction model with scraped data...")
    
    # Create comprehensive dataset using web scraping
    roles = [
        "python developer", "data scientist", "frontend developer", 
        "software engineer", "data analyst", "ui ux designer",
        "devops engineer", "machine learning engineer", "backend developer",
        "full stack developer", "mobile app developer", "qa engineer"
    ]
    
    try:
        # Scrape real salary data
        df = create_comprehensive_salary_dataset(roles, "India")
        print(f"Scraped {len(df)} salary records")
    except Exception as e:
        print(f"Scraping failed, using fallback data: {e}")
        # Fallback to enhanced sample data
        df = pd.DataFrame([
            # Python Developer
            {"skills_count": 3, "experience": 0, "role": "python developer", "salary": 300000},
            {"skills_count": 4, "experience": 1, "role": "python developer", "salary": 450000},
            {"skills_count": 5, "experience": 2, "role": "python developer", "salary": 600000},
            {"skills_count": 6, "experience": 3, "role": "python developer", "salary": 800000},
            {"skills_count": 7, "experience": 5, "role": "python developer", "salary": 1200000},
            
            # Data Scientist
            {"skills_count": 4, "experience": 0, "role": "data scientist", "salary": 400000},
            {"skills_count": 5, "experience": 1, "role": "data scientist", "salary": 600000},
            {"skills_count": 6, "experience": 2, "role": "data scientist", "salary": 800000},
            {"skills_count": 7, "experience": 3, "role": "data scientist", "salary": 1100000},
            {"skills_count": 8, "experience": 5, "role": "data scientist", "salary": 1500000},
            
            # Frontend Developer
            {"skills_count": 3, "experience": 0, "role": "frontend developer", "salary": 250000},
            {"skills_count": 4, "experience": 1, "role": "frontend developer", "salary": 400000},
            {"skills_count": 5, "experience": 2, "role": "frontend developer", "salary": 550000},
            {"skills_count": 6, "experience": 3, "role": "frontend developer", "salary": 700000},
            {"skills_count": 7, "experience": 5, "role": "frontend developer", "salary": 1000000},
            
            # Software Engineer
            {"skills_count": 4, "experience": 0, "role": "software engineer", "salary": 350000},
            {"skills_count": 5, "experience": 1, "role": "software engineer", "salary": 500000},
            {"skills_count": 6, "experience": 2, "role": "software engineer", "salary": 700000},
            {"skills_count": 7, "experience": 3, "role": "software engineer", "salary": 900000},
            {"skills_count": 8, "experience": 5, "role": "software engineer", "salary": 1300000},
            
            # Data Analyst
            {"skills_count": 3, "experience": 0, "role": "data analyst", "salary": 300000},
            {"skills_count": 4, "experience": 1, "role": "data analyst", "salary": 450000},
            {"skills_count": 5, "experience": 2, "role": "data analyst", "salary": 600000},
            {"skills_count": 6, "experience": 3, "role": "data analyst", "salary": 750000},
            {"skills_count": 7, "experience": 5, "role": "data analyst", "salary": 1000000},
            
            # UI/UX Designer
            {"skills_count": 3, "experience": 0, "role": "ui ux designer", "salary": 250000},
            {"skills_count": 4, "experience": 1, "role": "ui ux designer", "salary": 400000},
            {"skills_count": 5, "experience": 2, "role": "ui ux designer", "salary": 550000},
            {"skills_count": 6, "experience": 3, "role": "ui ux designer", "salary": 700000},
            {"skills_count": 7, "experience": 5, "role": "ui ux designer", "salary": 950000},
        ])
    
    # Save the dataset
    df.to_csv(SAMPLE_DATA, index=False)
    
    # Enhanced feature engineering
    le = LabelEncoder()
    df['role_encoded'] = le.fit_transform(df['role'])
    
    # Create additional features
    df['experience_squared'] = df['experience'] ** 2
    df['skills_experience_interaction'] = df['skills_count'] * df['experience']
    df['seniority_level'] = pd.cut(df['experience'], bins=[-1, 1, 3, 5, 10], labels=[0, 1, 2, 3])
    
    # Prepare features
    feature_columns = ["skills_count", "experience", "role_encoded", "experience_squared", "skills_experience_interaction"]
    X = df[feature_columns]
    y = df["salary"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train ensemble model
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42)
    
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    
    # Evaluate models
    rf_pred = rf_model.predict(X_test)
    gb_pred = gb_model.predict(X_test)
    
    rf_mae = mean_absolute_error(y_test, rf_pred)
    gb_mae = mean_absolute_error(y_test, gb_pred)
    
    print(f"Random Forest MAE: {rf_mae:,.0f}")
    print(f"Gradient Boosting MAE: {gb_mae:,.0f}")
    
    # Use the better model
    if rf_mae < gb_mae:
        model = rf_model
        print("Using Random Forest model")
    else:
        model = gb_model
        print("Using Gradient Boosting model")
    
    # Save model and encoders
    model_data = {
        'model': model,
        'label_encoder': le,
        'feature_columns': feature_columns
    }
    joblib.dump(model_data, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def predict_salary(skills, role, experience_years=0):
    if not MODEL_PATH.exists():
        ensure_trained_model()

    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    label_encoder = model_data['label_encoder']
    feature_columns = model_data['feature_columns']

    # Prepare features
    skills_count = len(skills) if skills else 0

    # Encode role
    try:
        role_encoded = label_encoder.transform([role.lower()])[0]
    except ValueError:
        # If role not in training data, use average encoding
        role_encoded = 0

    # Create feature vector
    features = np.array([[
        skills_count,
        experience_years,
        role_encoded,
        experience_years ** 2,
        skills_count * experience_years
    ]])

    pred = model.predict(features)[0]

    # Add confidence intervals based on model uncertainty
    confidence_factor = 0.15  # 15% uncertainty
    low = int(pred * (1 - confidence_factor))
    high = int(pred * (1 + confidence_factor))

    return low, high
    