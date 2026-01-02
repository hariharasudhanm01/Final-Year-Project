import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from typing import List, Dict, Tuple
import re

def scrape_glassdoor_salaries(role: str, location: str = "India") -> List[Dict]:
    """
    Scrape salary data from Glassdoor (mock implementation for demo).
    In production, you'd need proper API access or web scraping with proper headers.
    """
    try:
        # Mock data based on real salary ranges for Indian market
        mock_salaries = {
            "python developer": [
                {"experience": 0, "salary": 300000, "company_size": "startup"},
                {"experience": 1, "salary": 450000, "company_size": "startup"},
                {"experience": 2, "salary": 600000, "company_size": "mid"},
                {"experience": 3, "salary": 800000, "company_size": "mid"},
                {"experience": 5, "salary": 1200000, "company_size": "large"}
            ],
            "data scientist": [
                {"experience": 0, "salary": 400000, "company_size": "startup"},
                {"experience": 1, "salary": 600000, "company_size": "startup"},
                {"experience": 2, "salary": 800000, "company_size": "mid"},
                {"experience": 3, "salary": 1100000, "company_size": "mid"},
                {"experience": 5, "salary": 1500000, "company_size": "large"}
            ],
            "frontend developer": [
                {"experience": 0, "salary": 250000, "company_size": "startup"},
                {"experience": 1, "salary": 400000, "company_size": "startup"},
                {"experience": 2, "salary": 550000, "company_size": "mid"},
                {"experience": 3, "salary": 700000, "company_size": "mid"},
                {"experience": 5, "salary": 1000000, "company_size": "large"}
            ],
            "software engineer": [
                {"experience": 0, "salary": 350000, "company_size": "startup"},
                {"experience": 1, "salary": 500000, "company_size": "startup"},
                {"experience": 2, "salary": 700000, "company_size": "mid"},
                {"experience": 3, "salary": 900000, "company_size": "mid"},
                {"experience": 5, "salary": 1300000, "company_size": "large"}
            ],
            "data analyst": [
                {"experience": 0, "salary": 300000, "company_size": "startup"},
                {"experience": 1, "salary": 450000, "company_size": "startup"},
                {"experience": 2, "salary": 600000, "company_size": "mid"},
                {"experience": 3, "salary": 750000, "company_size": "mid"},
                {"experience": 5, "salary": 1000000, "company_size": "large"}
            ],
            "ui ux designer": [
                {"experience": 0, "salary": 250000, "company_size": "startup"},
                {"experience": 1, "salary": 400000, "company_size": "startup"},
                {"experience": 2, "salary": 550000, "company_size": "mid"},
                {"experience": 3, "salary": 700000, "company_size": "mid"},
                {"experience": 5, "salary": 950000, "company_size": "large"}
            ],
            "devops engineer": [
                {"experience": 0, "salary": 400000, "company_size": "startup"},
                {"experience": 1, "salary": 600000, "company_size": "startup"},
                {"experience": 2, "salary": 800000, "company_size": "mid"},
                {"experience": 3, "salary": 1000000, "company_size": "mid"},
                {"experience": 5, "salary": 1400000, "company_size": "large"}
            ],
            "machine learning engineer": [
                {"experience": 0, "salary": 500000, "company_size": "startup"},
                {"experience": 1, "salary": 700000, "company_size": "startup"},
                {"experience": 2, "salary": 900000, "company_size": "mid"},
                {"experience": 3, "salary": 1200000, "company_size": "mid"},
                {"experience": 5, "salary": 1600000, "company_size": "large"}
            ]
        }
        
        role_lower = role.lower()
        for key, salaries in mock_salaries.items():
            if key in role_lower or role_lower in key:
                return salaries
        
        # Default fallback
        return [
            {"experience": 0, "salary": 300000, "company_size": "startup"},
            {"experience": 1, "salary": 450000, "company_size": "startup"},
            {"experience": 2, "salary": 600000, "company_size": "mid"},
            {"experience": 3, "salary": 800000, "company_size": "mid"},
            {"experience": 5, "salary": 1100000, "company_size": "large"}
        ]
        
    except Exception as e:
        print(f"Error scraping salaries: {e}")
        return []

def scrape_payscale_salaries(role: str, location: str = "India") -> List[Dict]:
    """
    Scrape salary data from PayScale (mock implementation).
    """
    try:
        # Simulate API delay
        time.sleep(random.uniform(1, 2))
        
        # Mock data with more granular experience levels
        base_salaries = {
            "python developer": 350000,
            "data scientist": 450000,
            "frontend developer": 300000,
            "software engineer": 400000,
            "data analyst": 350000,
            "ui ux designer": 300000,
            "devops engineer": 450000,
            "machine learning engineer": 500000
        }
        
        role_lower = role.lower()
        base_salary = 350000  # default
        
        for key, salary in base_salaries.items():
            if key in role_lower or role_lower in key:
                base_salary = salary
                break
        
        # Generate salary progression
        salaries = []
        for exp in range(0, 8):
            multiplier = 1 + (exp * 0.15) + (exp * exp * 0.02)  # exponential growth
            salary = int(base_salary * multiplier)
            salaries.append({
                "experience": exp,
                "salary": salary,
                "source": "payscale"
            })
        
        return salaries
        
    except Exception as e:
        print(f"Error scraping PayScale: {e}")
        return []

def scrape_indeed_salaries(role: str, location: str = "India") -> List[Dict]:
    """
    Scrape salary data from Indeed (mock implementation).
    """
    try:
        time.sleep(random.uniform(0.5, 1.5))
        
        # Mock Indeed salary data with company size variations
        company_multipliers = {
            "startup": 0.8,
            "mid": 1.0,
            "large": 1.3,
            "faang": 1.8
        }
        
        base_salaries = {
            "python developer": 400000,
            "data scientist": 500000,
            "frontend developer": 350000,
            "software engineer": 450000,
            "data analyst": 400000,
            "ui ux designer": 350000,
            "devops engineer": 500000,
            "machine learning engineer": 550000
        }
        
        role_lower = role.lower()
        base_salary = 400000  # default
        
        for key, salary in base_salaries.items():
            if key in role_lower or role_lower in key:
                base_salary = salary
                break
        
        salaries = []
        for exp in range(0, 6):
            for company_type, multiplier in company_multipliers.items():
                exp_multiplier = 1 + (exp * 0.2)
                salary = int(base_salary * exp_multiplier * multiplier)
                salaries.append({
                    "experience": exp,
                    "salary": salary,
                    "company_size": company_type,
                    "source": "indeed"
                })
        
        return salaries
        
    except Exception as e:
        print(f"Error scraping Indeed: {e}")
        return []

def aggregate_salary_data(role: str, location: str = "India") -> pd.DataFrame:
    """
    Aggregate salary data from multiple sources and create a comprehensive dataset.
    """
    all_salaries = []
    
    # Scrape from multiple sources
    sources = [
        scrape_glassdoor_salaries(role, location),
        scrape_payscale_salaries(role, location),
        scrape_indeed_salaries(role, location)
    ]
    
    for source_data in sources:
        all_salaries.extend(source_data)
    
    if not all_salaries:
        # Fallback to default data
        all_salaries = [
            {"experience": 0, "salary": 300000, "company_size": "startup"},
            {"experience": 1, "salary": 450000, "company_size": "startup"},
            {"experience": 2, "salary": 600000, "company_size": "mid"},
            {"experience": 3, "salary": 800000, "company_size": "mid"},
            {"experience": 5, "salary": 1100000, "company_size": "large"}
        ]
    
    # Convert to DataFrame
    df = pd.DataFrame(all_salaries)
    
    # Add derived features
    df['skills_count'] = len(role.split()) + 3  # Mock skills count based on role complexity
    df['location_factor'] = 1.0 if location.lower() in ['bangalore', 'mumbai', 'delhi'] else 0.8
    df['company_size_numeric'] = df['company_size'].map({
        'startup': 1, 'mid': 2, 'large': 3, 'faang': 4
    }).fillna(2)
    
    return df

def create_comprehensive_salary_dataset(roles: List[str], location: str = "India") -> pd.DataFrame:
    """
    Create a comprehensive salary dataset by scraping multiple roles.
    """
    all_data = []
    
    for role in roles:
        print(f"Scraping salary data for: {role}")
        role_data = aggregate_salary_data(role, location)
        role_data['role'] = role
        all_data.append(role_data)
        time.sleep(1)  # Rate limiting
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    else:
        # Fallback comprehensive dataset
        return pd.DataFrame([
            {"experience": 0, "salary": 300000, "skills_count": 3, "role": "python developer"},
            {"experience": 1, "salary": 450000, "skills_count": 4, "role": "python developer"},
            {"experience": 2, "salary": 600000, "skills_count": 5, "role": "python developer"},
            {"experience": 0, "salary": 400000, "skills_count": 4, "role": "data scientist"},
            {"experience": 1, "salary": 600000, "skills_count": 5, "role": "data scientist"},
            {"experience": 2, "salary": 800000, "skills_count": 6, "role": "data scientist"},
            {"experience": 0, "salary": 250000, "skills_count": 3, "role": "frontend developer"},
            {"experience": 1, "salary": 400000, "skills_count": 4, "role": "frontend developer"},
            {"experience": 2, "salary": 550000, "skills_count": 5, "role": "frontend developer"},
        ])

if __name__ == "__main__":
    # Test the scraper
    roles = ["python developer", "data scientist", "frontend developer"]
    df = create_comprehensive_salary_dataset(roles)
    print("Sample salary data:")
    print(df.head())
    print(f"\nTotal records: {len(df)}")


