import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from typing import List, Dict, Tuple
import re
from ddgs import DDGS

def extract_salary_from_text(text: str) -> List[int]:
    """
    Extract salary amounts from text using regex patterns.
    Returns list of salary values in INR (converts from various formats).
    """
    salaries = []
    text_lower = text.lower()
    
    # Pattern 1: ₹X Lakhs, ₹X LPA, ₹X per annum
    patterns = [
        r'₹\s*(\d+(?:\.\d+)?)\s*(?:lakh|lpa|lakhs|per\s*annum)',
        r'(\d+(?:\.\d+)?)\s*(?:lakh|lpa|lakhs)\s*(?:per\s*annum|p\.?a\.?)',
        r'₹\s*(\d{1,2}(?:,\d{3})*(?:\.\d+)?)\s*(?:per\s*annum|p\.?a\.?|lpa)',
        r'(\d{1,2}(?:,\d{3})*(?:\.\d+)?)\s*(?:lpa|lakh|per\s*annum)',
        # Direct numbers in lakhs range (3-50 lakhs)
        r'(\d{1,2}(?:\.\d+)?)\s*(?:lakh|lpa)',
        # Full numbers (300000, 5,00,000, etc.)
        r'₹?\s*(\d{1,2}(?:,\d{2}){0,2}(?:,\d{3})*)\s*(?:per\s*annum|p\.?a\.?|lpa|inr)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            try:
                # Remove commas and convert to float
                num_str = match.replace(',', '')
                num = float(num_str)
                
                # If number is less than 100, assume it's in lakhs
                if num < 100:
                    salary = int(num * 100000)  # Convert lakhs to rupees
                else:
                    salary = int(num)
                
                # Filter reasonable salary ranges (1L to 50L for India)
                if 100000 <= salary <= 5000000:
                    salaries.append(salary)
            except (ValueError, AttributeError):
                continue
    
    return salaries

def scrape_salary_from_ddg(role: str, location: str = "India") -> List[Dict]:
    """
    Search for salary information using DuckDuckGo and extract salary data.
    """
    salaries = []
    try:
        queries = [
            f"{role} salary {location}",
            f"{role} average salary {location}",
            f"{role} salary range {location}",
            f"{role} fresher salary {location}",
            f"{role} experience salary {location}"
        ]
        
        with DDGS() as ddgs:
            for query in queries:
                try:
                    results = list(ddgs.text(query, max_results=10))
                    for result in results:
                        text = f"{result.get('title', '')} {result.get('body', '')}"
                        extracted_salaries = extract_salary_from_text(text)
                        
                        for salary in extracted_salaries:
                            # Estimate experience based on query context
                            exp = 0
                            if 'fresher' in query.lower() or 'entry' in query.lower():
                                exp = 0
                            elif 'senior' in query.lower() or 'experienced' in query.lower():
                                exp = 5
                            elif 'mid' in query.lower() or '2-3' in query.lower():
                                exp = 2
                            else:
                                exp = random.choice([0, 1, 2, 3, 5])
                            
                            salaries.append({
                                "experience": exp,
                                "salary": salary,
                                "source": "ddg_search"
                            })
                    
                    time.sleep(1)  # Rate limiting
                except Exception as e:
                    print(f"Error in DDG query '{query}': {e}")
                    continue
                    
    except Exception as e:
        print(f"Error in DDG salary search: {e}")
    
    return salaries

def scrape_ambitionbox_salaries(role: str, location: str = "India") -> List[Dict]:
    """
    Scrape salary data from AmbitionBox (popular Indian salary site).
    """
    salaries = []
    try:
        # AmbitionBox URL structure
        role_encoded = role.replace(' ', '-').lower()
        url = f"https://www.ambitionbox.com/salaries/{role_encoded}-salaries"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for salary information in various formats
            # AmbitionBox typically shows salary ranges
            salary_elements = soup.find_all(text=re.compile(r'₹?\s*\d+.*lakh|lpa|per\s*annum', re.I))
            
            for elem in salary_elements:
                extracted = extract_salary_from_text(str(elem))
                for salary in extracted:
                    salaries.append({
                        "experience": 2,  # Default, will be refined
                        "salary": salary,
                        "source": "ambitionbox"
                    })
        
        time.sleep(2)  # Rate limiting
        
    except Exception as e:
        print(f"Error scraping AmbitionBox: {e}")
    
    return salaries

def scrape_naukri_salary_insights(role: str, location: str = "India") -> List[Dict]:
    """
    Search Naukri.com for salary insights using DuckDuckGo.
    """
    salaries = []
    try:
        query = f"site:naukri.com {role} salary {location}"
        
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            for result in results:
                text = f"{result.get('title', '')} {result.get('body', '')}"
                extracted = extract_salary_from_text(text)
                
                for salary in extracted:
                    salaries.append({
                        "experience": 2,
                        "salary": salary,
                        "source": "naukri"
                    })
        
        time.sleep(1)
        
    except Exception as e:
        print(f"Error scraping Naukri insights: {e}")
    
    return salaries

def scrape_glassdoor_salaries(role: str, location: str = "India") -> List[Dict]:
    """
    Attempt to scrape salary data from Glassdoor (with proper headers and error handling).
    Falls back to search-based approach if direct scraping fails.
    """
    salaries = []
    
    try:
        # Try direct search approach first
        query = f"site:glassdoor.com {role} salary {location}"
        
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            for result in results:
                text = f"{result.get('title', '')} {result.get('body', '')}"
                extracted = extract_salary_from_text(text)
                
                for salary in extracted:
                    salaries.append({
                        "experience": 2,
                        "salary": salary,
                        "source": "glassdoor"
                    })
        
        time.sleep(1)
        
    except Exception as e:
        print(f"Error scraping Glassdoor: {e}")
    
    return salaries

def scrape_payscale_salaries(role: str, location: str = "India") -> List[Dict]:
    """
    Scrape salary data from PayScale using search-based approach.
    """
    salaries = []
    try:
        query = f"site:payscale.com {role} salary {location}"
        
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            for result in results:
                text = f"{result.get('title', '')} {result.get('body', '')}"
                extracted = extract_salary_from_text(text)
                
                for salary in extracted:
                    salaries.append({
                        "experience": 2,
                        "salary": salary,
                        "source": "payscale"
                    })
        
        time.sleep(1)
        
    except Exception as e:
        print(f"Error scraping PayScale: {e}")
    
    return salaries

def scrape_indeed_salaries(role: str, location: str = "India") -> List[Dict]:
    """
    Scrape salary data from Indeed job postings using search.
    """
    salaries = []
    try:
        # Search for job postings with salary information
        queries = [
            f"site:indeed.co.in {role} salary {location}",
            f"site:indeed.com {role} salary {location}"
        ]
        
        with DDGS() as ddgs:
            for query in queries:
                try:
                    results = list(ddgs.text(query, max_results=8))
                    for result in results:
                        text = f"{result.get('title', '')} {result.get('body', '')}"
                        extracted = extract_salary_from_text(text)
                        
                        for salary in extracted:
                            # Try to infer experience from job title
                            exp = 0
                            title_lower = result.get('title', '').lower()
                            if 'fresher' in title_lower or 'entry' in title_lower or 'intern' in title_lower:
                                exp = 0
                            elif 'senior' in title_lower or 'lead' in title_lower or 'principal' in title_lower:
                                exp = 5
                            elif 'mid' in title_lower or '2+' in title_lower or '3+' in title_lower:
                                exp = 2
                            else:
                                exp = random.choice([0, 1, 2, 3])
                            
                            salaries.append({
                                "experience": exp,
                                "salary": salary,
                                "source": "indeed"
                            })
                    
                    time.sleep(1)
                except Exception as e:
                    print(f"Error in Indeed query: {e}")
                    continue
        
    except Exception as e:
        print(f"Error scraping Indeed: {e}")
    
    return salaries

def aggregate_salary_data(role: str, location: str = "India") -> pd.DataFrame:
    """
    Aggregate salary data from multiple real-world sources and create a comprehensive dataset.
    """
    all_salaries = []
    
    print(f"  Scraping salary data for {role} in {location}...")
    
    # Scrape from multiple sources
    sources = [
        ("DuckDuckGo Search", scrape_salary_from_ddg),
        ("AmbitionBox", scrape_ambitionbox_salaries),
        ("Naukri", scrape_naukri_salary_insights),
        ("Glassdoor", scrape_glassdoor_salaries),
        ("PayScale", scrape_payscale_salaries),
        ("Indeed", scrape_indeed_salaries),
    ]
    
    for source_name, scrape_func in sources:
        try:
            print(f"    Trying {source_name}...")
            source_data = scrape_func(role, location)
            if source_data:
                all_salaries.extend(source_data)
                print(f"    Found {len(source_data)} salary records from {source_name}")
        except Exception as e:
            print(f"    Error with {source_name}: {e}")
            continue
    
    # If we got real data, use it; otherwise fall back
    if not all_salaries:
        print(f"    No real salary data found, using fallback data")
        # Fallback to realistic default data based on role
        all_salaries = get_fallback_salary_data(role)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_salaries)
    
    if df.empty:
        df = pd.DataFrame(get_fallback_salary_data(role))
    
    # Clean and process the data
    # Remove outliers (salaries outside reasonable range for India: 1L to 50L)
    df = df[(df['salary'] >= 100000) & (df['salary'] <= 5000000)]
    
    # Group by experience and calculate average salaries
    if 'experience' in df.columns:
        # Fill missing experience with median
        df['experience'] = df['experience'].fillna(df['experience'].median())
        df['experience'] = df['experience'].astype(int)
    else:
        df['experience'] = 2  # Default experience
    
    # Add derived features
    df['skills_count'] = len(role.split()) + 3  # Estimate based on role complexity
    df['location_factor'] = 1.0 if location.lower() in ['bangalore', 'mumbai', 'delhi', 'hyderabad', 'pune'] else 0.85
    
    # Add company_size if missing
    if 'company_size' not in df.columns:
        df['company_size'] = 'mid'  # Default
    
    df['company_size_numeric'] = df['company_size'].map({
        'startup': 1, 'mid': 2, 'large': 3, 'faang': 4
    }).fillna(2)
    
    return df

def get_fallback_salary_data(role: str) -> List[Dict]:
    """
    Get fallback salary data based on role type (used when scraping fails).
    These are realistic estimates for Indian market.
    """
    role_lower = role.lower()
    
    # Base salary ranges by role category
    role_salaries = {
        "python developer": {"base": 350000, "growth": 0.25},
        "data scientist": {"base": 450000, "growth": 0.30},
        "frontend developer": {"base": 300000, "growth": 0.22},
        "software engineer": {"base": 400000, "growth": 0.25},
        "data analyst": {"base": 350000, "growth": 0.23},
        "ui ux designer": {"base": 300000, "growth": 0.22},
        "devops engineer": {"base": 450000, "growth": 0.28},
        "machine learning engineer": {"base": 500000, "growth": 0.30},
        "backend developer": {"base": 400000, "growth": 0.25},
        "full stack developer": {"base": 450000, "growth": 0.27},
        "mobile app developer": {"base": 380000, "growth": 0.24},
        "qa engineer": {"base": 320000, "growth": 0.22},
    }
    
    # Find matching role
    base_salary = 350000
    growth_rate = 0.25
    
    for key, values in role_salaries.items():
        if key in role_lower or role_lower in key:
            base_salary = values["base"]
            growth_rate = values["growth"]
            break
    
    # Generate salary progression
    salaries = []
    for exp in [0, 1, 2, 3, 5]:
        multiplier = 1 + (exp * growth_rate) + (exp * exp * 0.02)
        salary = int(base_salary * multiplier)
        salaries.append({
            "experience": exp,
            "salary": salary,
            "company_size": "mid",
            "source": "fallback"
        })
    
    return salaries

def create_comprehensive_salary_dataset(roles: List[str], location: str = "India") -> pd.DataFrame:
    """
    Create a comprehensive salary dataset by scraping real salary data for multiple roles.
    """
    all_data = []
    
    print(f"\n=== Starting Real Salary Data Collection ===")
    print(f"Location: {location}")
    print(f"Roles to scrape: {len(roles)}\n")
    
    for i, role in enumerate(roles, 1):
        print(f"[{i}/{len(roles)}] Processing: {role}")
        try:
            role_data = aggregate_salary_data(role, location)
            if not role_data.empty:
                role_data['role'] = role
                all_data.append(role_data)
                print(f"  ✓ Collected {len(role_data)} salary records for {role}\n")
            else:
                print(f"  ⚠ No data collected for {role}, using fallback\n")
        except Exception as e:
            print(f"  ✗ Error processing {role}: {e}\n")
            continue
        
        # Rate limiting between roles
        if i < len(roles):
            time.sleep(2)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n=== Collection Complete ===")
        print(f"Total salary records collected: {len(combined_df)}")
        print(f"Unique roles: {combined_df['role'].nunique()}")
        print(f"Experience levels: {sorted(combined_df['experience'].unique())}")
        print(f"Salary range: ₹{combined_df['salary'].min():,} - ₹{combined_df['salary'].max():,}\n")
        return combined_df
    else:
        print("\n⚠ No data collected from any source, using comprehensive fallback dataset")
        # Comprehensive fallback dataset
        fallback_data = []
        for role in roles:
            fallback_data.extend(get_fallback_salary_data(role))
        
        df = pd.DataFrame(fallback_data)
        for role in roles:
            if role not in df['role'].values if 'role' in df.columns else True:
                role_fallback = get_fallback_salary_data(role)
                for entry in role_fallback:
                    entry['role'] = role
                fallback_data.extend(role_fallback)
        
        return pd.DataFrame(fallback_data)

if __name__ == "__main__":
    # Test the scraper
    roles = ["python developer", "data scientist", "frontend developer"]
    df = create_comprehensive_salary_dataset(roles)
    print("Sample salary data:")
    print(df.head())
    print(f"\nTotal records: {len(df)}")


