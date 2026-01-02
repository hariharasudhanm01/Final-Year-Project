import requests
import json
from pathlib import Path
from typing import List, Dict, Tuple
import time

# Course data structure
COURSE_DATA = {
    "Python": {
        "free": [
            {
                "title": "Python for Beginners - Complete Course",
                "platform": "YouTube",
                "url": "https://www.youtube.com/watch?v=kqtD5dpn9C8",
                "duration": "4.5 hours",
                "rating": 4.8
            },
            {
                "title": "Python Crash Course for Beginners",
                "platform": "freeCodeCamp",
                "url": "https://www.freecodecamp.org/news/python-crash-course/",
                "duration": "3 hours",
                "rating": 4.7
            }
        ],
        "paid": [
            {
                "title": "Complete Python Bootcamp",
                "platform": "Udemy",
                "url": "https://www.udemy.com/course/complete-python-bootcamp/",
                "price": "$89.99",
                "duration": "22 hours",
                "rating": 4.6
            },
            {
                "title": "Python for Data Science and Machine Learning",
                "platform": "Coursera",
                "url": "https://www.coursera.org/learn/python-for-data-science",
                "price": "$49/month",
                "duration": "6 weeks",
                "rating": 4.5
            }
        ]
    },
    "SQL": {
        "free": [
            {
                "title": "SQL Tutorial for Beginners",
                "platform": "YouTube",
                "url": "https://www.youtube.com/watch?v=HXV3zeQKqGY",
                "duration": "4.5 hours",
                "rating": 4.8
            },
            {
                "title": "Learn SQL with W3Schools",
                "platform": "W3Schools",
                "url": "https://www.w3schools.com/sql/",
                "duration": "Self-paced",
                "rating": 4.5
            }
        ],
        "paid": [
            {
                "title": "The Complete SQL Bootcamp",
                "platform": "Udemy",
                "url": "https://www.udemy.com/course/the-complete-sql-bootcamp/",
                "price": "$89.99",
                "duration": "9 hours",
                "rating": 4.7
            }
        ]
    },
    "Machine Learning": {
        "free": [
            {
                "title": "Machine Learning Course for Beginners",
                "platform": "YouTube",
                "url": "https://www.youtube.com/watch?v=NWONeJKn6kc",
                "duration": "11.5 hours",
                "rating": 4.8
            },
            {
                "title": "Machine Learning Crash Course",
                "platform": "Google",
                "url": "https://developers.google.com/machine-learning/crash-course",
                "duration": "15 hours",
                "rating": 4.9
            }
        ],
        "paid": [
            {
                "title": "Machine Learning A-Z",
                "platform": "Udemy",
                "url": "https://www.udemy.com/course/machinelearning/",
                "price": "$89.99",
                "duration": "40 hours",
                "rating": 4.6
            },
            {
                "title": "Machine Learning Specialization",
                "platform": "Coursera",
                "url": "https://www.coursera.org/specializations/machine-learning-introduction",
                "price": "$49/month",
                "duration": "3 months",
                "rating": 4.8
            }
        ]
    },
    "React": {
        "free": [
            {
                "title": "React Tutorial for Beginners",
                "platform": "YouTube",
                "url": "https://www.youtube.com/watch?v=DLX62G4lc44",
                "duration": "8 hours",
                "rating": 4.7
            },
            {
                "title": "React Official Tutorial",
                "platform": "React Docs",
                "url": "https://react.dev/learn",
                "duration": "Self-paced",
                "rating": 4.9
            }
        ],
        "paid": [
            {
                "title": "React - The Complete Guide",
                "platform": "Udemy",
                "url": "https://www.udemy.com/course/react-the-complete-guide-incl-redux/",
                "price": "$89.99",
                "duration": "48 hours",
                "rating": 4.7
            }
        ]
    },
    "Docker": {
        "free": [
            {
                "title": "Docker Tutorial for Beginners",
                "platform": "YouTube",
                "url": "https://www.youtube.com/watch?v=pTFZFxd4hOI",
                "duration": "2 hours",
                "rating": 4.6
            }
        ],
        "paid": [
            {
                "title": "Docker and Kubernetes: The Complete Guide",
                "platform": "Udemy",
                "url": "https://www.udemy.com/course/docker-and-kubernetes-the-complete-guide/",
                "price": "$89.99",
                "duration": "20 hours",
                "rating": 4.7
            }
        ]
    },
    "AWS": {
        "free": [
            {
                "title": "AWS Free Tier Tutorial",
                "platform": "YouTube",
                "url": "https://www.youtube.com/watch?v=ulprqHHWlng",
                "duration": "3 hours",
                "rating": 4.5
            }
        ],
        "paid": [
            {
                "title": "AWS Certified Solutions Architect",
                "platform": "Udemy",
                "url": "https://www.udemy.com/course/aws-certified-solutions-architect-associate/",
                "price": "$89.99",
                "duration": "27 hours",
                "rating": 4.6
            }
        ]
    },
    "Figma": {
        "free": [
            {
                "title": "Figma Tutorial for Beginners",
                "platform": "YouTube",
                "url": "https://www.youtube.com/watch?v=FTqijk6T9Gc",
                "duration": "2.5 hours",
                "rating": 4.7
            }
        ],
        "paid": [
            {
                "title": "Figma UI/UX Design Essentials",
                "platform": "Udemy",
                "url": "https://www.udemy.com/course/figma-ui-ux-design-essentials/",
                "price": "$89.99",
                "duration": "12 hours",
                "rating": 4.6
            }
        ]
    },
    "Git": {
        "free": [
            {
                "title": "Git Tutorial for Beginners",
                "platform": "YouTube",
                "url": "https://www.youtube.com/watch?v=8JJ101D3knE",
                "duration": "1.5 hours",
                "rating": 4.8
            }
        ],
        "paid": [
            {
                "title": "Git Complete: The definitive, step-by-step guide",
                "platform": "Udemy",
                "url": "https://www.udemy.com/course/git-complete/",
                "price": "$89.99",
                "duration": "6.5 hours",
                "rating": 4.7
            }
        ]
    },
    "JavaScript": {
        "free": [
            {
                "title": "JavaScript Tutorial for Beginners",
                "platform": "YouTube",
                "url": "https://www.youtube.com/watch?v=W6NZfCO5SIk",
                "duration": "3.5 hours",
                "rating": 4.8
            }
        ],
        "paid": [
            {
                "title": "The Complete JavaScript Course 2024",
                "platform": "Udemy",
                "url": "https://www.udemy.com/course/the-complete-javascript-course/",
                "price": "$89.99",
                "duration": "69 hours",
                "rating": 4.7
            }
        ]
    },
    "HTML": {
        "free": [
            {
                "title": "HTML Tutorial for Beginners",
                "platform": "YouTube",
                "url": "https://www.youtube.com/watch?v=qz0aGYrrlhU",
                "duration": "2 hours",
                "rating": 4.7
            }
        ],
        "paid": [
            {
                "title": "HTML5 and CSS3 Fundamentals",
                "platform": "Udemy",
                "url": "https://www.udemy.com/course/html5-and-css3-fundamentals/",
                "price": "$89.99",
                "duration": "8 hours",
                "rating": 4.6
            }
        ]
    },
    "CSS": {
        "free": [
            {
                "title": "CSS Tutorial for Beginners",
                "platform": "YouTube",
                "url": "https://www.youtube.com/watch?v=1Rs2ND1ryYc",
                "duration": "2.5 hours",
                "rating": 4.7
            }
        ],
        "paid": [
            {
                "title": "CSS - The Complete Guide 2024",
                "platform": "Udemy",
                "url": "https://www.udemy.com/course/css-the-complete-guide-incl-flexbox-grid-sass/",
                "price": "$89.99",
                "duration": "22 hours",
                "rating": 4.7
            }
        ]
    },
    "Excel": {
        "free": [
            {
                "title": "Excel Tutorial for Beginners",
                "platform": "YouTube",
                "url": "https://www.youtube.com/watch?v=Vl0H-qTclOg",
                "duration": "1.5 hours",
                "rating": 4.6
            }
        ],
        "paid": [
            {
                "title": "Microsoft Excel - Excel from Beginner to Advanced",
                "platform": "Udemy",
                "url": "https://www.udemy.com/course/microsoft-excel-2013-from-beginner-to-advanced-and-beyond/",
                "price": "$89.99",
                "duration": "18 hours",
                "rating": 4.6
            }
        ]
    },
    "Tableau": {
        "free": [
            {
                "title": "Tableau Tutorial for Beginners",
                "platform": "YouTube",
                "url": "https://www.youtube.com/watch?v=6mBtTNbykVI",
                "duration": "2 hours",
                "rating": 4.5
            }
        ],
        "paid": [
            {
                "title": "Tableau 2024 A-Z: Hands-On Tableau Training for Data Science",
                "platform": "Udemy",
                "url": "https://www.udemy.com/course/tableau10/",
                "price": "$89.99",
                "duration": "9 hours",
                "rating": 4.6
            }
        ]
    },
    "Power BI": {
        "free": [
            {
                "title": "Power BI Tutorial for Beginners",
                "platform": "YouTube",
                "url": "https://www.youtube.com/watch?v=1cgs6TuWkBs",
                "duration": "2.5 hours",
                "rating": 4.6
            }
        ],
        "paid": [
            {
                "title": "Microsoft Power BI - The Complete Masterclass",
                "platform": "Udemy",
                "url": "https://www.udemy.com/course/microsoft-power-bi-up-running-with-power-bi-desktop/",
                "price": "$89.99",
                "duration": "10 hours",
                "rating": 4.7
            }
        ]
    }
}

def get_course_recommendations(skill: str) -> Dict:
    """
    Get course recommendations for a specific skill.
    Returns both free and paid options.
    """
    skill_lower = skill.lower()
    
    # Direct match
    if skill_lower in COURSE_DATA:
        return COURSE_DATA[skill_lower]
    
    # Fuzzy matching for similar skills
    for key, courses in COURSE_DATA.items():
        if skill_lower in key.lower() or key.lower() in skill_lower:
            return courses
    
    # Enhanced fallback with more comprehensive courses
    return {
        "free": [
            {
                "title": f"{skill} Tutorial for Beginners",
                "platform": "YouTube",
                "url": f"https://www.youtube.com/results?search_query={skill}+tutorial+beginner",
                "duration": "2-4 hours",
                "rating": 4.2
            },
            {
                "title": f"Learn {skill} - Free Course",
                "platform": "freeCodeCamp",
                "url": f"https://www.freecodecamp.org/news/{skill.lower().replace(' ', '-')}-tutorial/",
                "duration": "3-6 hours",
                "rating": 4.3
            },
            {
                "title": f"{skill} Documentation & Tutorials",
                "platform": "Official Docs",
                "url": f"https://www.google.com/search?q={skill}+official+documentation",
                "duration": "Self-paced",
                "rating": 4.5
            }
        ],
        "paid": [
            {
                "title": f"Complete {skill} Bootcamp 2024",
                "platform": "Udemy",
                "url": f"https://www.udemy.com/courses/search/?q={skill}",
                "price": "$89.99",
                "duration": "15-25 hours",
                "rating": 4.4
            },
            {
                "title": f"{skill} Specialization",
                "platform": "Coursera",
                "url": f"https://www.coursera.org/courses?query={skill}",
                "price": "$49/month",
                "duration": "2-4 months",
                "rating": 4.5
            },
            {
                "title": f"Professional {skill} Certification",
                "platform": "LinkedIn Learning",
                "url": f"https://www.linkedin.com/learning/search?keywords={skill}",
                "price": "$29.99/month",
                "duration": "10-20 hours",
                "rating": 4.3
            }
        ]
    }

def scrape_udemy_courses(skill: str, max_results: int = 3) -> List[Dict]:
    """
    Scrape Udemy courses for a specific skill.
    Note: This is a simplified version. Real implementation would need proper API access.
    """
    try:
        # Simulate API call delay
        time.sleep(0.5)
        
        # Mock data for demonstration
        mock_courses = [
            {
                "title": f"Complete {skill} Course - From Zero to Hero",
                "platform": "Udemy",
                "url": f"https://www.udemy.com/course/{skill.lower().replace(' ', '-')}-complete/",
                "price": "$89.99",
                "duration": "15 hours",
                "rating": 4.5,
                "students": "50,000+"
            },
            {
                "title": f"{skill} Bootcamp 2024",
                "platform": "Udemy", 
                "url": f"https://www.udemy.com/course/{skill.lower().replace(' ', '-')}-bootcamp/",
                "price": "$79.99",
                "duration": "20 hours",
                "rating": 4.6,
                "students": "30,000+"
            }
        ]
        return mock_courses[:max_results]
    except Exception as e:
        print(f"Error scraping Udemy courses: {e}")
        return []

def scrape_coursera_courses(skill: str, max_results: int = 3) -> List[Dict]:
    """
    Scrape Coursera courses for a specific skill.
    Note: This is a simplified version. Real implementation would need proper API access.
    """
    try:
        time.sleep(0.5)
        
        mock_courses = [
            {
                "title": f"{skill} Specialization",
                "platform": "Coursera",
                "url": f"https://www.coursera.org/specializations/{skill.lower().replace(' ', '-')}",
                "price": "$49/month",
                "duration": "3 months",
                "rating": 4.7,
                "university": "Top University"
            }
        ]
        return mock_courses[:max_results]
    except Exception as e:
        print(f"Error scraping Coursera courses: {e}")
        return []

def get_enhanced_course_recommendations(skill: str) -> Dict:
    """
    Get enhanced course recommendations including scraped data.
    """
    # Get base recommendations
    base_courses = get_course_recommendations(skill)
    
    # Add scraped courses
    udemy_courses = scrape_udemy_courses(skill)
    coursera_courses = scrape_coursera_courses(skill)
    
    # Combine with existing paid courses
    enhanced_paid = base_courses.get("paid", []) + udemy_courses + coursera_courses
    
    return {
        "free": base_courses.get("free", []),
        "paid": enhanced_paid
    }

def create_learning_path(missing_skills: List[str], have_skills: List[str] = None) -> List[Dict]:
    """
    Create a personalized learning path by ordering missing skills.
    """
    # Skill dependency mapping
    skill_dependencies = {
        "Machine Learning": ["Python", "Statistics", "Linear Algebra"],
        "Deep Learning": ["Machine Learning", "Python", "TensorFlow"],
        "Data Science": ["Python", "SQL", "Statistics"],
        "Web Development": ["HTML", "CSS", "JavaScript"],
        "React": ["JavaScript", "HTML", "CSS"],
        "Django": ["Python", "SQL", "HTML"],
        "Flask": ["Python", "SQL"],
        "AWS": ["Linux", "Networking"],
        "Docker": ["Linux", "Command Line"],
        "Kubernetes": ["Docker", "Linux"],
        "DevOps": ["Linux", "Git", "Docker"],
        "Data Analysis": ["Python", "SQL", "Excel"],
        "UI Design": ["Figma", "Adobe XD"],
        "UX Design": ["User Research", "Figma"],
        "Mobile Development": ["JavaScript", "React Native"],
        "Blockchain": ["JavaScript", "Solidity"],
        "Cybersecurity": ["Linux", "Networking", "Python"]
    }
    
    # Create learning order based on dependencies
    learning_path = []
    remaining_skills = missing_skills.copy()
    have_set = set((have_skills or []))
    
    while remaining_skills:
        # Find skills that can be learned (dependencies are met)
        learnable = []
        for skill in remaining_skills:
            dependencies = skill_dependencies.get(skill, [])
            if all(dep in have_set for dep in dependencies):
                learnable.append(skill)
        
        if not learnable:
            # If no skills are learnable, add remaining skills
            learnable = remaining_skills
        
        # Add learnable skills to path
        for skill in learnable:
            courses = get_enhanced_course_recommendations(skill)
            learning_path.append({
                "skill": skill,
                "courses": courses,
                "dependencies": skill_dependencies.get(skill, [])
            })
            have_set.add(skill)
            remaining_skills.remove(skill)
    
    return learning_path
