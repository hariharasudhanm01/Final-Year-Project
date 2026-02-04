import mysql.connector 
from mysql.connector import Error 
import hashlib
import secrets
from datetime import datetime, timedelta
import os
from typing import Optional, Dict, List

class DatabaseManager:
    def __init__(self):
        self.connection = None
        self.connect()
    
    def connect(self):
        """Connect to MySQL database"""
        try:
            # Database configuration - update these with your MySQL credentials
            config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'database': os.getenv('DB_NAME', 'internship_recommender'),
                'user': os.getenv('DB_USER', 'root'),
                'password': os.getenv('DB_PASSWORD', ''),
                'port': int(os.getenv('DB_PORT', 3306)),
                'charset': 'utf8mb4',
                'collation': 'utf8mb4_unicode_ci'
            }
            
            self.connection = mysql.connector.connect(**config)
            if self.connection.is_connected():
                print("Connected to MySQL database")
                self.create_tables()
                self._ensure_migrations()
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            # Fallback to SQLite for development
            self.setup_sqlite_fallback()
    
    def setup_sqlite_fallback(self):
        """Fallback to SQLite if MySQL is not available"""
        try:
            import sqlite3
            # Use check_same_thread=False to allow cross-thread usage
            self.connection = sqlite3.connect('internship_recommender.db', check_same_thread=False)
            print("Using SQLite fallback database")
            self.create_tables()
            self._ensure_migrations()
        except Exception as e:
            print(f"Error setting up SQLite fallback: {e}")
    
    def _ensure_migrations(self):
        """Ensure migrations are run even if tables already exist"""
        cursor = None
        try:
            cursor = self.connection.cursor()
            is_sqlite = 'sqlite' in str(type(self.connection)).lower()
            self._run_migrations(cursor, is_sqlite)
            self.connection.commit()
        except Exception as e:
            print(f"Error ensuring migrations: {e}")
        finally:
            if cursor:
                cursor.close()
    
    def create_tables(self):
        """Create database tables if they don't exist"""
        cursor = None
        try:
            cursor = self.connection.cursor()
            
            # Check if we're using SQLite or MySQL
            is_sqlite = 'sqlite' in str(type(self.connection)).lower()
            
            if is_sqlite:
                # SQLite syntax
                create_users_table = """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    full_name TEXT,
                    user_role TEXT DEFAULT 'student',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
                """
                
                create_profiles_table = """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    degree TEXT,
                    study_year INTEGER,
                    sector TEXT,
                    stream TEXT,
                    skills TEXT,
                    resume_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
                
                create_sessions_table = """
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_token TEXT UNIQUE NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
                
                create_recommendations_table = """
                CREATE TABLE IF NOT EXISTS recommendation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    location TEXT,
                    skills_used TEXT,
                    missing_skills TEXT,
                    salary_range_low INTEGER,
                    salary_range_high INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            else:
                # MySQL syntax
                create_users_table = """
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    salt VARCHAR(32) NOT NULL,
                    full_name VARCHAR(100),
                    user_role VARCHAR(20) DEFAULT 'student',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
                """
                
                create_profiles_table = """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    degree VARCHAR(50),
                    study_year INT,
                    sector VARCHAR(50),
                    stream VARCHAR(50),
                    skills TEXT,
                    resume_path VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
                
                create_sessions_table = """
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    session_token VARCHAR(64) UNIQUE NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
                
                create_recommendations_table = """
                CREATE TABLE IF NOT EXISTS recommendation_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    role VARCHAR(100) NOT NULL,
                    location VARCHAR(100),
                    skills_used TEXT,
                    missing_skills TEXT,
                    salary_range_low INT,
                    salary_range_high INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            
            # Execute table creation
            cursor.execute(create_users_table)
            cursor.execute(create_profiles_table)
            cursor.execute(create_sessions_table)
            cursor.execute(create_recommendations_table)
            
            # Run migrations for existing databases
            self._run_migrations(cursor, is_sqlite)
            
            self.connection.commit()
            print("Database tables created successfully")
            
        except Exception as e:
            print(f"Error creating tables: {e}")
        finally:
            if cursor:
                cursor.close()
    
    def _run_migrations(self, cursor, is_sqlite):
        """Run database migrations for existing databases"""
        try:
            # Check if users table exists first
            if is_sqlite:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
                if not cursor.fetchone():
                    return  # Table doesn't exist yet, will be created with new schema
                
                # Check if user_role column exists in users table
                cursor.execute("PRAGMA table_info(users)")
                columns = [column[1] for column in cursor.fetchall()]
                
                if 'user_role' not in columns:
                    print("Migrating: Adding user_role column to users table...")
                    cursor.execute("ALTER TABLE users ADD COLUMN user_role TEXT DEFAULT 'student'")
                    # Update existing rows to have 'student' as default
                    cursor.execute("UPDATE users SET user_role = 'student' WHERE user_role IS NULL")
                    print("Migration completed: user_role column added")
            else:
                # For MySQL, check if table exists
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_SCHEMA = DATABASE() 
                    AND TABLE_NAME = 'users'
                """)
                if cursor.fetchone()[0] == 0:
                    return  # Table doesn't exist yet
                
                # Check if column exists
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_SCHEMA = DATABASE() 
                    AND TABLE_NAME = 'users' 
                    AND COLUMN_NAME = 'user_role'
                """)
                if cursor.fetchone()[0] == 0:
                    print("Migrating: Adding user_role column to users table...")
                    cursor.execute("ALTER TABLE users ADD COLUMN user_role VARCHAR(20) DEFAULT 'student'")
                    # Update existing rows to have 'student' as default
                    cursor.execute("UPDATE users SET user_role = 'student' WHERE user_role IS NULL")
                    print("Migration completed: user_role column added")
        except Exception as e:
            print(f"Error running migrations: {e}")
    
    def _get_placeholder(self):
        """Get the correct placeholder for the current database"""
        is_sqlite = 'sqlite' in str(type(self.connection)).lower()
        return '?' if is_sqlite else '%s'
    
    def _get_cursor(self, dictionary=False):
        """Get cursor with appropriate settings for current database"""
        is_sqlite = 'sqlite' in str(type(self.connection)).lower()
        if is_sqlite:
            if dictionary:
                # For SQLite, use row_factory to get dictionary-like results
                cursor = self.connection.cursor()
                cursor.row_factory = lambda cursor, row: dict(zip([col[0] for col in cursor.description], row))
                return cursor
            else:
                return self.connection.cursor()
        else:
            # MySQL supports dictionary parameter
            return self.connection.cursor(dictionary=dictionary)
    
    def hash_password(self, password: str, salt: str = None) -> tuple:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        password_hash = hashlib.pbkdf2_hmac('sha256', 
                                          password.encode('utf-8'), 
                                          salt.encode('utf-8'), 
                                          100000)
        return password_hash.hex(), salt
    
    def verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """Verify password against stored hash"""
        password_hash, _ = self.hash_password(password, salt)
        return password_hash == stored_hash
    
    def create_user(self, username: str, email: str, password: str, full_name: str = None, user_role: str = 'student') -> bool:
        """Create a new user"""
        cursor = None
        try:
            cursor = self._get_cursor()
            
            # Get correct placeholder for current database
            placeholder = self._get_placeholder()
            
            # Check if user already exists
            cursor.execute(f"SELECT id FROM users WHERE username = {placeholder} OR email = {placeholder}", (username, email))
            if cursor.fetchone():
                return False
            
            # Hash password
            password_hash, salt = self.hash_password(password)
            
            # Insert user
            cursor.execute(f"""
                INSERT INTO users (username, email, password_hash, salt, full_name, user_role)
                VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
            """, (username, email, password_hash, salt, full_name, user_role))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            print(f"Error creating user: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user and return user data"""
        cursor = None
        try:
            cursor = self._get_cursor(dictionary=True)
            
            placeholder = self._get_placeholder()
            cursor.execute(f"""
                SELECT id, username, email, password_hash, salt, full_name, user_role, is_active
                FROM users WHERE username = {placeholder} OR email = {placeholder}
            """, (username, username))
            
            user = cursor.fetchone()
            if user and self.verify_password(password, user['password_hash'], user['salt']):
                return user
            return None
            
        except Error as e:
            print(f"Error authenticating user: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    def create_session(self, user_id: int) -> str:
        """Create a new session for user"""
        cursor = None
        try:
            cursor = self._get_cursor()
            
            # Generate session token
            session_token = secrets.token_hex(32)
            expires_at = datetime.now() + timedelta(days=7)  # 7 days expiry
            
            # Insert session
            placeholder = self._get_placeholder()
            cursor.execute(f"""
                INSERT INTO user_sessions (user_id, session_token, expires_at)
                VALUES ({placeholder}, {placeholder}, {placeholder})
            """, (user_id, session_token, expires_at))
            
            self.connection.commit()
            return session_token
            
        except Error as e:
            print(f"Error creating session: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    def get_user_by_session(self, session_token: str) -> Optional[Dict]:
        """Get user by session token"""
        cursor = None
        try:
            cursor = self._get_cursor(dictionary=True)
            
            placeholder = self._get_placeholder()
            is_sqlite = 'sqlite' in str(type(self.connection)).lower()
            if is_sqlite:
                cursor.execute(f"""
                    SELECT u.id, u.username, u.email, u.full_name, u.user_role, u.is_active
                    FROM users u
                    JOIN user_sessions s ON u.id = s.user_id
                    WHERE s.session_token = {placeholder} AND s.expires_at > datetime('now') AND u.is_active = 1
                """, (session_token,))
            else:
                cursor.execute(f"""
                    SELECT u.id, u.username, u.email, u.full_name, u.user_role, u.is_active
                    FROM users u
                    JOIN user_sessions s ON u.id = s.user_id
                    WHERE s.session_token = {placeholder} AND s.expires_at > NOW() AND u.is_active = 1
                """, (session_token,))
            
            user = cursor.fetchone()
            return user
            
        except Error as e:
            print(f"Error getting user by session: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    def update_user_profile(self, user_id: int, profile_data: Dict) -> bool:
        """Update user profile"""
        cursor = None
        try:
            cursor = self._get_cursor()
            
            # Check if profile exists
            placeholder = self._get_placeholder()
            cursor.execute(f"SELECT id FROM user_profiles WHERE user_id = {placeholder}", (user_id,))
            profile_exists = cursor.fetchone()
            
            if profile_exists:
                # Update existing profile
                cursor.execute(f"""
                    UPDATE user_profiles 
                    SET degree = {placeholder}, study_year = {placeholder}, sector = {placeholder}, stream = {placeholder}, 
                        skills = {placeholder}, resume_path = {placeholder}, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = {placeholder}
                """, (profile_data.get('degree'), profile_data.get('study_year'),
                      profile_data.get('sector'), profile_data.get('stream'),
                      profile_data.get('skills'), profile_data.get('resume_path'), user_id))
            else:
                # Create new profile
                cursor.execute(f"""
                    INSERT INTO user_profiles (user_id, degree, study_year, sector, stream, skills, resume_path)
                    VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
                """, (user_id, profile_data.get('degree'), profile_data.get('study_year'),
                      profile_data.get('sector'), profile_data.get('stream'),
                      profile_data.get('skills'), profile_data.get('resume_path')))
            
            self.connection.commit()
            return True
            
        except Error as e:
            print(f"Error updating profile: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
    
    def get_user_profile(self, user_id: int) -> Optional[Dict]:
        """Get user profile"""
        cursor = None
        try:
            cursor = self._get_cursor(dictionary=True)
            
            placeholder = self._get_placeholder()
            cursor.execute(f"""
                SELECT * FROM user_profiles WHERE user_id = {placeholder}
            """, (user_id,))
            
            profile = cursor.fetchone()
            return profile
            
        except Error as e:
            print(f"Error getting profile: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    def save_recommendation_history(self, user_id: int, role: str, location: str, 
                                  skills_used: List[str], missing_skills: List[str],
                                  salary_range: tuple) -> bool:
        """Save recommendation history"""
        cursor = None
        try:
            cursor = self._get_cursor()
            
            placeholder = self._get_placeholder()
            cursor.execute(f"""
                INSERT INTO recommendation_history 
                (user_id, role, location, skills_used, missing_skills, salary_range_low, salary_range_high)
                VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
            """, (user_id, role, location, 
                  ','.join(skills_used), ','.join(missing_skills),
                  salary_range[0], salary_range[1]))
            
            self.connection.commit()
            return True
            
        except Error as e:
            print(f"Error saving recommendation history: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
    
    def get_recommendation_history(self, user_id: int, limit: int = 10) -> List[Dict]:
        """Get user's recommendation history"""
        cursor = None
        try:
            cursor = self._get_cursor(dictionary=True)
            
            placeholder = self._get_placeholder()
            cursor.execute(f"""
                SELECT * FROM recommendation_history 
                WHERE user_id = {placeholder} 
                ORDER BY created_at DESC 
                LIMIT {placeholder}
            """, (user_id, limit))
            
            history = cursor.fetchall()
            return history
            
        except Error as e:
            print(f"Error getting recommendation history: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
    
    def search_candidates(self, skills=None, role=None, location=None, degree=None, stream=None, limit=50):
        """Search candidates based on filters"""
        cursor = None
        try:
            cursor = self._get_cursor(dictionary=True)
            placeholder = self._get_placeholder()
            is_sqlite = 'sqlite' in str(type(self.connection)).lower()
            
            query = """
                SELECT u.id, u.username, u.email, u.full_name, u.created_at,
                       p.degree, p.study_year, p.sector, p.stream, p.skills, p.resume_path
                FROM users u
                LEFT JOIN user_profiles p ON u.id = p.user_id
                WHERE u.user_role = 'student' AND u.is_active = 1
            """
            params = []
            
            if skills:
                skills_list = [s.strip().lower() for s in skills.split(',') if s.strip()]
                if skills_list:
                    if is_sqlite:
                        query += " AND ("
                        for i, skill in enumerate(skills_list):
                            if i > 0:
                                query += " OR "
                            query += f"LOWER(p.skills) LIKE {placeholder}"
                            params.append(f"%{skill}%")
                        query += ")"
                    else:
                        query += " AND ("
                        for i, skill in enumerate(skills_list):
                            if i > 0:
                                query += " OR "
                            query += f"LOWER(p.skills) LIKE {placeholder}"
                            params.append(f"%{skill}%")
                        query += ")"
            
            if degree:
                query += f" AND p.degree = {placeholder}"
                params.append(degree)
            
            if stream:
                query += f" AND p.stream = {placeholder}"
                params.append(stream)
            
            query += f" ORDER BY u.created_at DESC LIMIT {placeholder}"
            params.append(limit)
            
            cursor.execute(query, params)
            return cursor.fetchall()
            
        except Exception as e:
            print(f"Error searching candidates: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
    
    def get_all_candidates(self, limit=100):
        """Get all candidate profiles"""
        cursor = None
        try:
            cursor = self._get_cursor(dictionary=True)
            placeholder = self._get_placeholder()
            
            query = f"""
                SELECT u.id, u.username, u.email, u.full_name, u.created_at,
                       p.degree, p.study_year, p.sector, p.stream, p.skills, p.resume_path
                FROM users u
                LEFT JOIN user_profiles p ON u.id = p.user_id
                WHERE u.user_role = 'student' AND u.is_active = 1
                ORDER BY u.created_at DESC
                LIMIT {placeholder}
            """
            cursor.execute(query, (limit,))
            return cursor.fetchall()
        except Exception as e:
            print(f"Error getting candidates: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
    
    def get_candidate_by_id(self, candidate_id):
        """Get detailed candidate profile"""
        cursor = None
        try:
            cursor = self._get_cursor(dictionary=True)
            placeholder = self._get_placeholder()
            
            query = f"""
                SELECT u.id, u.username, u.email, u.full_name, u.created_at,
                       p.degree, p.study_year, p.sector, p.stream, p.skills, p.resume_path
                FROM users u
                LEFT JOIN user_profiles p ON u.id = p.user_id
                WHERE u.id = {placeholder} AND u.user_role = 'student' AND u.is_active = 1
            """
            cursor.execute(query, (candidate_id,))
            return cursor.fetchone()
        except Exception as e:
            print(f"Error getting candidate: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    def create_job_posting(self, hr_user_id, title, description, required_skills, location, salary_range_low=None, salary_range_high=None):
        """Create a new job posting"""
        cursor = None
        try:
            cursor = self._get_cursor()
            placeholder = self._get_placeholder()
            is_sqlite = 'sqlite' in str(type(self.connection)).lower()
            
            if is_sqlite:
                create_jobs_table = """
                CREATE TABLE IF NOT EXISTS job_postings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hr_user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    required_skills TEXT,
                    location TEXT,
                    salary_range_low INTEGER,
                    salary_range_high INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (hr_user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            else:
                create_jobs_table = """
                CREATE TABLE IF NOT EXISTS job_postings (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    hr_user_id INT NOT NULL,
                    title VARCHAR(200) NOT NULL,
                    description TEXT,
                    required_skills TEXT,
                    location VARCHAR(100),
                    salary_range_low INT,
                    salary_range_high INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    FOREIGN KEY (hr_user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            cursor.execute(create_jobs_table)
            
            cursor.execute(f"""
                INSERT INTO job_postings (hr_user_id, title, description, required_skills, location, salary_range_low, salary_range_high)
                VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
            """, (hr_user_id, title, description, required_skills, location, salary_range_low, salary_range_high))
            
            self.connection.commit()
            return cursor.lastrowid
        except Exception as e:
            print(f"Error creating job posting: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    def get_job_postings(self, hr_user_id=None):
        """Get job postings"""
        cursor = None
        try:
            cursor = self._get_cursor(dictionary=True)
            placeholder = self._get_placeholder()
            
            if hr_user_id:
                query = f"""
                    SELECT * FROM job_postings 
                    WHERE hr_user_id = {placeholder} AND is_active = 1
                    ORDER BY created_at DESC
                """
                cursor.execute(query, (hr_user_id,))
            else:
                query = """
                    SELECT * FROM job_postings 
                    WHERE is_active = 1
                    ORDER BY created_at DESC
                """
                cursor.execute(query)
            
            return cursor.fetchall()
        except Exception as e:
            print(f"Error getting job postings: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
    
    def get_job_posting_by_id(self, job_id):
        """Get a specific job posting"""
        cursor = None
        try:
            cursor = self._get_cursor(dictionary=True)
            placeholder = self._get_placeholder()
            
            query = f"SELECT * FROM job_postings WHERE id = {placeholder} AND is_active = 1"
            cursor.execute(query, (job_id,))
            return cursor.fetchone()
        except Exception as e:
            print(f"Error getting job posting: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    def update_user_role(self, user_id: int, new_role: str) -> bool:
        """Update user role"""
        cursor = None
        try:
            if new_role not in ['student', 'hr']:
                return False
            
            cursor = self._get_cursor()
            placeholder = self._get_placeholder()
            
            cursor.execute(f"""
                UPDATE users 
                SET user_role = {placeholder}
                WHERE id = {placeholder}
            """, (new_role, user_id))
            
            self.connection.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error updating user role: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
    
    def get_user_by_username(self, username: str):
        """Get user by username"""
        cursor = None
        try:
            cursor = self._get_cursor(dictionary=True)
            placeholder = self._get_placeholder()
            
            cursor.execute(f"""
                SELECT id, username, email, full_name, user_role, is_active
                FROM users 
                WHERE username = {placeholder} OR email = {placeholder}
            """, (username, username))
            
            return cursor.fetchone()
        except Exception as e:
            print(f"Error getting user by username: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    def create_resume_modifications_table(self):
        """Create table for resume modifications history"""
        cursor = None
        try:
            cursor = self._get_cursor()
            is_sqlite = 'sqlite' in str(type(self.connection)).lower()
            
            if is_sqlite:
                create_table = """
                CREATE TABLE IF NOT EXISTS resume_modifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    original_resume_path TEXT,
                    modified_resume_path TEXT,
                    modifications_json TEXT,
                    ats_score_before REAL,
                    ats_score_after REAL,
                    job_description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            else:
                create_table = """
                CREATE TABLE IF NOT EXISTS resume_modifications (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    original_resume_path VARCHAR(255),
                    modified_resume_path VARCHAR(255),
                    modifications_json TEXT,
                    ats_score_before DECIMAL(5,2),
                    ats_score_after DECIMAL(5,2),
                    job_description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            
            cursor.execute(create_table)
            self.connection.commit()
        except Exception as e:
            print(f"Error creating resume_modifications table: {e}")
        finally:
            if cursor:
                cursor.close()
    
    def save_resume_modification(self, user_id, original_path, modified_path, modifications, score_before, score_after, job_desc):
        """Save resume modification record"""
        cursor = None
        try:
            cursor = self._get_cursor()
            placeholder = self._get_placeholder()
            
            import json
            modifications_json = json.dumps(modifications)
            
            cursor.execute(f"""
                INSERT INTO resume_modifications 
                (user_id, original_resume_path, modified_resume_path, modifications_json, 
                 ats_score_before, ats_score_after, job_description)
                VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
            """, (user_id, original_path, modified_path, modifications_json, score_before, score_after, job_desc))
            
            self.connection.commit()
            return cursor.lastrowid
        except Exception as e:
            print(f"Error saving resume modification: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    def get_modification_history(self, user_id, limit=10):
        """Get user's resume modification history"""
        cursor = None
        try:
            cursor = self._get_cursor(dictionary=True)
            placeholder = self._get_placeholder()
            
            cursor.execute(f"""
                SELECT * FROM resume_modifications 
                WHERE user_id = {placeholder} 
                ORDER BY created_at DESC 
                LIMIT {placeholder}
            """, (user_id, limit))
            
            return cursor.fetchall()
        except Exception as e:
            print(f"Error getting modification history: {e}")
            return []
        finally:
            if cursor:
                cursor.close()

    def reset_user_data(self, user_id: int) -> bool:
        """
        Hard reset: Clears all user data (Profile, Recommendations, Enhancements) except the user account itself.
        """
        cursor = None
        try:
            cursor = self._get_cursor()
            placeholder = self._get_placeholder()
            
            # 1. Delete Profile Data (Cascades usually handle this if user is deleted, but here we just want to clear data)
            # Actually, let's just delete the profile record entirely to force a fresh start
            cursor.execute(f"DELETE FROM user_profiles WHERE user_id = {placeholder}", (user_id,))
            
            # 2. Delete Recommendation History
            cursor.execute(f"DELETE FROM recommendation_history WHERE user_id = {placeholder}", (user_id,))
            
            # 3. Delete Resume Modifications (if table exists)
            try:
                cursor.execute(f"DELETE FROM resume_modifications WHERE user_id = {placeholder}", (user_id,))
            except Exception:
                pass # Table might not exist yet
                
            self.connection.commit()
            return True
        except Exception as e:
            print(f"Error resetting user data: {e}")
            self.connection.rollback()
            return False
        finally:
            if cursor:
                cursor.close()

    def close(self):
        """Close database connection"""
        if self.connection and hasattr(self.connection, 'is_connected') and self.connection.is_connected():
            self.connection.close()
            print("Database connection closed")
        elif self.connection:
             try:
                self.connection.close() # For SQLite
             except:
                 pass

# Global database instance
db = DatabaseManager()
