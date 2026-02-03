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
