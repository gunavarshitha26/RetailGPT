import sqlite3
import datetime
import bcrypt
from backend.config import settings
import os

def hash_password(password: str) -> str:
    """Hashes a password using bcrypt."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verifies a password against a bcrypt hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def init_db():
    """Initializes the database schema."""
    conn = sqlite3.connect(settings.DATABASE_PATH)
    cursor = conn.cursor()
    
    # [UPDATED] Users table with Name and Email
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            username TEXT UNIQUE NOT NULL, 
            password_hash TEXT NOT NULL
        )
    ''')
    
    # Uploads table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            username TEXT, 
            filename TEXT, 
            rows INTEGER, 
            upload_time TEXT
        )
    ''')
    
    # New User Files table (Data Hub)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            username TEXT, 
            filename TEXT, 
            file_type TEXT, 
            upload_date TEXT,
            status TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def create_user(name: str, email: str, username: str, password: str) -> bool:
    conn = sqlite3.connect(settings.DATABASE_PATH)
    cursor = conn.cursor()
    try:
        hashed = hash_password(password)
        cursor.execute(
            "INSERT INTO users (name, email, username, password_hash) VALUES (?, ?, ?, ?)", 
            (name, email, username, hashed)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username: str, password: str) -> dict:
    conn = sqlite3.connect(settings.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=?", (username,))
    user = cursor.fetchone()
    conn.close()
    
    if user and verify_password(password, user["password_hash"]):
        return {"name": user["name"], "email": user["email"], "username": user["username"]}
    return None

def save_upload_metadata(username: str, filename: str, rows: int):
    conn = sqlite3.connect(settings.DATABASE_PATH)
    cursor = conn.cursor()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO uploads (username, filename, rows, upload_time) VALUES (?, ?, ?, ?)", 
        (username, filename, rows, now)
    )
    conn.commit()
    conn.close()

def get_user_history(username: str) -> list:
    conn = sqlite3.connect(settings.DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT filename, upload_time FROM uploads WHERE username=? ORDER BY id DESC", 
        (username,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    # Return unique files, newest first
    unique_files = list(dict.fromkeys([r[0] for r in rows]))
    return unique_files

# --- Data Hub File Management ---
def add_user_file(username: str, filename: str, file_type: str, status: str = "Processing"):
    conn = sqlite3.connect(settings.DATABASE_PATH)
    cursor = conn.cursor()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO user_files (username, filename, file_type, upload_date, status) VALUES (?, ?, ?, ?, ?)", 
        (username, filename, file_type, now, status)
    )
    conn.commit()
    conn.close()

def get_user_files(username: str) -> list:
    conn = sqlite3.connect(settings.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, filename, file_type, upload_date, status FROM user_files WHERE username=? ORDER BY id DESC", 
        (username,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_user_file_record(username: str, filename: str):
    conn = sqlite3.connect(settings.DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM user_files WHERE username=? AND filename=?", (username, filename))
    conn.commit()
    conn.close()

# --- Migration Helper ---
def recreate_db():
    """Drops and recreates the database (Development Only)"""
    if os.path.exists(settings.DATABASE_PATH):
        os.rename(settings.DATABASE_PATH, settings.DATABASE_PATH + ".bak")
        print("Backed up old database.")
    init_db()
    print("Initialized new database schema.")