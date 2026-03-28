
import sqlite3
import json
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "gigafactory_live.db"
USERS_PATH = BASE_DIR / "users.json"

def migrate():
    # 1. Connect to SQLite
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # 2. Create users table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT NOT NULL,
            initials TEXT,
            email TEXT,
            department TEXT,
            role TEXT DEFAULT 'user',
            pages TEXT, -- Store as JSON string or comma separated
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 3. Load from users.json
    if USERS_PATH.exists():
        with open(USERS_PATH, encoding="utf-8") as f:
            data = json.load(f)
            users = data.get("users", [])
            
            for user in users:
                # Check if user already exists
                cursor.execute("SELECT 1 FROM users WHERE username = ?", (user["username"],))
                if cursor.fetchone():
                    print(f"User {user['username']} already exists, skipping.")
                    continue
                
                # Pages needs to be serialized
                pages_str = ",".join(user.get("pages", ["dashboard", "profile"]))
                
                cursor.execute('''
                    INSERT INTO users (username, password, name, initials, email, department, role, pages)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user["username"],
                    user["password"],
                    user["name"],
                    user["initials"],
                    user["email"],
                    user["department"],
                    user["role"],
                    pages_str
                ))
                print(f"Migrated user: {user['username']}")
    
    conn.commit()
    conn.close()
    print("Migration completed successfully.")

if __name__ == "__main__":
    migrate()
