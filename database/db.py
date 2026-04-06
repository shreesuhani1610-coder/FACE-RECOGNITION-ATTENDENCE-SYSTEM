import sqlite3
from datetime import datetime

DB_NAME = "attendance.db"

def get_connection():
    return sqlite3.connect(DB_NAME)

def create_table():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            date TEXT,
            time TEXT,
            UNIQUE(name, date)
        )
    """)

    conn.commit()
    conn.close()

def insert_attendance(name):
    conn = get_connection()
    cursor = conn.cursor()

    date = datetime.now().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H:%M:%S")

    try:
        cursor.execute("""
            INSERT OR IGNORE INTO attendance (name, date, time)
            VALUES (?, ?, ?)
        """, (name, date, time))

        conn.commit()
    except:
        pass

    conn.close()

def get_all_records():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM attendance")
    data = cursor.fetchall()

    conn.close()
    return data