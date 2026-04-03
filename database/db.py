"""
db.py - Database module for Face Recognition Attendance System
Handles all SQLite database operations for storing attendance records.
"""

import sqlite3
from datetime import datetime


# Name of the database file that will be created in the project folder
DATABASE_NAME = "attendance.db"


# ──────────────────────────────────────────────
# 1. Create a database connection
# ──────────────────────────────────────────────
def get_connection():
    """
    Create and return a connection to the SQLite database.
    The database file 'attendance.db' is created automatically
    if it does not already exist.
    """
    connection = sqlite3.connect(DATABASE_NAME)
    return connection


# ──────────────────────────────────────────────
# 2. Create the attendance table (if not exists)
# ──────────────────────────────────────────────
def create_table():
    """
    Create the 'attendance' table in the database.
    This function is safe to call multiple times —
    it will NOT overwrite existing data.

    Table columns:
        id   - Auto-incremented primary key
        name - Name of the recognized person
        date - Date of attendance  (e.g. '2025-04-04')
        time - Time of attendance  (e.g. '09:30:15')
    """
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id    INTEGER PRIMARY KEY AUTOINCREMENT,
            name  TEXT    NOT NULL,
            date  TEXT    NOT NULL,
            time  TEXT    NOT NULL,

            -- Prevent the same person from being marked twice on the same date
            UNIQUE (name, date)
        )
    """)

    connection.commit()   # Save the changes
    connection.close()    # Always close when done
    print("[DB] Table 'attendance' is ready.")


# ──────────────────────────────────────────────
# 3. Insert an attendance record
# ──────────────────────────────────────────────
def insert_attendance(name, date=None, time=None):
    """
    Insert a new attendance record for a person.

    Parameters:
        name (str) : Full name of the recognized person.
        date (str) : Date string in 'YYYY-MM-DD' format.
                     Defaults to today's date if not provided.
        time (str) : Time string in 'HH:MM:SS' format.
                     Defaults to the current time if not provided.

    Behaviour:
        - If the person has already been marked present on the same date,
          the record is IGNORED (no duplicate, no crash).
        - Returns True  if the record was inserted successfully.
        - Returns False if a duplicate was detected.
    """
    # Use current date/time when not supplied by the caller
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    if time is None:
        time = datetime.now().strftime("%H:%M:%S")

    connection = get_connection()
    cursor = connection.cursor()

    try:
        # INSERT OR IGNORE silently skips the row when the
        # UNIQUE constraint (name, date) would be violated.
        cursor.execute("""
            INSERT OR IGNORE INTO attendance (name, date, time)
            VALUES (?, ?, ?)
        """, (name, date, time))

        connection.commit()

        if cursor.rowcount == 1:
            # rowcount == 1  →  a new row was actually inserted
            print(f"[DB] Attendance marked  →  {name}  |  {date}  |  {time}")
            return True
        else:
            # rowcount == 0  →  duplicate was silently ignored
            print(f"[DB] Already marked today  →  {name}  |  {date}")
            return False

    except sqlite3.Error as error:
        print(f"[DB] Database error: {error}")
        return False

    finally:
        connection.close()   # Always close, even if an error occurred


# ──────────────────────────────────────────────
# 4. (Bonus) Fetch all attendance records
# ──────────────────────────────────────────────
def get_all_records():
    """
    Retrieve every row from the attendance table and return
    them as a list of tuples: (id, name, date, time).

    Useful for debugging or displaying a report.
    """
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("SELECT * FROM attendance ORDER BY date, time")
    records = cursor.fetchall()

    connection.close()
    return records


# ──────────────────────────────────────────────
# Example usage (runs only when executed directly)
# Import this file in recognize.py — the block below
# will NOT run automatically when imported.
# ──────────────────────────────────────────────
if __name__ == "__main__":

    # Step 1 – Make sure the table exists
    create_table()

    # Step 2 – Mark attendance (date & time auto-filled)
    insert_attendance("Alice Johnson")
    insert_attendance("Bob Smith")

    # Step 3 – Try inserting Alice again on the same day (duplicate → ignored)
    insert_attendance("Alice Johnson")

    # Step 4 – Insert with a custom date and time
    insert_attendance("Charlie Brown", date="2025-04-01", time="08:45:00")

    # Step 5 – Print all stored records
    print("\n── All Attendance Records ──")
    for row in get_all_records():
        print(f"  ID: {row[0]}  |  Name: {row[1]}  |  Date: {row[2]}  |  Time: {row[3]}")