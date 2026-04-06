import cv2
import json
import sys
import os
import tkinter as tk
from tkinter import messagebox

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database.db import create_table, insert_attendance

# Paths
MODEL_PATH = "face_model.yml"
LABEL_MAP_PATH = "label_map.json"

# Load model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

# Load label map
with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

label_map = {int(k): v for k, v in label_map.items()}

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Create DB table
create_table()

# Tkinter root (hidden)
root = tk.Tk()
root.withdraw()

# Webcam
cap = cv2.VideoCapture(0)

print("Starting recognition... Press Q to quit")

marked = set()

def show_popup(name):
    """
    Show popup with student ID + name
    """
    try:
        # Split ID and name
        if "_" in name:
            student_id, student_name = name.split("_", 1)
        else:
            student_id = "N/A"
            student_name = name

        message = f"ID: {student_id}\nName: {student_name}\n\n✅ Attendance marked for today"

        messagebox.showinfo("Attendance Marked", message)

    except Exception as e:
        print("Popup error:", e)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200,200))

        label, confidence = recognizer.predict(face)

        if confidence < 100:
            name = label_map[label]

            if name not in marked:
                insert_attendance(name)
                marked.add(name)

                # 🔥 SHOW POPUP
                show_popup(name)

            display_text = name
            color = (0,255,0)
        else:
            display_text = "Unknown"
            color = (0,0,255)

        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, display_text, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()