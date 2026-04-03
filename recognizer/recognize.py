"""
============================================================
  Face Recognition Attendance System
  Uses: OpenCV LBPH Face Recognizer + Haar Cascade
============================================================
  Requirements:
    - face_model.yml   : Pre-trained LBPH model
    - label_map.json   : Mapping of label IDs to student names
    - OpenCV with contrib modules (pip install opencv-contrib-python)

  Run:
    python recognize.py

  Press 'q' to quit.
============================================================
"""

import cv2
import json
import os
from datetime import datetime
from database.db import create_table, insert_attendance 


# ──────────────────────────────────────────────
# 1. LOAD LABEL MAP  (id → student name)
# ──────────────────────────────────────────────
LABEL_MAP_PATH = "label_map.json"
MODEL_PATH     = "face_model.yml"

if not os.path.exists(LABEL_MAP_PATH):
    print(f"[ERROR] Label map not found: {LABEL_MAP_PATH}")
    exit(1)

if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model file not found: {MODEL_PATH}")
    exit(1)

with open(LABEL_MAP_PATH, "r") as f:
    raw_map = json.load(f)
    label_map = {int(k): v for k, v in raw_map["id_to_name"].items()}

print("[INFO] Label map loaded:", label_map)

# ──────────────────────────────────────────────
# 2. LOAD THE TRAINED LBPH MODEL
# ──────────────────────────────────────────────
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)
print("[INFO] Trained model loaded successfully.")

# ──────────────────────────────────────────────
# 3. LOAD HAAR CASCADE FOR FACE DETECTION
# ──────────────────────────────────────────────
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("[ERROR] Could not load Haar Cascade classifier.")
    exit(1)

print("[INFO] Haar Cascade loaded.")

# ──────────────────────────────────────────────
# 4. OPEN WEBCAM
# ──────────────────────────────────────────────
create_table() # Ensure the attendance table exists before starting recognition
cap = cv2.VideoCapture(0)   # 0 = default webcam

if not cap.isOpened():
    print("[ERROR] Cannot open webcam.")
    exit(1)

print("[INFO] Webcam opened. Starting attendance system...")
print("[INFO] Press 'q' to quit.\n")

# ──────────────────────────────────────────────
# 5. ATTENDANCE TRACKING (avoid duplicates)
# ──────────────────────────────────────────────
# Stores names that have already been marked this session
marked_attendance = set()

# Confidence threshold: lower value = better match
# LBPH confidence: 0 is perfect, >100 is usually unknown
CONFIDENCE_THRESHOLD = 100

# ──────────────────────────────────────────────
# 6. REAL-TIME RECOGNITION LOOP
# ──────────────────────────────────────────────
while True:
    ret, frame = cap.read()

    if not ret:
        print("[WARNING] Failed to grab frame. Retrying...")
        continue

    # --- Convert frame to grayscale (required for detection & recognition) ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Detect faces in the grayscale frame ---
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,    # How much image size is reduced at each scale
        minNeighbors=5,     # Higher = fewer but stronger detections
        minSize=(60, 60)    # Minimum face size to detect
    )

    # --- Process each detected face ---
    for (x, y, w, h) in faces:

        # Crop the face region from the grayscale frame
        face_roi = gray[y:y+h, x:x+w]

        # Resize to 200×200 (must match training size)
        face_resized = cv2.resize(face_roi, (200, 200))

        # --- Predict using the LBPH recognizer ---
        label_id, confidence = recognizer.predict(face_resized)

        # --- Decide: recognized or unknown? ---
        if confidence < CONFIDENCE_THRESHOLD:
            # Known person
            full_name = label_map.get(label_id, "Unknown")
            name = full_name.split("_", 1)[-1]   # removes ID part
            display_text = f"{name}  ({confidence:.1f})"
            box_color  = (0, 200, 0)    # Green rectangle
            text_color = (0, 200, 0)

            # --- Mark attendance (only once per session) ---
            if name not in marked_attendance:
                marked_attendance.add(name)
                now  = datetime.now()
                date = now.strftime("%Y-%m-%d")
                time = now.strftime("%H:%M:%S")
                insert_attendance(name, date, time)
        else:
            # Unknown / low confidence
            name         = "Unknown"
            display_text = f"Unknown  ({confidence:.1f})"
            box_color    = (0, 0, 200)   # Red rectangle
            text_color   = (0, 0, 200)

        # --- Draw bounding box around the face ---
        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)

        # --- Display name above the bounding box ---
        cv2.putText(
            frame,
            display_text,
            (x, y - 10),               # Position: just above the box
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,                        # Font scale
            text_color,
            2,                          # Thickness
            cv2.LINE_AA
        )

    # --- Show a small legend in the top-left corner ---
    cv2.putText(frame, "Attendance System  |  Press 'q' to quit",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # --- Display the frame ---
    cv2.imshow("Face Recognition Attendance System", frame)

    # --- Exit on pressing 'q' ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n[INFO] 'q' pressed. Exiting...")
        break

# ──────────────────────────────────────────────
# 7. CLEANUP
# ──────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()

# ──────────────────────────────────────────────
# 8. PRINT FINAL ATTENDANCE SUMMARY
# ──────────────────────────────────────────────
print("\n" + "="*50)
print("       ATTENDANCE SUMMARY FOR THIS SESSION")
print("="*50)

if marked_attendance:
    for i, name in enumerate(sorted(marked_attendance), start=1):
        print(f"  {i}. {name}")
else:
    print("  No attendance marked in this session.")

print("="*50)
print("[INFO] System shut down. Goodbye!")