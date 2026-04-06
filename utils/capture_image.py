import cv2
import os
import time

def create_student_folder(name, base_dir="dataset"):
    path = os.path.join(base_dir, name)
    os.makedirs(path, exist_ok=True)
    return path

def capture_faces(name, num_images=30):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    folder = create_student_folder(name)

    cap = cv2.VideoCapture(0)

    count = 0
    last_time = 0

    print("Starting capture... Press Q to quit")

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            if time.time() - last_time > 0.3:
                count += 1
                last_time = time.time()

                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (200,200))

                filename = os.path.join(folder, f"{name}_{count}.jpg")
                cv2.imwrite(filename, face)

                print(f"Captured {count}/30")

        cv2.imshow("Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    student_id = input("Enter student ID: ")
    name = input("Enter student name: ")

    safe_name = f"{student_id}_{name}".replace(" ", "_").lower()
    capture_faces(safe_name)