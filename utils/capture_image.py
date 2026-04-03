"""
Face Image Capture Tool using OpenCV
Captures 30 face images per student and saves them in a named folder.
"""

import cv2
import os
import time


def create_student_folder(safe_name: str, base_dir: str = "student_faces") -> str:
    folder_path = os.path.join(base_dir, safe_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def capture_faces(
    safe_name: str,
    num_images: int = 30,
    base_dir: str = "student_faces",
    delay_between_captures: float = 0.3,
):
    """
    Capture face images from webcam for a student.

    Args:
        safe_name          : Sanitized name of the student (used as folder name).
        num_images         : Number of images to capture (default 30).
        base_dir           : Root directory to store all student folders.
        delay_between_captures: Seconds to wait between consecutive captures.
    """
    # Load OpenCV's pre-trained Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Create save folder
    save_folder = create_student_folder(safe_name, base_dir)
    print(f"\n📁  Saving images to: {save_folder}")

    # Open webcam (index 0 = default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌  Error: Cannot access webcam. Check if it is connected.")
        return

    # Optional: improve resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    captured = 0
    last_capture_time = 0

    print(f"\n🎥  Starting capture for: {safe_name}")
    print("     Look at the camera. Press  Q  to quit early.\n")

    while captured < num_images:
        ret, frame = cap.read()
        if not ret:
            print("❌  Failed to grab frame. Exiting.")
            break

        # Mirror the frame for a natural selfie feel
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

        display = frame.copy()

        for x, y, w, h in faces:
            # Draw rectangle around detected face
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Auto-capture with delay
            current_time = time.time()
            if current_time - last_capture_time >= delay_between_captures:
                captured += 1
                last_capture_time = current_time

                # Crop and save the face region with a small padding
                pad = 20
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(frame.shape[1], x + w + pad)
                y2 = min(frame.shape[0], y + h + pad)

                face_img = frame[y1:y2, x1:x2]
                filename = os.path.join(
                    save_folder, f"{safe_name}_{captured:03d}.jpg"
                )
                cv2.imwrite(filename, face_img)
                print(f"  ✅  Captured {captured}/{num_images}  →  {filename}")

            # Break loop if done
            if captured >= num_images:
                break

        # Overlay status on the live feed
        status_text = f"Student: {safe_name}  |  Captured: {captured}/{num_images}"
        cv2.putText(
            display, status_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2
        )

        if len(faces) == 0:
            cv2.putText(
                display, "No face detected — adjust position",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

        cv2.imshow("Face Capture — Press Q to quit", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n⚠️   Capture interrupted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n✅  Done! Saved {captured} image(s) for '{safe_name}' in '{save_folder}'\n")


def main():
    print("=" * 55)
    print("       Face Image Capture Tool — OpenCV")
    print("=" * 55)

    while True:
        safe_name = input("\nEnter student name (or 'quit' to exit): ").strip()
        safe_name = safe_name.strip().replace(" ", "_").lower()

        if safe_name.lower() in ("quit", "q", "exit"):
            print("👋  Goodbye!")
            break

        if not safe_name:
            print("⚠️   Name cannot be empty. Try again.")
            continue

        capture_faces(
            safe_name=safe_name,
            num_images=30,
            base_dir="student_faces",
            delay_between_captures=0.3,
        )

        another = input("Capture another student? (y/n): ").strip().lower()
        if another != "y":
            print("👋  Goodbye!")
            break


if __name__ == "__main__":
    main()