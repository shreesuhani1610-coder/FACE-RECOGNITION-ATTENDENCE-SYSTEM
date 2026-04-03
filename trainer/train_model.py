"""
train_model.py
--------------
Trains an LBPH Face Recognition model using images captured by capture_image.py.

Folder structure expected:
    student_faces/
    ├── alice/
    │   ├── alice_001.jpg
    │   └── ...
    └── bob/
        ├── bob_001.jpg
        └── ...

Outputs:
    face_model.yml   — trained LBPH model
    label_map.json   — mapping between numeric IDs and student names
"""

import cv2
import os
import json
import numpy as np


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATASET_DIR   = "dataset"   # folder created by capture_image.py
MODEL_PATH    = "face_model.yml"  # output: trained model
LABEL_MAP_PATH = "label_map.json" # output: id ↔ name mapping


# ─────────────────────────────────────────────
# STEP 1 — Discover students (subfolders)
# ─────────────────────────────────────────────
def get_student_folders(dataset_dir: str) -> list[str]:
    """
    Return a sorted list of subfolder names inside dataset_dir.
    Each subfolder = one student.
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"Dataset folder '{dataset_dir}' not found. "
            "Run capture_image.py first."
        )

    folders = [
        name for name in sorted(os.listdir(dataset_dir))
        if os.path.isdir(os.path.join(dataset_dir, name))
    ]

    if not folders:
        raise ValueError(f"No student subfolders found inside '{dataset_dir}'.")

    return folders


# ─────────────────────────────────────────────
# STEP 2 — Build label map  (name ↔ numeric id)
# ─────────────────────────────────────────────
def build_label_map(student_folders: list[str]) -> dict:
    """
    Assign a unique integer ID to every student name.

    Returns:
        {
          "id_to_name": {0: "alice", 1: "bob", ...},
          "name_to_id": {"alice": 0, "bob": 1, ...}
        }
    """
    id_to_name = {idx: name for idx, name in enumerate(student_folders)}
    name_to_id = {name: idx for idx, name in id_to_name.items()}
    return {"id_to_name": id_to_name, "name_to_id": name_to_id}


# ─────────────────────────────────────────────
# STEP 3 — Load images and assign labels
# ─────────────────────────────────────────────
def load_training_data(
    dataset_dir: str,
    label_map: dict,
) -> tuple[list[np.ndarray], list[int]]:
    """
    Walk through every student folder, load images as grayscale,
    and pair each image with its numeric label.

    Returns:
        faces  — list of grayscale numpy arrays (one per image)
        labels — list of integer IDs matching each face
    """
    name_to_id = label_map["name_to_id"]
    faces: list[np.ndarray] = []
    labels: list[int] = []

    total_students = len(name_to_id)

    for student_name, student_id in name_to_id.items():
        folder_path = os.path.join(dataset_dir, student_name)
        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not image_files:
            print(f"  ⚠️  No images found for '{student_name}' — skipping.")
            continue

        print(f"\n  👤 Loading [{student_id + 1}/{total_students}] "
              f"'{student_name}'  (ID={student_id})  —  {len(image_files)} image(s)")

        loaded = 0
        for filename in image_files:
            img_path = os.path.join(folder_path, filename)

            # Read image in grayscale directly
            gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if gray_img is None:
                print(f"    ⚠️  Could not read '{filename}' — skipping.")
                continue

            faces.append(gray_img)
            labels.append(student_id)
            loaded += 1

        print(f"    ✅ {loaded} image(s) loaded successfully.")

    return faces, labels


# ─────────────────────────────────────────────
# STEP 4 — Train the LBPH model
# ─────────────────────────────────────────────
def train_lbph_model(
    faces: list[np.ndarray],
    labels: list[int],
) -> cv2.face.LBPHFaceRecognizer:
    """
    Create and train an LBPH (Local Binary Patterns Histogram)
    face recognizer with the provided faces and labels.

    LBPH works well with small datasets and handles lighting
    variations better than Eigenfaces / Fisherfaces.
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,       # radius of the circular LBP pattern
        neighbors=8,    # number of sample points on the circle
        grid_x=8,       # number of cells in horizontal direction
        grid_y=8,       # number of cells in vertical direction
    )

    print("\n  🧠  Training LBPH model ...")
    recognizer.train(faces, np.array(labels))
    print("  ✅  Training complete!")

    return recognizer


# ─────────────────────────────────────────────
# STEP 5 — Save model and label map
# ─────────────────────────────────────────────
def save_model(recognizer: cv2.face.LBPHFaceRecognizer, model_path: str) -> None:
    """Save the trained LBPH model to a .yml file."""
    recognizer.save(model_path)
    print(f"\n  💾  Model saved  →  '{model_path}'")


def save_label_map(label_map: dict, label_map_path: str) -> None:
    """
    Save the label map to a JSON file.
    JSON keys must be strings, so convert int keys before saving.
    """
    serialisable = {
        "id_to_name": {str(k): v for k, v in label_map["id_to_name"].items()},
        "name_to_id": label_map["name_to_id"],
    }
    with open(label_map_path, "w") as f:
        json.dump(serialisable, f, indent=4)
    print(f"  💾  Label map saved  →  '{label_map_path}'")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("       Face Recognition — Model Trainer")
    print("=" * 55)

    # 1. Discover students
    print(f"\n📂  Scanning dataset folder: '{DATASET_DIR}' ...")
    student_folders = get_student_folders(DATASET_DIR)
    print(f"  Found {len(student_folders)} student(s): {', '.join(student_folders)}")

    # 2. Build label map
    label_map = build_label_map(student_folders)
    print("\n🏷️   Label assignments:")
    for name, idx in label_map["name_to_id"].items():
        print(f"    {idx:>3}  →  {name}")

    # 3. Load images
    print("\n🖼️   Loading training images ...")
    faces, labels = load_training_data(DATASET_DIR, label_map)

    if not faces:
        print("\n❌  No valid images loaded. Aborting.")
        return

    print(f"\n📊  Total images loaded: {len(faces)}")

    # 4. Train model
    recognizer = train_lbph_model(faces, labels)

    # 5. Save outputs
    print("\n💾  Saving outputs ...")
    save_model(recognizer, MODEL_PATH)
    save_label_map(label_map, LABEL_MAP_PATH)

    # Summary
    print("\n" + "=" * 55)
    print("  🎉  Training Summary")
    print("=" * 55)
    print(f"  Students trained : {len(student_folders)}")
    print(f"  Total images     : {len(faces)}")
    print(f"  Model file       : {MODEL_PATH}")
    print(f"  Label map file   : {LABEL_MAP_PATH}")
    print("=" * 55)
    print("\n✅  All done! You can now use these files in recognize.py\n")


if __name__ == "__main__":
    main()