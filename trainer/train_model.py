import cv2
import os
import json
import numpy as np

DATASET_DIR = "dataset"
MODEL_PATH = "face_model.yml"
LABEL_MAP_PATH = "label_map.json"

def train():
    faces = []
    labels = []
    label_map = {}

    folders = [f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))]

    for i, name in enumerate(folders):
        label_map[i] = name
        folder_path = os.path.join(DATASET_DIR, name)

        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (200,200))

            faces.append(img)
            labels.append(i)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    recognizer.save(MODEL_PATH)

    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f)

    print("Training complete!")

if __name__ == "__main__":
    train()