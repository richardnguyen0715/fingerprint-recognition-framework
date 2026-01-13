# src/data/dataset.py
import cv2
from pathlib import Path

def load_fingerprint(path, size=(256, 256)):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    img = img.astype("float32") / 255.0
    return img


def load_fvc2004_db(db_path):
    images = {}
    for img_path in Path(db_path).glob("*.tif"):
        subject = img_path.stem.split("_")[0]
        images.setdefault(subject, []).append(img_path)
    return images
