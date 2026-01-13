# experiments/exp_01_ssim_fvc2004.py
import csv
from src.data.dataset import load_fingerprint
from src.baselines.ssim import ssim_score
from src.evaluation.metrics import compute_eer

y_true, y_score = [], []

with open("data/pairs/fvc2004_db1_pairs.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        img1 = load_fingerprint(row["img1"])
        img2 = load_fingerprint(row["img2"])

        score = ssim_score(img1, img2)

        y_true.append(int(row["label"]))
        y_score.append(score)

eer = compute_eer(y_true, y_score)
print(f"EER (SSIM baseline): {eer:.4f}")
