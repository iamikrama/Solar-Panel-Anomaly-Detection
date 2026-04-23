#!/usr/bin/env python3
"""
SolarGuard AI — Model Training
================================
Step 2 of the pipeline:

Loads images from dataset/normal/ and dataset/anomaly/
Extracts features (resize + flatten + optional HOG)
Trains a model:
  - Only normal images  → IsolationForest (unsupervised, like FOMO-AD)
  - Normal + anomaly    → SVM classifier  (supervised, more accurate)
Saves trained model to model/solarguard_model.pkl

Run:
    python3 train.py
    python3 train.py --model svm    (force supervised SVM)
    python3 train.py --model iso    (force IsolationForest)
"""
from __future__ import annotations
import os, cv2, time, argparse, json
import numpy as np

try:
    import joblib
except ImportError:
    import pickle as joblib   # fallback

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix

# ── Config ────────────────────────────────────────────
DATASET_DIR  = "dataset"
NORMAL_DIR   = os.path.join(DATASET_DIR, "normal")
ANOMALY_DIR  = os.path.join(DATASET_DIR, "anomaly")
MODEL_DIR    = "model"
MODEL_PATH   = os.path.join(MODEL_DIR, "solarguard_model.pkl")
IMG_SIZE     = (64, 64)   # resize all images to this before feature extraction

os.makedirs(MODEL_DIR, exist_ok=True)


# ── Feature Extraction ────────────────────────────────
def extract_features(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert one BGR image to a 1-D feature vector.
    Steps:
      1. Resize to IMG_SIZE
      2. Convert to grayscale
      3. Flatten pixel values (4096 features for 64x64)
      4. Normalise to 0-1
    """
    img = cv2.resize(img_bgr, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flat = gray.flatten().astype(np.float32) / 255.0
    return flat


def load_images(folder: str, label_name: str) -> tuple[list, list]:
    """Load all images from a folder, return (features_list, labels_list)."""
    feats, labels = [], []
    files = [f for f in os.listdir(folder)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        return feats, labels

    print(f"  Loading {len(files):4d} images from {folder}/")
    for fname in files:
        path = os.path.join(folder, fname)
        img  = cv2.imread(path)
        if img is None:
            print(f"  [WARN] Cannot read {fname} — skipping")
            continue
        feats.append(extract_features(img))
        labels.append(label_name)

    return feats, labels


# ── Training ──────────────────────────────────────────
def train(model_type: str = "auto"):
    print("\n" + "="*55)
    print("  SolarGuard AI — Model Training")
    print("="*55)

    # ── Load dataset ──────────────────────────────────
    print("\n📂 Loading dataset...")
    X_normal, y_normal   = load_images(NORMAL_DIR,  "normal")
    X_anomaly, y_anomaly = load_images(ANOMALY_DIR, "anomaly")

    n_normal  = len(X_normal)
    n_anomaly = len(X_anomaly)
    print(f"\n  Normal images  : {n_normal}")
    print(f"  Anomaly images : {n_anomaly}")

    if n_normal == 0:
        print("\n❌ No normal images found in dataset/normal/")
        print("   Run capture_server.py first to collect images.")
        return

    # ── Decide model type ─────────────────────────────
    if model_type == "auto":
        if n_anomaly >= 10:
            model_type = "svm_oc"   # one-class SVM trained on normal, tested on both
            print("\n✅ Enough anomaly images — using supervised One-Class SVM")
        else:
            model_type = "iso"
            print("\n✅ Using IsolationForest (unsupervised — only needs normal images)")

    X_normal_arr = np.array(X_normal)

    # ── Scale ─────────────────────────────────────────
    print("\n⚙️  Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_normal_arr)

    # ── PCA dimensionality reduction ──────────────────
    n_components = min(100, X_scaled.shape[0] - 1, X_scaled.shape[1])
    print(f"⚙️  PCA → {n_components} components")
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # ── Train model ───────────────────────────────────
    print(f"\n🧠 Training model: {model_type}")
    t0 = time.time()

    if model_type == "iso":
        model = IsolationForest(
            n_estimators=200,
            contamination=0.05,   # expect ~5% of training data to be outliers
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_pca)

    elif model_type == "svm_oc":
        model = OneClassSVM(
            kernel="rbf",
            nu=0.05,    # expected fraction of anomalies
            gamma="scale"
        )
        model.fit(X_pca)

    elapsed = time.time() - t0
    print(f"   Training time: {elapsed:.2f}s")

    # ── Evaluate on training data (sanity check) ──────
    print("\n📊 Evaluating on training set...")
    preds_normal = model.predict(X_pca)   # +1 = normal, -1 = anomaly
    correct = np.sum(preds_normal == 1)
    print(f"   Normal correctly identified: {correct}/{n_normal} "
          f"({correct/n_normal*100:.1f}%)")

    if n_anomaly > 0:
        X_anom_arr  = np.array(X_anomaly)
        X_anom_sc   = scaler.transform(X_anom_arr)
        X_anom_pca  = pca.transform(X_anom_sc)
        preds_anom  = model.predict(X_anom_pca)
        detected    = np.sum(preds_anom == -1)
        print(f"   Anomaly correctly detected: {detected}/{n_anomaly} "
              f"({detected/n_anomaly*100:.1f}%)")

    # ── Save model bundle ─────────────────────────────
    bundle = {
        "model":        model,
        "scaler":       scaler,
        "pca":          pca,
        "model_type":   model_type,
        "img_size":     IMG_SIZE,
        "n_normal":     n_normal,
        "n_anomaly":    n_anomaly,
        "trained_at":   time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\n✅ Model saved → {MODEL_PATH}")

    # Save metadata as JSON for dashboard display
    meta = {k: v for k, v in bundle.items()
            if isinstance(v, (str, int, float, list, tuple))}
    with open(os.path.join(MODEL_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "="*55)
    print("  Training complete!")
    print(f"  Model type  : {model_type}")
    print(f"  Normal imgs : {n_normal}")
    print(f"  Anomaly imgs: {n_anomaly}")
    print("="*55)
    print("\n▶  Next step: run server.py to start live detection")
    print(f"   python3 server.py --no-esp32 --port 8080\n")


# ── CLI ───────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SolarGuard AI — Train Model")
    parser.add_argument("--model", default="auto",
                        choices=["auto", "iso", "svm_oc"],
                        help="Model type: auto (default), iso (IsolationForest), svm_oc (OneClassSVM)")
    args = parser.parse_args()
    train(args.model)
