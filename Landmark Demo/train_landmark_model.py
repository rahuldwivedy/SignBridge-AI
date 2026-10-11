"""
SignBridge AI — Train Landmark-Based Alphabet Classifier (with augmentation)

Trains a classifier on the landmark data you collected with collect_landmarks.py.
This version adds synthetic augmentation (small random rotations + noise) to each
training sample. This matters because extract_landmark_vector() only normalizes
translation and scale, NOT rotation — so a Random Forest trained on raw landmarks
can be thrown off by small hand-tilt differences between your recording session
and your live demo. Augmenting with rotated/noisy copies makes the model far more
robust to that drift, which is the most likely cause of letters like L/M/N/S
getting confused live.

Run with:  python train_landmark_model.py

Requires: landmark_data.csv (created by collect_landmarks.py) in the same folder.
Saves: models/landmark_rf_model.pkl and models/landmark_label_encoder.pkl
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix

DATA_CSV = "landmark_data.csv"
MODEL_DIR = "models"

# --- Augmentation settings ---
AUGMENTS_PER_SAMPLE = 4     # how many synthetic variants to generate per real sample
ROTATION_RANGE_DEG = 20     # max +/- rotation applied (in-plane, around wrist)
NOISE_STD = 0.02            # gaussian jitter added to normalized coords


def augment_sample(vec, rotation_range_deg=ROTATION_RANGE_DEG, noise_std=NOISE_STD):
    """Applies a random small in-plane rotation (around the wrist, which is the
    origin after normalization) plus gaussian noise to a 63-dim landmark vector.
    This simulates the natural variation in hand tilt/orientation you'll see live
    that isn't present in a single recording session."""
    coords = vec.reshape(21, 3).copy()

    angle = np.radians(np.random.uniform(-rotation_range_deg, rotation_range_deg))
    c, s = np.cos(angle), np.sin(angle)
    x, y = coords[:, 0].copy(), coords[:, 1].copy()
    coords[:, 0] = x * c - y * s
    coords[:, 1] = x * s + y * c

    coords += np.random.normal(0, noise_std, coords.shape)
    return coords.flatten()


def augment_dataset(X, y, n_augments=AUGMENTS_PER_SAMPLE):
    """Expands X, y with synthetic augmented copies. Only call this on TRAINING
    data (after train_test_split) so the test set stays a clean, honest measure
    of real-world performance."""
    aug_X, aug_y = [X], [y]
    for _ in range(n_augments):
        aug_X.append(np.array([augment_sample(row) for row in X]))
        aug_y.append(y)
    return np.vstack(aug_X), np.concatenate(aug_y)


def main():
    if not os.path.exists(DATA_CSV):
        print(f"ERROR: {DATA_CSV} not found. Run collect_landmarks.py first.")
        return

    df = pd.read_csv(DATA_CSV)
    print(f"Loaded {len(df)} samples across {df['label'].nunique()} letters")
    print(df['label'].value_counts().sort_index())

    low_count_letters = df['label'].value_counts()[df['label'].value_counts() < 20]
    if len(low_count_letters) > 0:
        print("\nWarning: these letters have under 20 samples — consider recording more:")
        print(low_count_letters)

    X = df.drop('label', axis=1).values
    y_raw = df['label'].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nBefore augmentation: training on {len(X_train)} samples")
    X_train, y_train = augment_dataset(X_train, y_train)
    print(f"After augmentation ({AUGMENTS_PER_SAMPLE}x + original): training on {len(X_train)} samples")
    print(f"Testing on {len(X_test)} samples (untouched, real data only)...")

    clf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    print(f"\nMacro F1 on held-out (real, unaugmented) test set: {macro_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("\nConfusion matrix (rows=true, cols=predicted) — check which letters swap:")
    cm = confusion_matrix(y_test, y_pred)
    print("Letters:", list(le.classes_))
    print(cm)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "landmark_rf_model.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(MODEL_DIR, "landmark_label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    print(f"\nSaved model + label encoder to {MODEL_DIR}/")
    print("Ready to plug into streamlit_app_landmark.py")


if __name__ == "__main__":
    main()
