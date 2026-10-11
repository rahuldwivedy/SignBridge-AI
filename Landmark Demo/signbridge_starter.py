"""
===============================================================================
SignBridge AI — Starter Pipeline Script
Capstone Project 2 (PGA 3.0: Machine Learning & Deep Learning)
===============================================================================
This script provides a complete end-to-end foundation for SignBridge AI:
1. Filters the WLASL dataset annotations down to a 25-30 word subset.
2. Extracts 21 3D hand landmarks (63 coordinates) per frame using Google MediaPipe.
3. Defines a 2D CNN for static ASL alphabet gestures (Sign Language MNIST).
4. Defines an LSTM Recurrent Neural Network for dynamic ASL sign sequences.
===============================================================================
"""

import os
import json
import numpy as np

# -----------------------------------------------------------------------------
# 1. Target 25-30 Word Glosses (High Frequency ASL Words)
# -----------------------------------------------------------------------------
TARGET_GLOSSES = [
    "hello", "thankyou", "yes", "no", "please", "sorry", "help", "eat",
    "drink", "water", "book", "computer", "family", "friend", "house", "love",
    "school", "work", "time", "day", "night", "happy", "sad", "like", "want",
    "father", "mother", "more", "finish", "again"
]

def filter_wlasl_json(json_path, output_json_path):
    """
    Reads the full WLASL_v0.3.json annotation file and filters it down
    to only the chosen target glosses (25-30 words).
    """
    if not os.path.exists(json_path):
        print(f"[-] Annotation file '{json_path}' not found. Download it from Kaggle (risangbaskoro/wlasl-processed).")
        return None

    with open(json_path, 'r') as f:
        wlasl_data = json.load(f)

    filtered_data = []
    total_videos = 0

    for item in wlasl_data:
        gloss = item.get('gloss', '').lower()
        if gloss in TARGET_GLOSSES:
            instances = item.get('instances', [])
            total_videos += len(instances)
            filtered_data.append({
                'gloss': gloss,
                'instances': instances
            })

    print(f"[+] Successfully filtered WLASL dataset:")
    print(f"    - Target Words (Glosses): {len(filtered_data)}")
    print(f"    - Total Video Samples: {total_videos}")

    with open(output_json_path, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    print(f"[+] Saved filtered annotations to: {output_json_path}")
    return filtered_data


# -----------------------------------------------------------------------------
# 2. Google MediaPipe Hand Landmark Extraction Pipeline
# -----------------------------------------------------------------------------
def extract_hand_landmarks(frame_bgr, mp_hands_solution):
    """
    Processes a single BGR image/video frame through MediaPipe Hands.
    Returns a flattened 63-dimensional array of normalized (x, y, z) coordinates
    for 21 hand keypoints, or zeros if no hand is detected.
    """
    import cv2
    
    # Convert BGR to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = mp_hands_solution.process(frame_rgb)
    
    landmarks_vector = np.zeros(21 * 3)  # 63 features
    
    if results.multi_hand_landmarks:
        # Take the primary detected hand
        primary_hand = results.multi_hand_landmarks[0]
        coords = []
        for lm in primary_hand.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        landmarks_vector = np.array(coords)
        
    return landmarks_vector


# -----------------------------------------------------------------------------
# 3. Model Architecture Helpers (TensorFlow / PyTorch placeholders)
# -----------------------------------------------------------------------------
def build_static_alphabet_cnn(input_shape=(28, 28, 1), num_classes=24):
    """
    2D CNN model for static ASL alphabet gestures (Sign Language MNIST).
    Input: 28x28 grayscale pixel array.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models

        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.Flatten(),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("[+] Static Alphabet 2D-CNN Model compiled successfully.")
        return model
    except ImportError:
        print("[!] TensorFlow not installed in current environment. Install via pip install tensorflow.")
        return None

def build_dynamic_sequence_lstm(sequence_length=30, feature_dim=63, num_classes=30):
    """
    LSTM Recurrent Neural Network for dynamic ASL sign sequences.
    Input shape: (30 frames, 63 coordinates per frame).
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models

        model = models.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=(sequence_length, feature_dim)),
            layers.Dropout(0.2),
            layers.LSTM(128, return_sequences=False),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("[+] Dynamic Sequence LSTM Model compiled successfully.")
        return model
    except ImportError:
        print("[!] TensorFlow not installed in current environment. Install via pip install tensorflow.")
        return None


if __name__ == "__main__":
    print("=" * 70)
    print("  SignBridge AI — Pipeline Starter Script")
    print("=" * 70)
    print(f"Target Vocabulary ({len(TARGET_GLOSSES)} words):")
    print(", ".join(TARGET_GLOSSES[:10]) + ", ...")
    print("\nNext Steps:")
    print("1. Download 'datamunge/sign-language-mnist' from Kaggle for static ASL alphabet.")
    print("2. Download 'risangbaskoro/wlasl-processed' from Kaggle for word-level videos.")
    print("3. Run this script to extract MediaPipe landmarks & train models!")
