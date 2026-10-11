"""
SignBridge AI — Landmark Data Collection Tool

Collects your own labeled hand-landmark samples via webcam, to train a classifier
that matches YOUR camera, lighting, and hand — closing the domain-gap problem the
pixel-based Sign MNIST model had.

Run with:  python collect_landmarks.py

Controls:
- Press a letter key (a-i, k-y — no j, z, since those need motion) to record
  5 seconds of samples for that letter. Move your hand slightly during recording
  (small rotations/distance changes) so the model sees some natural variation.
- Press ESC to quit and save everything to landmark_data.csv
  (NOTE: this used to be 'q', but 'q' is also a valid ASL letter in this list —
  that meant pressing 'q' always quit instead of ever recording the letter Q.
  Moved to ESC so every letter, including Q, can actually be recorded.)

Tip: aim for at least 40-50 samples per letter. Re-press a letter's key anytime
to add MORE samples for it (they accumulate, not overwrite).
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

VALID_LETTERS = [c for c in "abcdefghiklmnopqrstuvwxy"]  # no j, z (motion signs)
RECORD_SECONDS = 5
OUTPUT_CSV = "landmark_data.csv"


def extract_landmark_vector(hand_landmarks):
    """63-dim vector: 21 landmarks x (x,y,z), normalized for position AND scale
    (relative to wrist, divided by hand size) so it generalizes across distance
    from camera and position in frame."""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    coords -= coords[0]  # translate relative to wrist
    scale = np.linalg.norm(coords, axis=1).max()
    if scale > 0:
        coords /= scale
    return coords.flatten()


def main():
    counts = {letter: 0 for letter in VALID_LETTERS}
    rows = []

    # Resume from existing file if it exists, so re-running doesn't lose past sessions
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                rows.append(row)
                label = row[-1]
                if label in counts:
                    counts[label] += 1
        print(f"Loaded {len(rows)} existing samples from {OUTPUT_CSV}")

    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)

    print("\nControls: press a-i, k-y to record that letter for 5s. Press ESC to save & quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # mirror — must match the Streamlit app's flip

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        # Show sample counts on screen
        y_offset = 20
        for letter in VALID_LETTERS:
            color = (0, 255, 0) if counts[letter] >= 40 else (0, 200, 255)
            cv2.putText(frame, f"{letter.upper()}:{counts[letter]}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y_offset += 20
            if y_offset > 460:
                y_offset = 20

        cv2.putText(frame, "Press a-y letter to record, ESC to quit", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("SignBridge - Landmark Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        elif chr(key) in VALID_LETTERS:
            letter = chr(key)
            print(f"Recording '{letter}' for {RECORD_SECONDS}s — move your hand slightly...")
            start = time.time()
            recorded_this_session = 0

            while time.time() - start < RECORD_SECONDS:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    vec = extract_landmark_vector(hand_landmarks)
                    rows.append(list(vec) + [letter])
                    counts[letter] += 1
                    recorded_this_session += 1

                remaining = RECORD_SECONDS - (time.time() - start)
                cv2.putText(frame, f"RECORDING '{letter.upper()}' — {remaining:.1f}s left",
                            (10, frame.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("SignBridge - Landmark Collector", frame)
                cv2.waitKey(1)

            print(f"  -> Captured {recorded_this_session} samples for '{letter}' (total: {counts[letter]})")

    cap.release()
    cv2.destroyAllWindows()

    # Save to CSV
    header = [f"f{i}" for i in range(63)] + ["label"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} total samples to {OUTPUT_CSV}")
    print("Sample counts per letter:")
    for letter, count in counts.items():
        flag = "OK" if count >= 40 else "LOW - consider recording more"
        print(f"  {letter.upper()}: {count}  [{flag}]")


if __name__ == "__main__":
    main()
