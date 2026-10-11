"""
SignBridge AI — Live ASL Alphabet Recognition (Landmark-Based, with smoothing)

Uses MediaPipe hand-landmark coordinates + a Random Forest classifier trained on
YOUR own webcam data. This version adds:
  1. Temporal smoothing (majority vote over a rolling window of recent frames)
     so a single noisy/misclassified frame doesn't flash a wrong letter on screen.
  2. A confidence threshold — low-confidence single-frame predictions don't count
     as a "vote", which helps filter out transitional hand shapes (mid-motion).
  3. A live debug panel showing the top-3 predicted letters + probabilities for
     the current frame, so you can see exactly what the model is confusing
     something with if it still misfires — very useful while testing before
     your presentation.

Run locally with:  streamlit run streamlit_app_landmark.py

Folder structure expected:
    signbridge-ai/
    ├── streamlit_app_landmark.py
    └── models/
        ├── landmark_rf_model.pkl
        └── landmark_label_encoder.pkl
"""

import pickle
from collections import deque, Counter

import numpy as np
import cv2
import streamlit as st
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.set_page_config(page_title="SignBridge AI", layout="wide")

MODEL_PATH = "models/landmark_rf_model.pkl"
ENCODER_PATH = "models/landmark_label_encoder.pkl"

# --- Smoothing settings — tune these live if needed ---
BUFFER_SIZE = 12          # how many recent frames to consider
MIN_VOTE_FRACTION = 0.5   # fraction of buffer that must agree to display a letter
MIN_FRAME_CONFIDENCE = 0.4  # per-frame predictions below this don't count as a vote


@st.cache_resource
def load_assets():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    return model, le


model, le = load_assets()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_landmark_vector(hand_landmarks):
    """Must exactly match the normalization used in collect_landmarks.py."""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    coords -= coords[0]
    scale = np.linalg.norm(coords, axis=1).max()
    if scale > 0:
        coords /= scale
    return coords.flatten()


class ASLProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.recent_votes = deque(maxlen=BUFFER_SIZE)

        # Stable, smoothed values — this is what gets displayed
        self.stable_letter = ""
        self.stable_confidence = 0.0

        # Raw single-frame values — for the debug panel only
        self.last_top3 = []  # list of (letter, prob) tuples

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # mirror — must match collection script
        h, w, _ = img.shape

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            vec = extract_landmark_vector(hand_landmarks).reshape(1, -1)
            probs = model.predict_proba(vec)[0]

            top3_idx = np.argsort(probs)[::-1][:3]
            self.last_top3 = [(le.inverse_transform([i])[0].upper(), float(probs[i])) for i in top3_idx]

            pred_idx = int(top3_idx[0])
            confidence = float(probs[pred_idx])
            letter = le.inverse_transform([pred_idx])[0].upper()

            # Only "vote" with this frame if it's confident enough
            if confidence >= MIN_FRAME_CONFIDENCE:
                self.recent_votes.append(letter)

            if len(self.recent_votes) > 0:
                counts = Counter(self.recent_votes)
                top_letter, top_count = counts.most_common(1)[0]
                if top_count / len(self.recent_votes) >= MIN_VOTE_FRACTION:
                    self.stable_letter = top_letter
                    # report average confidence among votes for this letter this window
                    self.stable_confidence = top_count / len(self.recent_votes)

            xs = [lm.x * w for lm in hand_landmarks.landmark]
            ys = [lm.y * h for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(xs)) - 20, int(max(xs)) + 20
            y_min, y_max = int(min(ys)) - 20, int(max(ys)) + 20

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            if self.stable_letter:
                cv2.putText(
                    img,
                    f"{self.stable_letter} ({self.stable_confidence*100:.0f}%)",
                    (x_min, max(30, y_min - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3,
                )
        else:
            self.recent_votes.clear()
            self.stable_letter = ""
            self.stable_confidence = 0.0
            self.last_top3 = []

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.title("🤟 SignBridge AI — Live ASL Alphabet Recognition")
st.markdown(
    "Landmark-based recognition — trained on real webcam data, not just a benchmark dataset. "
    "Show a static ASL fingerspelling letter to your webcam and hold it steady. "
    "**Note:** J and Z aren't supported (they require motion, not a static pose)."
)

col1, col2 = st.columns([3, 1])

with col1:
    webrtc_ctx = webrtc_streamer(
        key="signbridge-asl-landmark",
        video_processor_factory=ASLProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("Tips")
    st.markdown(
        "- Plain background helps\n"
        "- Keep your hand centered in frame\n"
        "- Good, even lighting improves accuracy\n"
        "- Hold the pose steady for ~1 second — the display now smooths over "
        "a short window of frames rather than reacting instantly, so give it "
        "a moment to lock in"
    )

st.divider()
st.subheader("🔍 Debug: current frame's top-3 predictions")
st.caption(
    "Useful while testing before your presentation — if a letter keeps showing up "
    "as a close second/third for the wrong sign, that tells you exactly which "
    "letters need more/varied training samples."
)
debug_placeholder = st.empty()

if webrtc_ctx.state.playing:
    import time as _time
    while webrtc_ctx.state.playing:
        if webrtc_ctx.video_processor and webrtc_ctx.video_processor.last_top3:
            top3 = webrtc_ctx.video_processor.last_top3
            lines = "  |  ".join(f"**{letter}**: {prob*100:.0f}%" for letter, prob in top3)
            debug_placeholder.markdown(lines)
        _time.sleep(0.2)
