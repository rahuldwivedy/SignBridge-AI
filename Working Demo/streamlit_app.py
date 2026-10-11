"""
SignBridge AI — Live ASL Alphabet Recognition (Streamlit + MediaPipe + CNN)

Run locally with:  streamlit run streamlit_app.py

Folder structure expected:
    signbridge-ai/
    ├── streamlit_app.py
    └── models/
        ├── asl_alphabet_cnn.keras
        └── label_encoder.pkl
"""

import pickle
import numpy as np
import cv2
import streamlit as st
import mediapipe as mp
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.set_page_config(page_title="SignBridge AI", layout="wide")

MODEL_PATH = "models/asl_alphabet_cnn.keras"
ENCODER_PATH = "models/label_encoder.pkl"


@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    return model, le


model, le = load_assets()

# Sign MNIST excludes J and Z (they require motion). Map raw label -> letter.
ALL_LETTERS = [chr(65 + i) for i in range(26) if chr(65 + i) not in ("J", "Z")]


def decoded_label_to_letter(raw_label):
    # Sign MNIST's raw labels are direct alphabet positions: A=0, B=1, ... Z=25.
    # J (9) and Z (25) simply never appear in the data, but the numbering itself
    # is NOT compressed to remove their slots — so no re-indexing is needed here.
    if 0 <= raw_label <= 25:
        return chr(65 + raw_label)
    return "?"


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

PADDING = 30  # pixels of margin around the detected hand box


class ASLProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.last_letter = ""
        self.last_confidence = 0.0
        self.last_crop = None  # stores the exact 28x28 grayscale image fed to the model, for debugging

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # mirror horizontally — corrects orientation mismatch with training data
        h, w, _ = img.shape

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            xs = [lm.x * w for lm in hand_landmarks.landmark]
            ys = [lm.y * h for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(xs)) - PADDING, int(max(xs)) + PADDING
            y_min, y_max = int(min(ys)) - PADDING, int(max(ys)) + PADDING
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)

            crop = img[y_min:y_max, x_min:x_max]

            if crop.size > 0:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

                # Pad to a square FIRST so resizing doesn't stretch/distort the hand shape.
                ch, cw = gray.shape
                side = max(ch, cw)
                top = (side - ch) // 2
                bottom = side - ch - top
                left = (side - cw) // 2
                right = side - cw - left
                square = cv2.copyMakeBorder(
                    gray, top, bottom, left, right,
                    borderType=cv2.BORDER_CONSTANT, value=0
                )

                resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
                normalized = resized.astype("float32") / 255.0
                model_input = normalized.reshape(1, 28, 28, 1)

                preds = model.predict(model_input, verbose=0)
                pred_idx = int(np.argmax(preds))
                confidence = float(np.max(preds))

                raw_label = le.inverse_transform([pred_idx])[0]
                letter = decoded_label_to_letter(raw_label)

                self.last_letter = letter
                self.last_confidence = confidence
                self.last_crop = resized.copy()  # what the model actually saw, for the debug panel

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{self.last_letter} ({self.last_confidence*100:.0f}%)",
                (x_min, max(30, y_min - 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
            )
        else:
            self.last_letter = ""
            self.last_confidence = 0.0

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.title("🤟 SignBridge AI — Live ASL Alphabet Recognition")
st.markdown(
    "Show a static ASL fingerspelling letter to your webcam. "
    "Green box = detected hand region fed into the CNN. "
    "**Note:** J and Z aren't supported (they require motion, not a static pose)."
)

col1, col2 = st.columns([3, 1])

with col1:
    webrtc_ctx = webrtc_streamer(
        key="signbridge-asl",
        video_processor_factory=ASLProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("Tips")
    st.markdown(
        "- Plain background helps\n"
        "- Keep your hand centered in frame\n"
        "- Good, even lighting improves accuracy\n"
        "- Hold the pose steady for ~1 second"
    )

st.divider()
st.subheader("🔍 Debug: what the model actually sees")
st.caption(
    "This is the exact 28x28 grayscale crop fed into the CNN. If it looks unrecognizable as "
    "a hand shape (too much background, wrong framing, oddly cropped), that's usually why "
    "predictions look wrong — not the model itself."
)
debug_placeholder = st.empty()

if webrtc_ctx.state.playing:
    import time as _time
    while webrtc_ctx.state.playing:
        if webrtc_ctx.video_processor and webrtc_ctx.video_processor.last_crop is not None:
            crop = webrtc_ctx.video_processor.last_crop
            debug_placeholder.image(
                crop,
                width=200,
                caption=f"Model input — predicted: {webrtc_ctx.video_processor.last_letter}",
                clamp=True,
            )
        _time.sleep(0.3)
