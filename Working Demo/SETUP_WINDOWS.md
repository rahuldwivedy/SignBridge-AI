# SignBridge AI — Local Demo Setup (Windows)

## 1. Folder structure

Create a project folder and arrange files like this:

```
signbridge-ai\
├── streamlit_app.py
├── requirements.txt
└── models\
    ├── asl_alphabet_cnn.keras
    └── label_encoder.pkl
```

Put the two model files you already downloaded from Google Drive into the `models\` folder.

## 2. Create a virtual environment (Command Prompt or PowerShell)

```
cd path\to\signbridge-ai
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` appear at the start of your terminal prompt once it's active.

## 3. Install dependencies

```
pip install -r requirements.txt
```

This will take a few minutes — mediapipe, tensorflow, and streamlit-webrtc are sizeable packages.

## 4. Run the app

```
streamlit run streamlit_app.py
```

This opens a browser tab (usually `http://localhost:8501`). Click **START** on the video widget and allow camera access when Windows/Chrome prompts you.

## Common Windows issues

- **"No module named cv2"** → the venv wasn't activated when you ran `pip install`. Re-activate with `venv\Scripts\activate` and reinstall.
- **Camera doesn't start / black screen** → check Windows Settings → Privacy & Security → Camera → make sure "Let apps access your camera" and browser access are both turned on.
- **streamlit-webrtc fails to connect** → close other apps using the webcam (Zoom, Teams, etc.) and refresh the browser tab.
- **Predictions look wrong / laggy** → good, plain-background lighting matters a lot since the model was trained on clean Sign MNIST images, not real-world backgrounds. This gap between training data and live webcam conditions is worth mentioning as a limitation in your write-up.

## Stopping the app

`Ctrl + C` in the terminal, then `deactivate` to exit the virtual environment.
