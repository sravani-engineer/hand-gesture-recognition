import streamlit as st
import cv2
import joblib
import numpy as np
import tempfile
from collections import Counter, deque

# -----------------------------
# SAFE IMPORT (FIXED)
# -----------------------------
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    USE_MEDIAPIPE = True
except Exception as e:
    USE_MEDIAPIPE = False

# DEBUG (IMPORTANT)
st.write("MediaPipe Available:", USE_MEDIAPIPE)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Gesture Recognition", layout="centered")

# -----------------------------
# Header
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.title("🖐 Hand Gesture Recognition")
    st.subheader("Real-Time Robust Gesture Detection")

with col2:
    st.metric("Model", "Random Forest")
    st.metric("Dataset", "22K Frames")

st.markdown("---")

# -----------------------------
# Cloud Warning
# -----------------------------
if not USE_MEDIAPIPE:
    st.warning("""
⚠️ Live hand detection is disabled in this cloud deployment.

Reason:
MediaPipe is not supported in this environment.

👉 Run locally for full functionality.
""")

# -----------------------------
# Info
# -----------------------------
st.info("⚠️ Tested under real-world variations: lighting, background, and user differences.")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📌 About")
st.sidebar.write("""
- 22K frames dataset  
- 12 sessions  
- 5 users  
- Domain shift tested  
""")

st.sidebar.warning("""
⚠️ Limitations:
- Low light reduces accuracy  
- Confusion: open vs four  
- Sensitive to distance  
""")

# -----------------------------
# Instructions
# -----------------------------
st.markdown("## 🎮 Instructions")
st.info("""
1. Upload a gesture video  
2. Ensure hand is clearly visible  
3. Try different lighting  
4. Test different backgrounds  
""")

st.markdown("---")

# -----------------------------
# System Design
# -----------------------------
st.markdown("## 🧠 System Design")
st.code("Video → MediaPipe → Landmarks → Model → Prediction")

st.markdown("---")

# -----------------------------
# Load Model
# -----------------------------
try:
    model = joblib.load("models/gesture_model.pkl")
except:
    st.error("❌ Model not found!")
    st.stop()

# -----------------------------
# Upload
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload Gesture Video", type=["mp4", "avi", "mov"])

# ============================================================
# MAIN LOGIC
# ============================================================
if uploaded_file is not None:

    # ============================================================
    # FULL PIPELINE (ONLY IF MEDIAPIPE WORKS)
    # ============================================================
    if USE_MEDIAPIPE:

        hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )

        def extract_features(landmarks):
            landmarks = np.array(landmarks)
            base = landmarks[0]
            landmarks = landmarks - base

            max_val = np.max(np.abs(landmarks))
            if max_val != 0:
                landmarks = landmarks / max_val

            return landmarks.flatten().reshape(1, -1)

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)

        if not cap.isOpened():
            st.error("❌ Cannot open video")
            st.stop()

        st.markdown("### ⚙️ System Status")
        st.write("Processing frames...")

        stframe = st.empty()
        progress = st.progress(0)

        window = deque(maxlen=10)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1

        frame_count = 0

        st.markdown("---")
        st.markdown("## 🎯 Live Prediction")

        prediction_box = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:

                hand = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                landmarks = [[lm.x, lm.y, lm.z] for lm in hand.landmark]
                features = extract_features(landmarks)

                pred = model.predict(features)[0]
                window.append(pred)

                stable_pred = Counter(window).most_common(1)[0][0]

                # HERO OUTPUT
                prediction_box.markdown(f"# 🖐 {stable_pred}")
                st.caption("Stable prediction across last 10 frames")

            else:
                prediction_box.markdown("## ❌ No Hand Detected")

            stframe.image(frame, channels="BGR")
            progress.progress(min(frame_count / total_frames, 1.0))

        cap.release()
        st.success("✅ Processing complete")

    # ============================================================
    # FALLBACK MODE
    # ============================================================
    else:
        st.info("Demo mode: MediaPipe not available")

        st.markdown("## 🎯 Demo Output")
        st.markdown("# 🖐 Gesture Model Ready")
        st.caption("Run locally for full detection")

# -----------------------------
# Final Section
# -----------------------------
st.markdown("---")
st.markdown("## 🏁 Final Output")

st.write("✔ Model loaded")
st.write("✔ App running")