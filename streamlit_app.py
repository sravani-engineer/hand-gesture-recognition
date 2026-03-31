import streamlit as st
import cv2
import joblib
import numpy as np
import tempfile
from collections import Counter, deque

# -----------------------------
# SAFE IMPORT (CRITICAL FIX)
# -----------------------------
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    USE_MEDIAPIPE = True
except:
    USE_MEDIAPIPE = False

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
# Cloud Warning (IMPORTANT)
# -----------------------------
if not USE_MEDIAPIPE:
    st.warning("""
⚠️ Live hand detection is disabled in this cloud deployment.

Reason:
MediaPipe is not supported in this environment (Python 3.14).

👉 To test full real-time gesture detection:
Run this app locally.
""")

# -----------------------------
# Robustness Highlight
# -----------------------------
st.info("⚠️ This model is tested under real-world variations: lighting, background, and user differences.")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📌 About")
st.sidebar.write("""
- 22K frames dataset  
- 12 sessions  
- 5 users  
- Tested under domain shift  
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
3. Try different lighting conditions  
4. Test different backgrounds  
""")

st.markdown("---")

# -----------------------------
# System Design (BIG BOOST)
# -----------------------------
st.markdown("## 🧠 System Design")
st.code("""
Video → MediaPipe → Landmarks → Model → Prediction
""")

st.markdown("---")

# -----------------------------
# Load Model (MUST WORK)
# -----------------------------
MODEL_PATH = "models/gesture_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except:
    st.error("❌ Model not found!")
    st.stop()

# -----------------------------
# Upload
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload Gesture Video", type=["mp4", "avi", "mov"])

# ============================================================
# FULL PIPELINE (ONLY IF MEDIAPIPE AVAILABLE)
# ============================================================
if uploaded_file is not None:

    if USE_MEDIAPIPE:

        hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        mp_draw = mp.solutions.drawing_utils

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
        st.write("Processing frames and detecting hand landmarks...")

        stframe = st.empty()
        progress = st.progress(0)

        gesture_counter = Counter()
        confidence_list = []
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

                if hasattr(model, "predict_proba"):
                    confidence = np.max(model.predict_proba(features))
                    confidence_list.append(confidence)
                else:
                    confidence = None

                stable_pred = Counter(window).most_common(1)[0][0]

                # 🔥 HERO OUTPUT
                prediction_box.markdown(f"# 🖐 {stable_pred}")

                if confidence is not None:
                    st.metric("Confidence", f"{round(confidence*100,2)}%")

                st.caption("Stable prediction across last 10 frames")

            else:
                prediction_box.markdown("## ❌ No Hand Detected")

            stframe.image(frame, channels="BGR")
            progress.progress(min(frame_count / total_frames, 1.0))

        cap.release()

        st.markdown("---")
        st.success("✅ Processing complete")

    # ============================================================
    # DEMO MODE (CLOUD FALLBACK)
    # ============================================================
    else:
        st.info("Demo mode: UI + pipeline only (no live detection)")

        st.markdown("## 🎯 Demo Prediction")
        st.markdown("# 🖐 Gesture Recognition Model Ready")

        st.metric("Confidence", "N/A")
        st.caption("Run locally for real-time predictions")

# ============================================================
# FINAL RESULT SECTION
# ============================================================
st.markdown("---")
st.markdown("## 🏁 Final Output")

st.write("✔ Model loaded successfully")
st.write("✔ System ready for inference")