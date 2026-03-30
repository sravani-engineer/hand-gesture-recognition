import streamlit as st
import cv2
import mediapipe as mp
import joblib
import numpy as np
import tempfile
from collections import Counter, deque

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Gesture Recognition", layout="centered")

# -----------------------------
# Header (Dashboard Style)
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
# Highlight Robustness
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
# Load Model
# -----------------------------
MODEL_PATH = "models/gesture_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except:
    st.error("❌ Model not found!")
    st.stop()

# -----------------------------
# MediaPipe Setup
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(landmarks):
    landmarks = np.array(landmarks)

    base = landmarks[0]
    landmarks = landmarks - base

    max_val = np.max(np.abs(landmarks))
    if max_val != 0:
        landmarks = landmarks / max_val

    return landmarks.flatten().reshape(1, -1)

# -----------------------------
# Upload
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload Gesture Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    if not cap.isOpened():
        st.error("❌ Cannot open video")
        st.stop()

    # -----------------------------
    # System Status
    # -----------------------------
    st.markdown("---")
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
    detected_frames = 0

    # -----------------------------
    # Live Prediction (Hero Section)
    # -----------------------------
    st.markdown("---")
    st.markdown("## 🎯 Live Prediction")
    st.caption("Real-time prediction with temporal smoothing")

    prediction_box = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_count += 1

        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            detected_frames += 1

            hand = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            landmarks = [[lm.x, lm.y, lm.z] for lm in hand.landmark]
            features = extract_features(landmarks)

            try:
                pred = model.predict(features)[0]
                window.append(pred)

                # Confidence
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(features)
                    confidence = np.max(probs)
                    confidence_list.append(confidence)
                else:
                    confidence = None

                stable_pred = Counter(window).most_common(1)[0][0]

            except:
                stable_pred = "Error"
                confidence = None

            gesture_counter[stable_pred] += 1

            # -----------------------------
            # HERO OUTPUT (FINAL POLISH)
            # -----------------------------
            prediction_box.markdown(f"# 🖐 {stable_pred}")

            if confidence is not None:
                st.metric("Confidence", f"{round(confidence*100,2)}%")

            st.caption("Stable prediction across last 10 frames")

            cv2.putText(
                frame,
                stable_pred,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        else:
            prediction_box.markdown("## ❌ No Hand Detected")

        stframe.image(frame, channels="BGR")
        progress.progress(min(frame_count / total_frames, 1.0))

    cap.release()

    st.markdown("---")
    st.success("✅ Processing complete")

    # -----------------------------
    # Final Result
    # -----------------------------
    st.markdown("## 🎯 Final Result")

    if gesture_counter:
        final_gesture = max(gesture_counter, key=gesture_counter.get)
        st.markdown(f"## 🏁 Final Prediction: {final_gesture}")

        if confidence_list:
            avg_conf = np.mean(confidence_list)
            st.write(f"Average Confidence: {round(avg_conf*100,2)}%")

    else:
        st.error("❌ No gesture detected")

    # -----------------------------
    # Processing Details
    # -----------------------------
    st.markdown("---")
    st.markdown("## 📊 Processing Details")

    st.write(f"Total Frames: {frame_count}")
    st.write(f"Frames with Hand Detected: {detected_frames}")

    if frame_count > 0:
        detection_rate = (detected_frames / frame_count) * 100
        st.write(f"Detection Rate: {round(detection_rate, 2)}%")

    # -----------------------------
    # Prediction Summary
    # -----------------------------
    st.markdown("---")
    st.markdown("## 📊 Prediction Summary")

    for g, count in gesture_counter.items():
        st.write(f"{g}: {count} frames")