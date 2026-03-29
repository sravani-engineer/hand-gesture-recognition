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

st.title("🖐 Hand Gesture Recognition System")
st.caption("Robust gesture recognition evaluated under real-world variations")

# -----------------------------
# Supported Gestures
# -----------------------------
st.markdown("## 🎮 Supported Gestures")
st.write("✊ Fist | ✋ Open Palm | ☝ Index | 🖖 Four Fingers (thumb down) | 🤘 Small (thumb & pinky up, middle 3 down)")

st.info("Model trained on multi-user, multi-condition dataset (~22K frames)")
st.markdown("💡 Tip: Upload videos with different lighting or backgrounds to test robustness")

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = "models/gesture_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except:
    st.error("❌ Model not found! Check models/gesture_model.pkl")
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
# Upload Video
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload a gesture video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    if not cap.isOpened():
        st.error("❌ Cannot open video")
        st.stop()

    stframe = st.empty()
    status_text = st.empty()
    progress = st.progress(0)

    gesture_counter = Counter()
    confidence_list = []
    window = deque(maxlen=5)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 1

    frame_count = 0
    detected_frames = 0

    st.info("Processing video...")

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

                final_pred = Counter(window).most_common(1)[0][0]

            except:
                final_pred = "Error"
                confidence = None

            gesture_counter[final_pred] += 1

            # Live prediction display
            if confidence is not None:
                status_text.markdown(
                    f"### 🧠 Prediction: **{final_pred}** ({round(confidence*100,2)}%)"
                )
            else:
                status_text.markdown(f"### 🧠 Prediction: **{final_pred}**")

            cv2.putText(
                frame,
                final_pred,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        else:
            status_text.markdown("### ❌ No Hand Detected")

        stframe.image(frame, channels="BGR")
        progress.progress(min(frame_count / total_frames, 1.0))

    cap.release()

    st.success("✅ Processing complete")

    # -----------------------------
    # Final Result
    # -----------------------------
    st.markdown("## 🎯 Final Result")

    if gesture_counter:
        final_gesture = max(gesture_counter, key=gesture_counter.get)
        st.success(f"Predicted Gesture: {final_gesture}")

        if confidence_list:
            avg_conf = np.mean(confidence_list)
            st.write(f"Confidence: {round(avg_conf*100,2)}%")

        st.caption("Prediction based on majority voting across frames for stability.")

    else:
        st.error("❌ No valid gesture detected")

    # -----------------------------
    # Processing Details (renamed)
    # -----------------------------
    st.markdown("## 📊 Processing Details")

    st.write(f"Total Frames: {frame_count}")
    st.write(f"Frames with Hand Detected: {detected_frames}")

    if frame_count > 0:
        detection_rate = (detected_frames / frame_count) * 100
        st.write(f"Detection Rate: {round(detection_rate, 2)}%")

    # -----------------------------
    # Prediction Summary
    # -----------------------------
    st.markdown("## 📊 Prediction Summary")

    for g, count in gesture_counter.items():
        st.write(f"{g}: {count} frames")