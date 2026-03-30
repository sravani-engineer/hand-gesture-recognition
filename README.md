# 🖐️ Hand Gesture Recognition with Cross-Condition Robustness Evaluation

> A hand gesture recognition system explicitly evaluated under real-world domain shifts using session-based validation.

---

## 🚀 Key Highlights

- 📊 Evaluated across **12 sessions, 5 users, and ~22,000 frames**
- 🔬 Uses **session-based splitting** to simulate real-world deployment
- ⚠️ Demonstrates **performance drop under domain shift (100% → 88%)**
- 📉 Includes **confusion matrix and failure analysis**
- 🧪 Explicitly avoids **data leakage using session-based validation**
- 🎯 Focused on **robustness, not just accuracy**

---

## 🎥 Demo

<p align="center">
  <img src="results/demo.gif" width="600"/>
</p>

---

## 🌐 Live Demo (Streamlit App)

🚀 An interactive web application is available to test the model in real time.

👉 **Live App:** https://your-app-name.streamlit.app  

### 🖥️ Features
- Real-time gesture prediction from uploaded videos  
- Confidence score for each prediction  
- Temporal smoothing for stable predictions  
- Visual hand landmark detection using MediaPipe  
- Displays processing statistics (frame count, detection rate)  

### 🎯 What You Can Test
- Different lighting conditions  
- Background variations  
- Distance from camera  
- Gesture stability across frames  

### ⚠️ Known Limitations
- Reduced accuracy in low lighting  
- Confusion between similar gestures (open vs four)  
- Sensitive to hand distance and visibility  

💡 This app demonstrates how the model behaves under **real-world conditions**, not just controlled environments.

<p align="center">
  <img src="results/streamlit_ui.png" width="700"/>
</p>

---

## 📌 Overview

🚨 Unlike typical gesture recognition projects that rely on random train-test splits, this system uses **session-based evaluation** to simulate real-world deployment conditions.

The model is explicitly tested under **domain shifts** such as lighting variation, background clutter, distance, and user differences.

This project focuses on **robustness, generalization, and failure analysis**, not just high accuracy.

---

## 🎯 Problem Statement

Most gesture recognition systems perform well in controlled environments but fail under:

- Different lighting conditions  
- Background variations  
- Distance changes  
- Different users and hand variations  

👉 This project evaluates how well a model performs under these real-world challenges.

---

## 🚀 Solution

- Extracted **21 hand landmarks** using MediaPipe  
- Applied normalization for scale invariance  
- Trained a **Random Forest classifier**  
- Designed a **session-based evaluation strategy**  
- Measured performance under different real-world conditions  

---

## 📦 Dataset Design

- Total Sessions: **12**
- Users: **5 individuals**
- Gestures: **5 static classes (fist, open, index, four, small)**
- Total Samples: **~22,000 frames**

### Data Collection Strategy

The dataset was intentionally collected across multiple sessions to introduce real-world variability:

- Backgrounds: plain and cluttered indoor environments  
- Lighting Conditions: bright, dim, and natural light  
- User Variation: different hand shapes and sizes  
- Pose Variation: changes in hand position, orientation, and distance  

### Objective

The dataset is designed to evaluate gesture recognition performance under **real-world conditions**, rather than controlled environments.

---

## 📊 Dataset Strengths & Limitations

### Strengths
- Multi-user dataset improves **potential for generalization across users**  
- Environmental diversity (lighting + background)  
- Large sample size (~22K frames)  

### Limitations
- Limited to static gestures  
- No temporal modeling of dynamic gestures  

---

## 🧪 Evaluation Strategy (Key Design)

🚨 Instead of random train-test splits, this project uses **session-based splitting**:

- Train and test data come from **different sessions**
- Prevents **data leakage**
- Simulates **real-world deployment scenarios**
- Captures **domain shift across environments**

This ensures performance reflects **true generalization**, not memorization.

---

## 📊 Results

### Overall Performance
- Overall Accuracy: **~93–94%**

### Robustness Evaluation

| Condition | Description | Accuracy |
|----------|------------|---------|
| Controlled Environment | Uniform background, consistent lighting, single user | 1.0000 |
| Moderate Variation | Slight lighting changes, simple background variations | 0.9999 |
| Challenging Conditions | Low light, cluttered background, multiple users, varying distances | 0.8842 |

📉 **Performance Drop:** ~12% under real-world conditions

### Summary Statistics
- Minimum Accuracy: **0.8842**  
- Maximum Accuracy: **1.0000**  
- Average Accuracy: **0.9614**

---

## 🧠 Key Insights

- Session-based evaluation reveals performance gaps hidden by random splits  
- Landmark-based features are sensitive to lighting and distance variations  
- Similar gestures (open vs four) lead to misclassification due to feature similarity  
- Model generalizes across users but degrades under challenging conditions  

---

## 📉 Confusion Matrix

<p align="center">
  <img src="results/confusion_matrix.png" width="500"/>
</p>

---

## ⚠️ Failure Analysis

- Confusion between **open** and **four** due to similar finger configurations  
- Reduced landmark stability in **low lighting conditions**  
- Performance degradation when hand occupies **smaller region of frame**  
- Sensitivity to **partial occlusions and motion blur**  

---

## 💼 Why This Matters

This project simulates real-world ML deployment challenges where models must handle **distribution shifts**.

It demonstrates:

- Handling **domain shift**  
- Avoiding **data leakage**  
- Performing **failure analysis**  

👉 These are critical for production-level ML systems.

---

## 📂 Project Structure
robust-hand-gesture-recognition/
│
├── streamlit_app.py
├── models/
│ └── gesture_model.pkl
│
├── src/
│ ├── extract_landmarks.py
│ ├── preprocess.py
│ ├── train_model.py
│ ├── evaluate.py
│ ├── evaluate_conditions.py
│ ├── realtime_inference.py
│ └── test_hand_detection.py
│
├── results/
│ ├── demo.gif
│ ├── confusion_matrix.png
│ └── streamlit_ui.png
│
├── requirements.txt
├── README.md
└── .gitignore

---

## ⚙️ How to Run

### 1. Clone repository

```bash
git clone https://github.com/sravani-engineer/robust-hand-gesture-recognition.git
cd robust-hand-gesture-recognition
### 2. Create virtual environment
python -m venv venv
venv\Scripts\activate
### 3. Install dependencies
pip install -r requirements.txt
### 4. Run Streamlit App
streamlit run streamlit_app.py

🔄 Pipeline

Video Input → Landmark Extraction → Preprocessing → Model Training → Evaluation → Real-time Inference


---
