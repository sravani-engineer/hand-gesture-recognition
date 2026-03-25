# 🖐️ Evaluating Hand Gesture Recognition Robustness Under Real-World Conditions

> A robustness-focused hand gesture recognition system evaluated under real-world domain shifts using session-based validation.

---

## 🎥 Demo

<p align="center">
  <img src="results/demo.gif" width="600"/>
</p>

---

## 📌 Overview

🚨 Unlike most hand gesture recognition projects that rely on random train-test splits, this system uses **session-based evaluation to simulate real-world deployment conditions**.

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

## 📦 Dataset Details

- Total samples: **22,255**  
- Total sessions: **12**  

Data collected across:
- Multiple users  
- Different lighting conditions  
- Various backgrounds  
- Different camera qualities  

Each session represents a **distinct real-world scenario** used for evaluation.

---

## 🛠 Tech Stack

- Python  
- OpenCV  
- MediaPipe  
- Scikit-learn  
- NumPy, Pandas  
- Matplotlib, Seaborn  

---

## 📊 Results

### Overall Performance

- Accuracy: **~93–94%**

---

### Robustness Evaluation

| Condition | Description | Accuracy |
|----------|------------|---------|
| Controlled Environment | Uniform background, consistent lighting, single user | 1.0000 |
| Moderate Variation | Slight lighting changes, simple background variations | 0.9999 |
| Challenging Conditions | Low light, cluttered background, multiple users, varying distances, low-quality webcam | 0.8842 |

🔥 **Key Result:** Accuracy drops from **1.0000 → 0.8842** under challenging real-world conditions.

📉 **Key Insight:**  
Model performance decreases by ~12% under real-world variations, demonstrating the impact of **domain shift** on gesture recognition systems.

📌 **Note:**  
Near-perfect accuracy in controlled environments is expected due to limited variation. This highlights the gap between controlled and real-world performance.

---

### 📊 Summary Statistics

- Minimum Accuracy: **0.8842**  
- Maximum Accuracy: **1.0000**  
- Average Accuracy: **0.9614**

---

## 📉 Confusion Matrix

<p align="center">
  <img src="results/confusion_matrix.png" width="500"/>
</p>

---

## ⚠️ Failure Analysis

- Confusion between **open** and **four** gestures due to similar finger patterns  
- Performance drops in challenging conditions (~88%) due to:
  - Reduced landmark stability in low lighting  
  - Lower resolution impact at far distances  
- Model sensitive to **partial occlusions**  

---

## 🧪 Evaluation Strategy

🚨 **Key Design Choice: Session-Based Split**

This project avoids common data leakage issues by using **session-based splitting instead of random splitting**.

- Ensures model is tested on **unseen environments**  
- Captures **domain shift across conditions**  
- Simulates real-world deployment scenarios  

Additional details:
- Condition-wise evaluation performed  
- Metrics used: Accuracy, Confusion Matrix  

---

## 🧠 Key Learnings

- Generalization > raw accuracy  
- Landmark normalization improves performance  
- Distance and background affect prediction stability  
- Real-world ML systems must be evaluated beyond standard train-test splits  

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

```
robust-hand-gesture-recognition/
│
├── src/
│   ├── extract_landmarks.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate.py
│   ├── evaluate_conditions.py
│   ├── realtime_inference.py
│   └── test_hand_detection.py
│
├── results/
│   ├── demo.gif
│   └── confusion_matrix.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ How to Run

### 1. Clone repository

```bash
git clone https://github.com/sravani-engineer/robust-hand-gesture-recognition.git
cd robust-hand-gesture-recognition
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run real-time inference

```bash
python src/realtime_inference.py
```

---

## 🔄 Pipeline

Video Input → Landmark Extraction → Preprocessing → Model Training → Evaluation → Real-time Inference
