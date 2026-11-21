# 🎭 Real-Time Emotion Detection System (ML-Based)

A machine learning-based computer vision project that detects human emotions in real-time using a webcam. The system uses **HOG (Histogram of Oriented Gradients)** for feature extraction and an **SVM (Support Vector Machine)** classifier for prediction.

---

## 🚀 Features

* 🎥 Real-time emotion detection using webcam
* 🧠 Machine Learning pipeline (HOG + SVM)
* 😊 Supports multiple emotions (e.g., happy, sad, angry, neutral)
* ⚡ Optimized dataset preprocessing and augmentation
* 📊 Achieved ~82% accuracy on test data
* 💾 Model persistence using joblib

---

## 🛠️ Tech Stack

* Python
* OpenCV
* NumPy
* Scikit-learn
* Scikit-image

---

## 📂 Project Structure

```
emotion-detection/
│
├── data/                 # Original datase       
├── emotion_model.py     # Trained SVM model
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

### 1. Data Preprocessing

* Images are resized (72×72)
* Faces are detected using Haar Cascade
* Dataset is balanced and reduced per class

### 2. Feature Extraction

* HOG (Histogram of Oriented Gradients) is used to extract facial features

### 3. Data Augmentation

* Horizontal flipping
* Rotation
* Brightness adjustment

### 4. Model Training

* Model: SVM with RBF kernel
* Hyperparameter tuning using GridSearchCV
* StandardScaler applied to features

### 5. Real-Time Detection

* Webcam input processed frame-by-frame
* Face detected → features extracted → emotion predicted

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/emotion-detection.git
cd emotion-detection
```

### 2. Install dependencies

```bash
pip install opencv-python numpy scikit-learn scikit-image joblib
```

### 3. Train the model

```bash
python emotion_detection.py
```

---

## 📈 Results

* Accuracy: **~82%**
* Improved from initial ~40% through:

  * Face detection
  * Data augmentation
  * Hyperparameter tuning
  * Improved HOG configuration

---

## ⚠️ Limitations

* Performance depends on lighting and camera quality
* Struggles with subtle emotions (e.g., neutral vs sad)
* HOG features have limitations compared to deep learning

---

## 🚀 Future Improvements

* 🔥 Replace HOG with CNN (Deep Learning)
* 🌐 Deploy as a web application (Flask/React)
* 📱 Mobile integration
* 🎯 Improve accuracy using pre-trained models

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit a pull request.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

Sunaina Rajput

---

## ⭐ Acknowledgements

* OpenCV for computer vision tools
* Scikit-learn for ML models
* Public facial emotion datasets
