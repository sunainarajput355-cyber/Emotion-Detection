import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import joblib

# ================= CONFIG =================
DATASET_PATH = "data_reduced"
IMG_SIZE = 72   # increased size

# ================= FACE DETECTOR =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ================= HOG FEATURE FUNCTION =================
def extract_features(img):
    return hog(
        img,
        orientations=12,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

# ================= LOAD DATA =================
X = []
y = []

print("Loading dataset...")

for emotion in os.listdir(DATASET_PATH):
    emotion_path = os.path.join(DATASET_PATH, emotion)

    if not os.path.isdir(emotion_path):
        continue

    for img_name in os.listdir(emotion_path):

        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(emotion_path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # -------- FACE DETECTION --------
        faces = face_cascade.detectMultiScale(img, 1.3, 5)

        if len(faces) > 0:
            x, y_, w, h = faces[0]
            img = img[y_:y_+h, x:x+w]
        else:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # -------- ORIGINAL --------
        X.append(extract_features(img))
        y.append(emotion)

        # -------- AUGMENTATION --------

        # Flip
        flipped = cv2.flip(img, 1)
        X.append(extract_features(flipped))
        y.append(emotion)

        # Rotation
        rows, cols = img.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 10, 1)
        rotated = cv2.warpAffine(img, M, (cols, rows))
        X.append(extract_features(rotated))
        y.append(emotion)

        # Brightness
        bright = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
        X.append(extract_features(bright))
        y.append(emotion)

        if len(X) % 3000 == 0:
            print(f"Processed {len(X)} samples")

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("Dataset loaded!")
print("Total samples:", len(X))
print("Class distribution:", Counter(y))

# ================= TRAIN TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ================= SCALING =================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================= GRID SEARCH =================
print("Running GridSearch (this may take time)...")

param_grid = {
    'C': [1, 5, 10],
    'gamma': ['scale', 0.01],
    'kernel': ['rbf']
}

grid = GridSearchCV(
    SVC(class_weight='balanced'),
    param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

model = grid.best_estimator_

# ================= EVALUATION =================
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ================= SAVE =================
joblib.dump(model, "emotion_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved successfully!")

import cv2
import numpy as np
from skimage.feature import hog
import joblib

# ================= LOAD MODEL =================
model = joblib.load("emotion_model.pkl")
scaler = joblib.load("scaler.pkl")

IMG_SIZE = 72   # MUST match training

# ================= FACE DETECTOR =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ================= HOG FUNCTION =================
def extract_features(img):
    return hog(
        img,
        orientations=12,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

# ================= WEBCAM =================
cap = cv2.VideoCapture(0)

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        try:
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        except:
            continue

        # -------- FEATURE EXTRACTION --------
        features = extract_features(face)
        features = scaler.transform([features])

        # -------- PREDICTION --------
        prediction = model.predict(features)[0]

        # -------- DRAW --------
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, prediction, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Emotion Detector (ML)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()