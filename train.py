import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

from utils import extract_features

DATASET_PATH = "dataset/"

emotions = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

X, y = [], []

# Load dataset
for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)

            emotion_code = file.split("-")[2]
            emotion = emotions.get(emotion_code)

            features = extract_features(file_path)

            if features is not None:
                X.append(features)
                y.append(emotion)

# Convert to numpy
X = np.array(X)
y = np.array(y)

print("Dataset loaded:", X.shape)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/emotion_model.pkl")

print("Model saved!")