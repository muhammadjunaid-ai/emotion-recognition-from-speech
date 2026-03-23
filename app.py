import streamlit as st
import numpy as np
import joblib
import tempfile

from utils import extract_features

# Load model
model = joblib.load("model/emotion_model.pkl")

st.title("🎤 Emotion Recognition from Speech")

st.write("Upload a WAV audio file to detect emotion")

audio_file = st.file_uploader("Upload Audio", type=["wav"])

if audio_file is not None:
    st.audio(audio_file)

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_file.read())
        temp_path = tmp.name

    # Extract features
    features = extract_features(temp_path)

    if features is not None:
        features = np.expand_dims(features, axis=0)
        prediction = model.predict(features)

        st.success(f"Predicted Emotion: {prediction[0]}")
    else:
        st.error("Could not process audio")