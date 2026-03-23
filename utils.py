import librosa
import numpy as np

def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, duration=3, offset=0.5)

        mfcc = np.mean(
            librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T,
            axis=0
        )

        return mfcc

    except Exception as e:
        print("Error:", e)
        return None