import pandas as pd
import numpy as np
import librosa
import joblib

class Feature:
    def __init__(self, file_path):
        self.y, self.sr = librosa.load(file_path)
        self.length = len(self.y)
    def chroma_stft(self):
        chroma_stft = librosa.feature.chroma_stft(y=self.y, sr=self.sr)
        chroma_stft_mean = np.mean(chroma_stft)
        chroma_stft_var = np.var(chroma_stft)
        return chroma_stft_mean, chroma_stft_var

    def rms(self):
        rms = librosa.feature.rms(y=self.y)
        rms_mean = np.mean(rms)
        rms_var = np.var(rms)
        return rms_mean, rms_var

    def spectral_centroid(self):
        spectral_centroid = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)
        spectral_centroid_mean = np.mean(spectral_centroid)
        spectral_centroid_var = np.var(spectral_centroid)
        return spectral_centroid_mean, spectral_centroid_var

    def spectral_bandwidth(self):
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=self.y, sr=self.sr)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        spectral_bandwidth_var = np.var(spectral_bandwidth)
        return spectral_bandwidth_mean, spectral_bandwidth_var

    def spectral_rolloff(self):
        spectral_rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr)
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        spectral_rolloff_var = np.var(spectral_rolloff)
        return spectral_rolloff_mean, spectral_rolloff_var

    def zero_crossing_rate(self):
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=self.y)
        zero_crossing_rate_mean = np.mean(zero_crossing_rate)
        zero_crossing_rate_var = np.var(zero_crossing_rate)
        return zero_crossing_rate_mean, zero_crossing_rate_var

    def harmonic(self):
        chroma_cens = librosa.feature.chroma_cens(y=self.y, sr=self.sr)
        harmonic_mean = np.mean(chroma_cens)
        harmonic_var = np.var(chroma_cens)
        return harmonic_mean, harmonic_var

    def mfccs(self):
        mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr)
        mfccs_mean = np.mean(mfccs,axis=1)
        mfccs_var = np.var(mfccs,axis=1)
        return mfccs_mean.flatten(), mfccs_var.flatten()

    def tempo(self):
        tempo, beat_frames = librosa.beat.beat_track(y=self.y, sr=self.sr)
        tempo_mean = np.mean(tempo)
        return tempo_mean

    def extract_features(self):
        chroma_stft_mean, chroma_stft_var = self.chroma_stft()
        rms_mean, rms_var = self.rms()
        spectral_centroid_mean, spectral_centroid_var = self.spectral_centroid()
        spectral_bandwidth_mean, spectral_bandwidth_var = self.spectral_bandwidth()
        spectral_rolloff_mean, spectral_rolloff_var = self.spectral_rolloff()
        zero_crossing_rate_mean, zero_crossing_rate_var = self.zero_crossing_rate()
        harmonic_mean, harmonic_var = self.harmonic()
        mfccs_mean, mfccs_var = self.mfccs()
        tempo_mean = self.tempo()
        k=[]
        k.append(self.length)
        k.append(chroma_stft_mean)
        k.append(chroma_stft_var)
        k.append(rms_mean)
        k.append(rms_var)
        k.append(spectral_centroid_mean)
        k.append(spectral_centroid_var)
        k.append(spectral_bandwidth_mean)
        k.append(spectral_bandwidth_var)
        k.append(spectral_rolloff_mean)
        k.append(spectral_rolloff_var)
        k.append(zero_crossing_rate_mean)
        k.append(zero_crossing_rate_var)
        k.append(harmonic_mean)
        k.append(harmonic_var)
        k.append(tempo_mean)
        for m in mfccs_mean:
            k.append(m)
        for m1 in mfccs_var:
            k.append(m1)
        return k
       

model = joblib.load('/home/kushagra/Documents/code/AI/project/music_genere_classification/best_rf_model.pkl')
scaler=joblib.load('/home/kushagra/Documents/code/AI/project/music_genere_classification/scaler.pkl')
def extract_features(file_name):
    try:
        feature_instance = Feature(file_name)  # Use a different variable name
        features = feature_instance.extract_features()
        features=[features]
        scaler_f=scaler.transform(features)
        answers = model.predict(scaler_f)
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}")
        print(e)
        return None
    return answers

path = "/home/kushagra/Documents/code/AI/project/music_genere_classification/Data/genres_original/jazz/jazz.00001.wav"
print(extract_features(path))
