# ============================================================
# SPEECH TEST.PY
# ============================================================

import os
import warnings

import librosa
import numpy as np

import torch
import torch.nn as nn

from transformers import (
    HubertModel,
    AutoFeatureExtractor
)

warnings.filterwarnings("ignore")

# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ============================================================
# LABELS
# ============================================================

emotion_map = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "pleasant_surprise",
    6: "sad"
}

# ============================================================
# AUDIO CONFIG
# ============================================================

TARGET_SR = 16000

MAX_AUDIO_LENGTH = (
    4 * TARGET_SR
)

# ============================================================
# FEATURE EXTRACTOR
# ============================================================

feature_extractor = (
    AutoFeatureExtractor.from_pretrained(
        "facebook/hubert-base-ls960"
    )
)

# ============================================================
# AUDIO PREPROCESSING
# ============================================================

def preprocess_audio(audio_path):

    signal, sr = librosa.load(
        audio_path,
        sr=TARGET_SR
    )

    signal, _ = librosa.effects.trim(
        signal
    )

    signal = librosa.util.normalize(
        signal
    )

    if len(signal) > MAX_AUDIO_LENGTH:

        signal = signal[
            :MAX_AUDIO_LENGTH
        ]

    else:

        padding = (
            MAX_AUDIO_LENGTH
            - len(signal)
        )

        signal = np.pad(
            signal,
            (0, padding)
        )

    return signal.astype(
        np.float32
    )

# ============================================================
# MODEL
# ============================================================

class HuBERTBiLSTM(nn.Module):

    def __init__(
        self,
        hidden_size=256,
        dropout=0.3,
        num_classes=7
    ):

        super().__init__()

        self.hubert = HubertModel.from_pretrained(
            "facebook/hubert-base-ls960"
        )

        self.bilstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.fc1 = nn.Linear(
            hidden_size * 2,
            256
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            dropout
        )

        self.fc2 = nn.Linear(
            256,
            num_classes
        )

    def forward(
        self,
        input_values
    ):

        outputs = self.hubert(
            input_values
        )

        hidden_states = (
            outputs.last_hidden_state
        )

        lstm_out, _ = self.bilstm(
            hidden_states
        )

        embedding = torch.mean(
            lstm_out,
            dim=1
        )

        x = self.fc1(
            embedding
        )

        x = self.relu(x)

        x = self.dropout(x)

        logits = self.fc2(x)

        return logits

# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = (
    "pretrained_models/speech_model.pth"
)

model = HuBERTBiLSTM()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(
    DEVICE
)

model.eval()

print("Speech Model Loaded")

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def speech_predict(audio_path):

    audio = preprocess_audio(
        audio_path
    )

    audio = torch.tensor(
        audio,
        dtype=torch.float
    ).unsqueeze(0)

    audio = audio.to(
        DEVICE
    )

    with torch.no_grad():

        logits = model(
            audio
        )

        probs = torch.softmax(
            logits,
            dim=1
        )

        confidence, pred = (
            torch.max(
                probs,
                dim=1
            )
        )

    emotion = emotion_map[
        pred.item()
    ]

    confidence = (
        confidence.item()
        * 100
    )

    return {
        "emotion": emotion,
        "confidence": round(
            confidence,
            2
        )
    }

# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    audio_path = "sample.wav"

    result = speech_predict(
        audio_path
    )

    print(result)