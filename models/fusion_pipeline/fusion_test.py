# ============================================================
# FUSION TEST.PY
# ============================================================

import warnings

import librosa
import numpy as np

import torch
import torch.nn as nn

from transformers import (
    HubertModel,
    BertModel,
    BertTokenizer
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
MAX_AUDIO_LENGTH = 4 * TARGET_SR

# ============================================================
# TOKENIZER
# ============================================================

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

MAX_LEN = 16

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

        signal = signal[:MAX_AUDIO_LENGTH]

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
# SPEECH MODEL
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

    def forward(self, input_values):

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

        return logits, x

# ============================================================
# TEXT MODEL
# ============================================================

class BERTEmotionClassifier(nn.Module):

    def __init__(
        self,
        dropout=0.3,
        num_classes=7
    ):

        super().__init__()

        self.bert = BertModel.from_pretrained(
            "bert-base-uncased"
        )

        self.fc1 = nn.Linear(
            768,
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
        input_ids,
        attention_mask
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_embedding = (
            outputs.last_hidden_state[:,0,:]
        )

        x = self.fc1(
            cls_embedding
        )

        x = self.relu(x)

        x = self.dropout(x)

        logits = self.fc2(x)

        return logits, cls_embedding

# ============================================================
# FUSION MODEL
# ============================================================

class FusionEmotionModel(nn.Module):

    def __init__(
        self,
        dropout=0.3,
        num_classes=7
    ):

        super().__init__()

        self.fc1 = nn.Linear(
            1024,
            512
        )

        self.relu1 = nn.ReLU()

        self.dropout1 = nn.Dropout(
            dropout
        )

        self.fc2 = nn.Linear(
            512,
            256
        )

        self.relu2 = nn.ReLU()

        self.dropout2 = nn.Dropout(
            dropout
        )

        self.fc3 = nn.Linear(
            256,
            num_classes
        )

    def forward(
        self,
        speech_embedding,
        text_embedding
    ):

        fused = torch.cat(
            [
                speech_embedding,
                text_embedding
            ],
            dim=1
        )

        x = self.fc1(fused)

        x = self.relu1(x)

        x = self.dropout1(x)

        x = self.fc2(x)

        x = self.relu2(x)

        x = self.dropout2(x)

        logits = self.fc3(x)

        return logits

# ============================================================
# LOAD MODELS
# ============================================================

speech_model = HuBERTBiLSTM()
speech_model.load_state_dict(
    torch.load(
        "pretrained_models/speech_model.pth",
        map_location=DEVICE
    )
)

speech_model = speech_model.to(DEVICE)
speech_model.eval()

text_model = BERTEmotionClassifier()
text_model.load_state_dict(
    torch.load(
        "pretrained_models/text_model.pth",
        map_location=DEVICE
    )
)

text_model = text_model.to(DEVICE)
text_model.eval()

fusion_model = FusionEmotionModel()
fusion_model.load_state_dict(
    torch.load(
        "pretrained_models/fusion_model.pth",
        map_location=DEVICE
    )
)

fusion_model = fusion_model.to(DEVICE)
fusion_model.eval()

print("Fusion Model Loaded")

# ============================================================
# PREDICTION
# ============================================================

def fusion_predict(
    audio_path=None,
    text=None
):

    # -------------------------
    # AUDIO ONLY
    # -------------------------

    if audio_path and not text:

        from models.speech_pipeline.speech_test import (
            speech_predict
        )

        return speech_predict(
            audio_path
        )

    # -------------------------
    # TEXT ONLY
    # -------------------------

    if text and not audio_path:

        from models.text_pipeline.text_test import (
            text_predict
        )

        return text_predict(
            text
        )

    # -------------------------
    # TRUE FUSION
    # -------------------------

    audio = preprocess_audio(
        audio_path
    )

    audio = torch.tensor(
        audio,
        dtype=torch.float
    ).unsqueeze(0).to(DEVICE)

    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    input_ids = encoding[
        "input_ids"
    ].to(DEVICE)

    attention_mask = encoding[
        "attention_mask"
    ].to(DEVICE)

    with torch.no_grad():

        _, speech_embedding = (
            speech_model(audio)
        )

        _, text_embedding = (
            text_model(
                input_ids,
                attention_mask
            )
        )

        logits = fusion_model(
            speech_embedding,
            text_embedding
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

    result = fusion_predict(
        audio_path="sample.wav",
        text="I am feeling happy today"
    )

    print(result)