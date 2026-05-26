# ============================================================
# TEXT TEST.PY
# ============================================================

import warnings

import torch
import torch.nn as nn

from transformers import (
    BertTokenizer,
    BertModel
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
# TOKENIZER
# ============================================================

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

MAX_LEN = 16

# ============================================================
# MODEL
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
            outputs.last_hidden_state[:, 0, :]
        )

        x = self.fc1(
            cls_embedding
        )

        x = self.relu(x)

        x = self.dropout(x)

        logits = self.fc2(x)

        return logits

# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = (
    "pretrained_models/text_model.pth"
)

model = BERTEmotionClassifier()

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

print("Text Model Loaded")

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def text_predict(text):

    encoding = tokenizer(
        text,
        add_special_tokens=True,
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

        logits = model(
            input_ids,
            attention_mask
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

    sample_text = (
        "I am feeling very happy today"
    )

    result = text_predict(
        sample_text
    )

    print(result)