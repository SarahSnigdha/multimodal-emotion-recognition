# README FILE

## 1. Project Overview

This project implements a Multimodal Emotion Recognition System that predicts human emotions using speech, text, and multimodal fusion techniques.

The system analyzes speech signals using HuBERT-based embeddings and textual content using BERT-based embeddings. A fusion model combines information from both modalities to improve emotion classification performance.

The project combines speech processing, natural language processing, deep learning, and multimodal learning through an interactive Flask-based web application.

---

## 2. Objectives

- Recognize human emotions from speech signals.
- Recognize emotions from textual input.
- Combine speech and text information using multimodal fusion.
- Compare Speech-only, Text-only, and Fusion models.
- Visualize learned emotional representations using t-SNE.
- Provide a simple and user-friendly Flask-based web application.

---

## 3. Technical Architecture

| Component | Technology Used |
|------------|----------------|
| Speech Embedding | HuBERT Base |
| Text Embedding | BERT Base Uncased |
| Speech Model | HuBERT + Classification Head |
| Text Model | BERT + Classification Head |
| Fusion Model | Feature-Level Fusion |
| Audio Processing | Librosa |
| Deep Learning Framework | PyTorch |
| Backend Framework | Flask |
| UI | HTML + CSS |

---

## 4. Application Workflow

Audio Upload / Text Input

↓

Feature Extraction

(HuBERT / BERT)

↓

Emotion Classification

↓

Feature Fusion

↓

Final Emotion Prediction

↓

Result Display on Web Interface

---

## 5. Dataset Summary

### Dataset Used

Toronto Emotional Speech Set (TESS)

### Emotion Classes

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Pleasant Surprise
- Sad

### Input Features

- HuBERT speech embeddings
- BERT text embeddings

### Trained Models

Stored inside:

```text
pretrained_models/
├── speech_model.pth
├── text_model.pth
└── fusion_model.pth
```
## Pretrained Models

The trained model files exceed GitHub's file size limits.

Download them from:

https://drive.google.com/drive/folders/15mCrq212aiaFHj47bODxfZwzA2WI6DBb?usp=drive_link

Place the files inside:

```text
pretrained_models/
```
---

## 6. System Requirements & Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 7. How to Run the Application

### Step 1: Open the Project

Open the project folder in Visual Studio Code.

### Step 2: Open Terminal

Open a new terminal:

Terminal → New Terminal

### Step 3: Activate Virtual Environment

For PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

For Command Prompt (cmd):

```cmd
.venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Application

```bash
python main.py
```

### Step 6: Open Browser

Visit:

```text
http://127.0.0.1:5000
```

Upload an audio file and/or enter text, then click **Predict Emotion** to view Speech, Text, and Fusion predictions.
### Note

The speech and text model files exceed GitHub's file size limit. Download them from the Google Drive link above and place them inside:

```text
pretrained_models/
```
---

## 8. Project Folder Structure

```text
project/
│
├── models/
│   │
│   ├── speech_pipeline/
│   │   ├── speech_train.py
│   │   └── speech_test.py
│   │
│   ├── text_pipeline/
│   │   ├── text_train.py
│   │   └── text_test.py
│   │
│   └── fusion_pipeline/
│       ├── fusion_train.py
│       └── fusion_test.py
│
├── pretrained_models/
│   ├── speech_model.pth
│   ├── text_model.pth
│   └── fusion_model.pth
│
├── templates/
│   └── index.html
│
├── uploads/
│
├── Results/
│   ├── confusion_matrices/
│   ├── accuracy_tables/
│   └── tsne_plots/
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 9. Evaluation

| Model | Accuracy |
|---------|---------|
| Speech Model | 83.10% |
| Text Model | 14.29% |
| Fusion Model | 86.19% |

### Key Findings

- Speech provided the strongest emotional cues and achieved high classification performance.
- HuBERT successfully captured acoustic characteristics such as pitch, prosody, speaking rate, and vocal energy.
- The text model achieved limited performance because the dataset primarily contained isolated words with little emotional context.
- Increasing text model complexity did not significantly improve performance.
- The fusion model achieved the highest overall accuracy by combining speech and text representations.
- Fusion helped correct errors made by individual modalities and produced more confident predictions.
- t-SNE visualizations showed clear emotion clusters for Speech and Fusion representations.

---

## 10. Future Improvements

- Use emotionally rich text datasets.
- Support multilingual emotion recognition.
- Integrate facial expression analysis.
- Explore transformer-based fusion techniques.
- Add microphone-based real-time emotion prediction.
- Deploy the system as a cloud-based application.

---

## 11. References

- HuBERT: Self-Supervised Speech Representation Learning
- BERT: Bidirectional Encoder Representations from Transformers
- PyTorch Documentation
- Hugging Face Transformers Documentation
- Flask Documentation
- Toronto Emotional Speech Set (TESS)

---

## Authors

Sarah Snigdha Chikelay (23E51A66A3)


Department of Computer Science and Engineering – AI & ML

Hyderabad Institute of Technology and Management