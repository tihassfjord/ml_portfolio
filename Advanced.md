
**Descriptions will be natural, direct, and in third person—no “I, tihassfjord,” just smart project blurbs.**
All code and docs still use **tihassfjord** for authorship, print statements, etc.

**Advanced projects are tagged as `highlight` in their READMEs.**
Each project includes:

* Project description (for portfolio/publication)
* README.md (with `highlight` tag)
* Directory layout suggestion
* Code or notebook starter snippet, always personalized

---

# 📁 tihassfjord’s Advanced Machine Learning Portfolio

<details>
<summary><strong>Advanced & Highlight Projects</strong> (click to expand/collapse)</summary>

---

## 1. Build Your Own Neural Network (**highlight**)

### **Project Description**

A minimal neural network is implemented from scratch—no frameworks, just NumPy. Forward and backward passes, manual weight updates, and full training loop. This project proves a deep understanding of how neural networks work under the hood.

### **README.md**

```markdown
# Build Your Own Neural Network (highlight) — tihassfjord

## Goal
Implement a simple fully-connected neural network for classification using only NumPy.

## Dataset
- MNIST (or small subset, can use sklearn’s digits)

## Requirements
- Python 3.8+
- numpy

## How to Run
`python simple_nn_tihassfjord.py`

## Example Output
```

Epoch 1/20 - Loss: 1.85 - Accuracy: 0.54
...
Final Test Accuracy: 0.92

```
```

### **Directory Structure**

```
nn-from-scratch-tihassfjord/
│
├── simple_nn_tihassfjord.py
└── README.md
```

### **Sample Code**

```python
# simple_nn_tihassfjord.py

"""
Neural Network from scratch by tihassfjord
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

print("Training neural net (tihassfjord style).")

# Load data
digits = load_digits()
X = digits.data / 16.0
y = OneHotEncoder(sparse=False).fit_transform(digits.target.reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Model params
n_input, n_hidden, n_output = 64, 32, 10
W1 = np.random.randn(n_input, n_hidden) * 0.1
b1 = np.zeros(n_hidden)
W2 = np.random.randn(n_hidden, n_output) * 0.1
b2 = np.zeros(n_output)

def relu(x): return np.maximum(0, x)
def softmax(x): return np.exp(x)/np.exp(x).sum(axis=1, keepdims=True)

def forward(X):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    out = softmax(z2)
    return a1, out

def compute_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-7), axis=1))

# Training loop
lr = 0.1
for epoch in range(20):
    a1, y_pred = forward(X_train)
    loss = compute_loss(y_train, y_pred)
    # Backprop
    dz2 = y_pred - y_train
    dW2 = a1.T @ dz2 / len(X_train)
    db2 = dz2.mean(axis=0)
    da1 = dz2 @ W2.T
    dz1 = da1 * (a1 > 0)
    dW1 = X_train.T @ dz1 / len(X_train)
    db1 = dz1.mean(axis=0)
    # Update
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1
    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.2f}")

# Evaluate
_, test_pred = forward(X_test)
acc = (test_pred.argmax(1) == y_test.argmax(1)).mean()
print(f"Test Accuracy (tihassfjord): {acc:.2f}")
```

---

## 2. Real-time Face Recognition System (**highlight**)

### **Project Description**

A complete pipeline for detecting and recognizing faces from a live camera stream using OpenCV and deep learning embeddings. Useful as a demo of both computer vision and real-time ML engineering.

### **README.md**

```markdown
# Real-time Face Recognition System (highlight) — tihassfjord

## Goal
Detect and recognize faces in real time from webcam feed.

## Dataset
- Labeled Faces in the Wild (LFW) for training (sklearn or download)

## Requirements
- Python 3.8+
- opencv-python
- numpy
- face_recognition

## How to Run
Install requirements, then:
`python face_recognition_tihassfjord.py`

## Example Output
```

\[Display window with bounding boxes and names, or "Unknown" if not matched]

```
```

### **Directory Structure**

```
face-recognition-tihassfjord/
│
├── face_recognition_tihassfjord.py
├── faces/
│   └── [known faces .jpg]
└── README.md
```

### **Sample Code**

```python
# face_recognition_tihassfjord.py

"""
Real-time face recognition (tihassfjord)
"""

import cv2
import face_recognition
import os

print("Starting real-time face recognition (tihassfjord).")

# Load known faces
known_encodings = []
known_names = []
for filename in os.listdir("faces"):
    img = face_recognition.load_image_file(f"faces/{filename}")
    encoding = face_recognition.face_encodings(img)[0]
    known_encodings.append(encoding)
    known_names.append(filename.split('.')[0])

# Start webcam
video = cv2.VideoCapture(0)
while True:
    ret, frame = video.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locs = face_recognition.face_locations(rgb)
    face_encs = face_recognition.face_encodings(rgb, face_locs)
    for (top, right, bottom, left), enc in zip(face_locs, face_encs):
        matches = face_recognition.compare_faces(known_encodings, enc)
        name = "Unknown"
        if True in matches:
            name = known_names[matches.index(True)]
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, f"{name} (tihassfjord)", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    cv2.imshow("tihassfjord Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
```

---

## 3. Automated ML Pipeline (**highlight**)

### **Project Description**

End-to-end AutoML pipeline for tabular data. The script automates preprocessing, model selection, hyperparameter tuning, and reporting. Great for fast, reproducible baselines and competitions.

### **README.md**

```markdown
# Automated ML Pipeline (highlight) — tihassfjord

## Goal
Automate data cleaning, feature engineering, model selection, and evaluation for tabular data.

## Dataset
- Any tabular CSV (plug in your own)

## Requirements
- Python 3.8+
- scikit-learn
- pandas
- numpy

## How to Run
`python automl_pipeline_tihassfjord.py data/your_data.csv`

## Example Output
```

Best model: RandomForestClassifier, ROC AUC: 0.87

```
```

### **Directory Structure**

```
automl-pipeline-tihassfjord/
│
├── automl_pipeline_tihassfjord.py
├── data/
│   └── your_data.csv
└── README.md
```

### **Sample Code**

```python
# automl_pipeline_tihassfjord.py

"""
Automated ML Pipeline by tihassfjord
"""

import sys
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

print("Running AutoML pipeline (tihassfjord).")
df = pd.read_csv(sys.argv[1])
target = df.columns[-1]  # Assume last column is target
X = df.iloc[:,:-1]
y = df[target]

# Preprocessing
X = pd.get_dummies(X)
X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
models = {
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'LogisticRegression': LogisticRegression(max_iter=500)
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:,1]
    score = roc_auc_score(y_test, preds)
    results[name] = score
best = max(results, key=results.get)
print(f"Best model: {best}, ROC AUC: {results[best]:.2f}")
```

---

## 4. Language Model From Scratch (**highlight**)

### **Project Description**

Implements a simple character-level language model in pure Python. Useful for understanding how text generation and sequence modeling works at the most basic level.

### **README.md**

```markdown
# Language Model From Scratch (highlight) — tihassfjord

## Goal
Train a basic character-level language model for text generation.

## Dataset
- Any text file (sample provided)

## Requirements
- Python 3.8+
- numpy

## How to Run
`python char_lm_tihassfjord.py data/input.txt`

## Example Output
```

Generated text: "The quick brown fox jumps over the lazy dog..."

```
```

### **Directory Structure**

```
char-lm-tihassfjord/
│
├── char_lm_tihassfjord.py
├── data/
│   └── input.txt
└── README.md
```

### **Sample Code**

```python
# char_lm_tihassfjord.py

"""
Simple Char-level LM by tihassfjord
"""

import numpy as np
import sys

print("Training char-level language model (tihassfjord).")
with open(sys.argv[1], encoding='utf8') as f:
    text = f.read()
chars = sorted(set(text))
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}
vocab_size = len(chars)
X = np.array([stoi[c] for c in text[:-1]])
Y = np.array([stoi[c] for c in text[1:]])
W = np.random.randn(vocab_size, vocab_size) * 0.01

for epoch in range(100):
    # Forward
    logits = W[X]
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(len(Y)), Y]).mean()
    # Backward
    grad = probs
    grad[np.arange(len(Y)), Y] -= 1
    dW = X[:,None] == np.arange(vocab_size)
    gradW = dW.T @ grad / len(Y)
    W -= 0.1 * gradW
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, loss {loss:.3f}")

# Sampling
def sample(length=100, start_ix=0):
    ix = start_ix
    out = []
    for _ in range(length):
        p = np.exp(W[ix]) / np.exp(W[ix]).sum()
        ix = np.random.choice(np.arange(vocab_size), p=p)
        out.append(itos[ix])
    return ''.join(out)

print("Generated text (tihassfjord):")
print(sample(200, X[0]))
```

---

## 5. A/B Testing Framework (**highlight**)

### **Project Description**

A lightweight but robust Python toolkit for running, analyzing, and visualizing A/B tests with statistical rigor. Useful for product analytics, marketing, or any scientific decision-making.

### **README.md**

```markdown
# A/B Testing Framework (highlight) — tihassfjord

## Goal
Design and analyze A/B tests with statistical significance testing.

## Dataset
- Synthetic example or your own experiments

## Requirements
- Python 3.8+
- numpy
- scipy
- matplotlib

## How to Run
`python ab_test_framework_tihassfjord.py`

## Example Output
```

P-value: 0.042
Significant difference detected.

```
```

### **Directory Structure**

```
ab-test-framework-tihassfjord/
│
├── ab_test_framework_tihassfjord.py
└── README.md
```

### **Sample Code**

```python
# ab_test_framework_tihassfjord.py

"""
A/B testing toolkit by tihassfjord
"""

import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

print("Running A/B test (tihassfjord).")
A = np.random.binomial(1, 0.11, 1000)
B = np.random.binomial(1, 0.13, 1000)
t, p = ttest_ind(A, B)
print(f"P-value: {p:.4f}")
if p < 0.05:
    print("Significant difference detected.")
else:
    print("No significant difference.")

plt.bar(['A', 'B'], [A.mean(), B.mean()])
plt.title("A/B Test Results (tihassfjord)")
plt.show()
```

---

## 6. Image Generation System (**highlight**)

### **Project Description**

A pipeline for generating images using a pretrained GAN (e.g., DCGAN on MNIST or CelebA). Can be extended to style transfer or text-to-image if desired.

### **README.md**

```markdown
# Image Generation System (highlight) — tihassfjord

## Goal
Generate synthetic images using GANs.

## Dataset
- MNIST (default, extend to CelebA as needed)

## Requirements
- Python 3.8+
- tensorflow or pytorch
- matplotlib
- numpy

## How to Run
`python gan_mnist_tihassfjord.py`

## Example Output
[Image grid of generated digits]
```

### **Directory Structure**

```
image-gen-tihassfjord/
│
├── gan_mnist_tihassfjord.py
└── README.md
```

### **Sample Code**

```python
# gan_mnist_tihassfjord.py

"""
Simple DCGAN for MNIST by tihassfjord
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

print("Training DCGAN on MNIST (tihassfjord).")
(X_train, _), _ = tf.keras.datasets.mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = X_train[..., np.newaxis]

def make_generator():
    model = tf.keras.Sequential([
        layers.Dense(7*7*128, input_dim=100),
        layers.Reshape((7,7,128)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(64, 5, strides=2, padding='same'),
        layers.ReLU(),
        layers.Conv2DTranspose(1, 5, strides=2, padding='same', activation='tanh'),
    ])
    return model

def make_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, 5, strides=2, padding='same', input_shape=[28,28,1]),
        layers.LeakyReLU(),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

generator = make_generator()
discriminator = make_discriminator()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

@tf.function
def train_step(images, batch_size=64):
    noise = tf.random.normal([batch_size, 100])
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        gen_images = generator(noise)
        real_output = discriminator(images)
        fake_output = discriminator(gen_images)
        d_loss = cross_entropy(tf.ones_like(real_output), real_output) + \
                 cross_entropy(tf.zeros_like(fake_output), fake_output)
        g_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    gradients_of_gen = g_tape.gradient(g_loss, generator.trainable_variables)
    gradients_of_disc = d_tape.gradient(d_loss, discriminator.trainable_variables)
    generator.optimizer.apply_gradients(zip(gradients_of_gen, generator.trainable_variables))
    discriminator.optimizer.apply_gradients(zip(gradients_of_disc, discriminator.trainable_variables))

generator.optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator.optimizer = tf.keras.optimizers.Adam(1e-4)

for epoch in range(3):
    for i in range(0, len(X_train), 64):
        images = X_train[i:i+64]
        train_step(images)
    print(f"Epoch {epoch} complete (tihassfjord)")

noise = tf.random.normal([16, 100])
gen_images = generator(noise).numpy().squeeze()
fig, axs = plt.subplots(4,4)
for i, ax in enumerate(axs.flatten()):
    ax.imshow(gen_images[i], cmap='gray')
    ax.axis('off')
plt.suptitle("DCGAN MNIST (tihassfjord)")
plt.show()
```

---

## 7. Multi-language NLP Pipeline (**highlight**)

### **Project Description**

A robust NLP pipeline for multiple languages, including tokenization, translation, and sentiment or NER using Hugging Face transformers. A showcase of practical language model engineering.

### **README.md**

```markdown
# Multi-language NLP Pipeline (highlight) — tihassfjord

## Goal
Process and analyze text in multiple languages using modern NLP models.

## Dataset
- Sample text files in several languages

## Requirements
- Python 3.8+
- transformers
- torch

## How to Run
`python multilingual_nlp_tihassfjord.py data/your_text.txt`

## Example Output
```

Detected language: fr
Translation: "Hello, how are you?"
Sentiment: positive

```
```

### **Directory Structure**

```
multilingual-nlp-tihassfjord/
│
```

Absolutely! Here are the rest of the **Advanced (“highlight”) Projects** for your portfolio, continuing from **Reinforcement Learning Game AI**.
(If you want more code, details, or extra comments on any project, just shout.)

---

## 8. Reinforcement Learning Game AI (**highlight**)

### **Project Description**

A Deep Q-Network (DQN) agent is trained to play CartPole using PyTorch and Gym. This project demonstrates reinforcement learning from scratch, value function approximation, and how to let machines learn by reward and punishment.

### **README.md**

```markdown
# Reinforcement Learning Game AI (highlight) — tihassfjord

## Goal
Train a DQN agent to solve the CartPole-v1 environment using deep RL.

## Dataset
- OpenAI Gym CartPole-v1 (simulated)

## Requirements
- Python 3.8+
- gym
- torch
- numpy

## How to Run
`python rl_cartpole_tihassfjord.py`

## Example Output
```

Episode 10, Reward: 19
Episode 200, Reward: 200
Environment solved!

```
```

### **Directory Structure**

```
rl-game-ai-tihassfjord/
│
├── rl_cartpole_tihassfjord.py
└── README.md
```

### **Sample Code**

```python
# rl_cartpole_tihassfjord.py

"""
DQN agent for CartPole by tihassfjord
"""

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

print("Training DQN agent on CartPole (tihassfjord).")

env = gym.make('CartPole-v1')
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_size, 128), nn.ReLU(),
            nn.Linear(128, n_actions))
    def forward(self, x):
        return self.fc(x)

qnet = QNet()
optimizer = optim.Adam(qnet.parameters(), lr=0.001)
criterion = nn.MSELoss()

def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    with torch.no_grad():
        return qnet(torch.FloatTensor(state)).argmax().item()

memory = []
max_mem = 10000
batch_size = 64

for episode in range(201):
    state = env.reset()
    total_reward = 0
    for t in range(200):
        action = select_action(state, epsilon=max(0.1, 1-episode/200))
        next_state, reward, done, _ = env.step(action)
        memory.append((state, action, reward, next_state, done))
        if len(memory) > max_mem:
            memory.pop(0)
        state = next_state
        total_reward += reward
        if done:
            break
        # Training step
        if len(memory) > batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.BoolTensor(dones)
            q_values = qnet(states).gather(1, actions).squeeze()
            with torch.no_grad():
                max_q_next = qnet(next_states).max(1)[0]
            targets = rewards + (1 - dones.float()) * 0.99 * max_q_next
            loss = criterion(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    if episode % 10 == 0:
        print(f"Episode {episode}, Reward: {total_reward}")
    if total_reward >= 200:
        print("Environment solved! (tihassfjord)")
        break
```

---

## 9. Real-time Fraud Detection System (**highlight**)

### **Project Description**

A streaming fraud detection engine that uses online learning and anomaly detection on transaction data. Designed for deployment and continuous monitoring of financial streams.

### **README.md**

```markdown
# Real-time Fraud Detection System (highlight) — tihassfjord

## Goal
Detect fraudulent transactions in real time using ML and anomaly detection.

## Dataset
- Credit Card Fraud Detection (Kaggle), or simulate streaming batches

## Requirements
- Python 3.8+
- pandas
- scikit-learn
- numpy

## How to Run
`python fraud_detect_stream_tihassfjord.py`

## Example Output
```

Streaming batch 5 — Fraud detected in transaction 839

```
```

### **Directory Structure**

```
fraud-detect-tihassfjord/
│
├── fraud_detect_stream_tihassfjord.py
└── README.md
```

### **Sample Code**

```python
# fraud_detect_stream_tihassfjord.py

"""
Streaming fraud detection by tihassfjord
"""

import pandas as pd
from sklearn.ensemble import IsolationForest

print("Running real-time fraud detection (tihassfjord).")
df = pd.read_csv('creditcard.csv')
model = IsolationForest(contamination=0.001, random_state=42)
batch_size = 1000
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    X = batch.drop(columns=['Class'])
    model.fit(X)
    preds = model.predict(X)
    flagged = batch[preds==-1]
    for idx in flagged.index:
        print(f"Fraud detected in transaction {idx} (tihassfjord)")
```

---

## 10. Build Your Own AutoML (**highlight**)

### **Project Description**

A customizable AutoML engine: given a tabular dataset, the script tries several preprocessing, feature selection, and model combinations, picks the best, and outputs code for re-use. All logic is open and editable.

### **README.md**

```markdown
# Build Your Own AutoML (highlight) — tihassfjord

## Goal
Automate end-to-end ML: preprocessing, model selection, and evaluation, producing best model code.

## Dataset
- Any tabular CSV (user-provided)

## Requirements
- Python 3.8+
- pandas
- scikit-learn
- numpy

## How to Run
`python automl_custom_tihassfjord.py data/your_data.csv`

## Example Output
```

Best pipeline: RandomForest + MinMaxScaler, ROC AUC: 0.88
Pipeline code exported to: best\_pipeline\_tihassfjord.py

```
```

### **Directory Structure**

```
custom-automl-tihassfjord/
│
├── automl_custom_tihassfjord.py
├── data/
│   └── your_data.csv
└── README.md
```

### **Sample Code**

```python
# automl_custom_tihassfjord.py

"""
Custom AutoML by tihassfjord
"""

import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score

print("Custom AutoML search (tihassfjord).")
df = pd.read_csv(sys.argv[1])
target = df.columns[-1]
X = df.drop(columns=[target])
y = df[target]
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

pipelines = [
    ('rf_minmax', Pipeline([('scaler', MinMaxScaler()), ('clf', RandomForestClassifier())])),
    ('rf_standard', Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier())])),
    ('log_minmax', Pipeline([('scaler', MinMaxScaler()), ('clf', LogisticRegression(max_iter=500))]))
]

results = {}
for name, pipe in pipelines:
    pipe.fit(X_train, y_train)
    preds = pipe.predict_proba(X_test)[:,1]
    score = roc_auc_score(y_test, preds)
    results[name] = score

best = max(results, key=results.get)
print(f"Best pipeline: {best}, ROC AUC: {results[best]:.2f}")
# Optionally, save the best pipeline
import joblib
joblib.dump(pipelines[[x[0] for x in pipelines].index(best)][1], 'best_pipeline_tihassfjord.pkl')
```

---

## 11. MLOps Pipeline (**highlight**)

### **Project Description**

Shows how to deploy, monitor, and retrain an ML model in production with CI/CD hooks, versioning, and performance logging. Example: housing price predictor with model registry.

### **README.md**

```markdown
# MLOps Pipeline (highlight) — tihassfjord

## Goal
Automate deployment, logging, and monitoring for a regression model in a production-like setup.

## Dataset
- Housing prices (sample CSV)

## Requirements
- Python 3.8+
- scikit-learn
- pandas
- mlflow
- fastapi

## How to Run
Start the API: `uvicorn mlops_api_tihassfjord:app`
Log models: `mlflow ui` and run notebook

## Example Output
```

\[API] POST /predict — returns predicted price, logs to MLflow

```
```

### **Directory Structure**

```
mlops-pipeline-tihassfjord/
│
├── mlops_api_tihassfjord.py
├── train_model_tihassfjord.ipynb
└── README.md
```

### **Sample Code (FastAPI and MLflow parts)**

```python
# mlops_api_tihassfjord.py

"""
MLOps FastAPI for inference and logging (tihassfjord)
"""

from fastapi import FastAPI
import joblib
import mlflow
import pandas as pd

app = FastAPI()
model = joblib.load("best_pipeline_tihassfjord.pkl")

@app.post("/predict")
def predict(features: dict):
    df = pd.DataFrame([features])
    prediction = model.predict(df)[0]
    mlflow.log_metric("prediction", prediction)
    return {"prediction": prediction}
```

*Example notebook would handle retraining, logging, and versioning the model using MLflow UI.*

---

## 12. Distributed ML System (**highlight**)

### **Project Description**

Distributed training with PyTorch Lightning or Ray on synthetic or real data. Example: speed up training with data parallelism across CPUs/GPUs.

### **README.md**

```markdown
# Distributed ML System (highlight) — tihassfjord

## Goal
Train ML models faster by distributing work across multiple CPUs/GPUs.

## Dataset
- Synthetic or real (e.g., MNIST)

## Requirements
- Python 3.8+
- torch
- pytorch-lightning
- ray

## How to Run
`python distributed_training_tihassfjord.py`

## Example Output
```

\[Worker 1] Training complete, accuracy: 0.98
\[Worker 2] Training complete, accuracy: 0.98

```
```

### **Directory Structure**

```
distributed-ml-tihassfjord/
│
├── distributed_training_tihassfjord.py
└── README.md
```

### **Sample Code (Ray + PyTorch Lightning)**

```python
# distributed_training_tihassfjord.py

"""
Distributed ML example with Ray (tihassfjord)
"""

import ray
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

print("Distributed training with Ray (tihassfjord).")
ray.init(ignore_reinit_error=True)

@ray.remote
class Trainer:
    def train(self):
        X = torch.randn(1000, 10)
        y = (X.sum(axis=1) > 0).float()
        model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = nn.BCELoss()
        for epoch in range(5):
            optimizer.zero_grad()
            pred = model(X).squeeze()
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
        acc = ((model(X).squeeze() > 0.5) == y).float().mean().item()
        print(f"[{ray.get_runtime_context().current_actor.name}] Training complete, accuracy: {acc:.2f}")
        return acc

worker1 = Trainer.options(name="Worker 1").remote()
worker2 = Trainer.options(name="Worker 2").remote()
ray.get([worker1.train.remote(), worker2.train.remote()])
ray.shutdown()
```

---

</details>

---

