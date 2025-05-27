Here are the **Intermediate Projects** — each with:

* Personal project description in first person (as "tihassfjord")
* README.md
* Directory suggestion
* Sample code (Python or Jupyter, always name-tagged!)
* Best practices, clarity, and a bit of extra flavor

---

# 📁 tihassfjord’s Intermediate Machine Learning Projects

<details>
<summary><strong>Intermediate Projects</strong> (click to expand/collapse)</summary>

---

## 1. Build Your Own Linear Regression

### **Project Description**

> I, tihassfjord, wanted to demystify the magic behind linear regression. Instead of just calling scikit-learn, I implemented it from scratch—gradient descent and all—so I know exactly what’s under the hood.

### **README.md**

```markdown
# tihassfjord's Linear Regression From Scratch

## Goal
Implement ordinary least squares linear regression without using libraries like scikit-learn.

## Dataset
- Boston Housing (or generate synthetic data)

## Requirements
- Python 3.8+
- numpy
- matplotlib

## How to Run
1. `python linear_regression_tihassfjord.py`

## Example Output
```

Iteration 0, Loss: 600.12
...
Final Loss: 23.07
Coefficients: \[19.8, -0.3, 3.1, ...]

```
```

### **Directory Structure**

```
linear-regression-tihassfjord/
│
├── linear_regression_tihassfjord.py
└── README.md
```

### **Key Code Snippet:**

```python
# linear_regression_tihassfjord.py

"""
tihassfjord's Linear Regression (from scratch!)
"""

import numpy as np

print("tihassfjord: Training a linear regression model the hard way.")

# Synthetic data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2.5 * X.flatten() + np.random.randn(100) * 2

# Add bias term
X_b = np.c_[np.ones((100, 1)), X]

# Parameters
theta = np.zeros(X_b.shape[1])
learning_rate = 0.01

for epoch in range(500):
    gradients = -2/len(X_b) * X_b.T @ (y - X_b @ theta)
    theta -= learning_rate * gradients
    if epoch % 100 == 0:
        loss = np.mean((y - X_b @ theta) ** 2)
        print(f"Epoch {epoch}, Loss: {loss:.2f}")

print("Final coefficients (tihassfjord):", theta)
```

---

## 2. Image Classification System

### **Project Description**

> I, tihassfjord, built a classic image classifier for the MNIST digits dataset. This is where computer vision careers begin! I kept it simple and clear, and left plenty of room for extension.

### **README.md**

```markdown
# tihassfjord's Image Classification System

## Goal
Train a neural network to classify handwritten digits.

## Dataset
- MNIST (downloaded automatically)

## Requirements
- Python 3.8+
- tensorflow OR pytorch
- matplotlib

## How to Run
1. `python mnist_classifier_tihassfjord.py`

## Example Output
```

Epoch 1/10: Loss=0.34, Accuracy=0.91
...
Test Accuracy: 0.98

```
```

### **Directory Structure**

```
image-classification-tihassfjord/
│
├── mnist_classifier_tihassfjord.py
└── README.md
```

### **Key Code Snippet:**

```python
# mnist_classifier_tihassfjord.py

"""
tihassfjord's MNIST Classifier
"""

import tensorflow as tf
from tensorflow.keras import layers

print("tihassfjord: Starting MNIST training.")

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train, X_test = X_train/255.0, X_test/255.0

model = tf.keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, validation_split=0.1)
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"tihassfjord's MNIST Test Accuracy: {test_acc:.2f}")
```

---

## 3. Sentiment Analysis System

### **Project Description**

> I, tihassfjord, taught a computer to understand human feelings—at least enough to spot positive vs. negative movie reviews. Fast, practical, and surprisingly insightful for something trained on internet comments.

### **README.md**

```markdown
# tihassfjord's Sentiment Analysis System

## Goal
Classify text reviews as positive or negative using NLP.

## Dataset
- IMDB Movie Reviews (Keras datasets)

## Requirements
- Python 3.8+
- tensorflow
- numpy

## How to Run
1. `python sentiment_analysis_tihassfjord.py`

## Example Output
```

Epoch 1/3: Loss=0.44, Accuracy=0.80
Test Accuracy: 0.86

```
```

### **Directory Structure**

```
sentiment-analysis-tihassfjord/
│
├── sentiment_analysis_tihassfjord.py
└── README.md
```

### **Key Code Snippet:**

```python
# sentiment_analysis_tihassfjord.py

"""
tihassfjord's Sentiment Analysis with IMDB Reviews
"""

import tensorflow as tf
from tensorflow.keras import layers

print("tihassfjord: Training a text sentiment model.")

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=200)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=200)

model = tf.keras.Sequential([
    layers.Embedding(10000, 16),
    layers.GlobalAveragePooling1D(),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, batch_size=512, validation_split=0.1)
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"tihassfjord's IMDB Test Accuracy: {test_acc:.2f}")
```

---

## 4. Stock Price Predictor

### **Project Description**

> I, tihassfjord, gave in to curiosity: can ML predict the stock market? (Spoiler: don’t trust any model blindly). This project explores time series prediction and why skepticism pays off.

### **README.md**

```markdown
# tihassfjord's Stock Price Predictor

## Goal
Forecast stock prices using historical data and ML.

## Dataset
- Yahoo Finance (e.g., AAPL.csv, download manually or via yfinance)

## Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib

## How to Run
1. Download a CSV (e.g., `AAPL.csv`) to `data/`
2. `python stock_predictor_tihassfjord.py`

## Example Output
```

Predicted close for next day: 155.41

```
```

### **Directory Structure**

```
stock-price-predictor-tihassfjord/
│
├── data/
│   └── AAPL.csv
├── stock_predictor_tihassfjord.py
└── README.md
```

### **Key Code Snippet:**

```python
# stock_predictor_tihassfjord.py

"""
tihassfjord's Stock Price Predictor
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

print("tihassfjord: Trying to forecast stock prices—handle with care.")

df = pd.read_csv('data/AAPL.csv')
df['Return'] = df['Close'].pct_change()
df = df.dropna()

# Features: lagged returns
for lag in range(1, 6):
    df[f'Lag_{lag}'] = df['Return'].shift(lag)
df = df.dropna()

X = df[[f'Lag_{lag}' for lag in range(1, 6)]]
y = df['Return']

model = LinearRegression()
model.fit(X, y)
prediction = model.predict([X.iloc[-1]])[0]
print(f"tihassfjord's Next-Day Return Prediction: {prediction:.4f}")
```

---

## 5. Recommendation System

### **Project Description**

> I, tihassfjord, built a recommendation engine—Netflix-style. Here I show how collaborative filtering works, how to handle sparse data, and how to recommend stuff to strangers on the internet (hopefully better than YouTube).

### **README.md**

```markdown
# tihassfjord's Recommendation System

## Goal
Build a simple movie recommendation system using matrix factorization.

## Dataset
- MovieLens 100k (ml-100k, download from GroupLens)

## Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn

## How to Run
1. Place `u.data` (MovieLens) in `data/`
2. `python recommender_tihassfjord.py`

## Example Output
```

Top 5 movies for user 10: \['Star Wars', 'Toy Story', ...]

```
```

### **Directory Structure**

```
recommendation-system-tihassfjord/
│
├── data/
│   └── u.data
├── recommender_tihassfjord.py
└── README.md
```

### **Key Code Snippet:**

```python
# recommender_tihassfjord.py

"""
tihassfjord's Movie Recommendation System
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

print("tihassfjord: Recommending movies like a machine.")

ratings = pd.read_csv('data/u.data', sep='\t', names=['user','movie','rating','timestamp'])
user_movie = ratings.pivot(index='user', columns='movie', values='rating').fillna(0)

svd = TruncatedSVD(n_components=20, random_state=42)
latent = svd.fit_transform(user_movie)
preds = np.dot(latent, svd.components_)

def recommend_movies(user_id, top_n=5):
    user_idx = user_id - 1
    recs = preds[user_idx].argsort()[::-1][:top_n]
    print(f"tihassfjord's recommendations for user {user_id}: {recs}")

recommend_movies(10)
```

</details>

---
