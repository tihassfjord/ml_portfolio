Below is a **collapsible Markdown document**: each section is organized for copy-paste into your repo, blog, or wherever you want.
You’ll get for **each project**:

* Original project description (“I, Tor-Ivar, ...” style)
* README.md (goal, dataset, requirements, how-to, sample output)
* Directory structure suggestion
* Key code snippets/notebook cells (with your name!)

---

# 📁 Tor-Ivar’s Machine Learning Portfolio

<details>
<summary><strong>Beginner Projects</strong> (click to expand/collapse)</summary>

---

## 1. Exploratory Data Analysis (EDA) Portfolio

### **Project Description**

> I, Tor-Ivar, created this project to show my data-wrangling and visualization skills. Here, I take various real-world datasets, dig deep using Python, and turn chaos into insight—always with a bit of skepticism and humor. EDA is where every data science journey begins, so I wanted a solid showcase.

### **README.md**

```markdown
# Tor-Ivar's EDA Portfolio

## Goal
Perform and document exploratory data analysis (EDA) on multiple datasets, highlighting data cleaning, visualization, and insight extraction skills.

## Datasets Used
- Titanic (Kaggle)
- Iris (UCI)
- Any public CSV you like

## Requirements
- Python 3.8+
- pandas
- matplotlib
- seaborn
- jupyter

## How to Run
1. Clone this repo.
2. Install requirements: `pip install -r requirements.txt`
3. Run `jupyter notebook` and open any `.ipynb` file in `/notebooks`.

## Example Outputs
- Correlation heatmaps
- Distribution plots
- Insightful Markdown commentary

---
```

### **Directory Structure**

```
eda-portfolio-tor-ivar/
│
├── data/
│   ├── titanic.csv
│   └── iris.csv
│
├── notebooks/
│   ├── 01_titanic_eda_tor_ivar.ipynb
│   ├── 02_iris_eda_tor_ivar.ipynb
│   └── ...
│
├── requirements.txt
└── README.md
```

### **Key Code Snippet: (Titanic EDA example)**

```python
# notebooks/01_titanic_eda_tor_ivar.ipynb

# Title cell
# Tor-Ivar's Titanic EDA Notebook

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("👋 Hello, I'm Tor-Ivar, and I'm about to dig into the Titanic dataset.")

df = pd.read_csv('../data/titanic.csv')
print(df.head())

# Distribution of ages
plt.figure(figsize=(8,4))
sns.histplot(df['Age'].dropna(), kde=True)
plt.title("Age Distribution - Titanic (Tor-Ivar)")
plt.show()

# Survival by gender
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title("Survival by Gender - Tor-Ivar")
plt.show()
```

---

## 2. Iris Flower Classification

### **Project Description**

> This is Tor-Ivar’s rite of passage into machine learning: classifying the famous iris dataset. If you haven’t done this, are you even a data scientist? Here, I train a simple classifier, and more importantly, I *explain* what’s going on—no hand-waving.

### **README.md**

```markdown
# Tor-Ivar's Iris Flower Classifier

## Goal
Build and explain a simple classifier for the Iris dataset using scikit-learn.

## Dataset
- Iris Dataset (from scikit-learn datasets)

## Requirements
- Python 3.8+
- scikit-learn
- pandas
- matplotlib

## How to Run
1. `python iris_classifier_tor_ivar.py`

## Example Output
```

Training accuracy: 0.97
Test accuracy: 1.00
Predicted: setosa, Versicolor, Virginica...

```
```

### **Directory Structure**

```
iris-flower-classifier-tor-ivar/
│
├── iris_classifier_tor_ivar.py
└── README.md
```

### **Key Code Snippet:**

```python
# iris_classifier_tor_ivar.py

"""
Tor-Ivar's Iris Flower Classification Script
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Tor-Ivar: Loading Iris data 🚀")
data = load_iris(as_frame=True)
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Tor-Ivar's Iris Classifier Accuracy: {accuracy_score(y_test, y_pred):.2f}")
```

---

## 3. Titanic Survival Prediction

### **Project Description**

> I, Tor-Ivar, attempt the classic: predicting who survives the Titanic. This is more than a Kaggle meme—it's about practicing feature engineering, handling missing data, and making a solid, explainable model.

### **README.md**

```markdown
# Tor-Ivar's Titanic Survival Prediction

## Goal
Predict passenger survival on the Titanic with interpretable ML models.

## Dataset
- Titanic Dataset (Kaggle)

## Requirements
- Python 3.8+
- pandas
- scikit-learn
- matplotlib

## How to Run
1. Put `titanic.csv` in the `data/` folder.
2. Run: `python titanic_predict_tor_ivar.py`

## Example Output
```

Validation accuracy: 0.79
Feature importances: Sex, Age, Fare...

```
```

### **Directory Structure**

```
titanic-survival-tor-ivar/
│
├── data/
│   └── titanic.csv
├── titanic_predict_tor_ivar.py
└── README.md
```

### **Key Code Snippet:**

```python
# titanic_predict_tor_ivar.py

"""
Tor-Ivar's Titanic Survival Predictor
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("Tor-Ivar: Ready to see who lives and dies (in the data).")
df = pd.read_csv('data/titanic.csv')

df['Sex'] = df['Sex'].map({'male':0, 'female':1})
df['Age'].fillna(df['Age'].median(), inplace=True)
features = ['Pclass', 'Sex', 'Age', 'Fare']

X = df[features]
y = df['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
acc = model.score(X_val, y_val)
print(f"Tor-Ivar's Model Validation Accuracy: {acc:.2f}")
```

---

## 4. Housing Price Predictor

### **Project Description**

> I, Tor-Ivar, take on the real estate market. Here, I build a regression model to predict house prices—a favorite ML exercise. The point: get hands-on with regression, learn what drives price, and avoid overfitting traps.

### **README.md**

```markdown
# Tor-Ivar's Housing Price Predictor

## Goal
Predict house prices based on features like size, location, and amenities.

## Dataset
- Ames Housing Dataset (Kaggle) or any housing.csv

## Requirements
- Python 3.8+
- pandas
- scikit-learn
- matplotlib

## How to Run
1. Place `housing.csv` in `data/`.
2. Run: `python housing_price_tor_ivar.py`

## Example Output
```

RMSE: 35000.0
Top features: GrLivArea, OverallQual, Neighborhood...

```
```

### **Directory Structure**

```
housing-price-predictor-tor-ivar/
│
├── data/
│   └── housing.csv
├── housing_price_tor_ivar.py
└── README.md
```

### **Key Code Snippet:**

```python
# housing_price_tor_ivar.py

"""
Tor-Ivar's Housing Price Predictor
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

print("Tor-Ivar: Predicting house prices like a skeptical economist.")
df = pd.read_csv('data/housing.csv')

features = ['GrLivArea', 'OverallQual', 'GarageCars', 'TotalBsmtSF']
X = df[features]
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
print(f"Tor-Ivar's Model RMSE: {rmse:.0f}")
```

---

## 5. Customer Churn Predictor

### **Project Description**

> I, Tor-Ivar, wanted to know: can we spot which customers are about to leave? I built a churn prediction model, focused on clarity and honest feature engineering—no black boxes, just practical insights for business decisions.

### **README.md**

```markdown
# Tor-Ivar's Customer Churn Predictor

## Goal
Predict whether a customer will churn using classical ML techniques.

## Dataset
- Telco Customer Churn Dataset (Kaggle) or churn.csv

## Requirements
- Python 3.8+
- pandas
- scikit-learn
- matplotlib

## How to Run
1. Put `churn.csv` in `data/`.
2. Run: `python churn_predictor_tor_ivar.py`

## Example Output
```

ROC AUC: 0.81
Confusion matrix: \[\[...]]

```
```

### **Directory Structure**

```
customer-churn-tor-ivar/
│
├── data/
│   └── churn.csv
├── churn_predictor_tor_ivar.py
└── README.md
```

### **Key Code Snippet:**

```python
# churn_predictor_tor_ivar.py

"""
Tor-Ivar's Customer Churn Predictor
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

print("Tor-Ivar: Predicting which customers are ready to bail.")
df = pd.read_csv('data/churn.csv')

df['gender'] = df['gender'].map({'Male':0, 'Female':1})
features = ['tenure', 'MonthlyCharges', 'gender']
target = 'Churn'
df[target] = df[target].map({'Yes':1, 'No':0})

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
probs = model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, probs)
print(f"Tor-Ivar's Model ROC AUC: {auc:.2f}")
```

</details>

---
