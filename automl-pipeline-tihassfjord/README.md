# Automated ML Pipeline (highlight) — tihassfjord

## Goal
Automate data cleaning, feature engineering, model selection, and evaluation for tabular data.

## Dataset
- Any tabular CSV (plug in your own)
- Sample dataset included for demonstration

## Requirements
- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## How to Run
```bash
# Use provided sample data
python automl_pipeline_tihassfjord.py

# Or use your own CSV file
python automl_pipeline_tihassfjord.py data/your_data.csv
```

## Example Output
```
AutoML Pipeline by tihassfjord
==============================
Dataset: sample_data.csv
Samples: 1000, Features: 10

Data Preprocessing:
✓ Handled missing values
✓ Encoded categorical features  
✓ Scaled numerical features

Model Selection:
RandomForest: ROC AUC = 0.87 ± 0.03
GradientBoosting: ROC AUC = 0.85 ± 0.04
LogisticRegression: ROC AUC = 0.83 ± 0.02

Best model: RandomForest, ROC AUC: 0.87
Model saved as: best_model_tihassfjord.pkl
```

## Project Structure
```
automl-pipeline-tihassfjord/
│
├── automl_pipeline_tihassfjord.py    # Main AutoML system
├── data/                             # Data directory
│   └── sample_data.csv              # Sample dataset
├── models/                          # Saved models directory
├── requirements.txt                 # Dependencies
└── README.md                       # This file
```

## Key Features
- Automatic data preprocessing
- Missing value imputation
- Feature scaling and encoding
- Model selection and comparison
- Cross-validation evaluation
- Hyperparameter tuning
- Model persistence
- Comprehensive reporting

## Learning Outcomes
- End-to-end ML pipeline design
- Automated preprocessing techniques
- Model selection strategies
- Cross-validation and evaluation
- Hyperparameter optimization
- Production-ready ML systems

---
*Project by tihassfjord - Advanced ML Portfolio*
