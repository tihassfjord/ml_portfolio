# 📁 tihassfjord's Complete Machine Learning Portfolio

Welcome to my comprehensive machine learning portfolio! This collection showcases fundamental and advanced ML skills through carefully crafted beginner and intermediate projects, demonstrating proficiency across multiple domains of machine learning.

## 🚀 Quick Start with UV

Each project can be set up quickly using `uv` (the fast Python package installer):

```powershell
# Navigate to any project
cd eda-portfolio-tihassfjord

# Create virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt

# Activate environment (Windows)
.venv\Scripts\activate

# Run the project
jupyter notebook  # For EDA projects
# OR
python script_name.py  # For standalone scripts
```

## 📊 Beginner Projects Overview

### 1. 🔍 [EDA Portfolio](./eda-portfolio-tihassfjord/)
**Exploratory Data Analysis showcase**
- **Datasets**: Titanic, Iris
- **Skills**: Data cleaning, visualization, statistical analysis
- **Tools**: Pandas, Matplotlib, Seaborn
- **Deliverables**: Interactive Jupyter notebooks with insights

### 2. 🌸 [Iris Classifier](./iris-flower-classifier-tihassfjord/)
**Classic ML classification problem**
- **Dataset**: Iris flowers (built-in scikit-learn)
- **Skills**: Classification, model evaluation
- **Algorithm**: Random Forest
- **Output**: Accuracy metrics and feature importance

### 3. 🚢 [Titanic Survival Predictor](./titanic-survival-tihassfjord/)
**Binary classification with feature engineering**
- **Dataset**: Titanic passenger data
- **Skills**: Feature engineering, handling missing data
- **Focus**: Interpretable models and real-world insights
- **Metrics**: Accuracy, confusion matrix, feature importance

### 4. 🏠 [Housing Price Predictor](./housing-price-predictor-tihassfjord/)
**Regression analysis for price prediction**
- **Dataset**: Synthetic housing data (realistic features)
- **Skills**: Regression, feature selection, evaluation
- **Algorithm**: Gradient Boosting
- **Metrics**: RMSE, R² score, prediction vs actual plots

### 5. 📱 [Customer Churn Predictor](./customer-churn-tihassfjord/)
**Business-focused classification problem**
- **Dataset**: Synthetic telecom customer data
- **Skills**: Binary classification, business metrics
- **Focus**: ROC AUC, precision/recall for business decisions
- **Output**: High-risk customer identification

## 🚀 Intermediate Projects Overview

### 6. 📈 [Advanced Linear Regression](./linear-regression-tihassfjord/)
**Mathematical foundations with gradient descent**
- **Implementation**: From-scratch gradient descent algorithm
- **Skills**: Mathematical optimization, feature scaling, regularization
- **Features**: Multiple regression types, polynomial features, cross-validation
- **Visualization**: Training curves, residual analysis, feature importance

### 7. 🖼️ [Image Classification System](./image-classification-tihassfjord/)
**Deep learning with CNNs and MLPs**
- **Architectures**: Convolutional Neural Networks, Multi-Layer Perceptrons
- **Dataset**: CIFAR-10 (10 object classes)
- **Skills**: Deep learning, computer vision, model comparison
- **Advanced Features**: Data augmentation, dropout, batch normalization

### 8. 💬 [Sentiment Analysis Engine](./sentiment-analysis-tihassfjord/)
**Natural Language Processing with deep learning**
- **Models**: LSTM networks, traditional ML (Naive Bayes, SVM)
- **Skills**: Text preprocessing, sequence modeling, NLP pipelines
- **Features**: Attention mechanisms, word embeddings, model comparison
- **Applications**: Real-time sentiment prediction, text classification

### 9. 📊 [Stock Price Predictor](./stock-price-predictor-tihassfjord/)
**Time series forecasting with LSTM networks**
- **Algorithms**: LSTM neural networks, technical analysis
- **Skills**: Time series analysis, financial modeling, feature engineering
- **Technical Indicators**: Moving averages, RSI, MACD, Bollinger Bands
- **Features**: Multi-step ahead forecasting, market event simulation

### 10. 🎬 [Movie Recommendation System](./recommendation-system-tihassfjord/)
**Collaborative filtering and content-based recommendations**
- **Methods**: User-based CF, item-based CF, content-based filtering, matrix factorization
- **Skills**: Recommendation engines, similarity metrics, hybrid systems
- **Features**: Cold start handling, recommendation diversity, performance evaluation
- **Algorithms**: SVD, cosine similarity, TF-IDF vectorization

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.8+**
- **Core Libraries**: pandas, scikit-learn, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Environment**: uv, jupyter

### Machine Learning & Deep Learning
- **Traditional ML**: Random Forest, Gradient Boosting, SVM, Naive Bayes
- **Deep Learning**: TensorFlow/Keras, LSTM, CNN, MLP
- **Specialized**: scikit-surprise (recommendations), NLTK (NLP)

### Advanced Techniques
- **Feature Engineering**: Polynomial features, technical indicators, text preprocessing
- **Model Evaluation**: Cross-validation, ROC curves, confusion matrices
- **Optimization**: Gradient descent, Adam optimizer, regularization
- **Data Generation**: Synthetic datasets with realistic patterns

## 📁 Project Structure

Each project follows a consistent structure:
```
project-name-tihassfjord/
├── README.md              # Project-specific documentation
├── requirements.txt       # Python dependencies
├── main_script.py        # Core implementation
├── data/                 # Dataset files
└── notebooks/           # Jupyter notebooks (where applicable)
```

## 🎯 Learning Outcomes

Through these projects, I demonstrate:

### Foundational Skills (Beginner Projects)
1. **Data Manipulation**: Cleaning, preprocessing, feature engineering
2. **Visualization**: Creating meaningful plots and insights
3. **Machine Learning**: Classification and regression fundamentals
4. **Model Evaluation**: Proper metrics and validation techniques
5. **Code Quality**: Clean, documented, reproducible code
6. **Business Understanding**: Translating ML results to actionable insights

### Advanced Skills (Intermediate Projects)
1. **Mathematical Foundations**: Implementing algorithms from scratch (gradient descent)
2. **Deep Learning**: CNNs, LSTMs, MLPs with TensorFlow/Keras
3. **Specialized Domains**: NLP, computer vision, time series, recommendation systems
4. **Advanced Techniques**: Regularization, attention mechanisms, ensemble methods
5. **System Design**: Hybrid models, fallback mechanisms, scalable architectures
6. **Research Skills**: Algorithm comparison, hyperparameter tuning, performance analysis

## 🔧 Setup Instructions

### Prerequisites
- Python 3.8 or higher
- uv (recommended) or pip

### Installation
1. Clone or download this portfolio
2. Navigate to any project directory
3. Set up virtual environment:
   ```powershell
   uv venv
   uv pip install -r requirements.txt
   .venv\Scripts\activate  # Windows
   ```

### Running Projects
- **EDA Projects**: `jupyter notebook` then open `.ipynb` files
- **Standalone Scripts**: `python script_name.py`

## 🏆 Advanced Highlights Portfolio

### 🌟 Advanced ML Projects (12 Highlights)

#### **1. 🧠 [Neural Network from Scratch](./nn-from-scratch-tihassfjord/)**
**Pure NumPy implementation of neural networks**
- **Skills**: Mathematical foundations, backpropagation, optimization
- **Features**: Multi-layer perceptrons, activation functions, MNIST classification
- **Highlight**: Complete from-scratch implementation without ML frameworks

#### **2. 👤 [Real-time Face Recognition System](./face-recognition-tihassfjord/)**
**Computer vision with webcam integration**
- **Skills**: OpenCV, face detection, real-time processing
- **Features**: Live webcam detection, face encoding, recognition pipeline
- **Highlight**: Production-ready real-time CV system

#### **3. 🤖 [Automated ML Pipeline](./automl-pipeline-tihassfjord/)**
**End-to-end AutoML system**
- **Skills**: Pipeline automation, model selection, hyperparameter tuning
- **Features**: Automated preprocessing, cross-validation, reporting
- **Highlight**: Complete AutoML workflow with intelligent automation

#### **4. 📝 [Character Language Model](./char-lm-tihassfjord/)**
**Text generation using pure NumPy**
- **Skills**: NLP, sequence modeling, character-level processing
- **Features**: Text generation, attention mechanisms, training from scratch
- **Highlight**: Language model implementation without deep learning frameworks

#### **5. 📊 [A/B Testing Framework](./ab-test-framework-tihassfjord/)**
**Statistical testing and experimentation**
- **Skills**: Statistical analysis, hypothesis testing, experimental design
- **Features**: Power analysis, confidence intervals, effect size calculation
- **Highlight**: Complete statistical testing toolkit for data science

#### **6. 🎨 [Image Generation System (DCGAN)](./image-gen-tihassfjord/)**
**Generative Adversarial Networks**
- **Skills**: Deep learning, generative models, adversarial training
- **Features**: DCGAN architecture, MNIST generation, GAN training
- **Highlight**: Advanced generative AI with custom GAN implementation

#### **7. 🎮 [Reinforcement Learning Game AI](./rl-game-ai-tihassfjord/)**
**DQN agent for game environments**
- **Skills**: Reinforcement learning, Q-learning, policy optimization
- **Features**: CartPole mastery, experience replay, target networks
- **Highlight**: Advanced RL with Deep Q-Networks

#### **8. 🌍 [Multi-language NLP Pipeline](./multilingual-nlp-tihassfjord/)**
**Advanced multilingual text processing**
- **Skills**: Transformer models, multilingual NLP, advanced preprocessing
- **Features**: Language detection, sentiment analysis, named entity recognition
- **Highlight**: Production-grade multilingual NLP system

#### **9. 🚨 [Real-time Fraud Detection System](./fraud-detection-tihassfjord/)**
**Streaming fraud detection**
- **Skills**: Real-time ML, ensemble methods, anomaly detection
- **Features**: Streaming processing, adaptive thresholds, alert systems
- **Highlight**: Enterprise-grade real-time ML system

#### **10. 🔧 [Build Your Own AutoML](./custom-automl-tihassfjord/)**
**Custom AutoML system from scratch**
- **Skills**: Meta-learning, automated feature engineering, code generation
- **Features**: Custom pipeline generation, intelligent preprocessing, automated reporting
- **Highlight**: Complete AutoML framework with advanced automation

#### **11. 🚀 [MLOps Pipeline](./mlops-pipeline-tihassfjord/)**
**Production ML deployment and monitoring**
- **Skills**: MLOps, FastAPI, model serving, monitoring
- **Features**: Model deployment, performance tracking, automated retraining, API endpoints
- **Highlight**: Enterprise MLOps with FastAPI and MLflow integration

#### **12. ⚡ [Distributed ML System](./distributed-ml-tihassfjord/)**
**Multi-framework distributed training**
- **Skills**: Distributed computing, parallel processing, scalable ML
- **Features**: Ray, Dask, PyTorch DDP, performance benchmarking, fault tolerance
- **Highlight**: Advanced distributed ML with multiple frameworks

### 🎯 Advanced Technical Skills Demonstrated
- **Mathematical Foundations**: Neural networks and optimization from scratch
- **Production Systems**: Real-time processing, monitoring, deployment
- **Advanced Algorithms**: GANs, reinforcement learning, distributed training
- **MLOps**: End-to-end ML lifecycle, monitoring, automation
- **System Architecture**: Scalable, fault-tolerant, enterprise-grade systems

---

## 📈 Portfolio Statistics

### 📊 Project Completion
- **Total Projects**: 27 (100% Complete)
- **Beginner Projects**: 5 ✅
- **Intermediate Projects**: 5 ✅
- **Advanced Highlight Projects**: 12 ✅
- **EDA Notebooks**: 5 ✅

### 🛠️ Technologies Mastered
- **Programming**: Python, NumPy, Pandas, Scikit-learn
- **Deep Learning**: TensorFlow/Keras, PyTorch, Neural Networks from Scratch
- **Computer Vision**: OpenCV, CNNs, GANs, Face Recognition
- **NLP**: LSTM, Transformers, Multilingual Processing, Text Generation
- **MLOps**: FastAPI, MLflow, Docker, Model Monitoring, CI/CD
- **Distributed Computing**: Ray, Dask, PyTorch DDP, Parallel Processing
- **Advanced ML**: Reinforcement Learning, AutoML, A/B Testing, Fraud Detection

### 🎯 Key Achievements
- ✅ **Mathematical Foundations**: Neural networks and optimization from scratch
- ✅ **Production Systems**: Real-time processing, monitoring, deployment
- ✅ **Advanced Algorithms**: GANs, reinforcement learning, distributed training
- ✅ **MLOps Excellence**: End-to-end ML lifecycle, monitoring, automation
- ✅ **System Architecture**: Scalable, fault-tolerant, enterprise-grade systems
- ✅ **Research & Innovation**: Custom implementations, framework comparisons

---

**Created by tihassfjord** | [GitHub Portfolio](https://github.com/tihassfjord) | *A comprehensive showcase of machine learning expertise*
