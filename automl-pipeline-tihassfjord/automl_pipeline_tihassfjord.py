"""
Automated ML Pipeline by tihassfjord
Complete end-to-end machine learning automation system
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class AutoMLPipeline:
    """Automated Machine Learning Pipeline"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.preprocessor = None
        self.results = {}
        
        # Create directories
        Path("models").mkdir(exist_ok=True)
        Path("reports").mkdir(exist_ok=True)
        
    def load_data(self, file_path=None):
        """Load and examine dataset"""
        if file_path is None:
            # Create sample dataset if none provided
            self._create_sample_dataset()
            file_path = "data/sample_data.csv"
        
        print(f"Loading dataset: {file_path}")
        self.df = pd.read_csv(file_path)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        return self.df
    
    def _create_sample_dataset(self):
        """Create a sample dataset for demonstration"""
        Path("data").mkdir(exist_ok=True)
        
        np.random.seed(self.random_state)
        n_samples = 1000
        
        # Generate synthetic data
        data = {
            'age': np.random.randint(18, 80, n_samples),
            'income': np.random.normal(50000, 20000, n_samples),
            'education_years': np.random.randint(10, 20, n_samples),
            'credit_score': np.random.randint(300, 850, n_samples),
            'employment_years': np.random.randint(0, 40, n_samples),
            'debt_ratio': np.random.uniform(0, 1, n_samples),
            'num_accounts': np.random.randint(1, 10, n_samples),
            'geography': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples),
            'gender': np.random.choice(['M', 'F'], n_samples),
        }
        
        # Create target variable (loan approval)
        df_temp = pd.DataFrame(data)
        
        # Create logical target based on features
        approval_score = (
            (df_temp['credit_score'] - 300) / 550 * 0.4 +
            np.clip(df_temp['income'] / 100000, 0, 1) * 0.3 +
            (1 - df_temp['debt_ratio']) * 0.2 +
            np.random.random(n_samples) * 0.1
        )
        
        data['loan_approved'] = (approval_score > 0.5).astype(int)
        
        # Add some missing values
        missing_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
        data['income'][missing_indices[:len(missing_indices)//2]] = np.nan
        
        df = pd.DataFrame(data)
        df.to_csv("data/sample_data.csv", index=False)
        print("Created sample dataset: data/sample_data.csv")
    
    def analyze_data(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*50)
        print("DATA ANALYSIS")
        print("="*50)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nMissing values:")
        missing = self.df.isnull().sum()
        print(missing[missing > 0])
        
        print(f"\nData types:")
        print(self.df.dtypes)
        
        # Identify target column (assume last column or 'target')
        if 'target' in self.df.columns:
            self.target_col = 'target'
        else:
            self.target_col = self.df.columns[-1]
        
        print(f"\nTarget column: {self.target_col}")
        print(f"Target distribution:")
        print(self.df[self.target_col].value_counts())
        
        # Separate features and target
        self.X = self.df.drop(columns=[self.target_col])
        self.y = self.df[self.target_col]
        
        # Identify column types
        self.numeric_features = self.X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = self.X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"\nNumeric features ({len(self.numeric_features)}): {self.numeric_features}")
        print(f"Categorical features ({len(self.categorical_features)}): {self.categorical_features}")
    
    def preprocess_data(self):
        """Preprocess the data"""
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Create preprocessing steps
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        print("✓ Created preprocessing pipeline")
        print("  - Numeric: Imputation (median) + StandardScaler")
        print("  - Categorical: Imputation (missing) + OneHotEncoder")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state, 
            stratify=self.y
        )
        
        print(f"✓ Split data: {self.X_train.shape[0]} train, {self.X_test.shape[0]} test")
    
    def define_models(self):
        """Define models to test"""
        self.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100, 
                random_state=self.random_state
            ),
            'LogisticRegression': LogisticRegression(
                max_iter=1000, 
                random_state=self.random_state
            ),
            'SVM': SVC(
                probability=True, 
                random_state=self.random_state
            )
        }
        
        print(f"\n✓ Defined {len(self.models)} models for comparison")
    
    def train_and_evaluate_models(self):
        """Train and evaluate all models"""
        print("\n" + "="*50)
        print("MODEL TRAINING & EVALUATION")
        print("="*50)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', model)
            ])
            
            # Cross-validation
            cv_scores = cross_val_score(
                pipeline, self.X_train, self.y_train, 
                cv=5, scoring='roc_auc'
            )
            
            # Fit on full training set
            pipeline.fit(self.X_train, self.y_train)
            
            # Test score
            test_score = roc_auc_score(
                self.y_test, 
                pipeline.predict_proba(self.X_test)[:, 1]
            )
            
            # Store results
            self.results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_score': test_score,
                'pipeline': pipeline
            }
            
            print(f"  CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(f"  Test Score: {test_score:.3f}")
            
            # Track best model
            if cv_scores.mean() > self.best_score:
                self.best_score = cv_scores.mean()
                self.best_model = pipeline
                self.best_model_name = name
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning on best model"""
        print("\n" + "="*50)
        print("HYPERPARAMETER TUNING")
        print("="*50)
        
        print(f"Tuning {self.best_model_name}...")
        
        # Define parameter grids
        param_grids = {
            'RandomForest': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5, 10]
            },
            'GradientBoosting': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 7]
            },
            'LogisticRegression': {
                'classifier__C': [0.1, 1, 10],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear']
            }
        }
        
        if self.best_model_name in param_grids:
            param_grid = param_grids[self.best_model_name]
            
            # Create base pipeline
            base_pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', self.models[self.best_model_name])
            ])
            
            # Grid search
            grid_search = GridSearchCV(
                base_pipeline, param_grid, 
                cv=3, scoring='roc_auc', 
                n_jobs=-1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            self.best_model = grid_search.best_estimator_
            tuned_score = grid_search.best_score_
            
            print(f"  Best parameters: {grid_search.best_params_}")
            print(f"  Tuned CV score: {tuned_score:.3f}")
            print(f"  Improvement: {tuned_score - self.best_score:.3f}")
        
        else:
            print(f"  No parameter grid defined for {self.best_model_name}")
    
    def generate_report(self):
        """Generate comprehensive report"""
        print("\n" + "="*50)
        print("FINAL REPORT")
        print("="*50)
        
        # Model comparison
        print("Model Comparison:")
        print("-" * 50)
        for name, result in self.results.items():
            print(f"{name:15}: CV={result['cv_mean']:.3f}±{result['cv_std']:.3f}, "
                  f"Test={result['test_score']:.3f}")
        
        print(f"\nBest Model: {self.best_model_name}")
        
        # Detailed evaluation of best model
        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        
        print(f"Test ROC AUC: {roc_auc_score(self.y_test, y_pred_proba):.3f}")
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Save confusion matrix plot
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {self.best_model_name} (tihassfjord)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('reports/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance (if available)
        if hasattr(self.best_model.named_steps['classifier'], 'feature_importances_'):
            self._plot_feature_importance()
    
    def _plot_feature_importance(self):
        """Plot feature importance"""
        classifier = self.best_model.named_steps['classifier']
        preprocessor = self.best_model.named_steps['preprocessor']
        
        # Get feature names
        feature_names = []
        
        # Numeric features
        feature_names.extend(self.numeric_features)
        
        # Categorical features (one-hot encoded)
        if self.categorical_features:
            cat_encoder = preprocessor.named_transformers_['cat']['onehot']
            cat_feature_names = cat_encoder.get_feature_names_out(self.categorical_features)
            feature_names.extend(cat_feature_names)
        
        # Get importances
        importances = classifier.feature_importances_
        
        # Plot top 15 features
        indices = np.argsort(importances)[::-1][:15]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importance - {self.best_model_name} (tihassfjord)')
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig('reports/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self):
        """Save the best model"""
        model_path = f"models/best_model_{self.best_model_name.lower()}_tihassfjord.pkl"
        joblib.dump(self.best_model, model_path)
        print(f"\n✓ Model saved: {model_path}")
        
        # Save metadata
        metadata = {
            'model_name': self.best_model_name,
            'cv_score': self.best_score,
            'test_score': roc_auc_score(
                self.y_test, 
                self.best_model.predict_proba(self.X_test)[:, 1]
            ),
            'features': {
                'numeric': self.numeric_features,
                'categorical': self.categorical_features
            }
        }
        
        import json
        with open('models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✓ Model metadata saved")
    
    def run_pipeline(self, file_path=None):
        """Run the complete AutoML pipeline"""
        print("AutoML Pipeline by tihassfjord")
        print("=" * 30)
        
        try:
            # Load and analyze data
            self.load_data(file_path)
            self.analyze_data()
            
            # Preprocess data
            self.preprocess_data()
            
            # Define and train models
            self.define_models()
            self.train_and_evaluate_models()
            
            # Hyperparameter tuning
            self.hyperparameter_tuning()
            
            # Generate report and save model
            self.generate_report()
            self.save_model()
            
            print(f"\n🎉 AutoML pipeline complete! Best model: {self.best_model_name}")
            
        except Exception as e:
            print(f"Error in AutoML pipeline: {e}")
            raise

def main():
    """Main function"""
    # Get data file from command line argument
    file_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Create and run AutoML pipeline
    automl = AutoMLPipeline()
    automl.run_pipeline(file_path)

if __name__ == "__main__":
    main()
