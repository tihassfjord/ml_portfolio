#!/usr/bin/env python3
"""
Build Your Own AutoML (highlight) — tihassfjord

Advanced AutoML system that automates the entire machine learning pipeline:
data preprocessing, feature engineering, model selection, hyperparameter tuning,
and code generation. Designed to compete with commercial AutoML solutions.

Author: tihassfjord
Project: Advanced ML Portfolio - Custom AutoML System
"""

import os
import sys
import time
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import json
import pickle
from pathlib import Path
import itertools

warnings.filterwarnings('ignore')

# Core libraries
import pandas as pd
import numpy as np

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score, classification_report
import joblib

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automl_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for ML models."""
    name: str
    model: Any
    param_grid: Dict
    task_type: str  # 'classification' or 'regression'
    preprocessing_requirements: List[str]

@dataclass 
class AutoMLResult:
    """Results from AutoML pipeline."""
    best_model: Any
    best_score: float
    best_params: Dict
    cv_scores: List[float]
    preprocessing_pipeline: Any
    feature_importance: Optional[Dict] = None
    training_time: Optional[float] = None
    generated_code: Optional[str] = None

class CustomAutoML:
    """
    Advanced AutoML system with comprehensive automation capabilities.
    """
    
    def __init__(self, task_type: str = 'auto', random_state: int = 42):
        """
        Initialize the AutoML system.
        
        Args:
            task_type: 'classification', 'regression', or 'auto'
            random_state: Random state for reproducibility
        """
        self.task_type = task_type
        self.random_state = random_state
        self.results = {}
        self.best_pipeline = None
        self.feature_columns = []
        self.target_column = None
        self.data_info = {}
        
        # Initialize model configurations
        self.model_configs = self._initialize_model_configs()
        
        logger.info("tihassfjord: Custom AutoML system initialized")
    
    def _initialize_model_configs(self) -> List[ModelConfig]:
        """Initialize model configurations with hyperparameter grids."""
        configs = []
        
        # Classification models
        classification_models = [
            ModelConfig(
                name='RandomForest_Classifier',
                model=RandomForestClassifier(random_state=self.random_state),
                param_grid={
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [None, 5, 10, 15],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4]
                },
                task_type='classification',
                preprocessing_requirements=['scaling_optional']
            ),
            ModelConfig(
                name='GradientBoosting_Classifier',
                model=GradientBoostingClassifier(random_state=self.random_state),
                param_grid={
                    'model__n_estimators': [50, 100, 200],
                    'model__learning_rate': [0.01, 0.1, 0.2],
                    'model__max_depth': [3, 5, 7]
                },
                task_type='classification',
                preprocessing_requirements=['scaling_optional']
            ),
            ModelConfig(
                name='LogisticRegression',
                model=LogisticRegression(random_state=self.random_state, max_iter=1000),
                param_grid={
                    'model__C': [0.01, 0.1, 1, 10, 100],
                    'model__penalty': ['l1', 'l2'],
                    'model__solver': ['liblinear', 'saga']
                },
                task_type='classification',
                preprocessing_requirements=['scaling_required']
            ),
            ModelConfig(
                name='SVM_Classifier',
                model=SVC(random_state=self.random_state, probability=True),
                param_grid={
                    'model__C': [0.1, 1, 10],
                    'model__kernel': ['rbf', 'linear'],
                    'model__gamma': ['scale', 'auto']
                },
                task_type='classification',
                preprocessing_requirements=['scaling_required']
            ),
            ModelConfig(
                name='KNN_Classifier',
                model=KNeighborsClassifier(),
                param_grid={
                    'model__n_neighbors': [3, 5, 7, 9],
                    'model__weights': ['uniform', 'distance'],
                    'model__metric': ['euclidean', 'manhattan']
                },
                task_type='classification',
                preprocessing_requirements=['scaling_required']
            ),
            ModelConfig(
                name='DecisionTree_Classifier',
                model=DecisionTreeClassifier(random_state=self.random_state),
                param_grid={
                    'model__max_depth': [None, 5, 10, 15, 20],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4]
                },
                task_type='classification',
                preprocessing_requirements=['scaling_optional']
            )
        ]
        
        # Regression models
        regression_models = [
            ModelConfig(
                name='RandomForest_Regressor',
                model=RandomForestRegressor(random_state=self.random_state),
                param_grid={
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [None, 5, 10, 15],
                    'model__min_samples_split': [2, 5, 10]
                },
                task_type='regression',
                preprocessing_requirements=['scaling_optional']
            ),
            ModelConfig(
                name='GradientBoosting_Regressor',
                model=GradientBoostingRegressor(random_state=self.random_state),
                param_grid={
                    'model__n_estimators': [50, 100, 200],
                    'model__learning_rate': [0.01, 0.1, 0.2],
                    'model__max_depth': [3, 5, 7]
                },
                task_type='regression',
                preprocessing_requirements=['scaling_optional']
            ),
            ModelConfig(
                name='LinearRegression',
                model=LinearRegression(),
                param_grid={},
                task_type='regression',
                preprocessing_requirements=['scaling_optional']
            ),
            ModelConfig(
                name='Ridge_Regression',
                model=Ridge(random_state=self.random_state),
                param_grid={
                    'model__alpha': [0.01, 0.1, 1, 10, 100]
                },
                task_type='regression',
                preprocessing_requirements=['scaling_required']
            ),
            ModelConfig(
                name='Lasso_Regression',
                model=Lasso(random_state=self.random_state, max_iter=1000),
                param_grid={
                    'model__alpha': [0.01, 0.1, 1, 10, 100]
                },
                task_type='regression',
                preprocessing_requirements=['scaling_required']
            ),
            ModelConfig(
                name='SVR',
                model=SVR(),
                param_grid={
                    'model__C': [0.1, 1, 10],
                    'model__kernel': ['rbf', 'linear'],
                    'model__gamma': ['scale', 'auto']
                },
                task_type='regression',
                preprocessing_requirements=['scaling_required']
            ),
            ModelConfig(
                name='KNN_Regressor',
                model=KNeighborsRegressor(),
                param_grid={
                    'model__n_neighbors': [3, 5, 7, 9],
                    'model__weights': ['uniform', 'distance']
                },
                task_type='regression',
                preprocessing_requirements=['scaling_required']
            )
        ]
        
        configs.extend(classification_models)
        configs.extend(regression_models)
        
        return configs
    
    def analyze_dataset(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Comprehensive dataset analysis and task type detection.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            
        Returns:
            Dictionary with dataset analysis results
        """
        logger.info("tihassfjord: Analyzing dataset...")
        
        analysis = {
            'shape': df.shape,
            'target_column': target_column,
            'feature_columns': [col for col in df.columns if col != target_column],
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'numerical_columns': [],
            'categorical_columns': [],
            'target_type': None,
            'task_type': None,
            'class_distribution': None,
            'target_statistics': None
        }
        
        # Identify column types
        for col in analysis['feature_columns']:
            if df[col].dtype in ['int64', 'float64']:
                analysis['numerical_columns'].append(col)
            else:
                analysis['categorical_columns'].append(col)
        
        # Analyze target variable
        target_series = df[target_column]
        unique_values = target_series.nunique()
        
        if target_series.dtype in ['int64', 'float64']:
            if unique_values < 20 and target_series.dtype == 'int64':
                # Likely classification (discrete integer values)
                analysis['target_type'] = 'categorical'
                analysis['task_type'] = 'classification'
                analysis['class_distribution'] = target_series.value_counts().to_dict()
            else:
                # Regression (continuous values)
                analysis['target_type'] = 'numerical'
                analysis['task_type'] = 'regression'
                analysis['target_statistics'] = {
                    'mean': float(target_series.mean()),
                    'std': float(target_series.std()),
                    'min': float(target_series.min()),
                    'max': float(target_series.max())
                }
        else:
            # Categorical target
            analysis['target_type'] = 'categorical'
            analysis['task_type'] = 'classification'
            analysis['class_distribution'] = target_series.value_counts().to_dict()
        
        # Override if task_type was manually specified
        if self.task_type != 'auto':
            analysis['task_type'] = self.task_type
        
        # Data quality assessment
        analysis['data_quality'] = {
            'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'constant_columns': [col for col in df.columns if df[col].nunique() <= 1],
            'high_cardinality_columns': [col for col in analysis['categorical_columns'] 
                                       if df[col].nunique() > 50]
        }
        
        self.data_info = analysis
        self.target_column = target_column
        self.feature_columns = analysis['feature_columns']
        
        logger.info(f"Dataset analysis complete: {analysis['task_type']} task with {len(analysis['feature_columns'])} features")
        return analysis
    
    def create_preprocessing_pipeline(self, df: pd.DataFrame, scaling_type: str = 'standard') -> ColumnTransformer:
        """
        Create preprocessing pipeline based on data analysis.
        
        Args:
            df: Input DataFrame
            scaling_type: Type of scaling ('standard', 'minmax', 'robust')
            
        Returns:
            ColumnTransformer for preprocessing
        """
        numerical_features = self.data_info['numerical_columns']
        categorical_features = self.data_info['categorical_columns']
        
        # Numerical preprocessing
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', self._get_scaler(scaling_type))
        ])
        
        # Categorical preprocessing
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        return preprocessor
    
    def _get_scaler(self, scaling_type: str):
        """Get scaler based on type."""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        return scalers.get(scaling_type, StandardScaler())
    
    def automated_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Automated feature engineering and selection.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("tihassfjord: Performing automated feature engineering...")
        
        df_engineered = df.copy()
        
        # Numerical feature engineering
        numerical_cols = self.data_info['numerical_columns']
        
        for col in numerical_cols:
            if df_engineered[col].nunique() > 10:  # Only for columns with sufficient variance
                # Log transformation for skewed data
                if df_engineered[col].min() > 0:
                    skewness = df_engineered[col].skew()
                    if abs(skewness) > 1:
                        df_engineered[f'{col}_log'] = np.log1p(df_engineered[col])
                
                # Polynomial features
                df_engineered[f'{col}_squared'] = df_engineered[col] ** 2
                df_engineered[f'{col}_sqrt'] = np.sqrt(np.abs(df_engineered[col]))
        
        # Interaction features (between top correlated features)
        if len(numerical_cols) >= 2:
            # Calculate correlations with target
            correlations = {}
            for col in numerical_cols:
                if self.data_info['task_type'] == 'regression':
                    corr = df_engineered[col].corr(df_engineered[self.target_column])
                else:
                    # For classification, use point-biserial correlation approximation
                    corr = df_engineered[col].corr(df_engineered[self.target_column].astype(float))
                correlations[col] = abs(corr) if not np.isnan(corr) else 0
            
            # Create interactions between top correlated features
            top_features = sorted(correlations.keys(), key=lambda x: correlations[x], reverse=True)[:3]
            for i, col1 in enumerate(top_features):
                for col2 in top_features[i+1:]:
                    df_engineered[f'{col1}_x_{col2}'] = df_engineered[col1] * df_engineered[col2]
        
        # Categorical feature engineering
        categorical_cols = self.data_info['categorical_columns']
        for col in categorical_cols:
            # Frequency encoding
            freq_encoding = df_engineered[col].value_counts().to_dict()
            df_engineered[f'{col}_frequency'] = df_engineered[col].map(freq_encoding)
        
        # Update feature columns list
        new_features = [col for col in df_engineered.columns if col != self.target_column]
        self.feature_columns = new_features
        
        # Update data info
        self.data_info['feature_columns'] = new_features
        self.data_info['numerical_columns'] = [col for col in new_features 
                                             if df_engineered[col].dtype in ['int64', 'float64']]
        self.data_info['categorical_columns'] = [col for col in new_features 
                                               if df_engineered[col].dtype not in ['int64', 'float64']]
        
        logger.info(f"Feature engineering complete: {len(new_features)} features (added {len(new_features) - len(df.columns) + 1})")
        return df_engineered
    
    def run_automl(self, df: pd.DataFrame, target_column: str, 
                   time_limit_minutes: int = 30, cv_folds: int = 5,
                   feature_engineering: bool = True, 
                   max_models: int = None) -> AutoMLResult:
        """
        Run the complete AutoML pipeline.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            time_limit_minutes: Maximum time to spend on search
            cv_folds: Number of cross-validation folds
            feature_engineering: Whether to perform automated feature engineering
            max_models: Maximum number of models to try (None for all)
            
        Returns:
            AutoMLResult with best model and results
        """
        start_time = time.time()
        logger.info("tihassfjord: Starting AutoML pipeline...")
        
        # Analyze dataset
        analysis = self.analyze_dataset(df, target_column)
        task_type = analysis['task_type']
        
        # Feature engineering
        if feature_engineering:
            df = self.automated_feature_engineering(df)
        
        # Prepare data
        X = df[self.feature_columns]
        y = df[target_column]
        
        # Handle categorical target for classification
        if task_type == 'classification' and y.dtype == 'object':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state,
            stratify=y if task_type == 'classification' else None
        )
        
        # Filter models by task type
        relevant_models = [config for config in self.model_configs 
                          if config.task_type == task_type]
        
        if max_models:
            relevant_models = relevant_models[:max_models]
        
        logger.info(f"Testing {len(relevant_models)} models for {task_type}")
        
        # Try different preprocessing combinations
        scaling_options = ['standard', 'minmax', 'robust', 'none']
        
        best_score = -np.inf if task_type == 'classification' else np.inf
        best_model = None
        best_params = None
        best_preprocessing = None
        all_results = {}
        
        time_limit_seconds = time_limit_minutes * 60
        
        for model_config in relevant_models:
            if time.time() - start_time > time_limit_seconds:
                logger.warning("Time limit reached, stopping model search")
                break
                
            logger.info(f"Testing {model_config.name}...")
            
            # Determine scaling options for this model
            if 'scaling_required' in model_config.preprocessing_requirements:
                current_scaling_options = [opt for opt in scaling_options if opt != 'none']
            elif 'scaling_optional' in model_config.preprocessing_requirements:
                current_scaling_options = scaling_options
            else:
                current_scaling_options = ['none']
            
            for scaling in current_scaling_options:
                try:
                    # Create preprocessing pipeline
                    if scaling == 'none':
                        # Simple imputation only
                        preprocessor = ColumnTransformer(
                            transformers=[
                                ('num', SimpleImputer(strategy='median'), self.data_info['numerical_columns']),
                                ('cat', Pipeline([
                                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                                ]), self.data_info['categorical_columns'])
                            ]
                        )
                    else:
                        preprocessor = self.create_preprocessing_pipeline(df, scaling)
                    
                    # Create full pipeline
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', model_config.model)
                    ])
                    
                    # Hyperparameter tuning
                    if model_config.param_grid:
                        # Use RandomizedSearchCV for faster search
                        search = RandomizedSearchCV(
                            pipeline, 
                            model_config.param_grid,
                            n_iter=20,  # Limit iterations for speed
                            cv=cv_folds,
                            scoring=self._get_scoring_metric(task_type),
                            random_state=self.random_state,
                            n_jobs=-1,
                            error_score='raise'
                        )
                    else:
                        # No hyperparameters to tune
                        search = pipeline
                    
                    # Fit model
                    search.fit(X_train, y_train)
                    
                    if hasattr(search, 'best_score_'):
                        score = search.best_score_
                        params = search.best_params_
                        fitted_model = search.best_estimator_
                    else:
                        # Cross-validation score for models without hyperparameter search
                        cv_scores = cross_val_score(
                            search, X_train, y_train, 
                            cv=cv_folds, 
                            scoring=self._get_scoring_metric(task_type)
                        )
                        score = cv_scores.mean()
                        params = {}
                        fitted_model = search
                        fitted_model.fit(X_train, y_train)
                    
                    # Check if this is the best model
                    is_better = (score > best_score) if task_type == 'classification' else (score < best_score)
                    
                    if is_better:
                        best_score = score
                        best_model = fitted_model
                        best_params = params
                        best_preprocessing = scaling
                    
                    # Store results
                    model_key = f"{model_config.name}_{scaling}"
                    all_results[model_key] = {
                        'score': score,
                        'params': params,
                        'scaling': scaling,
                        'model_name': model_config.name
                    }
                    
                    logger.info(f"  {scaling} scaling: score = {score:.4f}")
                    
                except Exception as e:
                    logger.warning(f"Model {model_config.name} with {scaling} scaling failed: {e}")
                    continue
        
        # Evaluate best model on test set
        test_score = None
        feature_importance = None
        
        if best_model is not None:
            y_test_pred = best_model.predict(X_test)
            
            if task_type == 'classification':
                test_score = accuracy_score(y_test, y_test_pred)
                try:
                    if hasattr(best_model.named_steps['model'], 'feature_importances_'):
                        # Get feature names after preprocessing
                        feature_names = self._get_feature_names(best_model, X)
                        importances = best_model.named_steps['model'].feature_importances_
                        feature_importance = dict(zip(feature_names, importances))
                except:
                    pass
            else:
                test_score = r2_score(y_test, y_test_pred)
                try:
                    if hasattr(best_model.named_steps['model'], 'feature_importances_'):
                        feature_names = self._get_feature_names(best_model, X)
                        importances = best_model.named_steps['model'].feature_importances_
                        feature_importance = dict(zip(feature_names, importances))
                except:
                    pass
        
        # Generate code
        generated_code = self._generate_code(best_model, best_preprocessing, task_type)
        
        training_time = time.time() - start_time
        
        result = AutoMLResult(
            best_model=best_model,
            best_score=best_score,
            best_params=best_params,
            cv_scores=[best_score],  # Simplified
            preprocessing_pipeline=best_model.named_steps['preprocessor'] if best_model else None,
            feature_importance=feature_importance,
            training_time=training_time,
            generated_code=generated_code
        )
        
        # Store all results
        self.results = all_results
        self.best_pipeline = best_model
        
        logger.info(f"AutoML complete! Best model: {best_score:.4f} score in {training_time:.1f}s")
        return result
    
    def _get_scoring_metric(self, task_type: str) -> str:
        """Get appropriate scoring metric for task type."""
        if task_type == 'classification':
            # Use accuracy for simplicity, could be made configurable
            return 'accuracy'
        else:
            # Use negative mean squared error for regression
            return 'neg_mean_squared_error'
    
    def _get_feature_names(self, pipeline, X):
        """Extract feature names from pipeline."""
        try:
            preprocessor = pipeline.named_steps['preprocessor']
            
            # Get numerical feature names
            num_features = self.data_info['numerical_columns']
            
            # Get categorical feature names (after one-hot encoding)
            cat_features = []
            if hasattr(preprocessor.named_transformers_['cat'], 'named_steps'):
                onehot = preprocessor.named_transformers_['cat'].named_steps['onehot']
                if hasattr(onehot, 'get_feature_names_out'):
                    cat_feature_names = onehot.get_feature_names_out(self.data_info['categorical_columns'])
                    cat_features = list(cat_feature_names)
            
            return num_features + cat_features
        except:
            # Fallback to generic names
            return [f'feature_{i}' for i in range(X.shape[1])]
    
    def _generate_code(self, model, preprocessing_type: str, task_type: str) -> str:
        """Generate Python code for the best model."""
        if model is None:
            return "# No successful model found"
        
        model_name = type(model.named_steps['model']).__name__
        
        code = f'''#!/usr/bin/env python3
"""
Generated AutoML Pipeline by tihassfjord
Best {task_type} model: {model_name}
Preprocessing: {preprocessing_type} scaling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.{model.named_steps['model'].__module__.split('.')[-1]} import {model_name}
from sklearn.metrics import accuracy_score, r2_score
import joblib

def load_and_preprocess_data(file_path, target_column):
    """Load and preprocess data."""
    df = pd.read_csv(file_path)
    
    # Feature engineering (add your custom features here)
    # ... (feature engineering code would be generated based on what was done)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y

def create_model():
    """Create the best model pipeline."""
    
    # Define preprocessing for numerical and categorical features
    numerical_features = {self.data_info['numerical_columns']}
    categorical_features = {self.data_info['categorical_columns']}
    
    # Numerical preprocessing
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))'''

        if preprocessing_type != 'none':
            scaler_map = {
                'standard': 'StandardScaler()',
                'minmax': 'MinMaxScaler()',
                'robust': 'RobustScaler()'
            }
            code += f''',
        ('scaler', {scaler_map[preprocessing_type]})'''
        
        code += '''
    ])
    
    # Categorical preprocessing
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Create full pipeline'''
        
        # Add model with best parameters
        if hasattr(model, 'best_params_') and model.best_params_:
            model_params = {k.replace('model__', ''): v for k, v in model.best_params_.items() 
                          if k.startswith('model__')}
            params_str = ', '.join([f'{k}={repr(v)}' for k, v in model_params.items()])
            code += f'''
    model = {model_name}({params_str})'''
        else:
            code += f'''
    model = {model_name}()'''
        
        code += '''
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return pipeline

def train_and_evaluate(X, y):
    """Train and evaluate the model."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = create_model()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate'''
        
        if task_type == 'classification':
            code += '''
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")'''
        else:
            code += '''
    r2 = r2_score(y_test, y_pred)
    print(f"R² Score: {r2:.4f}")'''
        
        code += '''
    
    # Save model
    joblib.dump(model, 'best_model_tihassfjord.pkl')
    print("Model saved as best_model_tihassfjord.pkl")
    
    return model

if __name__ == "__main__":
    # Usage example
    # X, y = load_and_preprocess_data('your_data.csv', 'target_column')
    # model = train_and_evaluate(X, y)
    print("AutoML Generated Code by tihassfjord")
    print("Replace the data loading section with your actual data.")
'''
        
        return code
    
    def save_results(self, output_dir: str = "automl_results"):
        """Save AutoML results and generated code."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save best model
        if self.best_pipeline:
            model_path = output_path / "best_model_tihassfjord.pkl"
            joblib.dump(self.best_pipeline, model_path)
            logger.info(f"Best model saved to {model_path}")
        
        # Save results summary
        results_path = output_path / "automl_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save data analysis
        analysis_path = output_path / "data_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(self.data_info, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
    
    def create_model_comparison_plot(self, save_path: str = "model_comparison.png"):
        """Create visualization comparing all tested models."""
        if not self.results:
            logger.warning("No results to plot")
            return
        
        # Prepare data for plotting
        model_names = []
        scores = []
        scalings = []
        
        for model_key, result in self.results.items():
            model_names.append(result['model_name'])
            scores.append(result['score'])
            scalings.append(result['scaling'])
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Group by model name for better visualization
        df_results = pd.DataFrame({
            'Model': model_names,
            'Score': scores,
            'Scaling': scalings
        })
        
        # Create bar plot
        sns.barplot(data=df_results, x='Model', y='Score', hue='Scaling')
        plt.title('AutoML Model Comparison - tihassfjord', fontsize=16, fontweight='bold')
        plt.xlabel('Model Type')
        plt.ylabel('Cross-Validation Score')
        plt.xticks(rotation=45)
        plt.legend(title='Preprocessing')
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Model comparison plot saved to {save_path}")


def create_sample_datasets():
    """Create sample datasets for testing."""
    np.random.seed(42)
    
    # Classification dataset
    n_samples = 1000
    n_features = 10
    
    # Generate features
    X_class = np.random.randn(n_samples, n_features)
    
    # Create target with some pattern
    y_class = (X_class[:, 0] + X_class[:, 1] * 0.5 + np.random.randn(n_samples) * 0.3 > 0).astype(int)
    
    # Add categorical features
    categories = np.random.choice(['A', 'B', 'C'], n_samples)
    
    df_class = pd.DataFrame(X_class, columns=[f'feature_{i}' for i in range(n_features)])
    df_class['category'] = categories
    df_class['target'] = y_class
    
    # Regression dataset
    X_reg = np.random.randn(n_samples, n_features)
    y_reg = X_reg[:, 0] * 2 + X_reg[:, 1] * 1.5 - X_reg[:, 2] * 0.5 + np.random.randn(n_samples) * 0.5
    
    df_reg = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(n_features)])
    df_reg['category'] = categories
    df_reg['target'] = y_reg
    
    # Save datasets
    os.makedirs('data', exist_ok=True)
    df_class.to_csv('data/classification_sample.csv', index=False)
    df_reg.to_csv('data/regression_sample.csv', index=False)
    
    logger.info("Sample datasets created: classification_sample.csv, regression_sample.csv")
    return df_class, df_reg


def demo_custom_automl():
    """Comprehensive demonstration of the custom AutoML system."""
    print("\n" + "="*70)
    print("🤖 CUSTOM AUTOML SYSTEM DEMO by tihassfjord")
    print("="*70)
    
    # Create sample datasets
    print("\n📊 Creating sample datasets...")
    df_class, df_reg = create_sample_datasets()
    
    # Demo 1: Classification
    print("\n🔍 Classification Task Demo")
    print("-" * 50)
    
    automl_classifier = CustomAutoML(task_type='auto')
    
    # Run AutoML on classification data
    result_class = automl_classifier.run_automl(
        df_class, 
        target_column='target',
        time_limit_minutes=5,
        cv_folds=3,
        feature_engineering=True,
        max_models=4  # Limit for demo speed
    )
    
    print(f"✅ Best Classification Model:")
    print(f"   Score: {result_class.best_score:.4f}")
    print(f"   Model: {type(result_class.best_model.named_steps['model']).__name__}")
    print(f"   Training Time: {result_class.training_time:.1f}s")
    
    if result_class.feature_importance:
        print(f"   Top Features: {list(result_class.feature_importance.keys())[:3]}")
    
    # Demo 2: Regression
    print("\n📈 Regression Task Demo")
    print("-" * 50)
    
    automl_regressor = CustomAutoML(task_type='auto')
    
    result_reg = automl_regressor.run_automl(
        df_reg,
        target_column='target',
        time_limit_minutes=5,
        cv_folds=3,
        feature_engineering=True,
        max_models=4
    )
    
    print(f"✅ Best Regression Model:")
    print(f"   Score: {result_reg.best_score:.4f}")
    print(f"   Model: {type(result_reg.best_model.named_steps['model']).__name__}")
    print(f"   Training Time: {result_reg.training_time:.1f}s")
    
    # Save results
    print("\n💾 Saving Results...")
    automl_classifier.save_results("automl_classification_results")
    automl_regressor.save_results("automl_regression_results")
    
    # Create visualizations
    print("\n📊 Creating Visualizations...")
    automl_classifier.create_model_comparison_plot("classification_comparison.png")
    automl_regressor.create_model_comparison_plot("regression_comparison.png")
    
    # Generate and save code
    print("\n💻 Generated Code Preview:")
    print("-" * 50)
    print(result_class.generated_code[:500] + "...")
    
    # Save generated code
    with open("generated_model_code_tihassfjord.py", "w") as f:
        f.write(result_class.generated_code)
    
    print(f"\n✅ Full generated code saved to: generated_model_code_tihassfjord.py")
    
    print("\n🎉 Custom AutoML System Demo Complete!")
    print("Author: tihassfjord | Advanced ML Portfolio")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('automl_results', exist_ok=True)
    
    # Run demonstration
    try:
        demo_custom_automl()
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
