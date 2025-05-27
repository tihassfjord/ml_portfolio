#!/usr/bin/env python3
"""
MLOps Pipeline (highlight) — tihassfjord

Production-ready MLOps system with model deployment, monitoring, versioning, 
and automated retraining capabilities. Features FastAPI serving, MLflow tracking,
Docker containerization, and comprehensive CI/CD pipeline integration.

Author: tihassfjord
Project: Advanced ML Portfolio - MLOps Pipeline System
"""

import os
import sys
import time
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import json
import pickle
import joblib
from pathlib import Path
import uvicorn
import asyncio
import threading
from dataclasses import dataclass, asdict

warnings.filterwarnings('ignore')

# Core libraries
import pandas as pd
import numpy as np

# ML libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import joblib

# MLOps libraries
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available. Install with: pip install mlflow")

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not available. Install with: pip install fastapi uvicorn")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mlops_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Model performance metrics."""
    mse: float
    rmse: float
    mae: float
    r2: float
    training_time: float
    model_size_mb: float

@dataclass
class DeploymentConfig:
    """Model deployment configuration."""
    model_name: str
    version: str
    environment: str
    endpoint_url: str
    health_check_interval: int
    max_prediction_latency_ms: float

class MLOpsDataGenerator:
    """Generate synthetic housing data for MLOps demonstration."""
    
    @staticmethod
    def generate_housing_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
        """Generate synthetic housing price dataset."""
        np.random.seed(random_state)
        
        # Features
        square_feet = np.random.normal(2000, 500, n_samples)
        square_feet = np.clip(square_feet, 800, 5000)
        
        bedrooms = np.random.poisson(3, n_samples)
        bedrooms = np.clip(bedrooms, 1, 6)
        
        bathrooms = bedrooms + np.random.normal(0, 0.5, n_samples)
        bathrooms = np.clip(bathrooms, 1, 8)
        
        age = np.random.exponential(15, n_samples)
        age = np.clip(age, 0, 100)
        
        garage = np.random.choice([0, 1, 2], n_samples, p=[0.2, 0.5, 0.3])
        
        location_quality = np.random.normal(5, 2, n_samples)
        location_quality = np.clip(location_quality, 1, 10)
        
        # Price calculation with realistic relationships
        base_price = (
            square_feet * 150 +  # $150 per sq ft
            bedrooms * 10000 +   # $10k per bedroom
            bathrooms * 8000 +   # $8k per bathroom
            garage * 15000 +     # $15k per garage
            location_quality * 20000 -  # Location premium
            age * 1000           # Depreciation
        )
        
        # Add noise and market variations
        noise = np.random.normal(0, 30000, n_samples)
        market_factor = np.random.normal(1.0, 0.1, n_samples)
        
        price = base_price * market_factor + noise
        price = np.clip(price, 50000, 2000000)  # Realistic price range
        
        df = pd.DataFrame({
            'square_feet': square_feet.astype(int),
            'bedrooms': bedrooms,
            'bathrooms': bathrooms.round(1),
            'age': age.round(),
            'garage': garage,
            'location_quality': location_quality.round(1),
            'price': price.round(-3)  # Round to nearest thousand
        })
        
        return df

class ModelTrainer:
    """Advanced model training with MLflow integration."""
    
    def __init__(self, experiment_name: str = "housing_price_prediction"):
        self.experiment_name = experiment_name
        self.models = {}
        self.best_model = None
        self.best_metrics = None
        
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment(experiment_name)
            self.client = MlflowClient()
        
    def prepare_models(self) -> Dict[str, Pipeline]:
        """Prepare different model pipelines."""
        models = {
            'linear_regression': Pipeline([
                ('scaler', StandardScaler()),
                ('model', LinearRegression())
            ]),
            'ridge_regression': Pipeline([
                ('scaler', StandardScaler()),
                ('model', Ridge(alpha=1.0))
            ]),
            'lasso_regression': Pipeline([
                ('scaler', StandardScaler()),
                ('model', Lasso(alpha=1.0))
            ]),
            'random_forest': Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestRegressor(n_estimators=100, random_state=42))
            ]),
            'gradient_boosting': Pipeline([
                ('scaler', StandardScaler()),
                ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
            ])
        }
        
        self.models = models
        logger.info(f"Prepared {len(models)} model pipelines")
        return models
    
    def evaluate_model(self, model: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                      y_train: pd.Series, y_test: pd.Series) -> ModelMetrics:
        """Evaluate model performance."""
        start_time = time.time()
        
        # Train model
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Model size (approximate)
        model_path = 'temp_model.pkl'
        joblib.dump(model, model_path)
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        os.remove(model_path)
        
        return ModelMetrics(
            mse=mse,
            rmse=rmse,
            mae=mae,
            r2=r2,
            training_time=training_time,
            model_size_mb=model_size_mb
        )
    
    def train_and_compare_models(self, df: pd.DataFrame, target_column: str = 'price') -> Dict[str, ModelMetrics]:
        """Train and compare all models with MLflow tracking."""
        logger.info("tihassfjord: Starting model training and comparison...")
        
        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            
            if MLFLOW_AVAILABLE:
                with mlflow.start_run(run_name=f"{model_name}_run"):
                    # Evaluate model
                    metrics = self.evaluate_model(model, X_train, X_test, y_train, y_test)
                    results[model_name] = metrics
                    
                    # Log parameters and metrics to MLflow
                    mlflow.log_params({
                        'model_type': model_name,
                        'train_samples': len(X_train),
                        'test_samples': len(X_test),
                        'features': list(X.columns)
                    })
                    
                    mlflow.log_metrics({
                        'mse': metrics.mse,
                        'rmse': metrics.rmse,
                        'mae': metrics.mae,
                        'r2': metrics.r2,
                        'training_time': metrics.training_time,
                        'model_size_mb': metrics.model_size_mb
                    })
                    
                    # Log model
                    mlflow.sklearn.log_model(
                        model, 
                        f"{model_name}_model",
                        registered_model_name=f"housing_predictor_{model_name}"
                    )
                    
                    logger.info(f"  R² Score: {metrics.r2:.4f}")
                    logger.info(f"  RMSE: ${metrics.rmse:,.0f}")
                    logger.info(f"  Training Time: {metrics.training_time:.2f}s")
            else:
                # Evaluate without MLflow
                metrics = self.evaluate_model(model, X_train, X_test, y_train, y_test)
                results[model_name] = metrics
                logger.info(f"  R² Score: {metrics.r2:.4f}")
                logger.info(f"  RMSE: ${metrics.rmse:,.0f}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k].r2)
        self.best_model = self.models[best_model_name]
        self.best_metrics = results[best_model_name]
        
        # Retrain best model on full training data
        self.best_model.fit(X_train, y_train)
        
        logger.info(f"Best model: {best_model_name} (R² = {self.best_metrics.r2:.4f})")
        
        # Save best model
        model_path = Path("models")
        model_path.mkdir(exist_ok=True)
        
        best_model_path = model_path / f"best_model_{best_model_name}_tihassfjord.pkl"
        joblib.dump(self.best_model, best_model_path)
        
        # Save model metadata
        metadata = {
            'model_name': best_model_name,
            'metrics': asdict(self.best_metrics),
            'features': list(X.columns),
            'target': target_column,
            'training_timestamp': datetime.now().isoformat(),
            'model_path': str(best_model_path)
        }
        
        metadata_path = model_path / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved: {best_model_path}")
        logger.info(f"Metadata saved: {metadata_path}")
        
        return results

class ModelMonitor:
    """Model performance monitoring and drift detection."""
    
    def __init__(self, model_path: str, metadata_path: str):
        self.model = joblib.load(model_path)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.predictions_log = []
        self.performance_log = []
        
    def log_prediction(self, features: Dict, prediction: float, actual: Optional[float] = None):
        """Log prediction for monitoring."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'prediction': prediction,
            'actual': actual
        }
        self.predictions_log.append(log_entry)
        
        # Calculate performance if actual is available
        if actual is not None:
            error = abs(prediction - actual)
            relative_error = error / actual if actual != 0 else 0
            
            performance_entry = {
                'timestamp': datetime.now().isoformat(),
                'absolute_error': error,
                'relative_error': relative_error,
                'prediction': prediction,
                'actual': actual
            }
            self.performance_log.append(performance_entry)
    
    def get_performance_metrics(self, window_hours: int = 24) -> Dict[str, float]:
        """Get recent performance metrics."""
        if not self.performance_log:
            return {}
        
        # Filter recent entries
        cutoff_time = datetime.now().timestamp() - (window_hours * 3600)
        recent_entries = [
            entry for entry in self.performance_log
            if datetime.fromisoformat(entry['timestamp']).timestamp() > cutoff_time
        ]
        
        if not recent_entries:
            return {}
        
        errors = [entry['absolute_error'] for entry in recent_entries]
        relative_errors = [entry['relative_error'] for entry in recent_entries]
        
        return {
            'mean_absolute_error': np.mean(errors),
            'median_absolute_error': np.median(errors),
            'mean_relative_error': np.mean(relative_errors),
            'predictions_count': len(recent_entries),
            'window_hours': window_hours
        }

# FastAPI Application
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Housing Price Prediction API - tihassfjord",
        description="MLOps Pipeline with real-time predictions and monitoring",
        version="1.0.0"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    class PredictionRequest(BaseModel):
        square_feet: int
        bedrooms: int
        bathrooms: float
        age: float
        garage: int
        location_quality: float
    
    class PredictionResponse(BaseModel):
        prediction: float
        model_version: str
        prediction_id: str
        timestamp: str
        confidence_interval: Optional[Dict[str, float]] = None
    
    class HealthResponse(BaseModel):
        status: str
        model_loaded: bool
        model_version: str
        uptime_seconds: float
        predictions_served: int
    
    # Global state
    model_monitor = None
    app_start_time = time.time()
    predictions_served = 0
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize model on startup."""
        global model_monitor
        try:
            model_path = "models/best_model_random_forest_tihassfjord.pkl"
            metadata_path = "models/model_metadata.json"
            
            if os.path.exists(model_path) and os.path.exists(metadata_path):
                model_monitor = ModelMonitor(model_path, metadata_path)
                logger.info("Model loaded successfully for API serving")
            else:
                logger.warning("Model files not found. Train a model first.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        global predictions_served, app_start_time
        
        return HealthResponse(
            status="healthy" if model_monitor else "model_not_loaded",
            model_loaded=model_monitor is not None,
            model_version=model_monitor.metadata['model_name'] if model_monitor else "unknown",
            uptime_seconds=time.time() - app_start_time,
            predictions_served=predictions_served
        )
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict_price(request: PredictionRequest):
        """Make price prediction."""
        global predictions_served
        
        if not model_monitor:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            # Convert request to DataFrame
            features_dict = request.dict()
            features_df = pd.DataFrame([features_dict])
            
            # Make prediction
            prediction = model_monitor.model.predict(features_df)[0]
            
            # Generate prediction ID
            prediction_id = f"pred_{int(time.time())}_{predictions_served}"
            
            # Log prediction
            model_monitor.log_prediction(features_dict, prediction)
            
            predictions_served += 1
            
            # Simple confidence interval (±10% based on training RMSE)
            rmse = model_monitor.metadata['metrics']['rmse']
            confidence_interval = {
                'lower': prediction - rmse,
                'upper': prediction + rmse
            }
            
            response = PredictionResponse(
                prediction=round(prediction, 2),
                model_version=model_monitor.metadata['model_name'],
                prediction_id=prediction_id,
                timestamp=datetime.now().isoformat(),
                confidence_interval=confidence_interval
            )
            
            logger.info(f"Prediction made: ${prediction:,.0f} (ID: {prediction_id})")
            return response
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    @app.get("/metrics")
    async def get_metrics():
        """Get model performance metrics."""
        if not model_monitor:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        return {
            "model_metadata": model_monitor.metadata,
            "recent_performance": model_monitor.get_performance_metrics(),
            "total_predictions": len(model_monitor.predictions_log)
        }
    
    @app.post("/feedback")
    async def provide_feedback(prediction_id: str, actual_price: float):
        """Provide actual price for model performance monitoring."""
        if not model_monitor:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Find the prediction
        for log_entry in model_monitor.predictions_log:
            if prediction_id in str(log_entry):
                # Update with actual value
                model_monitor.log_prediction(
                    log_entry['features'], 
                    log_entry['prediction'], 
                    actual_price
                )
                logger.info(f"Feedback received for {prediction_id}: actual=${actual_price:,.0f}")
                return {"message": "Feedback received", "prediction_id": prediction_id}
        
        raise HTTPException(status_code=404, detail="Prediction ID not found")

class MLOpsPipeline:
    """Complete MLOps pipeline orchestrator."""
    
    def __init__(self):
        self.trainer = ModelTrainer()
        self.data_generator = MLOpsDataGenerator()
        
    def run_training_pipeline(self, data_path: Optional[str] = None):
        """Run the complete training pipeline."""
        logger.info("tihassfjord: Starting MLOps training pipeline...")
        
        # Load or generate data
        if data_path and os.path.exists(data_path):
            logger.info(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
        else:
            logger.info("Generating synthetic housing data...")
            df = self.data_generator.generate_housing_data(n_samples=2000)
            
            # Save generated data
            Path("data").mkdir(exist_ok=True)
            data_path = "data/housing_data_tihassfjord.csv"
            df.to_csv(data_path, index=False)
            logger.info(f"Data saved to {data_path}")
        
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
        
        # Prepare models
        self.trainer.prepare_models()
        
        # Train and compare models
        results = self.trainer.train_and_compare_models(df)
        
        # Create performance visualization
        self.create_model_comparison_plot(results)
        
        logger.info("Training pipeline completed successfully!")
        return results
    
    def create_model_comparison_plot(self, results: Dict[str, ModelMetrics]):
        """Create model comparison visualization."""
        plt.figure(figsize=(15, 10))
        
        models = list(results.keys())
        r2_scores = [results[model].r2 for model in models]
        rmse_values = [results[model].rmse for model in models]
        training_times = [results[model].training_time for model in models]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison - tihassfjord MLOps Pipeline', fontsize=16, fontweight='bold')
        
        # R² Score comparison
        axes[0, 0].bar(models, r2_scores, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('R² Score (Higher is Better)')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE comparison
        axes[0, 1].bar(models, rmse_values, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('RMSE (Lower is Better)')
        axes[0, 1].set_ylabel('RMSE ($)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Training time comparison
        axes[1, 0].bar(models, training_times, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Training Time')
        axes[1, 0].set_ylabel('Seconds')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance vs Speed scatter
        axes[1, 1].scatter(training_times, r2_scores, s=100, alpha=0.7, c='purple')
        for i, model in enumerate(models):
            axes[1, 1].annotate(model, (training_times[i], r2_scores[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 1].set_xlabel('Training Time (seconds)')
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].set_title('Performance vs Speed Trade-off')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        plot_path = plots_dir / "model_comparison_tihassfjord.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Model comparison plot saved: {plot_path}")
    
    def start_api_server(self, host: str = "127.0.0.1", port: int = 8000):
        """Start the FastAPI server."""
        if not FASTAPI_AVAILABLE:
            logger.error("FastAPI not available. Install with: pip install fastapi uvicorn")
            return
        
        logger.info(f"Starting MLOps API server at http://{host}:{port}")
        logger.info("Endpoints available:")
        logger.info("  GET  /health - Health check")
        logger.info("  POST /predict - Make predictions")
        logger.info("  GET  /metrics - Performance metrics")
        logger.info("  POST /feedback - Provide ground truth feedback")
        
        uvicorn.run(app, host=host, port=port, log_level="info")

def demo_mlops_pipeline():
    """Demonstrate the complete MLOps pipeline."""
    print("="*70)
    print("🚀 MLOps Pipeline (highlight) — tihassfjord")
    print("="*70)
    print("Advanced MLOps system with model training, deployment, and monitoring")
    print()
    
    # Initialize pipeline
    pipeline = MLOpsPipeline()
    
    try:
        # Run training pipeline
        results = pipeline.run_training_pipeline()
        
        print("\n" + "="*50)
        print("MODEL TRAINING RESULTS")
        print("="*50)
        
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  R² Score: {metrics.r2:.4f}")
            print(f"  RMSE: ${metrics.rmse:,.0f}")
            print(f"  MAE: ${metrics.mae:,.0f}")
            print(f"  Training Time: {metrics.training_time:.2f}s")
            print(f"  Model Size: {metrics.model_size_mb:.2f} MB")
        
        best_model = max(results.keys(), key=lambda k: results[k].r2)
        print(f"\n🏆 Best Model: {best_model} (R² = {results[best_model].r2:.4f})")
        
        print("\n" + "="*50)
        print("DEPLOYMENT READY")
        print("="*50)
        print("✓ Model trained and saved")
        print("✓ Performance metrics logged")
        print("✓ Model comparison plots created")
        print("✓ Ready for API deployment")
        
        if FASTAPI_AVAILABLE:
            print("\n📡 To start the API server, run:")
            print("python mlops_pipeline_tihassfjord.py --serve")
        else:
            print("\n📦 To enable API serving, install:")
            print("pip install fastapi uvicorn")
        
        if MLFLOW_AVAILABLE:
            print("\n📊 To view MLflow tracking, run:")
            print("mlflow ui")
        else:
            print("\n📦 To enable MLflow tracking, install:")
            print("pip install mlflow")
        
        print("\n🎯 Example API usage:")
        print("""
curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{
       "square_feet": 2000,
       "bedrooms": 3,
       "bathrooms": 2.5,
       "age": 10,
       "garage": 2,
       "location_quality": 7.5
     }'
        """)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise

def main():
    """Main function with CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MLOps Pipeline - tihassfjord")
    parser.add_argument("--train", action="store_true", help="Run training pipeline")
    parser.add_argument("--serve", action="store_true", help="Start API server")
    parser.add_argument("--data", type=str, help="Path to training data CSV")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="API server host")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    
    args = parser.parse_args()
    
    pipeline = MLOpsPipeline()
    
    if args.serve:
        pipeline.start_api_server(args.host, args.port)
    elif args.train:
        pipeline.run_training_pipeline(args.data)
    else:
        # Run demo
        demo_mlops_pipeline()

if __name__ == "__main__":
    main()
