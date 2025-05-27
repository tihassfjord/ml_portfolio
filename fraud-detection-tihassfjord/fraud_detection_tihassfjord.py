#!/usr/bin/env python3
"""
Real-time Fraud Detection System (highlight) — tihassfjord

Advanced fraud detection system with real-time streaming capabilities,
anomaly detection, ensemble models, and comprehensive monitoring.
Designed for financial transaction monitoring and fraud prevention.

Author: tihassfjord
Project: Advanced ML Portfolio - Real-time Fraud Detection
"""

import os
import sys
import time
import logging
import warnings
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import json
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')

# Core libraries
import pandas as pd
import numpy as np
from collections import deque
import queue

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fraud_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TransactionData:
    """Data structure for transaction information."""
    transaction_id: str
    user_id: str
    amount: float
    merchant_category: str
    transaction_time: datetime
    location: str
    device_type: str
    is_weekend: bool
    hour_of_day: int
    days_since_last_transaction: float
    amount_zscore: float
    is_fraud: Optional[bool] = None
    fraud_score: Optional[float] = None

class FraudDetectionEngine:
    """
    Real-time fraud detection engine with multiple detection algorithms
    and streaming capabilities.
    """
    
    def __init__(self, model_save_path: str = "models"):
        """Initialize the fraud detection engine."""
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(exist_ok=True)
        
        # Initialize models
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
        # Real-time processing
        self.transaction_buffer = deque(maxlen=10000)
        self.alert_queue = queue.Queue()
        self.is_monitoring = False
        
        # Statistics tracking
        self.stats = {
            'total_transactions': 0,
            'fraud_detected': 0,
            'false_positives': 0,
            'processing_time_ms': deque(maxlen=1000),
            'hourly_stats': {}
        }
        
        logger.info("tihassfjord: Fraud detection engine initialized")
    
    def generate_synthetic_data(self, n_samples: int = 100000, fraud_rate: float = 0.02) -> pd.DataFrame:
        """
        Generate realistic synthetic transaction data for training and testing.
        
        Args:
            n_samples: Number of transactions to generate
            fraud_rate: Proportion of fraudulent transactions
            
        Returns:
            DataFrame with synthetic transaction data
        """
        np.random.seed(42)
        logger.info(f"tihassfjord: Generating {n_samples} synthetic transactions")
        
        # Generate base transaction data
        data = {
            'transaction_id': [f"TXN_{i:08d}" for i in range(n_samples)],
            'user_id': np.random.choice(range(1000, 10000), n_samples),
            'amount': np.random.lognormal(3, 1.5, n_samples),  # Log-normal distribution for amounts
            'merchant_category': np.random.choice(
                ['grocery', 'gas', 'restaurant', 'retail', 'online', 'entertainment', 'utilities'],
                n_samples,
                p=[0.25, 0.15, 0.20, 0.15, 0.10, 0.10, 0.05]
            ),
            'location': np.random.choice(
                ['domestic', 'international', 'neighboring_country'],
                n_samples,
                p=[0.85, 0.10, 0.05]
            ),
            'device_type': np.random.choice(
                ['mobile', 'desktop', 'tablet', 'atm'],
                n_samples,
                p=[0.60, 0.25, 0.10, 0.05]
            )
        }
        
        # Generate time-based features
        base_time = datetime.now() - timedelta(days=30)
        transaction_times = [
            base_time + timedelta(
                days=np.random.randint(0, 30),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60)
            )
            for _ in range(n_samples)
        ]
        
        data['transaction_time'] = transaction_times
        data['is_weekend'] = [t.weekday() >= 5 for t in transaction_times]
        data['hour_of_day'] = [t.hour for t in transaction_times]
        
        df = pd.DataFrame(data)
        
        # Calculate user-based features
        user_stats = df.groupby('user_id')['amount'].agg(['mean', 'std']).reset_index()
        user_stats['std'] = user_stats['std'].fillna(user_stats['mean'] * 0.5)
        df = df.merge(user_stats, on='user_id', suffixes=('', '_user_avg'))
        
        # Amount z-score relative to user's typical behavior
        df['amount_zscore'] = (df['amount'] - df['mean']) / df['std']
        df['amount_zscore'] = df['amount_zscore'].fillna(0)
        
        # Days since last transaction (simulate)
        df['days_since_last_transaction'] = np.random.exponential(2, n_samples)
        
        # Generate fraud labels with realistic patterns
        n_fraud = int(n_samples * fraud_rate)
        fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
        df['is_fraud'] = False
        df.loc[fraud_indices, 'is_fraud'] = True
        
        # Make fraudulent transactions more suspicious
        fraud_mask = df['is_fraud']
        
        # Fraudulent transactions tend to be larger
        df.loc[fraud_mask, 'amount'] *= np.random.uniform(2, 10, fraud_mask.sum())
        
        # More likely to be international
        df.loc[fraud_mask & (np.random.random(fraud_mask.sum()) < 0.3), 'location'] = 'international'
        
        # More likely at unusual hours
        unusual_hours = np.random.choice([2, 3, 4, 23, 0, 1], fraud_mask.sum())
        df.loc[fraud_mask, 'hour_of_day'] = unusual_hours
        
        # Recalculate z-scores after modification
        user_stats = df.groupby('user_id')['amount'].agg(['mean', 'std']).reset_index()
        user_stats['std'] = user_stats['std'].fillna(user_stats['mean'] * 0.5)
        df = df.merge(user_stats, on='user_id', suffixes=('', '_updated'))
        df['amount_zscore'] = (df['amount'] - df['mean_updated']) / df['std_updated']
        df['amount_zscore'] = df['amount_zscore'].fillna(0)
        
        # Clean up temporary columns
        df = df.drop(columns=['mean', 'std', 'mean_updated', 'std_updated'])
        
        logger.info(f"Generated dataset: {len(df)} transactions, {fraud_mask.sum()} fraudulent ({fraud_rate*100:.1f}%)")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for machine learning models.
        
        Args:
            df: Input DataFrame with transaction data
            
        Returns:
            DataFrame with processed features
        """
        features_df = df.copy()
        
        # Encode categorical variables
        categorical_columns = ['merchant_category', 'location', 'device_type']
        features_df = pd.get_dummies(features_df, columns=categorical_columns, prefix=categorical_columns)
        
        # Convert boolean to int
        features_df['is_weekend'] = features_df['is_weekend'].astype(int)
        
        # Time-based features
        if 'transaction_time' in features_df.columns:
            features_df['transaction_hour_sin'] = np.sin(2 * np.pi * features_df['hour_of_day'] / 24)
            features_df['transaction_hour_cos'] = np.cos(2 * np.pi * features_df['hour_of_day'] / 24)
        
        # Amount features
        features_df['log_amount'] = np.log1p(features_df['amount'])
        features_df['amount_squared'] = features_df['amount'] ** 2
        
        # Remove non-feature columns
        non_feature_cols = ['transaction_id', 'user_id', 'transaction_time', 'is_fraud']
        feature_cols = [col for col in features_df.columns if col not in non_feature_cols]
        
        self.feature_columns = feature_cols
        return features_df[feature_cols]
    
    def train_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train multiple fraud detection models.
        
        Args:
            df: Training data DataFrame
            
        Returns:
            Dictionary with training results
        """
        logger.info("tihassfjord: Training fraud detection models...")
        
        # Prepare features
        X = self.prepare_features(df)
        y = df['is_fraud'].astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize models
        models_config = {
            'isolation_forest': {
                'model': IsolationForest(contamination=0.02, random_state=42),
                'requires_scaling': True,
                'is_unsupervised': True
            },
            'random_forest': {
                'model': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
                'requires_scaling': False,
                'is_unsupervised': False
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
                'requires_scaling': True,
                'is_unsupervised': False
            }
        }
        
        results = {}
        
        for model_name, config in models_config.items():
            logger.info(f"Training {model_name}...")
            
            # Setup pipeline
            steps = []
            if config['requires_scaling']:
                scaler = StandardScaler()
                steps.append(('scaler', scaler))
                self.scalers[model_name] = scaler
            
            steps.append(('model', config['model']))
            pipeline = Pipeline(steps)
            
            # Train model
            if config['is_unsupervised']:
                # For unsupervised models like Isolation Forest
                pipeline.fit(X_train)
                y_pred = pipeline.predict(X_test)
                # Convert Isolation Forest output (-1, 1) to (1, 0)
                y_pred_binary = (y_pred == -1).astype(int)
                y_score = pipeline.decision_function(X_test)
                # Invert scores for consistency (higher = more fraudulent)
                y_score = -y_score
            else:
                # For supervised models
                pipeline.fit(X_train, y_train)
                y_pred_binary = pipeline.predict(X_test)
                y_score = pipeline.predict_proba(X_test)[:, 1]
            
            # Evaluate model
            auc_score = roc_auc_score(y_test, y_score)
            classification_rep = classification_report(y_test, y_pred_binary, output_dict=True)
            
            # Store model and results
            self.models[model_name] = pipeline
            results[model_name] = {
                'auc_score': auc_score,
                'classification_report': classification_rep,
                'test_predictions': y_pred_binary,
                'test_scores': y_score
            }
            
            logger.info(f"{model_name} - AUC: {auc_score:.3f}")
        
        # Save models
        self.save_models()
        
        # Create ensemble
        self._create_ensemble_model(X_test, y_test, results)
        
        return results
    
    def _create_ensemble_model(self, X_test: pd.DataFrame, y_test: pd.Series, 
                              individual_results: Dict) -> None:
        """Create an ensemble model from individual models."""
        logger.info("Creating ensemble model...")
        
        # Collect predictions from all models
        predictions = []
        weights = []
        
        for model_name, results in individual_results.items():
            predictions.append(results['test_scores'])
            weights.append(results['auc_score'])  # Weight by AUC score
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Create weighted ensemble predictions
        ensemble_scores = np.average(predictions, axis=0, weights=weights)
        ensemble_predictions = (ensemble_scores > 0.5).astype(int)
        
        # Evaluate ensemble
        ensemble_auc = roc_auc_score(y_test, ensemble_scores)
        ensemble_report = classification_report(y_test, ensemble_predictions, output_dict=True)
        
        # Store ensemble info
        self.models['ensemble'] = {
            'weights': weights,
            'model_names': list(individual_results.keys())
        }
        
        logger.info(f"Ensemble model - AUC: {ensemble_auc:.3f}")
        
        individual_results['ensemble'] = {
            'auc_score': ensemble_auc,
            'classification_report': ensemble_report,
            'test_predictions': ensemble_predictions,
            'test_scores': ensemble_scores
        }
    
    def predict_fraud(self, transaction_data: Union[Dict, pd.DataFrame, TransactionData]) -> Dict[str, Any]:
        """
        Predict fraud for a single transaction or batch.
        
        Args:
            transaction_data: Transaction data to analyze
            
        Returns:
            Dictionary with fraud prediction results
        """
        start_time = time.time()
        
        # Convert input to DataFrame
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        elif isinstance(transaction_data, TransactionData):
            df = pd.DataFrame([transaction_data.__dict__])
        else:
            df = transaction_data.copy()
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Get predictions from all models
        predictions = {}
        scores = {}
        
        for model_name, model in self.models.items():
            if model_name == 'ensemble':
                continue
                
            try:
                if 'isolation_forest' in model_name:
                    # Handle Isolation Forest
                    pred = model.predict(X)
                    score = -model.decision_function(X)  # Invert for consistency
                    pred_binary = (pred == -1).astype(int)
                else:
                    # Handle supervised models
                    pred_binary = model.predict(X)
                    score = model.predict_proba(X)[:, 1]
                
                predictions[model_name] = pred_binary[0] if len(pred_binary) == 1 else pred_binary
                scores[model_name] = score[0] if len(score) == 1 else score
                
            except Exception as e:
                logger.error(f"Prediction failed for {model_name}: {e}")
                predictions[model_name] = 0
                scores[model_name] = 0.0
        
        # Calculate ensemble prediction
        if 'ensemble' in self.models:
            ensemble_info = self.models['ensemble']
            model_scores = [scores[name] for name in ensemble_info['model_names']]
            ensemble_score = np.average(model_scores, weights=ensemble_info['weights'])
            ensemble_prediction = 1 if ensemble_score > 0.5 else 0
            
            predictions['ensemble'] = ensemble_prediction
            scores['ensemble'] = ensemble_score
        
        # Processing time
        processing_time = (time.time() - start_time) * 1000
        self.stats['processing_time_ms'].append(processing_time)
        
        # Determine final prediction (use ensemble if available, otherwise best individual model)
        if 'ensemble' in predictions:
            final_prediction = predictions['ensemble']
            final_score = scores['ensemble']
            method_used = 'ensemble'
        else:
            # Use the model with highest individual score
            best_model = max(scores.keys(), key=lambda k: scores[k])
            final_prediction = predictions[best_model]
            final_score = scores[best_model]
            method_used = best_model
        
        result = {
            'transaction_id': df['transaction_id'].iloc[0] if 'transaction_id' in df.columns else 'unknown',
            'is_fraud': bool(final_prediction),
            'fraud_score': float(final_score),
            'confidence': float(abs(final_score - 0.5) * 2),  # Distance from decision boundary
            'method_used': method_used,
            'all_predictions': predictions,
            'all_scores': scores,
            'processing_time_ms': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update statistics
        self.stats['total_transactions'] += 1
        if final_prediction:
            self.stats['fraud_detected'] += 1
        
        return result
    
    def start_real_time_monitoring(self, alert_threshold: float = 0.7) -> None:
        """
        Start real-time fraud monitoring in a separate thread.
        
        Args:
            alert_threshold: Fraud score threshold for alerts
        """
        def monitoring_loop():
            logger.info("tihassfjord: Starting real-time fraud monitoring...")
            self.is_monitoring = True
            
            while self.is_monitoring:
                try:
                    # Check if there are transactions to process
                    if self.transaction_buffer:
                        transaction = self.transaction_buffer.popleft()
                        result = self.predict_fraud(transaction)
                        
                        # Generate alert if needed
                        if result['fraud_score'] >= alert_threshold:
                            alert = {
                                'timestamp': datetime.now().isoformat(),
                                'transaction_id': result['transaction_id'],
                                'fraud_score': result['fraud_score'],
                                'confidence': result['confidence'],
                                'alert_level': 'HIGH' if result['fraud_score'] > 0.9 else 'MEDIUM'
                            }
                            self.alert_queue.put(alert)
                            logger.warning(f"FRAUD ALERT: Transaction {result['transaction_id']} - Score: {result['fraud_score']:.3f}")
                    
                    time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(1)
        
        # Start monitoring in background thread
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
    
    def stop_real_time_monitoring(self) -> None:
        """Stop real-time fraud monitoring."""
        self.is_monitoring = False
        logger.info("Real-time fraud monitoring stopped")
    
    def add_transaction(self, transaction: Union[Dict, TransactionData]) -> None:
        """Add a transaction to the processing buffer."""
        self.transaction_buffer.append(transaction)
    
    def get_alerts(self) -> List[Dict]:
        """Get all pending fraud alerts."""
        alerts = []
        while not self.alert_queue.empty():
            try:
                alerts.append(self.alert_queue.get_nowait())
            except queue.Empty:
                break
        return alerts
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        avg_processing_time = np.mean(self.stats['processing_time_ms']) if self.stats['processing_time_ms'] else 0
        
        stats = {
            'total_transactions_processed': self.stats['total_transactions'],
            'fraud_detected': self.stats['fraud_detected'],
            'fraud_rate': self.stats['fraud_detected'] / max(1, self.stats['total_transactions']),
            'average_processing_time_ms': avg_processing_time,
            'max_processing_time_ms': max(self.stats['processing_time_ms']) if self.stats['processing_time_ms'] else 0,
            'buffer_size': len(self.transaction_buffer),
            'pending_alerts': self.alert_queue.qsize(),
            'monitoring_active': self.is_monitoring
        }
        
        return stats
    
    def save_models(self) -> None:
        """Save trained models to disk."""
        for model_name, model in self.models.items():
            if model_name != 'ensemble':
                model_path = self.model_save_path / f"{model_name}_model.pkl"
                joblib.dump(model, model_path)
        
        # Save ensemble info and feature columns
        metadata = {
            'feature_columns': self.feature_columns,
            'ensemble_info': self.models.get('ensemble', {}),
            'model_names': list(self.models.keys())
        }
        
        metadata_path = self.model_save_path / "model_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Models saved to {self.model_save_path}")
    
    def load_models(self) -> None:
        """Load trained models from disk."""
        try:
            # Load metadata
            metadata_path = self.model_save_path / "model_metadata.pkl"
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.feature_columns = metadata['feature_columns']
            
            # Load individual models
            for model_name in metadata['model_names']:
                if model_name != 'ensemble':
                    model_path = self.model_save_path / f"{model_name}_model.pkl"
                    if model_path.exists():
                        self.models[model_name] = joblib.load(model_path)
            
            # Load ensemble info
            if metadata['ensemble_info']:
                self.models['ensemble'] = metadata['ensemble_info']
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def visualize_results(self, results: Dict, save_path: str = "fraud_detection_results.png") -> None:
        """
        Create visualization of fraud detection results.
        
        Args:
            results: Results from model training
            save_path: Path to save the visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Fraud Detection System Results - tihassfjord', fontsize=16, fontweight='bold')
        
        # Model performance comparison
        ax1 = axes[0, 0]
        model_names = list(results.keys())
        auc_scores = [results[name]['auc_score'] for name in model_names]
        
        bars = ax1.bar(model_names, auc_scores, color=['skyblue', 'lightgreen', 'coral', 'gold'][:len(model_names)])
        ax1.set_title('Model Performance (AUC Score)')
        ax1.set_ylabel('AUC Score')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, auc_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Processing time distribution
        ax2 = axes[0, 1]
        if self.stats['processing_time_ms']:
            ax2.hist(self.stats['processing_time_ms'], bins=30, alpha=0.7, color='lightblue')
            ax2.set_title('Processing Time Distribution')
            ax2.set_xlabel('Processing Time (ms)')
            ax2.set_ylabel('Frequency')
        
        # Statistics summary
        ax3 = axes[1, 0]
        stats = self.get_statistics()
        stats_labels = ['Total Transactions', 'Fraud Detected', 'Avg Process Time (ms)']
        stats_values = [
            stats['total_transactions_processed'],
            stats['fraud_detected'],
            stats['average_processing_time_ms']
        ]
        
        bars = ax3.bar(stats_labels, stats_values, color=['lightcoral', 'lightyellow', 'lightgreen'])
        ax3.set_title('System Statistics')
        
        # Add value labels
        for bar, value in zip(bars, stats_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stats_values) * 0.01,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # Fraud score distribution (if available)
        ax4 = axes[1, 1]
        if 'ensemble' in results:
            scores = results['ensemble']['test_scores']
            ax4.hist(scores, bins=30, alpha=0.7, color='orange', label='All Transactions')
            ax4.set_title('Fraud Score Distribution')
            ax4.set_xlabel('Fraud Score')
            ax4.set_ylabel('Frequency')
            ax4.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Results visualization saved to {save_path}")


def simulate_real_time_transactions(fraud_engine: FraudDetectionEngine, 
                                   n_transactions: int = 100,
                                   fraud_rate: float = 0.05) -> None:
    """
    Simulate real-time transaction stream for testing.
    
    Args:
        fraud_engine: Initialized fraud detection engine
        n_transactions: Number of transactions to simulate
        fraud_rate: Rate of fraudulent transactions
    """
    logger.info(f"tihassfjord: Simulating {n_transactions} real-time transactions")
    
    # Generate test transactions
    test_data = fraud_engine.generate_synthetic_data(n_transactions, fraud_rate)
    
    # Start monitoring
    fraud_engine.start_real_time_monitoring(alert_threshold=0.6)
    
    # Stream transactions
    for _, row in test_data.iterrows():
        transaction = TransactionData(
            transaction_id=row['transaction_id'],
            user_id=str(row['user_id']),
            amount=row['amount'],
            merchant_category=row['merchant_category'],
            transaction_time=row['transaction_time'],
            location=row['location'],
            device_type=row['device_type'],
            is_weekend=row['is_weekend'],
            hour_of_day=row['hour_of_day'],
            days_since_last_transaction=row['days_since_last_transaction'],
            amount_zscore=row['amount_zscore'],
            is_fraud=row['is_fraud']
        )
        
        fraud_engine.add_transaction(transaction)
        time.sleep(0.01)  # Small delay to simulate real-time stream
    
    # Wait for processing to complete
    time.sleep(2)
    
    # Get and display alerts
    alerts = fraud_engine.get_alerts()
    logger.info(f"Generated {len(alerts)} fraud alerts")
    
    for alert in alerts[-5:]:  # Show last 5 alerts
        logger.info(f"ALERT: {alert['transaction_id']} - Score: {alert['fraud_score']:.3f} - Level: {alert['alert_level']}")
    
    # Stop monitoring
    fraud_engine.stop_real_time_monitoring()


def demo_fraud_detection():
    """Comprehensive demonstration of the fraud detection system."""
    print("\n" + "="*70)
    print("🚨 REAL-TIME FRAUD DETECTION SYSTEM DEMO by tihassfjord")
    print("="*70)
    
    # Initialize system
    fraud_engine = FraudDetectionEngine()
    
    # Generate training data
    print("\n📊 Generating synthetic transaction data...")
    training_data = fraud_engine.generate_synthetic_data(n_samples=50000, fraud_rate=0.02)
    print(f"Generated {len(training_data)} transactions for training")
    
    # Train models
    print("\n🤖 Training fraud detection models...")
    results = fraud_engine.train_models(training_data)
    
    # Display training results
    print("\n📈 Model Performance Results:")
    print("-" * 50)
    for model_name, result in results.items():
        auc = result['auc_score']
        precision = result['classification_report']['1']['precision']
        recall = result['classification_report']['1']['recall']
        print(f"{model_name:20} | AUC: {auc:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}")
    
    # Test single transaction prediction
    print("\n🔍 Testing single transaction prediction...")
    test_transaction = {
        'transaction_id': 'TEST_001',
        'user_id': '5000',
        'amount': 2500.0,  # Unusually high amount
        'merchant_category': 'online',
        'transaction_time': datetime.now(),
        'location': 'international',  # International transaction
        'device_type': 'mobile',
        'is_weekend': False,
        'hour_of_day': 3,  # Unusual hour
        'days_since_last_transaction': 0.1,  # Very recent
        'amount_zscore': 3.5  # High z-score
    }
    
    prediction = fraud_engine.predict_fraud(test_transaction)
    print(f"Transaction ID: {prediction['transaction_id']}")
    print(f"Fraud Prediction: {'🚨 FRAUD' if prediction['is_fraud'] else '✅ LEGITIMATE'}")
    print(f"Fraud Score: {prediction['fraud_score']:.3f}")
    print(f"Confidence: {prediction['confidence']:.3f}")
    print(f"Processing Time: {prediction['processing_time_ms']:.1f}ms")
    
    # Real-time monitoring demo
    print("\n🔄 Real-time monitoring demonstration...")
    simulate_real_time_transactions(fraud_engine, n_transactions=500, fraud_rate=0.08)
    
    # Display system statistics
    print("\n📊 System Statistics:")
    stats = fraud_engine.get_statistics()
    print("-" * 50)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:30}: {value:.3f}")
        else:
            print(f"{key:30}: {value}")
    
    # Create visualizations
    print("\n📈 Creating performance visualizations...")
    fraud_engine.visualize_results(results)
    
    print("\n🎉 Fraud Detection System Demo Complete!")
    print("Author: tihassfjord | Advanced ML Portfolio")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Run demonstration
    try:
        demo_fraud_detection()
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
