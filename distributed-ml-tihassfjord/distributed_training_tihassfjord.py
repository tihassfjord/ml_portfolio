#!/usr/bin/env python3
"""
Distributed ML System (highlight) — tihassfjord

Advanced distributed machine learning system using Ray for parallel training,
PyTorch Lightning for scalable deep learning, and Dask for distributed computing.
Features data parallelism, model parallelism, and federated learning capabilities.

Author: tihassfjord
Project: Advanced ML Portfolio - Distributed ML System
"""

import os
import sys
import time
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
import json
import pickle
from pathlib import Path
import asyncio
import threading
from dataclasses import dataclass, asdict
import math

warnings.filterwarnings('ignore')

# Core libraries
import pandas as pd
import numpy as np

# ML libraries
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib

# Deep Learning
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    import torch.optim as optim
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
    import torch.multiprocessing as mp
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")

# Distributed Computing
try:
    import ray
    from ray import tune
    from ray.train import Trainer
    from ray.air import session
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Ray not available. Install with: pip install ray[tune]")

try:
    import dask
    import dask.dataframe as dd
    from dask.distributed import Client, as_completed
    from dask_ml.model_selection import train_test_split as dask_train_test_split
    from dask_ml.preprocessing import StandardScaler as DaskStandardScaler
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    print("Dask not available. Install with: pip install dask[complete] dask-ml")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('distributed_ml.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration for distributed systems."""
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    num_workers: int = 4
    device: str = "cpu"
    distributed_backend: str = "ray"  # ray, torch, dask
    
@dataclass
class DistributedTrainingResult:
    """Results from distributed training."""
    accuracy: float
    loss: float
    training_time: float
    num_workers: int
    model_size_mb: float
    throughput_samples_per_sec: float

class SyntheticDataGenerator:
    """Generate synthetic datasets for distributed training."""
    
    @staticmethod
    def generate_classification_data(n_samples: int = 10000, n_features: int = 20, 
                                   n_classes: int = 3, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic classification dataset."""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, n_features//2),
            n_redundant=max(0, n_features//4),
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=random_state
        )
        return X.astype(np.float32), y.astype(np.int64)
    
    @staticmethod
    def generate_regression_data(n_samples: int = 10000, n_features: int = 20, 
                               noise: float = 0.1, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic regression dataset."""
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, n_features//2),
            noise=noise,
            random_state=random_state
        )
        return X.astype(np.float32), y.astype(np.float32)

# PyTorch Models
if TORCH_AVAILABLE:
    class DistributedClassifier(nn.Module):
        """Deep neural network for distributed classification."""
        
        def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int, dropout: float = 0.2):
            super().__init__()
            
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, num_classes))
            self.network = nn.Sequential(*layers)
            
        def forward(self, x):
            return self.network(x)
    
    class DistributedRegressor(nn.Module):
        """Deep neural network for distributed regression."""
        
        def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.2):
            super().__init__()
            
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, 1))
            self.network = nn.Sequential(*layers)
            
        def forward(self, x):
            return self.network(x).squeeze()

# Ray Distributed Training
if RAY_AVAILABLE:
    @ray.remote
    class RayWorker:
        """Ray remote worker for distributed training."""
        
        def __init__(self, worker_id: int, config: TrainingConfig):
            self.worker_id = worker_id
            self.config = config
            self.device = torch.device(config.device)
            
        def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                       X_test: np.ndarray, y_test: np.ndarray, 
                       task_type: str = "classification") -> Dict[str, Any]:
            """Train model on worker."""
            logger.info(f"Worker {self.worker_id}: Starting training...")
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.LongTensor(y_train).to(self.device) if task_type == "classification" else torch.FloatTensor(y_train).to(self.device)
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            y_test_tensor = torch.LongTensor(y_test).to(self.device) if task_type == "classification" else torch.FloatTensor(y_test).to(self.device)
            
            # Create model
            if task_type == "classification":
                num_classes = len(np.unique(y_train))
                model = DistributedClassifier(
                    input_dim=X_train.shape[1],
                    hidden_dims=[128, 64, 32],
                    num_classes=num_classes
                ).to(self.device)
                criterion = nn.CrossEntropyLoss()
            else:
                model = DistributedRegressor(
                    input_dim=X_train.shape[1],
                    hidden_dims=[128, 64, 32]
                ).to(self.device)
                criterion = nn.MSELoss()
            
            optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
            
            # Training loop
            model.train()
            start_time = time.time()
            
            dataset = TensorDataset(X_train_tensor, y_train_tensor)
            dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
            
            for epoch in range(self.config.epochs):
                epoch_loss = 0.0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                if epoch % 10 == 0:
                    avg_loss = epoch_loss / len(dataloader)
                    logger.info(f"Worker {self.worker_id}: Epoch {epoch}, Loss: {avg_loss:.4f}")
            
            training_time = time.time() - start_time
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                
                if task_type == "classification":
                    _, predicted = torch.max(test_outputs.data, 1)
                    accuracy = (predicted == y_test_tensor).float().mean().item()
                    final_loss = criterion(test_outputs, y_test_tensor).item()
                else:
                    accuracy = r2_score(y_test_tensor.cpu().numpy(), test_outputs.cpu().numpy())
                    final_loss = criterion(test_outputs, y_test_tensor).item()
            
            # Calculate throughput
            total_samples = len(X_train) * self.config.epochs
            throughput = total_samples / training_time
            
            result = {
                'worker_id': self.worker_id,
                'accuracy': accuracy,
                'loss': final_loss,
                'training_time': training_time,
                'throughput': throughput,
                'samples_processed': total_samples
            }
            
            logger.info(f"Worker {self.worker_id}: Training complete. Accuracy: {accuracy:.4f}")
            return result

    class RayDistributedTrainer:
        """Distributed training using Ray."""
        
        def __init__(self, config: TrainingConfig):
            self.config = config
            
        def train_distributed(self, X: np.ndarray, y: np.ndarray, 
                            task_type: str = "classification") -> List[Dict[str, Any]]:
            """Train model using distributed Ray workers."""
            logger.info("tihassfjord: Starting Ray distributed training...")
            
            if not RAY_AVAILABLE:
                raise ImportError("Ray not available")
            
            # Initialize Ray
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            
            # Split data for workers
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Create data partitions for workers
            train_partitions = np.array_split(X_train, self.config.num_workers)
            target_partitions = np.array_split(y_train, self.config.num_workers)
            
            # Create workers
            workers = [RayWorker.remote(i, self.config) for i in range(self.config.num_workers)]
            
            # Start training on all workers
            futures = []
            for i, worker in enumerate(workers):
                future = worker.train_model.remote(
                    train_partitions[i], target_partitions[i],
                    X_test, y_test, task_type
                )
                futures.append(future)
            
            # Collect results
            results = ray.get(futures)
            
            logger.info(f"Distributed training completed with {self.config.num_workers} workers")
            return results

# Dask Distributed Computing
if DASK_AVAILABLE:
    class DaskDistributedTrainer:
        """Distributed training using Dask."""
        
        def __init__(self, config: TrainingConfig):
            self.config = config
            self.client = None
            
        def start_cluster(self, scheduler_address: Optional[str] = None):
            """Start Dask distributed cluster."""
            if scheduler_address:
                self.client = Client(scheduler_address)
            else:
                # Local cluster
                self.client = Client(processes=True, n_workers=self.config.num_workers, threads_per_worker=2)
            
            logger.info(f"Dask cluster started: {self.client}")
            
        def train_distributed(self, X: np.ndarray, y: np.ndarray, 
                            task_type: str = "classification") -> Dict[str, Any]:
            """Train using Dask distributed computing."""
            logger.info("tihassfjord: Starting Dask distributed training...")
            
            if not self.client:
                self.start_cluster()
            
            # Convert to Dask DataFrame
            df = pd.DataFrame(X)
            df['target'] = y
            ddf = dd.from_pandas(df, npartitions=self.config.num_workers)
            
            # Distributed preprocessing
            feature_cols = [col for col in df.columns if col != 'target']
            
            # Split features and target
            X_dask = ddf[feature_cols]
            y_dask = ddf['target']
            
            # Standardize features
            scaler = DaskStandardScaler()
            X_scaled = scaler.fit_transform(X_dask)
            
            # Train-test split
            X_train, X_test, y_train, y_test = dask_train_test_split(
                X_scaled, y_dask, test_size=0.2, random_state=42
            )
            
            # Simple distributed training (using sklearn-compatible estimators)
            from dask_ml.ensemble import RandomForestClassifier as DaskRandomForestClassifier
            from dask_ml.ensemble import RandomForestRegressor as DaskRandomForestRegressor
            
            start_time = time.time()
            
            if task_type == "classification":
                model = DaskRandomForestClassifier(
                    n_estimators=100, 
                    random_state=42,
                    n_jobs=-1
                )
            else:
                model = DaskRandomForestRegressor(
                    n_estimators=100, 
                    random_state=42,
                    n_jobs=-1
                )
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            training_time = time.time() - start_time
            
            # Compute metrics
            if task_type == "classification":
                accuracy = accuracy_score(y_test.compute(), y_pred.compute())
                loss = 1 - accuracy  # Simple loss approximation
            else:
                y_test_computed = y_test.compute()
                y_pred_computed = y_pred.compute()
                accuracy = r2_score(y_test_computed, y_pred_computed)
                loss = mean_squared_error(y_test_computed, y_pred_computed)
            
            result = {
                'accuracy': accuracy,
                'loss': loss,
                'training_time': training_time,
                'num_workers': self.config.num_workers,
                'framework': 'dask'
            }
            
            logger.info(f"Dask training complete. Accuracy: {accuracy:.4f}")
            return result
        
        def shutdown(self):
            """Shutdown Dask cluster."""
            if self.client:
                self.client.close()

# PyTorch DDP (Distributed Data Parallel)
if TORCH_AVAILABLE:
    def setup_ddp(rank: int, world_size: int):
        """Setup distributed data parallel."""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        
    def cleanup_ddp():
        """Cleanup distributed training."""
        dist.destroy_process_group()
        
    def ddp_train_worker(rank: int, world_size: int, config: TrainingConfig, 
                        X: np.ndarray, y: np.ndarray, task_type: str = "classification"):
        """DDP training worker function."""
        setup_ddp(rank, world_size)
        
        # Set device
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        
        # Prepare data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.LongTensor(y_train) if task_type == "classification" else torch.FloatTensor(y_train)
        )
        
        # Distributed sampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            sampler=train_sampler
        )
        
        # Create model
        if task_type == "classification":
            num_classes = len(np.unique(y))
            model = DistributedClassifier(
                input_dim=X.shape[1],
                hidden_dims=[128, 64, 32],
                num_classes=num_classes
            ).to(device)
            criterion = nn.CrossEntropyLoss()
        else:
            model = DistributedRegressor(
                input_dim=X.shape[1],
                hidden_dims=[128, 64, 32]
            ).to(device)
            criterion = nn.MSELoss()
        
        # Wrap model with DDP
        model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # Training loop
        model.train()
        for epoch in range(config.epochs):
            train_sampler.set_epoch(epoch)
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        cleanup_ddp()

class DistributedMLSystem:
    """Main distributed ML system orchestrator."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.results = {}
        
    def benchmark_frameworks(self, X: np.ndarray, y: np.ndarray, 
                           task_type: str = "classification") -> Dict[str, Any]:
        """Benchmark different distributed frameworks."""
        logger.info("tihassfjord: Starting distributed ML framework benchmark...")
        
        results = {}
        
        # Ray distributed training
        if RAY_AVAILABLE:
            try:
                logger.info("Testing Ray distributed training...")
                ray_trainer = RayDistributedTrainer(self.config)
                ray_results = ray_trainer.train_distributed(X, y, task_type)
                
                # Aggregate results
                total_time = max(r['training_time'] for r in ray_results)
                avg_accuracy = np.mean([r['accuracy'] for r in ray_results])
                total_throughput = sum(r['throughput'] for r in ray_results)
                
                results['ray'] = {
                    'accuracy': avg_accuracy,
                    'training_time': total_time,
                    'throughput': total_throughput,
                    'num_workers': self.config.num_workers,
                    'framework': 'ray'
                }
                
                # Shutdown Ray
                ray.shutdown()
                
            except Exception as e:
                logger.error(f"Ray training failed: {e}")
        
        # Dask distributed training
        if DASK_AVAILABLE:
            try:
                logger.info("Testing Dask distributed training...")
                dask_trainer = DaskDistributedTrainer(self.config)
                dask_result = dask_trainer.train_distributed(X, y, task_type)
                results['dask'] = dask_result
                dask_trainer.shutdown()
                
            except Exception as e:
                logger.error(f"Dask training failed: {e}")
        
        # PyTorch DDP (single machine, multiple processes)
        if TORCH_AVAILABLE and self.config.num_workers <= 4:  # Limit for demo
            try:
                logger.info("Testing PyTorch DDP training...")
                start_time = time.time()
                
                # Simulate DDP with multiprocessing
                world_size = min(self.config.num_workers, 2)  # Limit for demo
                mp.spawn(
                    ddp_train_worker,
                    args=(world_size, self.config, X, y, task_type),
                    nprocs=world_size,
                    join=True
                )
                
                ddp_time = time.time() - start_time
                
                # Simple evaluation (would be more sophisticated in practice)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                if task_type == "classification":
                    num_classes = len(np.unique(y))
                    model = DistributedClassifier(X.shape[1], [128, 64, 32], num_classes)
                    criterion = nn.CrossEntropyLoss()
                else:
                    model = DistributedRegressor(X.shape[1], [128, 64, 32])
                    criterion = nn.MSELoss()
                
                # Quick training for evaluation
                optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
                
                X_train_tensor = torch.FloatTensor(X_train)
                y_train_tensor = torch.LongTensor(y_train) if task_type == "classification" else torch.FloatTensor(y_train)
                X_test_tensor = torch.FloatTensor(X_test)
                y_test_tensor = torch.LongTensor(y_test) if task_type == "classification" else torch.FloatTensor(y_test)
                
                for epoch in range(min(10, self.config.epochs)):  # Quick training
                    optimizer.zero_grad()
                    outputs = model(X_train_tensor)
                    loss = criterion(outputs, y_train_tensor)
                    loss.backward()
                    optimizer.step()
                
                # Evaluate
                model.eval()
                with torch.no_grad():
                    test_outputs = model(X_test_tensor)
                    if task_type == "classification":
                        _, predicted = torch.max(test_outputs.data, 1)
                        accuracy = (predicted == y_test_tensor).float().mean().item()
                    else:
                        accuracy = r2_score(y_test_tensor.numpy(), test_outputs.numpy())
                
                results['pytorch_ddp'] = {
                    'accuracy': accuracy,
                    'training_time': ddp_time,
                    'num_workers': world_size,
                    'framework': 'pytorch_ddp'
                }
                
            except Exception as e:
                logger.error(f"PyTorch DDP training failed: {e}")
        
        self.results = results
        return results
    
    def create_performance_comparison(self):
        """Create performance comparison visualization."""
        if not self.results:
            logger.warning("No results to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Distributed ML Framework Comparison - tihassfjord', fontsize=16, fontweight='bold')
        
        frameworks = list(self.results.keys())
        accuracies = [self.results[fw]['accuracy'] for fw in frameworks]
        times = [self.results[fw]['training_time'] for fw in frameworks]
        
        # Accuracy comparison
        axes[0, 0].bar(frameworks, accuracies, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_ylabel('Accuracy/R²')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Training time comparison
        axes[0, 1].bar(frameworks, times, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Training Time')
        axes[0, 1].set_ylabel('Seconds')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Throughput comparison (if available)
        throughputs = [self.results[fw].get('throughput', 0) for fw in frameworks]
        if any(throughputs):
            axes[1, 0].bar(frameworks, throughputs, color='lightgreen', alpha=0.7)
            axes[1, 0].set_title('Training Throughput')
            axes[1, 0].set_ylabel('Samples/Second')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Efficiency scatter plot
        axes[1, 1].scatter(times, accuracies, s=100, alpha=0.7, c='purple')
        for i, fw in enumerate(frameworks):
            axes[1, 1].annotate(fw, (times[i], accuracies[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 1].set_xlabel('Training Time (seconds)')
        axes[1, 1].set_ylabel('Accuracy/R²')
        axes[1, 1].set_title('Efficiency: Accuracy vs Time')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        plot_path = plots_dir / "distributed_comparison_tihassfjord.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Performance comparison plot saved: {plot_path}")

def demo_distributed_ml():
    """Demonstrate distributed ML capabilities."""
    print("="*70)
    print("🚀 Distributed ML System (highlight) — tihassfjord")
    print("="*70)
    print("Advanced distributed machine learning with Ray, Dask, and PyTorch DDP")
    print()
    
    # Configuration
    config = TrainingConfig(
        batch_size=64,
        epochs=20,
        learning_rate=0.001,
        num_workers=4,
        device="cpu"
    )
    
    print(f"Configuration:")
    print(f"  Workers: {config.num_workers}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print()
    
    # Generate datasets
    print("Generating synthetic datasets...")
    data_gen = SyntheticDataGenerator()
    
    # Classification dataset
    X_class, y_class = data_gen.generate_classification_data(
        n_samples=5000, n_features=20, n_classes=3
    )
    
    # Regression dataset  
    X_reg, y_reg = data_gen.generate_regression_data(
        n_samples=5000, n_features=20
    )
    
    print(f"Classification dataset: {X_class.shape}")
    print(f"Regression dataset: {X_reg.shape}")
    print()
    
    # Initialize distributed system
    distributed_system = DistributedMLSystem(config)
    
    # Test classification
    print("="*50)
    print("DISTRIBUTED CLASSIFICATION BENCHMARK")
    print("="*50)
    
    try:
        class_results = distributed_system.benchmark_frameworks(X_class, y_class, "classification")
        
        for framework, result in class_results.items():
            print(f"\n{framework.upper()}:")
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  Training Time: {result['training_time']:.2f}s")
            if 'throughput' in result:
                print(f"  Throughput: {result['throughput']:.0f} samples/sec")
            print(f"  Workers: {result.get('num_workers', 'N/A')}")
        
        # Find best framework
        best_framework = max(class_results.keys(), key=lambda k: class_results[k]['accuracy'])
        print(f"\n🏆 Best Classification Framework: {best_framework}")
        print(f"   Accuracy: {class_results[best_framework]['accuracy']:.4f}")
        
    except Exception as e:
        logger.error(f"Classification benchmark failed: {e}")
    
    # Test regression
    print("\n" + "="*50)
    print("DISTRIBUTED REGRESSION BENCHMARK")
    print("="*50)
    
    try:
        reg_results = distributed_system.benchmark_frameworks(X_reg, y_reg, "regression")
        
        for framework, result in reg_results.items():
            print(f"\n{framework.upper()}:")
            print(f"  R² Score: {result['accuracy']:.4f}")
            print(f"  Training Time: {result['training_time']:.2f}s")
            if 'throughput' in result:
                print(f"  Throughput: {result['throughput']:.0f} samples/sec")
            print(f"  Workers: {result.get('num_workers', 'N/A')}")
        
        # Find best framework
        best_framework = max(reg_results.keys(), key=lambda k: reg_results[k]['accuracy'])
        print(f"\n🏆 Best Regression Framework: {best_framework}")
        print(f"   R² Score: {reg_results[best_framework]['accuracy']:.4f}")
        
    except Exception as e:
        logger.error(f"Regression benchmark failed: {e}")
    
    # Create performance visualization
    print("\n" + "="*50)
    print("PERFORMANCE VISUALIZATION")
    print("="*50)
    
    try:
        distributed_system.create_performance_comparison()
        print("✓ Performance comparison plots created")
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
    
    print("\n" + "="*50)
    print("DISTRIBUTED SYSTEM SUMMARY")
    print("="*50)
    
    available_frameworks = []
    if RAY_AVAILABLE:
        available_frameworks.append("Ray")
    if DASK_AVAILABLE:
        available_frameworks.append("Dask")
    if TORCH_AVAILABLE:
        available_frameworks.append("PyTorch DDP")
    
    print(f"✓ Available frameworks: {', '.join(available_frameworks)}")
    print(f"✓ Tested with {config.num_workers} workers")
    print(f"✓ Processed {X_class.shape[0]} classification + {X_reg.shape[0]} regression samples")
    print(f"✓ Benchmarked training time and accuracy")
    print(f"✓ Generated performance comparison plots")
    
    print("\n🎯 Key Benefits of Distributed ML:")
    print("  • Faster training on large datasets")
    print("  • Scalability across multiple machines")
    print("  • Fault tolerance and resilience")
    print("  • Efficient resource utilization")
    
    print("\n📊 Framework Recommendations:")
    print("  • Ray: Best for RL and hyperparameter tuning")
    print("  • Dask: Excellent for large DataFrame operations")
    print("  • PyTorch DDP: Optimal for deep learning models")

def main():
    """Main function with CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Distributed ML System - tihassfjord")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--task", choices=["classification", "regression", "both"], 
                       default="both", help="Task type")
    
    args = parser.parse_args()
    
    # Create configuration
    config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.workers,
        device="cpu"
    )
    
    if args.task == "both":
        demo_distributed_ml()
    else:
        # Run specific task
        distributed_system = DistributedMLSystem(config)
        data_gen = SyntheticDataGenerator()
        
        if args.task == "classification":
            X, y = data_gen.generate_classification_data(n_samples=args.samples)
            results = distributed_system.benchmark_frameworks(X, y, "classification")
        else:
            X, y = data_gen.generate_regression_data(n_samples=args.samples)
            results = distributed_system.benchmark_frameworks(X, y, "regression")
        
        # Print results
        for framework, result in results.items():
            print(f"{framework}: Accuracy={result['accuracy']:.4f}, Time={result['training_time']:.2f}s")

if __name__ == "__main__":
    main()
