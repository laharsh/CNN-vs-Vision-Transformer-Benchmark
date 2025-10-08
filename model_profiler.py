"""
Advanced Model Profiling System for CNN vs Vision Transformer Comparison

This module provides comprehensive profiling capabilities including:
1. Training speed comparison (epochs, batches, forward/backward passes)
2. Memory usage analysis (CPU, GPU, peak memory)
3. Bottleneck identification (data loading, computation, I/O)
4. Performance metrics (throughput, latency, efficiency)
5. Hardware utilization monitoring
6. Detailed timing breakdowns
"""

import torch
import torch.nn as nn
import time
import psutil
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import threading
import queue
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ProfilingMetrics:
    """Data class for storing profiling metrics"""
    model_name: str
    epoch: int
    batch_idx: int
    forward_time: float
    backward_time: float
    optimizer_time: float
    data_load_time: float
    total_batch_time: float
    cpu_memory_percent: float
    gpu_memory_mb: float
    gpu_memory_allocated_mb: float
    gpu_memory_reserved_mb: float
    gpu_utilization: float
    cpu_utilization: float
    batch_size: int
    sequence_length: Optional[int] = None
    patch_count: Optional[int] = None

class MemoryMonitor:
    """Real-time memory monitoring"""
    
    def __init__(self, interval=0.1):
        self.interval = interval
        self.monitoring = False
        self.memory_data = deque(maxlen=10000)
        self.thread = None
        
    def start_monitoring(self):
        """Start memory monitoring in background thread"""
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        if self.thread:
            self.thread.join()
            
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # CPU memory
                cpu_memory = psutil.virtual_memory().percent
                
                # GPU memory
                gpu_memory = 0
                gpu_allocated = 0
                gpu_reserved = 0
                
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                    gpu_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                    gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
                
                self.memory_data.append({
                    'timestamp': time.time(),
                    'cpu_memory': cpu_memory,
                    'gpu_memory': gpu_memory,
                    'gpu_allocated': gpu_allocated,
                    'gpu_reserved': gpu_reserved
                })
                
                time.sleep(self.interval)
            except Exception as e:
                print(f"Memory monitoring error: {e}")
                break

class ModelProfiler:
    """Comprehensive model profiling system"""
    
    def __init__(self, model_name: str, device: str = 'cpu'):
        self.model_name = model_name
        self.device = device
        self.metrics: List[ProfilingMetrics] = []
        self.memory_monitor = MemoryMonitor()
        self.timing_data = defaultdict(list)
        self.memory_data = defaultdict(list)
        self.performance_data = defaultdict(list)
        
        # Profiling state
        self.current_epoch = 0
        self.current_batch = 0
        self.start_times = {}
        
        # Performance tracking
        self.total_samples_processed = 0
        self.total_training_time = 0
        self.peak_memory_usage = 0
        
    def start_profiling(self):
        """Start comprehensive profiling"""
        self.memory_monitor.start_monitoring()
        self.total_training_time = time.time()
        
    def stop_profiling(self):
        """Stop profiling and collect final metrics"""
        self.memory_monitor.stop_monitoring()
        self.total_training_time = time.time() - self.total_training_time
        
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()
        
    def end_timer(self, operation: str) -> float:
        """End timing and return duration"""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.timing_data[operation].append(duration)
            return duration
        return 0.0
    
    def record_batch_metrics(self, batch_size: int, forward_time: float, 
                           backward_time: float, optimizer_time: float, 
                           data_load_time: float, sequence_length: int = None,
                           patch_count: int = None):
        """Record comprehensive batch metrics"""
        
        # Get current memory usage
        cpu_memory = psutil.virtual_memory().percent
        
        gpu_memory = 0
        gpu_allocated = 0
        gpu_reserved = 0
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            gpu_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
        
        # Calculate total batch time
        total_batch_time = forward_time + backward_time + optimizer_time + data_load_time
        
        # Create metrics record
        metrics = ProfilingMetrics(
            model_name=self.model_name,
            epoch=self.current_epoch,
            batch_idx=self.current_batch,
            forward_time=forward_time,
            backward_time=backward_time,
            optimizer_time=optimizer_time,
            data_load_time=data_load_time,
            total_batch_time=total_batch_time,
            cpu_memory_percent=cpu_memory,
            gpu_memory_mb=gpu_memory,
            gpu_memory_allocated_mb=gpu_allocated,
            gpu_memory_reserved_mb=gpu_reserved,
            gpu_utilization=0,  # Would need nvidia-ml-py for actual GPU utilization
            cpu_utilization=psutil.cpu_percent(),
            batch_size=batch_size,
            sequence_length=sequence_length,
            patch_count=patch_count
        )
        
        self.metrics.append(metrics)
        
        # Update performance tracking
        self.total_samples_processed += batch_size
        self.peak_memory_usage = max(self.peak_memory_usage, gpu_memory)
        
        # Store aggregated data
        self.memory_data['cpu_memory'].append(cpu_memory)
        self.memory_data['gpu_memory'].append(gpu_memory)
        self.memory_data['gpu_allocated'].append(gpu_allocated)
        self.memory_data['gpu_reserved'].append(gpu_reserved)
        
        self.performance_data['forward_time'].append(forward_time)
        self.performance_data['backward_time'].append(backward_time)
        self.performance_data['optimizer_time'].append(optimizer_time)
        self.performance_data['data_load_time'].append(data_load_time)
        self.performance_data['total_batch_time'].append(total_batch_time)
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.metrics:
            return {}
        
        summary = {
            'model_name': self.model_name,
            'total_training_time': self.total_training_time,
            'total_samples_processed': self.total_samples_processed,
            'throughput_samples_per_sec': self.total_samples_processed / self.total_training_time if self.total_training_time > 0 else 0,
            'peak_memory_usage_mb': self.peak_memory_usage,
            'total_batches': len(self.metrics),
            'total_epochs': max([m.epoch for m in self.metrics]) + 1 if self.metrics else 0
        }
        
        # Timing statistics
        timing_stats = {}
        for operation in ['forward_time', 'backward_time', 'optimizer_time', 'data_load_time', 'total_batch_time']:
            if operation in self.performance_data and self.performance_data[operation]:
                times = self.performance_data[operation]
                timing_stats[f'{operation}_mean'] = np.mean(times)
                timing_stats[f'{operation}_std'] = np.std(times)
                timing_stats[f'{operation}_min'] = np.min(times)
                timing_stats[f'{operation}_max'] = np.max(times)
                timing_stats[f'{operation}_p95'] = np.percentile(times, 95)
                timing_stats[f'{operation}_p99'] = np.percentile(times, 99)
        
        summary.update(timing_stats)
        
        # Memory statistics
        if self.memory_data['cpu_memory']:
            summary['cpu_memory_mean'] = np.mean(self.memory_data['cpu_memory'])
            summary['cpu_memory_max'] = np.max(self.memory_data['cpu_memory'])
        
        if self.memory_data['gpu_memory']:
            summary['gpu_memory_mean'] = np.mean(self.memory_data['gpu_memory'])
            summary['gpu_memory_max'] = np.max(self.memory_data['gpu_memory'])
        
        # Efficiency metrics
        if self.performance_data['total_batch_time']:
            total_compute_time = sum(self.performance_data['forward_time']) + sum(self.performance_data['backward_time'])
            total_data_time = sum(self.performance_data['data_load_time'])
            total_optimizer_time = sum(self.performance_data['optimizer_time'])
            
            summary['compute_efficiency'] = total_compute_time / (total_compute_time + total_data_time + total_optimizer_time)
            summary['data_loading_overhead'] = total_data_time / (total_compute_time + total_data_time + total_optimizer_time)
            summary['optimizer_overhead'] = total_optimizer_time / (total_compute_time + total_data_time + total_optimizer_time)
        
        return summary
    
    def plot_performance_analysis(self, save_path: Optional[str] = None):
        """Create comprehensive performance analysis plots"""
        if not self.metrics:
            print("No metrics available for plotting")
            return
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f'Performance Analysis: {self.model_name}', fontsize=16, fontweight='bold')
        
        # Timing analysis
        epochs = [m.epoch for m in self.metrics]
        forward_times = [m.forward_time for m in self.metrics]
        backward_times = [m.backward_time for m in self.metrics]
        optimizer_times = [m.optimizer_time for m in self.metrics]
        data_load_times = [m.data_load_time for m in self.metrics]
        
        # 1. Timing breakdown over epochs
        axes[0, 0].plot(epochs, forward_times, label='Forward', alpha=0.7)
        axes[0, 0].plot(epochs, backward_times, label='Backward', alpha=0.7)
        axes[0, 0].plot(epochs, optimizer_times, label='Optimizer', alpha=0.7)
        axes[0, 0].plot(epochs, data_load_times, label='Data Loading', alpha=0.7)
        axes[0, 0].set_title('Timing Breakdown Over Epochs')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Memory usage over time
        axes[0, 1].plot(epochs, [m.cpu_memory_percent for m in self.metrics], label='CPU Memory %', alpha=0.7)
        if torch.cuda.is_available():
            axes[0, 1].plot(epochs, [m.gpu_memory_mb for m in self.metrics], label='GPU Memory (MB)', alpha=0.7)
        axes[0, 1].set_title('Memory Usage Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Memory Usage')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Batch processing time distribution
        batch_times = [m.total_batch_time for m in self.metrics]
        axes[0, 2].hist(batch_times, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('Batch Processing Time Distribution')
        axes[0, 2].set_xlabel('Time (seconds)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Throughput over time
        batch_sizes = [m.batch_size for m in self.metrics]
        throughput = [bs / bt for bs, bt in zip(batch_sizes, batch_times)]
        axes[1, 0].plot(epochs, throughput, alpha=0.7)
        axes[1, 0].set_title('Throughput Over Time')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Samples/Second')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Memory efficiency
        if torch.cuda.is_available():
            memory_efficiency = [m.gpu_memory_mb / m.batch_size for m in self.metrics]
            axes[1, 1].plot(epochs, memory_efficiency, alpha=0.7)
            axes[1, 1].set_title('Memory Efficiency (MB per Sample)')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('MB per Sample')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Timing component pie chart
        total_forward = sum(forward_times)
        total_backward = sum(backward_times)
        total_optimizer = sum(optimizer_times)
        total_data = sum(data_load_times)
        
        sizes = [total_forward, total_backward, total_optimizer, total_data]
        labels = ['Forward', 'Backward', 'Optimizer', 'Data Loading']
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        axes[1, 2].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 2].set_title('Time Distribution')
        
        # 7. Performance trends
        window_size = 10
        if len(forward_times) >= window_size:
            forward_smooth = np.convolve(forward_times, np.ones(window_size)/window_size, mode='valid')
            backward_smooth = np.convolve(backward_times, np.ones(window_size)/window_size, mode='valid')
            
            axes[2, 0].plot(forward_smooth, label='Forward (smoothed)', alpha=0.7)
            axes[2, 0].plot(backward_smooth, label='Backward (smoothed)', alpha=0.7)
            axes[2, 0].set_title('Performance Trends (Smoothed)')
            axes[2, 0].set_xlabel('Batch')
            axes[2, 0].set_ylabel('Time (seconds)')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
        
        # 8. Resource utilization
        cpu_util = [m.cpu_utilization for m in self.metrics]
        axes[2, 1].plot(epochs, cpu_util, alpha=0.7, color='green')
        axes[2, 1].set_title('CPU Utilization')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('CPU %')
        axes[2, 1].grid(True, alpha=0.3)
        
        # 9. Summary statistics
        summary_text = f"""
        Total Training Time: {self.total_training_time:.2f}s
        Total Samples: {self.total_samples_processed:,}
        Throughput: {self.total_samples_processed/self.total_training_time:.2f} samples/sec
        Peak Memory: {self.peak_memory_usage:.2f} MB
        Total Batches: {len(self.metrics)}
        """
        
        axes[2, 2].text(0.1, 0.5, summary_text, transform=axes[2, 2].transAxes, 
                        fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[2, 2].set_title('Summary Statistics')
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_metrics(self, filepath: str):
        """Save metrics to JSON file"""
        metrics_dict = [asdict(metric) for metric in self.metrics]
        summary = self.get_performance_summary()
        
        data = {
            'summary': summary,
            'detailed_metrics': metrics_dict,
            'timing_data': dict(self.timing_data),
            'memory_data': dict(self.memory_data),
            'performance_data': dict(self.performance_data)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Metrics saved to {filepath}")

class ProfiledTrainer:
    """Trainer wrapper with integrated profiling"""
    
    def __init__(self, model, train_loader, test_loader, device, 
                 learning_rate=0.001, weight_decay=1e-4, 
                 use_mixed_precision=True, optimizer_type='adamw'):
        
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        
        # Initialize profiler
        model_name = type(model).__name__
        self.profiler = ModelProfiler(model_name, device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        
        if optimizer_type.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and torch.cuda.is_available() else None
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        
    def train_epoch_profiled(self, epoch: int):
        """Train one epoch with comprehensive profiling"""
        self.model.train()
        self.profiler.current_epoch = epoch
        self.profiler.start_profiling()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.profiler.current_batch = batch_idx
            
            # Data loading timing
            self.profiler.start_timer('data_load')
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            data_load_time = self.profiler.end_timer('data_load')
            
            # Forward pass timing
            self.profiler.start_timer('forward')
            if self.use_mixed_precision and self.scaler:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
            forward_time = self.profiler.end_timer('forward')
            
            # Backward pass timing
            self.profiler.start_timer('backward')
            if self.use_mixed_precision and self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            backward_time = self.profiler.end_timer('backward')
            
            # Optimizer timing
            self.profiler.start_timer('optimizer')
            if self.use_mixed_precision and self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            optimizer_time = self.profiler.end_timer('optimizer')
            
            # Record metrics
            batch_size = data.size(0)
            sequence_length = None
            patch_count = None
            
            # For ViT models, extract additional metrics
            if hasattr(self.model, 'patch_embedding'):
                patch_count = self.model.patch_embedding.n_patches
                sequence_length = patch_count + 1  # +1 for cls token
            
            self.profiler.record_batch_metrics(
                batch_size=batch_size,
                forward_time=forward_time,
                backward_time=backward_time,
                optimizer_time=optimizer_time,
                data_load_time=data_load_time,
                sequence_length=sequence_length,
                patch_count=patch_count
            )
            
            # Statistics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}: '
                      f'Loss={loss.item():.4f}, '
                      f'Acc={100.*correct/total:.2f}%, '
                      f'Forward={forward_time:.4f}s, '
                      f'Backward={backward_time:.4f}s')
        
        self.profiler.stop_profiling()
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = 100. * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def test_epoch_profiled(self):
        """Test one epoch with profiling"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                if self.use_mixed_precision and self.scaler:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(self.test_loader)
        epoch_accuracy = 100. * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def train_profiled(self, num_epochs: int, save_path: str = './profiling_results'):
        """Train with comprehensive profiling"""
        os.makedirs(save_path, exist_ok=True)
        
        print(f"🚀 Starting profiled training for {num_epochs} epochs...")
        print(f"Model: {type(self.model).__name__}")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_mixed_precision}")
        
        for epoch in range(num_epochs):
            print(f"\n📊 Epoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch_profiled(epoch)
            
            # Testing
            test_loss, test_acc = self.test_epoch_profiled()
            
            # Store history
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_acc)
            
            print(f"📈 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"📈 Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            
            # Save metrics every epoch
            metrics_file = os.path.join(save_path, f'{self.profiler.model_name}_epoch_{epoch+1}_metrics.json')
            self.profiler.save_metrics(metrics_file)
        
        # Final performance analysis
        summary = self.profiler.get_performance_summary()
        print(f"\n✅ Training completed!")
        print(f"📊 Performance Summary:")
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        # Save final results
        final_metrics_file = os.path.join(save_path, f'{self.profiler.model_name}_final_metrics.json')
        self.profiler.save_metrics(final_metrics_file)
        
        # Create performance plots
        plot_file = os.path.join(save_path, f'{self.profiler.model_name}_performance_analysis.png')
        self.profiler.plot_performance_analysis(plot_file)
        
        return summary

# Example usage
if __name__ == "__main__":
    from cnn_model import create_cnn_model
    from vit_model import create_vit_model
    from data_utils import load_cifar10_data
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_loader = load_cifar10_data(batch_size=64)
    
    # Test with CNN
    print("\n🔍 Profiling CNN...")
    cnn_model = create_cnn_model(num_classes=10, device=device)
    cnn_trainer = ProfiledTrainer(cnn_model, train_loader, test_loader, device)
    cnn_summary = cnn_trainer.train_profiled(num_epochs=2, save_path='./cnn_profiling')
    
    # Test with ViT
    print("\n🔍 Profiling ViT...")
    vit_model = create_vit_model(num_classes=10, device=device, model_size='small')
    vit_trainer = ProfiledTrainer(vit_model, train_loader, test_loader, device)
    vit_summary = vit_trainer.train_profiled(num_epochs=2, save_path='./vit_profiling')
    
    print("\n📊 Comparison Summary:")
    print("CNN Performance:")
    for key, value in cnn_summary.items():
        print(f"   {key}: {value}")
    
    print("\nViT Performance:")
    for key, value in vit_summary.items():
        print(f"   {key}: {value}")

