"""
Benchmark Runner for CNN vs Vision Transformer Comparison

This module orchestrates comprehensive benchmarking of CNN and ViT models including:
1. Automated model training with profiling
2. Performance comparison across multiple metrics
3. Memory usage analysis
4. Training speed comparison
5. Bottleneck identification
6. Comprehensive reporting
"""

import torch
import torch.nn as nn
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from cnn_model import create_cnn_model, get_model_info as get_cnn_info
from vit_model import create_vit_model, get_model_info as get_vit_info
from data_utils import load_cifar10_data
from model_profiler import ProfiledTrainer, ModelProfiler

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs"""
    num_epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    optimizer: str = 'adamw'
    use_mixed_precision: bool = True
    device: str = 'auto'
    save_path: str = './benchmark_results'
    model_sizes: List[str] = None
    
    def __post_init__(self):
        if self.model_sizes is None:
            self.model_sizes = ['small']
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    model_name: str
    model_type: str
    model_size: str
    training_time: float
    final_train_accuracy: float
    final_test_accuracy: float
    peak_memory_mb: float
    throughput_samples_per_sec: float
    total_parameters: int
    model_size_mb: float
    performance_summary: Dict[str, Any]
    detailed_metrics: List[Dict[str, Any]]

class ModelBenchmark:
    """Individual model benchmark runner"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device(config.device)
        
    def benchmark_cnn(self) -> BenchmarkResult:
        """Benchmark CNN model"""
        print("🔍 Benchmarking CNN Model...")
        
        # Create model
        model = create_cnn_model(num_classes=10, device=self.device)
        model_info = get_cnn_info(model)
        
        # Load data
        train_loader, test_loader = load_cifar10_data(
            batch_size=self.config.batch_size,
            num_workers=0  # Windows compatibility
        )
        
        # Create profiled trainer
        trainer = ProfiledTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=self.device,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            use_mixed_precision=self.config.use_mixed_precision,
            optimizer_type=self.config.optimizer
        )
        
        # Run training with profiling
        start_time = time.time()
        performance_summary = trainer.train_profiled(
            num_epochs=self.config.num_epochs,
            save_path=os.path.join(self.config.save_path, 'cnn')
        )
        training_time = time.time() - start_time
        
        # Extract detailed metrics
        detailed_metrics = [trainer.profiler.metrics[i].__dict__ for i in range(len(trainer.profiler.metrics))]
        
        # Create result
        result = BenchmarkResult(
            model_name='CNN',
            model_type='CNN',
            model_size='standard',
            training_time=training_time,
            final_train_accuracy=trainer.train_accuracies[-1] if trainer.train_accuracies else 0,
            final_test_accuracy=trainer.test_accuracies[-1] if trainer.test_accuracies else 0,
            peak_memory_mb=performance_summary.get('peak_memory_usage_mb', 0),
            throughput_samples_per_sec=performance_summary.get('throughput_samples_per_sec', 0),
            total_parameters=model_info['total_parameters'],
            model_size_mb=model_info['model_size_mb'],
            performance_summary=performance_summary,
            detailed_metrics=detailed_metrics
        )
        
        return result
    
    def benchmark_vit(self, model_size: str = 'small') -> BenchmarkResult:
        """Benchmark ViT model"""
        print(f"🔍 Benchmarking ViT Model ({model_size})...")
        
        # Create model
        model = create_vit_model(
            num_classes=10, 
            device=self.device, 
            model_size=model_size
        )
        model_info = get_vit_info(model)
        
        # Load data
        train_loader, test_loader = load_cifar10_data(
            batch_size=self.config.batch_size,
            num_workers=0  # Windows compatibility
        )
        
        # Create profiled trainer
        trainer = ProfiledTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=self.device,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            use_mixed_precision=self.config.use_mixed_precision,
            optimizer_type=self.config.optimizer
        )
        
        # Run training with profiling
        start_time = time.time()
        performance_summary = trainer.train_profiled(
            num_epochs=self.config.num_epochs,
            save_path=os.path.join(self.config.save_path, f'vit_{model_size}')
        )
        training_time = time.time() - start_time
        
        # Extract detailed metrics
        detailed_metrics = [trainer.profiler.metrics[i].__dict__ for i in range(len(trainer.profiler.metrics))]
        
        # Create result
        result = BenchmarkResult(
            model_name=f'ViT-{model_size}',
            model_type='ViT',
            model_size=model_size,
            training_time=training_time,
            final_train_accuracy=trainer.train_accuracies[-1] if trainer.train_accuracies else 0,
            final_test_accuracy=trainer.test_accuracies[-1] if trainer.test_accuracies else 0,
            peak_memory_mb=performance_summary.get('peak_memory_usage_mb', 0),
            throughput_samples_per_sec=performance_summary.get('throughput_samples_per_sec', 0),
            total_parameters=model_info['total_parameters'],
            model_size_mb=model_info['model_size_mb'],
            performance_summary=performance_summary,
            detailed_metrics=detailed_metrics
        )
        
        return result

class BenchmarkComparator:
    """Compare results from multiple benchmark runs"""
    
    def __init__(self, results: List[BenchmarkResult]):
        self.results = results
        
    def create_comparison_report(self, save_path: str):
        """Create comprehensive comparison report"""
        os.makedirs(save_path, exist_ok=True)
        
        # Generate comparison plots
        self.plot_performance_comparison(save_path)
        self.plot_memory_comparison(save_path)
        self.plot_timing_breakdown(save_path)
        self.plot_efficiency_metrics(save_path)
        
        # Generate summary report
        self.generate_summary_report(save_path)
        
    def plot_performance_comparison(self, save_path: str):
        """Plot performance comparison charts"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('CNN vs ViT Performance Comparison', fontsize=16, fontweight='bold')
        
        # Extract data
        model_names = [r.model_name for r in self.results]
        train_accuracies = [r.final_train_accuracy for r in self.results]
        test_accuracies = [r.final_test_accuracy for r in self.results]
        training_times = [r.training_time for r in self.results]
        throughputs = [r.throughput_samples_per_sec for r in self.results]
        peak_memories = [r.peak_memory_mb for r in self.results]
        parameters = [r.total_parameters for r in self.results]
        
        # 1. Accuracy comparison
        x_pos = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x_pos - width/2, train_accuracies, width, label='Train Accuracy', alpha=0.8)
        axes[0, 0].bar(x_pos + width/2, test_accuracies, width, label='Test Accuracy', alpha=0.8)
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_title('Final Accuracy Comparison')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Training time comparison
        axes[0, 1].bar(model_names, training_times, alpha=0.8, color='skyblue')
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylabel('Training Time (seconds)')
        axes[0, 1].set_title('Training Time Comparison')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Throughput comparison
        axes[0, 2].bar(model_names, throughputs, alpha=0.8, color='lightgreen')
        axes[0, 2].set_xlabel('Model')
        axes[0, 2].set_ylabel('Throughput (samples/sec)')
        axes[0, 2].set_title('Training Throughput Comparison')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Memory usage comparison
        axes[1, 0].bar(model_names, peak_memories, alpha=0.8, color='orange')
        axes[1, 0].set_xlabel('Model')
        axes[1, 0].set_ylabel('Peak Memory (MB)')
        axes[1, 0].set_title('Peak Memory Usage Comparison')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Model size comparison
        axes[1, 1].bar(model_names, parameters, alpha=0.8, color='purple')
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('Number of Parameters')
        axes[1, 1].set_title('Model Size Comparison')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Efficiency scatter plot (accuracy vs time)
        colors = ['red' if 'CNN' in name else 'blue' for name in model_names]
        axes[1, 2].scatter(training_times, test_accuracies, c=colors, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            axes[1, 2].annotate(name, (training_times[i], test_accuracies[i]), 
                              xytext=(5, 5), textcoords='offset points')
        axes[1, 2].set_xlabel('Training Time (seconds)')
        axes[1, 2].set_ylabel('Test Accuracy (%)')
        axes[1, 2].set_title('Efficiency: Accuracy vs Time')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_memory_comparison(self, save_path: str):
        """Plot detailed memory usage comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Memory Usage Analysis', fontsize=16, fontweight='bold')
        
        for i, result in enumerate(self.results):
            if result.detailed_metrics:
                # Extract memory data over time
                epochs = [m['epoch'] for m in result.detailed_metrics]
                cpu_memory = [m['cpu_memory_percent'] for m in result.detailed_metrics]
                gpu_memory = [m['gpu_memory_mb'] for m in result.detailed_metrics]
                
                # Plot CPU memory over time
                axes[0, 0].plot(epochs, cpu_memory, label=result.model_name, alpha=0.7)
                
                # Plot GPU memory over time
                if any(gpu_memory):  # Only plot if GPU memory data exists
                    axes[0, 1].plot(epochs, gpu_memory, label=result.model_name, alpha=0.7)
        
        axes[0, 0].set_title('CPU Memory Usage Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('CPU Memory (%)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('GPU Memory Usage Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('GPU Memory (MB)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Memory efficiency comparison
        model_names = [r.model_name for r in self.results]
        peak_memories = [r.peak_memory_mb for r in self.results]
        parameters = [r.total_parameters for r in self.results]
        
        # Memory per parameter
        memory_per_param = [mem / param for mem, param in zip(peak_memories, parameters)]
        
        axes[1, 0].bar(model_names, memory_per_param, alpha=0.8, color='lightcoral')
        axes[1, 0].set_title('Memory Efficiency (MB per Parameter)')
        axes[1, 0].set_xlabel('Model')
        axes[1, 0].set_ylabel('MB per Parameter')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Memory vs accuracy
        test_accuracies = [r.final_test_accuracy for r in self.results]
        colors = ['red' if 'CNN' in name else 'blue' for name in model_names]
        axes[1, 1].scatter(peak_memories, test_accuracies, c=colors, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            axes[1, 1].annotate(name, (peak_memories[i], test_accuracies[i]), 
                              xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_xlabel('Peak Memory (MB)')
        axes[1, 1].set_ylabel('Test Accuracy (%)')
        axes[1, 1].set_title('Memory vs Accuracy Trade-off')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'memory_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_timing_breakdown(self, save_path: str):
        """Plot detailed timing breakdown comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Timing Breakdown Analysis', fontsize=16, fontweight='bold')
        
        # Extract timing data
        model_names = [r.model_name for r in self.results]
        
        # Forward, backward, optimizer, data loading times
        forward_times = []
        backward_times = []
        optimizer_times = []
        data_load_times = []
        
        for result in self.results:
            if result.performance_summary:
                forward_times.append(result.performance_summary.get('forward_time_mean', 0))
                backward_times.append(result.performance_summary.get('backward_time_mean', 0))
                optimizer_times.append(result.performance_summary.get('optimizer_time_mean', 0))
                data_load_times.append(result.performance_summary.get('data_load_time_mean', 0))
        
        # Stacked bar chart of timing breakdown
        x_pos = np.arange(len(model_names))
        width = 0.6
        
        axes[0, 0].bar(x_pos, forward_times, width, label='Forward Pass', alpha=0.8)
        axes[0, 0].bar(x_pos, backward_times, width, bottom=forward_times, label='Backward Pass', alpha=0.8)
        axes[0, 0].bar(x_pos, optimizer_times, width, 
                      bottom=np.array(forward_times) + np.array(backward_times), 
                      label='Optimizer', alpha=0.8)
        axes[0, 0].bar(x_pos, data_load_times, width, 
                      bottom=np.array(forward_times) + np.array(backward_times) + np.array(optimizer_times), 
                      label='Data Loading', alpha=0.8)
        
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].set_title('Timing Breakdown per Batch')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Efficiency metrics
        compute_efficiencies = []
        data_overheads = []
        optimizer_overheads = []
        
        for result in self.results:
            if result.performance_summary:
                compute_efficiencies.append(result.performance_summary.get('compute_efficiency', 0))
                data_overheads.append(result.performance_summary.get('data_loading_overhead', 0))
                optimizer_overheads.append(result.performance_summary.get('optimizer_overhead', 0))
        
        # Efficiency comparison
        x_pos = np.arange(len(model_names))
        width = 0.25
        
        axes[0, 1].bar(x_pos - width, compute_efficiencies, width, label='Compute Efficiency', alpha=0.8)
        axes[0, 1].bar(x_pos, data_overheads, width, label='Data Loading Overhead', alpha=0.8)
        axes[0, 1].bar(x_pos + width, optimizer_overheads, width, label='Optimizer Overhead', alpha=0.8)
        
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylabel('Ratio')
        axes[0, 1].set_title('Training Efficiency Breakdown')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(model_names, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Throughput comparison
        throughputs = [r.throughput_samples_per_sec for r in self.results]
        axes[1, 0].bar(model_names, throughputs, alpha=0.8, color='lightgreen')
        axes[1, 0].set_xlabel('Model')
        axes[1, 0].set_ylabel('Throughput (samples/sec)')
        axes[1, 0].set_title('Training Throughput')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance vs efficiency scatter
        colors = ['red' if 'CNN' in name else 'blue' for name in model_names]
        axes[1, 1].scatter(compute_efficiencies, throughputs, c=colors, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            axes[1, 1].annotate(name, (compute_efficiencies[i], throughputs[i]), 
                              xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_xlabel('Compute Efficiency')
        axes[1, 1].set_ylabel('Throughput (samples/sec)')
        axes[1, 1].set_title('Efficiency vs Throughput')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'timing_breakdown.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_efficiency_metrics(self, save_path: str):
        """Plot efficiency and bottleneck analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Efficiency and Bottleneck Analysis', fontsize=16, fontweight='bold')
        
        model_names = [r.model_name for r in self.results]
        
        # Extract efficiency metrics
        accuracies = [r.final_test_accuracy for r in self.results]
        times = [r.training_time for r in self.results]
        memories = [r.peak_memory_mb for r in self.results]
        parameters = [r.total_parameters for r in self.results]
        
        # Accuracy per time
        accuracy_per_time = [acc / time for acc, time in zip(accuracies, times)]
        axes[0, 0].bar(model_names, accuracy_per_time, alpha=0.8, color='skyblue')
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('Accuracy per Second')
        axes[0, 0].set_title('Training Efficiency (Accuracy/Time)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy per memory
        accuracy_per_memory = [acc / mem for acc, mem in zip(accuracies, memories)]
        axes[0, 1].bar(model_names, accuracy_per_memory, alpha=0.8, color='lightcoral')
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylabel('Accuracy per MB')
        axes[0, 1].set_title('Memory Efficiency (Accuracy/Memory)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Accuracy per parameter
        accuracy_per_param = [acc / param for acc, param in zip(accuracies, parameters)]
        axes[1, 0].bar(model_names, accuracy_per_param, alpha=0.8, color='lightgreen')
        axes[1, 0].set_xlabel('Model')
        axes[1, 0].set_ylabel('Accuracy per Parameter')
        axes[1, 0].set_title('Parameter Efficiency (Accuracy/Parameter)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Overall efficiency radar chart (simplified as bar chart)
        efficiency_metrics = ['Accuracy/Time', 'Accuracy/Memory', 'Accuracy/Parameter']
        cnn_scores = [accuracy_per_time[0], accuracy_per_memory[0], accuracy_per_param[0]]
        vit_scores = [accuracy_per_time[1], accuracy_per_memory[1], accuracy_per_param[1]]
        
        x_pos = np.arange(len(efficiency_metrics))
        width = 0.35
        
        axes[1, 1].bar(x_pos - width/2, cnn_scores, width, label='CNN', alpha=0.8)
        axes[1, 1].bar(x_pos + width/2, vit_scores, width, label='ViT', alpha=0.8)
        axes[1, 1].set_xlabel('Efficiency Metric')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Overall Efficiency Comparison')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(efficiency_metrics, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'efficiency_metrics.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self, save_path: str):
        """Generate comprehensive summary report"""
        report = {
            'benchmark_summary': {
                'total_models_tested': len(self.results),
                'models': [r.model_name for r in self.results],
                'benchmark_date': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'performance_comparison': {},
            'recommendations': []
        }
        
        # Performance comparison
        for result in self.results:
            report['performance_comparison'][result.model_name] = {
                'final_test_accuracy': result.final_test_accuracy,
                'training_time': result.training_time,
                'throughput_samples_per_sec': result.throughput_samples_per_sec,
                'peak_memory_mb': result.peak_memory_mb,
                'total_parameters': result.total_parameters,
                'model_size_mb': result.model_size_mb
            }
        
        # Generate recommendations
        if len(self.results) >= 2:
            cnn_result = next((r for r in self.results if 'CNN' in r.model_name), None)
            vit_result = next((r for r in self.results if 'ViT' in r.model_name), None)
            
            if cnn_result and vit_result:
                # Accuracy comparison
                if cnn_result.final_test_accuracy > vit_result.final_test_accuracy:
                    report['recommendations'].append(
                        f"CNN achieves higher accuracy ({cnn_result.final_test_accuracy:.2f}% vs {vit_result.final_test_accuracy:.2f}%)"
                    )
                else:
                    report['recommendations'].append(
                        f"ViT achieves higher accuracy ({vit_result.final_test_accuracy:.2f}% vs {cnn_result.final_test_accuracy:.2f}%)"
                    )
                
                # Speed comparison
                if cnn_result.training_time < vit_result.training_time:
                    report['recommendations'].append(
                        f"CNN trains faster ({cnn_result.training_time:.2f}s vs {vit_result.training_time:.2f}s)"
                    )
                else:
                    report['recommendations'].append(
                        f"ViT trains faster ({vit_result.training_time:.2f}s vs {cnn_result.training_time:.2f}s)"
                    )
                
                # Memory comparison
                if cnn_result.peak_memory_mb < vit_result.peak_memory_mb:
                    report['recommendations'].append(
                        f"CNN uses less memory ({cnn_result.peak_memory_mb:.2f}MB vs {vit_result.peak_memory_mb:.2f}MB)"
                    )
                else:
                    report['recommendations'].append(
                        f"ViT uses less memory ({vit_result.peak_memory_mb:.2f}MB vs {cnn_result.peak_memory_mb:.2f}MB)"
                    )
                
                # Efficiency recommendations
                cnn_efficiency = cnn_result.final_test_accuracy / cnn_result.training_time
                vit_efficiency = vit_result.final_test_accuracy / vit_result.training_time
                
                if cnn_efficiency > vit_efficiency:
                    report['recommendations'].append(
                        f"CNN is more efficient overall (accuracy/time ratio: {cnn_efficiency:.4f} vs {vit_efficiency:.4f})"
                    )
                else:
                    report['recommendations'].append(
                        f"ViT is more efficient overall (accuracy/time ratio: {vit_efficiency:.4f} vs {cnn_efficiency:.4f})"
                    )
        
        # Save report
        report_path = os.path.join(save_path, 'benchmark_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n📊 BENCHMARK SUMMARY")
        print("=" * 50)
        print(f"Models tested: {len(self.results)}")
        print(f"Date: {report['benchmark_summary']['benchmark_date']}")
        print("\nPerformance Comparison:")
        for model_name, metrics in report['performance_comparison'].items():
            print(f"\n{model_name}:")
            print(f"  Accuracy: {metrics['final_test_accuracy']:.2f}%")
            print(f"  Training Time: {metrics['training_time']:.2f}s")
            print(f"  Throughput: {metrics['throughput_samples_per_sec']:.2f} samples/sec")
            print(f"  Peak Memory: {metrics['peak_memory_mb']:.2f}MB")
            print(f"  Parameters: {metrics['total_parameters']:,}")
        
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        
        print(f"\nDetailed report saved to: {report_path}")

def run_comprehensive_benchmark(config: BenchmarkConfig) -> List[BenchmarkResult]:
    """Run comprehensive benchmark comparing CNN and ViT models"""
    print("🚀 Starting Comprehensive CNN vs ViT Benchmark")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Device: {config.device}")
    print(f"  Mixed Precision: {config.use_mixed_precision}")
    print(f"  Model Sizes: {config.model_sizes}")
    print("=" * 60)
    
    # Create benchmark runner
    benchmark = ModelBenchmark(config)
    results = []
    
    # Benchmark CNN
    try:
        cnn_result = benchmark.benchmark_cnn()
        results.append(cnn_result)
        print(f"✅ CNN benchmark completed")
    except Exception as e:
        print(f"❌ CNN benchmark failed: {e}")
    
    # Benchmark ViT models
    for model_size in config.model_sizes:
        try:
            vit_result = benchmark.benchmark_vit(model_size)
            results.append(vit_result)
            print(f"✅ ViT-{model_size} benchmark completed")
        except Exception as e:
            print(f"❌ ViT-{model_size} benchmark failed: {e}")
    
    # Create comparison report
    if len(results) > 1:
        comparator = BenchmarkComparator(results)
        comparator.create_comparison_report(config.save_path)
        print(f"📊 Comparison report generated in: {config.save_path}")
    
    return results

# Example usage
if __name__ == "__main__":
    # Configure benchmark
    config = BenchmarkConfig(
        num_epochs=5,  # Quick demo
        batch_size=64,
        learning_rate=0.001,
        weight_decay=1e-4,
        optimizer='adamw',
        use_mixed_precision=True,
        device='auto',
        save_path='./benchmark_results',
        model_sizes=['small']  # Test small ViT for demo
    )
    
    # Run benchmark
    results = run_comprehensive_benchmark(config)
    
    print(f"\n🎉 Benchmark completed! Results for {len(results)} models:")
    for result in results:
        print(f"  {result.model_name}: {result.final_test_accuracy:.2f}% accuracy, "
              f"{result.training_time:.2f}s training time, "
              f"{result.peak_memory_mb:.2f}MB peak memory")

