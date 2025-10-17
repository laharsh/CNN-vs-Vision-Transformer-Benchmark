"""
Multi-GPU Benchmark Runner for CNN vs Vision Transformer Comparison

This module uses PyTorch DDP to run CNN and ViT benchmarks on separate GPUs in parallel.
Each model gets its own dedicated GPU for maximum parallelization.
"""

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
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
    save_path: str = './benchmark_results'
    model_sizes: List[str] = None
    world_size: int = 2  # Number of GPUs to use
    
    def __post_init__(self):
        if self.model_sizes is None:
            self.model_sizes = ['small']

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
    gpu_id: int
    performance_summary: Dict[str, Any]
    detailed_metrics: List[Dict[str, Any]]

def setup_ddp(rank: int, world_size: int):
    """Initialize the distributed environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group(
        backend='nccl',  # Use NCCL for GPU
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # Set the GPU for this process
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Clean up the distributed environment"""
    dist.destroy_process_group()

def benchmark_worker(rank: int, world_size: int, config: BenchmarkConfig, 
                     model_type: str, model_size: str, result_queue):
    """Worker function for benchmarking a single model on a dedicated GPU"""
    try:
        # Setup DDP
        setup_ddp(rank, world_size)
        device = torch.device(f'cuda:{rank}')
        
        print(f"🔧 [GPU {rank}] Starting {model_type} benchmark on {device}")
        
        # Create model based on type
        if model_type == 'CNN':
            model = create_cnn_model(num_classes=10, device=device)
            model_info = get_cnn_info(model)
            model_name = 'CNN'
        else:  # ViT
            model = create_vit_model(num_classes=10, device=device, model_size=model_size)
            model_info = get_vit_info(model)
            model_name = f'ViT-{model_size}'
        
        # Load data (each GPU gets its own data loaders)
        train_loader, test_loader = load_cifar10_data(
            batch_size=config.batch_size,
            num_workers=2  # Use some workers for data loading
        )
        
        # Create profiled trainer
        trainer = ProfiledTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            use_mixed_precision=config.use_mixed_precision,
            optimizer_type=config.optimizer
        )
        
        # Run training with profiling
        print(f"🏃 [GPU {rank}] Training {model_name}...")
        start_time = time.time()
        performance_summary = trainer.train_profiled(
            num_epochs=config.num_epochs,
            save_path=os.path.join(config.save_path, f'{model_type.lower()}_{rank}')
        )
        training_time = time.time() - start_time
        
        # Extract detailed metrics
        detailed_metrics = [trainer.profiler.metrics[i].__dict__ 
                          for i in range(len(trainer.profiler.metrics))]
        
        # Create result
        result = BenchmarkResult(
            model_name=model_name,
            model_type=model_type,
            model_size=model_size if model_type == 'ViT' else 'standard',
            training_time=training_time,
            final_train_accuracy=trainer.train_accuracies[-1] if trainer.train_accuracies else 0,
            final_test_accuracy=trainer.test_accuracies[-1] if trainer.test_accuracies else 0,
            peak_memory_mb=performance_summary.get('peak_memory_usage_mb', 0),
            throughput_samples_per_sec=performance_summary.get('throughput_samples_per_sec', 0),
            total_parameters=model_info['total_parameters'],
            model_size_mb=model_info['model_size_mb'],
            gpu_id=rank,
            performance_summary=performance_summary,
            detailed_metrics=detailed_metrics
        )
        
        print(f"✅ [GPU {rank}] {model_name} benchmark completed: "
              f"{result.final_test_accuracy:.2f}% accuracy in {training_time:.2f}s")
        
        # Convert result to dict and put in queue
        result_dict = asdict(result)
        result_queue.put(result_dict)
        
    except Exception as e:
        print(f"❌ [GPU {rank}] Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put(None)
    finally:
        # Cleanup DDP
        cleanup_ddp()

class ParallelBenchmarkRunner:
    """Orchestrates parallel benchmarking across multiple GPUs"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        
        # Check GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This runner requires GPUs.")
        
        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            raise RuntimeError(f"This runner requires at least 2 GPUs, but only {num_gpus} found.")
        
        print(f"🎮 Found {num_gpus} GPUs available")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    def run_parallel_benchmark(self) -> List[BenchmarkResult]:
        """Run CNN and ViT benchmarks in parallel on separate GPUs"""
        print("\n🚀 Starting Parallel Multi-GPU Benchmark")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  Epochs: {self.config.num_epochs}")
        print(f"  Batch Size: {self.config.batch_size}")
        print(f"  Learning Rate: {self.config.learning_rate}")
        print(f"  Mixed Precision: {self.config.use_mixed_precision}")
        print(f"  World Size: {self.config.world_size}")
        print("=" * 60)
        
        # Create result queue for inter-process communication
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        
        # Create process list
        processes = []
        
        # GPU 0: CNN
        p_cnn = ctx.Process(
            target=benchmark_worker,
            args=(0, self.config.world_size, self.config, 'CNN', 'standard', result_queue)
        )
        processes.append(p_cnn)
        
        # GPU 1: ViT
        for idx, model_size in enumerate(self.config.model_sizes):
            gpu_id = 1 + idx  # ViT models start from GPU 1
            if gpu_id >= self.config.world_size:
                print(f"⚠️ Not enough GPUs for ViT-{model_size}, skipping...")
                continue
            
            p_vit = ctx.Process(
                target=benchmark_worker,
                args=(gpu_id, self.config.world_size, self.config, 'ViT', model_size, result_queue)
            )
            processes.append(p_vit)
        
        # Start all processes
        print(f"\n🏁 Starting {len(processes)} parallel benchmark processes...")
        start_time = time.time()
        
        for p in processes:
            p.start()
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        total_time = time.time() - start_time
        print(f"\n⏱️ All benchmarks completed in {total_time:.2f}s (wall time)")
        
        # Collect results from queue
        results = []
        while not result_queue.empty():
            result_dict = result_queue.get()
            if result_dict is not None:
                # Convert dict back to BenchmarkResult
                result = BenchmarkResult(**result_dict)
                results.append(result)
        
        print(f"\n✅ Successfully collected {len(results)} benchmark results")
        
        # Calculate speedup from parallelization
        if len(results) >= 2:
            total_sequential_time = sum(r.training_time for r in results)
            speedup = total_sequential_time / total_time
            print(f"🚀 Parallel speedup: {speedup:.2f}x")
            print(f"   Sequential time would be: {total_sequential_time:.2f}s")
            print(f"   Parallel time: {total_time:.2f}s")
        
        return results

class BenchmarkComparator:
    """Compare results from multiple benchmark runs"""
    
    def __init__(self, results: List[BenchmarkResult]):
        self.results = results
    
    def create_comparison_report(self, save_path: str):
        """Create comprehensive comparison report"""
        os.makedirs(save_path, exist_ok=True)
        
        print("\n📊 Generating comparison visualizations...")
        
        # Generate comparison plots
        self.plot_performance_comparison(save_path)
        self.plot_gpu_utilization(save_path)
        
        # Generate summary report
        self.generate_summary_report(save_path)
    
    def plot_performance_comparison(self, save_path: str):
        """Plot performance comparison charts"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('CNN vs ViT Performance Comparison (Parallel Multi-GPU)', 
                    fontsize=16, fontweight='bold')
        
        # Extract data
        model_names = [f"{r.model_name}\n(GPU {r.gpu_id})" for r in self.results]
        train_accuracies = [r.final_train_accuracy for r in self.results]
        test_accuracies = [r.final_test_accuracy for r in self.results]
        training_times = [r.training_time for r in self.results]
        throughputs = [r.throughput_samples_per_sec for r in self.results]
        peak_memories = [r.peak_memory_mb for r in self.results]
        parameters = [r.total_parameters for r in self.results]
        
        # 1. Accuracy comparison
        x_pos = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x_pos - width/2, train_accuracies, width, 
                      label='Train Accuracy', alpha=0.8)
        axes[0, 0].bar(x_pos + width/2, test_accuracies, width, 
                      label='Test Accuracy', alpha=0.8)
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_title('Final Accuracy Comparison')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Training time comparison
        colors = ['skyblue' if 'CNN' in r.model_name else 'lightcoral' 
                 for r in self.results]
        axes[0, 1].bar(range(len(model_names)), training_times, 
                      alpha=0.8, color=colors)
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylabel('Training Time (seconds)')
        axes[0, 1].set_title('Training Time per Model')
        axes[0, 1].set_xticks(range(len(model_names)))
        axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Throughput comparison
        axes[0, 2].bar(range(len(model_names)), throughputs, 
                      alpha=0.8, color='lightgreen')
        axes[0, 2].set_xlabel('Model')
        axes[0, 2].set_ylabel('Throughput (samples/sec)')
        axes[0, 2].set_title('Training Throughput')
        axes[0, 2].set_xticks(range(len(model_names)))
        axes[0, 2].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Memory usage comparison
        axes[1, 0].bar(range(len(model_names)), peak_memories, 
                      alpha=0.8, color='orange')
        axes[1, 0].set_xlabel('Model')
        axes[1, 0].set_ylabel('Peak Memory (MB)')
        axes[1, 0].set_title('Peak GPU Memory Usage')
        axes[1, 0].set_xticks(range(len(model_names)))
        axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Model size comparison
        axes[1, 1].bar(range(len(model_names)), parameters, 
                      alpha=0.8, color='purple')
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('Number of Parameters')
        axes[1, 1].set_title('Model Size Comparison')
        axes[1, 1].set_xticks(range(len(model_names)))
        axes[1, 1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Efficiency scatter plot
        colors_scatter = ['red' if 'CNN' in r.model_name else 'blue' 
                         for r in self.results]
        axes[1, 2].scatter(training_times, test_accuracies, 
                          c=colors_scatter, s=150, alpha=0.7)
        for i, r in enumerate(self.results):
            axes[1, 2].annotate(r.model_name, 
                              (training_times[i], test_accuracies[i]), 
                              xytext=(5, 5), textcoords='offset points')
        axes[1, 2].set_xlabel('Training Time (seconds)')
        axes[1, 2].set_ylabel('Test Accuracy (%)')
        axes[1, 2].set_title('Efficiency: Accuracy vs Time')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'parallel_performance_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: parallel_performance_comparison.png")
        plt.close()
    
    def plot_gpu_utilization(self, save_path: str):
        """Plot GPU utilization timeline"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        fig.suptitle('GPU Utilization Timeline (Parallel Execution)', 
                    fontsize=14, fontweight='bold')
        
        # Create Gantt-like chart showing parallel execution
        colors = {'CNN': 'skyblue', 'ViT': 'lightcoral'}
        
        for result in self.results:
            color = colors.get(result.model_type, 'gray')
            ax.barh(
                result.gpu_id, 
                result.training_time,
                left=0,
                height=0.5,
                label=result.model_name,
                color=color,
                alpha=0.8
            )
            # Add text label
            ax.text(
                result.training_time / 2,
                result.gpu_id,
                f"{result.model_name}\n{result.training_time:.1f}s",
                ha='center',
                va='center',
                fontsize=10,
                fontweight='bold'
            )
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('GPU ID')
        ax.set_title('Parallel Training Timeline')
        ax.set_yticks(range(len(self.results)))
        ax.set_yticklabels([f"GPU {r.gpu_id}" for r in self.results])
        ax.grid(True, alpha=0.3, axis='x')
        
        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'gpu_utilization_timeline.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: gpu_utilization_timeline.png")
        plt.close()
    
    def generate_summary_report(self, save_path: str):
        """Generate comprehensive summary report"""
        report = {
            'benchmark_summary': {
                'total_models_tested': len(self.results),
                'models': [r.model_name for r in self.results],
                'gpus_used': list(set(r.gpu_id for r in self.results)),
                'benchmark_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'execution_mode': 'parallel_multi_gpu'
            },
            'performance_comparison': {},
            'parallelization_efficiency': {},
            'recommendations': []
        }
        
        # Performance comparison
        for result in self.results:
            report['performance_comparison'][result.model_name] = {
                'gpu_id': result.gpu_id,
                'final_test_accuracy': result.final_test_accuracy,
                'training_time': result.training_time,
                'throughput_samples_per_sec': result.throughput_samples_per_sec,
                'peak_memory_mb': result.peak_memory_mb,
                'total_parameters': result.total_parameters,
                'model_size_mb': result.model_size_mb
            }
        
        # Parallelization efficiency
        if len(self.results) >= 2:
            total_sequential_time = sum(r.training_time for r in self.results)
            max_parallel_time = max(r.training_time for r in self.results)
            speedup = total_sequential_time / max_parallel_time
            efficiency = speedup / len(self.results) * 100
            
            report['parallelization_efficiency'] = {
                'total_sequential_time_seconds': total_sequential_time,
                'actual_parallel_time_seconds': max_parallel_time,
                'speedup_factor': speedup,
                'parallel_efficiency_percent': efficiency
            }
        
        # Generate recommendations
        if len(self.results) >= 2:
            cnn_result = next((r for r in self.results if 'CNN' in r.model_name), None)
            vit_result = next((r for r in self.results if 'ViT' in r.model_name), None)
            
            if cnn_result and vit_result:
                if cnn_result.final_test_accuracy > vit_result.final_test_accuracy:
                    report['recommendations'].append(
                        f"CNN achieves higher accuracy ({cnn_result.final_test_accuracy:.2f}% "
                        f"vs {vit_result.final_test_accuracy:.2f}%)"
                    )
                else:
                    report['recommendations'].append(
                        f"ViT achieves higher accuracy ({vit_result.final_test_accuracy:.2f}% "
                        f"vs {cnn_result.final_test_accuracy:.2f}%)"
                    )
                
                report['recommendations'].append(
                    f"Parallel execution saved {total_sequential_time - max_parallel_time:.2f}s "
                    f"compared to sequential execution"
                )
        
        # Save report
        report_path = os.path.join(save_path, 'parallel_benchmark_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n📊 PARALLEL BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Models tested: {len(self.results)}")
        print(f"GPUs used: {len(set(r.gpu_id for r in self.results))}")
        print(f"Date: {report['benchmark_summary']['benchmark_date']}")
        
        if 'parallelization_efficiency' in report:
            eff = report['parallelization_efficiency']
            print(f"\nParallelization Efficiency:")
            print(f"  Sequential time: {eff['total_sequential_time_seconds']:.2f}s")
            print(f"  Parallel time: {eff['actual_parallel_time_seconds']:.2f}s")
            print(f"  Speedup: {eff['speedup_factor']:.2f}x")
            print(f"  Efficiency: {eff['parallel_efficiency_percent']:.1f}%")
        
        print("\nPerformance Comparison:")
        for model_name, metrics in report['performance_comparison'].items():
            print(f"\n{model_name} (GPU {metrics['gpu_id']}):")
            print(f"  Accuracy: {metrics['final_test_accuracy']:.2f}%")
            print(f"  Training Time: {metrics['training_time']:.2f}s")
            print(f"  Throughput: {metrics['throughput_samples_per_sec']:.2f} samples/sec")
            print(f"  Peak Memory: {metrics['peak_memory_mb']:.2f}MB")
        
        if report['recommendations']:
            print("\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  • {rec}")
        
        print(f"\nDetailed report saved to: {report_path}")

def run_parallel_benchmark(config: BenchmarkConfig) -> List[BenchmarkResult]:
    """Run comprehensive parallel benchmark comparing CNN and ViT models"""
    # Create parallel runner
    runner = ParallelBenchmarkRunner(config)
    
    # Run parallel benchmarks
    results = runner.run_parallel_benchmark()
    
    # Create comparison report
    if len(results) > 1:
        comparator = BenchmarkComparator(results)
        comparator.create_comparison_report(config.save_path)
        print(f"\n📊 Comparison report generated in: {config.save_path}")
    
    return results

# Example usage
if __name__ == "__main__":
    # Check for multi-GPU setup
    if torch.cuda.device_count() < 2:
        print("❌ This script requires at least 2 GPUs!")
        print(f"   Found: {torch.cuda.device_count()} GPU(s)")
        exit(1)
    
    # Configure benchmark
    config = BenchmarkConfig(
        num_epochs=5,
        batch_size=64,
        learning_rate=0.001,
        weight_decay=1e-4,
        optimizer='adamw',
        use_mixed_precision=True,
        save_path='./parallel_benchmark_results',
        model_sizes=['small'],  # Can add more: ['small', 'base']
        world_size=2  # Number of GPUs (1 for CNN, 1 for ViT)
    )
    
    # Run parallel benchmark
    results = run_parallel_benchmark(config)
    
    print(f"\n🎉 Parallel benchmark completed! Results for {len(results)} models:")
    for result in results:
        print(f"  {result.model_name} (GPU {result.gpu_id}): "
              f"{result.final_test_accuracy:.2f}% accuracy, "
              f"{result.training_time:.2f}s training time")