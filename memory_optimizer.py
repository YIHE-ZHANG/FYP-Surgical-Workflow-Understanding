import torch
import gc
import psutil
import logging
from typing import Dict, Optional, Tuple

class GPUMemoryOptimizer:
    """
    Provides dynamic memory monitoring, intelligent cleanup strategies,
    and adaptive batch size management for consumer-grade hardware.
    """
    
    def __init__(self, device: str = 'cuda', cleanup_threshold: float = 0.85, 
                 verbose: bool = True):
        """
        Initialize memory optimizer.
        
        Args:
            device: CUDA device identifier
            cleanup_threshold: Memory usage ratio that triggers cleanup
            verbose: Enable detailed logging
        """
        self.device = device
        self.cleanup_threshold = cleanup_threshold
        self.verbose = verbose
        
        if torch.cuda.is_available():
            self.total_memory = torch.cuda.get_device_properties(0).total_memory
            self.device_name = torch.cuda.get_device_name(0)
        else:
            self.total_memory = 0
            self.device_name = "CPU"
            
        self.cleanup_count = 0
        self.peak_memory = 0
        
        if verbose:
            print(f"Memory Optimizer initialized for {self.device_name}")
            print(f"Total GPU Memory: {self.total_memory / 1024**3:.2f} GB")
    
    def get_memory_stats(self) -> Optional[Dict[str, float]]:
        """
        Get comprehensive GPU memory statistics.
        
        Returns:
            Dictionary with memory usage information or None if CUDA unavailable
        """
        if not torch.cuda.is_available():
            return None
            
        allocated = torch.cuda.memory_allocated(self.device)
        cached = torch.cuda.memory_reserved(self.device)
        
        # Update peak memory tracking
        self.peak_memory = max(self.peak_memory, allocated)
        
        stats = {
            'allocated_mb': allocated / 1024**2,
            'cached_mb': cached / 1024**2,
            'total_mb': self.total_memory / 1024**2,
            'allocated_gb': allocated / 1024**3,
            'cached_gb': cached / 1024**3,
            'total_gb': self.total_memory / 1024**3,
            'usage_ratio': allocated / self.total_memory,
            'cache_ratio': cached / self.total_memory,
            'peak_mb': self.peak_memory / 1024**2
        }
        
        return stats
    
    def aggressive_cleanup(self) -> Tuple[float, float]:
        """
        Perform comprehensive memory cleanup.
        
        Returns:
            Tuple of (memory_before, memory_after) in GB
        """
        before_stats = self.get_memory_stats()
        memory_before = before_stats['allocated_gb'] if before_stats else 0
        
        # Multi-stage cleanup
        torch.cuda.empty_cache()  # Clear PyTorch cache
        gc.collect()              # Python garbage collection
        
        # Additional CUDA cleanup if available
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            torch.cuda.reset_peak_memory_stats()
        
        if hasattr(torch.cuda, 'synchronize'):
            torch.cuda.synchronize()  # Ensure all operations complete
            
        torch.cuda.empty_cache()  # Second pass cleanup
        
        after_stats = self.get_memory_stats()
        memory_after = after_stats['allocated_gb'] if after_stats else 0
        
        self.cleanup_count += 1
        
        if self.verbose:
            freed = memory_before - memory_after
            print(f"Cleanup #{self.cleanup_count}: Freed {freed:.3f} GB "
                  f"({memory_before:.3f} → {memory_after:.3f} GB)")
        
        return memory_before, memory_after
    
    def adaptive_batch_size(self, current_batch_size: int, 
                          target_memory_ratio: float = 0.75,
                          min_batch_size: int = 1,
                          max_batch_size: int = 16) -> int:
        """
        Dynamically adjust batch size based on current memory usage.
        
        Args:
            current_batch_size: Current batch size
            target_memory_ratio: Target memory usage (0.0-1.0)
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            
        Returns:
            Recommended new batch size
        """
        stats = self.get_memory_stats()
        if not stats:
            return current_batch_size
            
        usage_ratio = stats['usage_ratio']
        
        if usage_ratio > target_memory_ratio + 0.1:
            # Memory pressure - reduce batch size
            reduction_factor = min(0.8, target_memory_ratio / usage_ratio)
            new_batch_size = max(min_batch_size, 
                               int(current_batch_size * reduction_factor))
            
            if new_batch_size != current_batch_size:
                self.aggressive_cleanup()
                if self.verbose:
                    print(f"Memory pressure detected ({usage_ratio:.1%}): "
                          f"Reducing batch size {current_batch_size} → {new_batch_size}")
                          
        elif usage_ratio < target_memory_ratio - 0.2:
            # Low memory usage - can potentially increase
            increase_factor = min(1.25, target_memory_ratio / usage_ratio)
            new_batch_size = min(max_batch_size,
                               int(current_batch_size * increase_factor))
            
            if new_batch_size != current_batch_size and self.verbose:
                print(f"Low memory usage ({usage_ratio:.1%}): "
                      f"Could increase batch size {current_batch_size} → {new_batch_size}")
        else:
            new_batch_size = current_batch_size
            
        return new_batch_size
    
    def monitor_and_optimize(self, step: int, 
                           cleanup_frequency: int = 100,
                           report_frequency: int = 500) -> Optional[Dict[str, float]]:
        """
        Monitor memory usage and perform optimization as needed.
        
        Args:
            step: Current training step
            cleanup_frequency: How often to check for cleanup needs
            report_frequency: How often to report detailed statistics
            
        Returns:
            Memory statistics dictionary if monitoring occurred
        """
        stats = None
        
        # Periodic cleanup check
        if step % cleanup_frequency == 0:
            stats = self.get_memory_stats()
            
            if stats and stats['usage_ratio'] > self.cleanup_threshold:
                if self.verbose:
                    print(f"Step {step}: Memory usage {stats['usage_ratio']:.1%} "
                          f"exceeds threshold {self.cleanup_threshold:.1%}")
                self.aggressive_cleanup()
        
        # Detailed reporting
        if step % report_frequency == 0 and self.verbose:
            if not stats:
                stats = self.get_memory_stats()
            
            if stats:
                print(f"\n--- Memory Report (Step {step}) ---")
                print(f"Allocated: {stats['allocated_gb']:.2f} GB "
                      f"({stats['usage_ratio']:.1%})")
                print(f"Cached: {stats['cached_gb']:.2f} GB "
                      f"({stats['cache_ratio']:.1%})")
                print(f"Peak: {stats['peak_mb']:.1f} MB")
                print(f"Cleanups performed: {self.cleanup_count}")
                print("─" * 35)
                
        return stats
    
    def get_optimization_summary(self) -> Dict[str, any]:
        """
        Get summary of optimization activities.
        
        Returns:
            Summary dictionary with optimization statistics
        """
        current_stats = self.get_memory_stats()
        
        summary = {
            'total_cleanups': self.cleanup_count,
            'peak_memory_gb': self.peak_memory / 1024**3 if self.peak_memory else 0,
            'current_usage_gb': current_stats['allocated_gb'] if current_stats else 0,
            'current_usage_ratio': current_stats['usage_ratio'] if current_stats else 0,
            'device_name': self.device_name,
            'total_memory_gb': self.total_memory / 1024**3
        }
        
        return summary


def create_memory_optimizer(config, verbose: bool = True) -> GPUMemoryOptimizer:
    """
    Factory function to create memory optimizer based on configuration.
    
    Args:
        config: Training configuration object
        verbose: Enable verbose logging
        
    Returns:
        Configured GPUMemoryOptimizer instance
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Adjust thresholds based on GPU memory
    if torch.cuda.is_available():
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        if total_memory_gb <= 8:  # RTX 3060 8GB
            cleanup_threshold = 0.8
        elif total_memory_gb <= 12:  # RTX 3060 12GB
            cleanup_threshold = 0.85
        else:  # Higher-end cards
            cleanup_threshold = 0.9
    else:
        cleanup_threshold = 0.85
    
    optimizer = GPUMemoryOptimizer(
        device=device,
        cleanup_threshold=cleanup_threshold,
        verbose=verbose
    )
    
    if verbose:
        print(f"Memory optimizer configured with {cleanup_threshold:.1%} cleanup threshold")
    
    return optimizer


# Example integration patterns for constraint-aware training
class ConstraintAwareTrainingMixin:
    """
    Mixin class providing constraint-aware training utilities.
  
    """
    
    def __init__(self):
        self.memory_optimizer = None
        self.adaptive_batch_size = True
        self.memory_monitoring = True
    
    def setup_memory_optimization(self, config):
        """Setup memory optimization for training loop."""
        self.memory_optimizer = create_memory_optimizer(config, verbose=True)
        print("Advanced memory optimization enabled")
    
    def training_step_with_memory_optimization(self, step, batch, model, optimizer):
        """
        Example of how memory optimization could be integrated into training.
        
        """
        if self.memory_optimizer:
            # Monitor and optimize memory before processing batch
            self.memory_optimizer.monitor_and_optimize(step)
            
            # Adaptive batch size adjustment (would require dataloader rebuild)
            if self.adaptive_batch_size and step % 1000 == 0:
                current_batch_size = len(batch[0])
                new_batch_size = self.memory_optimizer.adaptive_batch_size(
                    current_batch_size
                )
                if new_batch_size != current_batch_size:
                    print(f"Recommended batch size change: {current_batch_size} → {new_batch_size}")
        
        # Regular training step would continue here...
        pass


if __name__ == "__main__":
    # Demonstration of memory optimizer capabilities
    print("GPU Memory Optimizer - Demonstration")
    print("=" * 50)
    
    optimizer = GPUMemoryOptimizer(verbose=True)
    
    if torch.cuda.is_available():
        # Simulate memory usage
        print("\nSimulating memory allocation...")
        dummy_tensors = []
        
        for i in range(5):
            tensor = torch.randn(1000, 1000, device='cuda')
            dummy_tensors.append(tensor)
            
            stats = optimizer.get_memory_stats()
            print(f"Allocation {i+1}: {stats['allocated_mb']:.1f} MB allocated")
        
        print("\nPerforming cleanup...")
        before, after = optimizer.aggressive_cleanup()
        
        print(f"\nOptimization Summary:")
        summary = optimizer.get_optimization_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")
            
        # Cleanup
        del dummy_tensors
        torch.cuda.empty_cache()
        
    else:
        print("CUDA not available - memory optimization would run in CPU mode")