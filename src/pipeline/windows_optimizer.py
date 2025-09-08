"""
Windows-specific optimizations for 32GB RAM systems
"""

import os
import psutil
import torch
import logging
from typing import Dict, Optional, Tuple
import gc
from pathlib import Path

logger = logging.getLogger(__name__)


class WindowsOptimizer:
    """
    Windows-specific optimizations for high-memory systems
    """
    
    def __init__(self, target_memory_usage: float = 0.8):
        """
        Initialize Windows optimizer
        
        Args:
            target_memory_usage: Target memory usage ratio (0.0-1.0)
        """
        self.target_memory_usage = target_memory_usage
        self.total_memory = psutil.virtual_memory().total
        self.available_memory = psutil.virtual_memory().available
        self.target_memory = int(self.total_memory * target_memory_usage)
        
        logger.info(f"Windows Optimizer initialized:")
        logger.info(f"  Total RAM: {self.total_memory / (1024**3):.1f} GB")
        logger.info(f"  Available RAM: {self.available_memory / (1024**3):.1f} GB")
        logger.info(f"  Target usage: {self.target_memory / (1024**3):.1f} GB ({target_memory_usage*100:.0f}%)")
        
        # Apply Windows-specific optimizations
        self._apply_system_optimizations()
        self._configure_torch_for_windows()
    
    def _apply_system_optimizations(self):
        """Apply Windows-specific system optimizations"""
        try:
            # Set environment variables for better performance
            os.environ['OMP_NUM_THREADS'] = str(min(8, psutil.cpu_count()))
            os.environ['MKL_NUM_THREADS'] = str(min(8, psutil.cpu_count()))
            os.environ['NUMEXPR_NUM_THREADS'] = str(min(8, psutil.cpu_count()))
            
            # Windows-specific optimizations
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async CUDA operations
            
            logger.info("Applied Windows system optimizations")
            
        except Exception as e:
            logger.warning(f"Failed to apply system optimizations: {e}")
    
    def _configure_torch_for_windows(self):
        """Configure PyTorch for optimal Windows performance"""
        try:
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Memory management
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Set memory fraction for CUDA
                torch.cuda.set_per_process_memory_fraction(0.9)
                logger.info("Configured CUDA for Windows optimization")
            
            # Enable mixed precision if available
            if hasattr(torch.cuda, 'amp'):
                logger.info("Mixed precision training available")
            
        except Exception as e:
            logger.warning(f"Failed to configure PyTorch: {e}")
    
    def get_optimal_batch_size(self, model_memory_estimate: int) -> int:
        """
        Calculate optimal batch size based on available memory
        
        Args:
            model_memory_estimate: Estimated memory usage per sample in MB
            
        Returns:
            Optimal batch size
        """
        available_memory_mb = self.available_memory / (1024**2)
        target_memory_mb = self.target_memory / (1024**2)
        
        # Reserve 20% for system operations
        usable_memory = target_memory_mb * 0.8
        
        # Calculate batch size
        batch_size = max(1, int(usable_memory / model_memory_estimate))
        
        # Cap at reasonable maximum
        batch_size = min(batch_size, 16)
        
        logger.info(f"Calculated optimal batch size: {batch_size}")
        return batch_size
    
    def get_optimal_tile_size(self, image_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculate optimal tile size for large image processing
        
        Args:
            image_size: (height, width) of the image
            
        Returns:
            Optimal tile size (height, width)
        """
        h, w = image_size
        
        # Base tile size for 32GB RAM
        base_tile_size = 1024
        
        # Adjust based on image size
        if h > 4000 or w > 4000:
            tile_size = 1536  # Larger tiles for very high-res images
        elif h > 2000 or w > 2000:
            tile_size = 1024  # Standard for high-res images
        else:
            tile_size = 512   # Smaller for normal images
        
        # Ensure tile size doesn't exceed image dimensions
        tile_h = min(tile_size, h)
        tile_w = min(tile_size, w)
        
        logger.info(f"Optimal tile size for {image_size}: ({tile_h}, {tile_w})")
        return (tile_h, tile_w)
    
    def get_optimal_workers(self) -> int:
        """
        Calculate optimal number of worker processes
        
        Returns:
            Optimal number of workers
        """
        cpu_count = psutil.cpu_count()
        memory_gb = self.total_memory / (1024**3)
        
        # For 32GB RAM systems, use more workers
        if memory_gb >= 32:
            workers = min(cpu_count, 12)  # Cap at 12 for stability
        elif memory_gb >= 16:
            workers = min(cpu_count, 8)
        else:
            workers = min(cpu_count, 4)
        
        logger.info(f"Optimal workers for {memory_gb:.1f}GB RAM: {workers}")
        return workers
    
    def monitor_memory_usage(self) -> Dict[str, float]:
        """
        Monitor current memory usage
        
        Returns:
            Dictionary with memory usage statistics
        """
        memory = psutil.virtual_memory()
        
        stats = {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent_used': memory.percent,
            'target_usage_gb': self.target_memory / (1024**3)
        }
        
        return stats
    
    def cleanup_memory(self):
        """Clean up memory and force garbage collection"""
        try:
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Memory cleanup completed")
            
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
    
    def should_use_tiling(self, image_size: Tuple[int, int]) -> bool:
        """
        Determine if image should be processed with tiling
        
        Args:
            image_size: (height, width) of the image
            
        Returns:
            True if tiling should be used
        """
        h, w = image_size
        pixel_count = h * w
        
        # Use tiling for images larger than 4MP
        use_tiling = pixel_count > 4_000_000
        
        if use_tiling:
            logger.info(f"Image size {image_size} requires tiling")
        
        return use_tiling
    
    def get_model_cache_config(self) -> Dict[str, int]:
        """
        Get optimal model caching configuration
        
        Returns:
            Dictionary with cache configuration
        """
        memory_gb = self.total_memory / (1024**3)
        
        if memory_gb >= 32:
            return {
                'max_models': 4,
                'cache_size_mb': 2048,
                'preload_models': True
            }
        elif memory_gb >= 16:
            return {
                'max_models': 2,
                'cache_size_mb': 1024,
                'preload_models': True
            }
        else:
            return {
                'max_models': 1,
                'cache_size_mb': 512,
                'preload_models': False
            }


def get_windows_optimizer() -> WindowsOptimizer:
    """
    Get Windows optimizer instance
    
    Returns:
        WindowsOptimizer instance
    """
    return WindowsOptimizer()


def optimize_for_windows_32gb():
    """
    Apply all Windows 32GB optimizations
    """
    optimizer = get_windows_optimizer()
    
    # Log system information
    stats = optimizer.monitor_memory_usage()
    logger.info(f"System Memory Stats: {stats}")
    
    return optimizer
