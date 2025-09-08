"""
Windows-optimized batch processing script for wallpaper compositing
Optimized for 32GB RAM systems
"""

import argparse
import logging
import time
import os
import sys
from pathlib import Path
from typing import List, Optional
import json
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

try:
    import torch
except ImportError:
    torch = None

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import (
    WallpaperCompositor,
    DebugVisualizer,
    load_image,
    ensure_dir,
    pick_random_room,
    get_first_room,
    get_wallpapers,
    create_output_paths,
    set_seed
)
from pipeline.windows_optimizer import get_windows_optimizer


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration for Windows"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_dir / 'wallpaper_pipeline.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def get_windows_paths():
    """Get Windows-optimized default paths"""
    return {
        'rooms': os.path.join(os.getcwd(), "src", "data", "rooms"),
        'wallpapers': os.path.join(os.getcwd(), "src", "data", "wallpapers"),
        'out': os.path.join(os.getcwd(), "src", "data", "out", "windows_optimized")
    }


def process_single_wallpaper(args):
    """Process a single wallpaper (for parallel processing)"""
    room_image, wallpaper_path, output_paths, compositor_config = args
    
    try:
        # Create compositor for this process
        compositor = WallpaperCompositor(**compositor_config)
        
        # Load wallpaper
        wallpaper_image = load_image(wallpaper_path)
        
        # Process
        result = compositor.composite_wallpaper(room_image, wallpaper_image, output_paths)
        result['wallpaper_path'] = wallpaper_path
        
        return result
        
    except Exception as e:
        logging.error(f"Error processing {wallpaper_path}: {e}")
        return {
            'success': False,
            'error': str(e),
            'wallpaper_path': wallpaper_path
        }


def main():
    """Main function for Windows-optimized batch processing"""
    parser = argparse.ArgumentParser(
        description="Windows-optimized wallpaper compositor with 32GB RAM support"
    )
    
    # Get Windows-optimized default paths
    default_paths = get_windows_paths()
    
    # Input/Output paths
    parser.add_argument(
        "--rooms-dir",
        type=str,
        default=default_paths['rooms'],
        help="Directory containing room images"
    )
    parser.add_argument(
        "--wallpapers-dir", 
        type=str,
        default=default_paths['wallpapers'],
        help="Directory containing wallpaper images"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=default_paths['out'],
        help="Output directory for results"
    )
    
    # Processing options
    parser.add_argument(
        "--num-wallpapers",
        type=int,
        default=10,
        help="Number of wallpapers to process"
    )
    parser.add_argument(
        "--room-pick",
        type=str,
        choices=["first", "random"],
        default="random",
        help="Room selection strategy"
    )
    parser.add_argument(
        "--use-depth",
        action="store_true",
        help="Enable depth estimation"
    )
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="Disable depth estimation"
    )
    
    # Windows-specific optimizations
    parser.add_argument(
        "--windows-optimized",
        action="store_true",
        default=True,
        help="Enable Windows 32GB RAM optimizations"
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=None,
        help="Number of parallel workers (auto-detected if not specified)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for processing (auto-calculated if not specified)"
    )
    parser.add_argument(
        "--memory-limit",
        type=float,
        default=0.8,
        help="Memory usage limit (0.0-1.0)"
    )
    
    # Device options
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Processing device"
    )
    
    # Quality settings
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Object detection confidence threshold"
    )
    parser.add_argument(
        "--feather-radius",
        type=int,
        default=8,
        help="Edge feathering radius"
    )
    
    # Behavior
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use fixed random seed"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save debug visualizations"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Initialize Windows optimizer
    if args.windows_optimized:
        optimizer = get_windows_optimizer()
        logger.info("Windows 32GB RAM optimizations enabled")
        
        # Log system information
        memory_stats = optimizer.monitor_memory_usage()
        logger.info(f"System Memory: {memory_stats['total_gb']:.1f}GB total, "
                   f"{memory_stats['available_gb']:.1f}GB available")
    
    # Set random seed
    if args.deterministic:
        set_seed(42)
        logger.info("Deterministic mode enabled")
    
    # Determine device
    device = args.device
    if device == 'auto':
        if args.windows_optimized and optimizer:
            # Use Windows optimizer for device detection
            if torch and torch.cuda.is_available():
                device = 'cuda'
                logger.info("Using CUDA for GPU acceleration")
            else:
                device = 'cpu'
                logger.info("Using CPU (CUDA not available)")
        else:
            device = 'cpu'
    
    # Calculate optimal parameters
    if args.windows_optimized and optimizer:
        if args.parallel_workers is None:
            args.parallel_workers = optimizer.get_optimal_workers()
        
        if args.batch_size is None:
            # Estimate memory per wallpaper (rough estimate)
            estimated_memory_mb = 500  # MB per wallpaper
            args.batch_size = optimizer.get_optimal_batch_size(estimated_memory_mb)
    
    logger.info(f"Processing configuration:")
    logger.info(f"  Device: {device}")
    logger.info(f"  Parallel workers: {args.parallel_workers}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Memory limit: {args.memory_limit*100:.0f}%")
    
    # Validate paths
    rooms_dir = Path(args.rooms_dir)
    wallpapers_dir = Path(args.wallpapers_dir)
    out_dir = Path(args.out_dir)
    
    if not rooms_dir.exists():
        logger.error(f"Rooms directory not found: {rooms_dir}")
        return 1
    
    if not wallpapers_dir.exists():
        logger.error(f"Wallpapers directory not found: {wallpapers_dir}")
        return 1
    
    # Create output directory
    ensure_dir(out_dir)
    ensure_dir(out_dir / 'composites')
    ensure_dir(out_dir / 'masks')
    ensure_dir(out_dir / 'debug')
    
    # Get room and wallpapers
    room_files = list(rooms_dir.glob("*.jpg")) + list(rooms_dir.glob("*.png"))
    wallpaper_files = list(wallpapers_dir.glob("*.jpg")) + list(wallpapers_dir.glob("*.png"))
    
    if not room_files:
        logger.error("No room images found")
        return 1
    
    if not wallpaper_files:
        logger.error("No wallpaper images found")
        return 1
    
    # Select room
    if args.room_pick == "first":
        room_path = get_first_room(rooms_dir)
    else:
        room_path = pick_random_room(rooms_dir)
    
    room_name = room_path.stem
    logger.info(f"Selected room: {room_name}")
    
    # Select wallpapers
    wallpaper_paths = get_wallpapers(wallpapers_dir, args.num_wallpapers)
    logger.info(f"Selected {len(wallpaper_paths)} wallpapers")
    
    # Load room image
    logger.info("Loading room image...")
    room_image = load_image(room_path)
    logger.info(f"Room image loaded: {room_image.shape}")
    
    # Check if tiling is needed
    if args.windows_optimized and optimizer:
        use_tiling = optimizer.should_use_tiling(room_image.shape[:2])
        if use_tiling:
            logger.info("Large image detected - will use tiling for processing")
    
    # Initialize compositor
    compositor_config = {
        'use_depth': args.use_depth and not args.no_depth,
        'device': device,
        'confidence_threshold': args.confidence_threshold,
        'feather_radius': args.feather_radius,
        'windows_optimized': args.windows_optimized
    }
    
    logger.info("Initializing compositor...")
    compositor = WallpaperCompositor(**compositor_config)
    
    # Process wallpapers
    logger.info("Starting batch processing...")
    start_time = time.time()
    
    results = []
    
    if args.parallel_workers > 1:
        # Parallel processing
        logger.info(f"Processing {len(wallpaper_paths)} wallpapers in parallel with {args.parallel_workers} workers")
        
        # Prepare arguments for parallel processing
        parallel_args = []
        for wallpaper_path in wallpaper_paths:
            wallpaper_name = wallpaper_path.stem
            output_paths = {
                'composite': out_dir / 'composites' / f"{room_name}__{wallpaper_name}.png",
                'wall_mask': out_dir / 'masks' / f"{wallpaper_name}.wall.png",
                'objects_mask': out_dir / 'masks' / f"{wallpaper_name}.objects.png",
                'debug_panel': out_dir / 'debug' / f"{room_name}__{wallpaper_name}.panel.png"
            }
            
            parallel_args.append((room_image, wallpaper_path, output_paths, compositor_config))
        
        # Use ThreadPoolExecutor for I/O bound tasks
        with ThreadPoolExecutor(max_workers=args.parallel_workers) as executor:
            results = list(executor.map(process_single_wallpaper, parallel_args))
    
    else:
        # Sequential processing
        logger.info(f"Processing {len(wallpaper_paths)} wallpapers sequentially")
        
        for i, wallpaper_path in enumerate(wallpaper_paths):
            logger.info(f"Processing wallpaper {i+1}/{len(wallpaper_paths)}: {wallpaper_path.name}")
            
            try:
                # Load wallpaper
                wallpaper_image = load_image(wallpaper_path)
                
                # Create output paths
                wallpaper_name = wallpaper_path.stem
                output_paths = {
                    'composite': out_dir / 'composites' / f"{room_name}__{wallpaper_name}.png",
                    'wall_mask': out_dir / 'masks' / f"{wallpaper_name}.wall.png",
                    'objects_mask': out_dir / 'masks' / f"{wallpaper_name}.objects.png",
                    'debug_panel': out_dir / 'debug' / f"{room_name}__{wallpaper_name}.panel.png"
                }
                
                # Process wallpaper
                result = compositor.composite_wallpaper(room_image, wallpaper_image, output_paths)
                result['wallpaper_path'] = wallpaper_path
                result['room_path'] = room_path
                
                results.append(result)
                
                if result['success']:
                    logger.info(f"Successfully processed: {wallpaper_name}")
                else:
                    logger.error(f"Failed to process: {wallpaper_name}")
                
                # Memory cleanup after each wallpaper
                if args.windows_optimized and optimizer:
                    optimizer.cleanup_memory()
                
            except Exception as e:
                logger.error(f"Error processing {wallpaper_path}: {e}")
                results.append({
                    'success': False,
                    'error': str(e),
                    'wallpaper_path': wallpaper_path,
                    'room_path': room_path
                })
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Generate processing report
    successful = sum(1 for r in results if r.get('success', False))
    failed = len(results) - successful
    
    report = {
        'processing_time_seconds': processing_time,
        'total_wallpapers': len(wallpaper_paths),
        'successful': successful,
        'failed': failed,
        'success_rate': successful / len(wallpaper_paths) if wallpaper_paths else 0,
        'room_name': room_name,
        'device_used': device,
        'windows_optimized': args.windows_optimized,
        'parallel_workers': args.parallel_workers,
        'batch_size': args.batch_size,
        'memory_stats': optimizer.monitor_memory_usage() if args.windows_optimized and optimizer else None,
        'results': results
    }
    
    # Save report
    report_path = out_dir / 'processing_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Log summary
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {processing_time:.2f} seconds")
    logger.info(f"Wallpapers processed: {len(wallpaper_paths)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {successful/len(wallpaper_paths)*100:.1f}%")
    logger.info(f"Average time per wallpaper: {processing_time/len(wallpaper_paths):.2f} seconds")
    logger.info(f"Report saved to: {report_path}")
    
    if args.windows_optimized and optimizer:
        final_memory_stats = optimizer.monitor_memory_usage()
        logger.info(f"Final memory usage: {final_memory_stats['used_gb']:.1f}GB / {final_memory_stats['total_gb']:.1f}GB")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
