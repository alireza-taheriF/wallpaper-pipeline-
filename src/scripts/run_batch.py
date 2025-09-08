"""
Main batch processing script for wallpaper compositing
"""

import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional
import json

from ..pipeline import (
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


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Production-grade wallpaper compositor with precise wall detection and object separation"
    )
    
    # Input/Output paths
    parser.add_argument(
        "--rooms-dir",
        type=str,
        default="/Users/alireza/Documents/wallpaper_pipeline/src/data/rooms",
        help="Directory containing room images"
    )
    parser.add_argument(
        "--wallpapers-dir", 
        type=str,
        default="/Users/alireza/Documents/wallpaper_pipeline/src/data/wallpapers",
        help="Directory containing wallpaper images"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/Users/alireza/Documents/wallpaper_pipeline/src/data/out",
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
        default="first",
        help="How to pick room: 'first' (deterministic) or 'random'"
    )
    parser.add_argument(
        "--use-depth",
        action="store_true",
        default=True,
        help="Use depth estimation for wall refinement"
    )
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="Disable depth estimation (faster but less accurate)"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic processing (set random seeds)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic processing"
    )
    
    # Model options
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run models on"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for object detection"
    )
    
    # Output options
    parser.add_argument(
        "--save-debug",
        action="store_true",
        default=True,
        help="Save debug panels"
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        default=True,
        help="Save intermediate masks"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Set random seed if deterministic
    if args.deterministic:
        set_seed(args.seed)
        logger.info(f"Set random seed to {args.seed} for deterministic processing")
    
    # Determine depth usage
    use_depth = args.use_depth and not args.no_depth
    
    # Validate input directories
    rooms_dir = Path(args.rooms_dir)
    wallpapers_dir = Path(args.wallpapers_dir)
    out_dir = Path(args.out_dir)
    
    if not rooms_dir.exists():
        logger.error(f"Rooms directory does not exist: {rooms_dir}")
        return 1
    
    if not wallpapers_dir.exists():
        logger.error(f"Wallpapers directory does not exist: {wallpapers_dir}")
        return 1
    
    # Create output directories
    ensure_dir(out_dir)
    ensure_dir(out_dir / "composites")
    ensure_dir(out_dir / "masks")
    ensure_dir(out_dir / "debug")
    
    logger.info("Starting wallpaper compositing pipeline...")
    logger.info(f"Rooms directory: {rooms_dir}")
    logger.info(f"Wallpapers directory: {wallpapers_dir}")
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Number of wallpapers: {args.num_wallpapers}")
    logger.info(f"Room selection: {args.room_pick}")
    logger.info(f"Use depth estimation: {use_depth}")
    logger.info(f"Device: {args.device}")
    
    try:
        # Pick room
        if args.room_pick == "random":
            room_path = pick_random_room(rooms_dir, args.seed if args.deterministic else None)
        else:
            room_path = get_first_room(rooms_dir)
        
        if room_path is None:
            logger.error("No room images found")
            return 1
        
        logger.info(f"Selected room: {room_path.name}")
        
        # Get wallpapers
        wallpaper_paths = get_wallpapers(wallpapers_dir, args.num_wallpapers)
        if not wallpaper_paths:
            logger.error("No wallpaper images found")
            return 1
        
        logger.info(f"Found {len(wallpaper_paths)} wallpapers")
        
        # Initialize compositor
        logger.info("Initializing wallpaper compositor...")
        compositor = WallpaperCompositor(
            use_depth=use_depth,
            device=args.device,
            confidence_threshold=args.confidence_threshold
        )
        
        # Initialize debug visualizer
        debug_visualizer = DebugVisualizer()
        
        # Process wallpapers
        logger.info("Starting batch processing...")
        start_time = time.time()
        
        results = compositor.process_batch(
            room_path=room_path,
            wallpaper_paths=wallpaper_paths,
            output_dir=out_dir,
            room_name=room_path.stem
        )
        
        processing_time = time.time() - start_time
        
        # Create debug panels for successful results
        if args.save_debug:
            logger.info("Creating debug panels...")
            for result in results:
                if result.get('success', False):
                    try:
                        # Load images for debug panel
                        original_image = load_image(result['room_path'])
                        composite_image = result['composite_image']
                        
                        # Create debug panel
                        debug_panel = debug_visualizer.create_debug_panel(
                            original_image=original_image,
                            wall_mask=result['wall_mask'],
                            objects_mask=result['objects_mask'],
                            polygon=result['polygon'],
                            composite_image=composite_image,
                            metadata=result.get('metadata', {})
                        )
                        
                        # Save debug panel
                        wallpaper_name = result['wallpaper_path'].stem
                        debug_path = out_dir / "debug" / f"{room_path.stem}__{wallpaper_name}.panel.png"
                        debug_visualizer.save_debug_panel(debug_panel, debug_path)
                        
                    except Exception as e:
                        logger.warning(f"Failed to create debug panel for {result['wallpaper_path'].name}: {e}")
        
        # Get processing statistics
        stats = compositor.get_processing_stats(results)
        
        # Print results
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total processing time: {processing_time:.2f} seconds")
        logger.info(f"Total wallpapers: {stats['total_wallpapers']}")
        logger.info(f"Successful: {stats['successful']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Success rate: {stats['success_rate']:.1%}")
        logger.info(f"Average wall area ratio: {stats['avg_wall_area_ratio']:.1%}")
        logger.info(f"Average objects area ratio: {stats['avg_objects_area_ratio']:.1%}")
        logger.info(f"Average polygon vertices: {stats['avg_polygon_vertices']:.1f}")
        logger.info(f"Fallback rate: {stats['fallback_rate']:.1%}")
        
        # Save processing report
        report = {
            'processing_time': processing_time,
            'room_path': str(room_path),
            'wallpaper_count': len(wallpaper_paths),
            'statistics': stats,
            'results': []
        }
        
        for result in results:
            result_summary = {
                'wallpaper_path': str(result['wallpaper_path']),
                'success': result.get('success', False),
                'metadata': result.get('metadata', {}),
                'error': result.get('error', None)
            }
            report['results'].append(result_summary)
        
        report_path = out_dir / "processing_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Processing report saved: {report_path}")
        
        # List successful outputs
        if stats['successful'] > 0:
            logger.info("\nSuccessful outputs:")
            for result in results:
                if result.get('success', False):
                    wallpaper_name = result['wallpaper_path'].stem
                    composite_path = out_dir / "composites" / f"{room_path.stem}__{wallpaper_name}.png"
                    logger.info(f"  - {composite_path}")
        
        # List failed outputs
        if stats['failed'] > 0:
            logger.info("\nFailed outputs:")
            for result in results:
                if not result.get('success', False):
                    wallpaper_name = result['wallpaper_path'].stem
                    error = result.get('error', 'Unknown error')
                    logger.info(f"  - {wallpaper_name}: {error}")
        
        logger.info("=" * 60)
        
        return 0 if stats['failed'] == 0 else 1
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
