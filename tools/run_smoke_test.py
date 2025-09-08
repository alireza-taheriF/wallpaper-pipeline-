# tools/run_smoke_test.py
import json, os, sys
from pathlib import Path
import logging

# Add src to path
sys.path.append('src')

from utils.io_any import imread_any  # sanity import
from pipeline.compositor import WallpaperCompositor

# Setup logging
OUT = Path("src/data/out/reports")
OUT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    filename=OUT / "smoke_test.log",
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

NAMES = [
 "download_14b8c046-93f8-4822-801e-2334d9dca696.webp",
 "download_1569ffce-8b33-4a0f-a73e-5267e57fc61d.webp",
 "download_1e0c70f6-e691-4e21-95f6-336d60f9619d.webp",
]

def run_single(room_path, wallpaper_path, output_dir):
    """Run pipeline for single wallpaper"""
    compositor = WallpaperCompositor(use_depth=False, device='cpu')
    
    # Load images
    room_image = imread_any(room_path)
    wallpaper_image = imread_any(wallpaper_path)
    
    # Create output paths
    wallpaper_name = Path(wallpaper_path).stem
    room_name = Path(room_path).stem
    
    output_paths = {
        'composite': output_dir / 'composites' / f"{room_name}__{wallpaper_name}.png",
        'wall_mask': output_dir / 'masks' / f"{wallpaper_name}.wall.png",
        'objects_mask': output_dir / 'masks' / f"{wallpaper_name}.objects.png",
        'debug_panel': output_dir / 'debug' / f"{room_name}__{wallpaper_name}.panel.png"
    }
    
    # Process wallpaper
    result = compositor.composite_wallpaper(room_image, wallpaper_image, output_paths)
    result['wallpaper_path'] = str(wallpaper_path)
    result['room_path'] = str(room_path)
    
    return result

# Main execution
room_path = Path("src/data/rooms/room10.jpg")
wallpaper_dir = Path("src/data/wallpapers")
output_dir = Path("src/data/out")

# Create output directories
(output_dir / 'composites').mkdir(parents=True, exist_ok=True)
(output_dir / 'masks').mkdir(parents=True, exist_ok=True)
(output_dir / 'debug').mkdir(parents=True, exist_ok=True)

results = []
for name in NAMES:
    wallpaper_path = wallpaper_dir / name
    if wallpaper_path.exists():
        logger.info(f"Processing {name}...")
        try:
            res = run_single(room_path, wallpaper_path, output_dir)
            results.append(res)
            logger.info(f"Successfully processed {name}")
        except Exception as e:
            logger.error(f"Failed to process {name}: {e}")
            results.append({'success': False, 'error': str(e), 'wallpaper_path': str(wallpaper_path)})
    else:
        logger.warning(f"Wallpaper not found: {name}")
        results.append({'success': False, 'error': 'File not found', 'wallpaper_path': str(wallpaper_path)})

# Save results (convert numpy arrays to lists for JSON serialization)
def convert_numpy(obj):
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    else:
        return obj

results_serializable = convert_numpy(results)
Path(OUT / "smoke_run.json").write_text(json.dumps(results_serializable, indent=2))
print("DONE smoke.")
