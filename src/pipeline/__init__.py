"""
Wallpaper Pipeline - Production-grade wallpaper compositor
"""

__version__ = "0.1.0"
__author__ = "ML Engineer"

from .io_utils import load_image, save_image, ensure_dir, get_image_files, pick_random_room, get_first_room, get_wallpapers, create_output_paths, set_seed
from .seg_semantic import SemanticSegmenter
from .seg_instances import InstanceSegmenter
from .depth_plane import DepthPlaneRefiner
from .wall_polygon import WallPolygonizer
from .illumination import IlluminationTransfer
from .compositor import WallpaperCompositor
from .visual_debug import DebugVisualizer

__all__ = [
    "load_image",
    "save_image", 
    "ensure_dir",
    "get_image_files",
    "pick_random_room",
    "get_first_room", 
    "get_wallpapers",
    "create_output_paths",
    "set_seed",
    "SemanticSegmenter",
    "InstanceSegmenter",
    "DepthPlaneRefiner",
    "WallPolygonizer",
    "IlluminationTransfer",
    "WallpaperCompositor",
    "DebugVisualizer",
]
