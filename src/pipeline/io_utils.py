"""
I/O utilities for image loading, saving, and directory management
"""

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_image(path: Union[str, Path], target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load image and convert to RGB format using robust IO
    
    Args:
        path: Path to image file
        target_size: Optional (width, height) to resize image
        
    Returns:
        RGB image as numpy array (H, W, 3)
    """
    from .utils.io_any import imread_any
    
    # Use robust image reader
    image = imread_any(path)
    
    # Resize if requested
    if target_size is not None:
        width, height = target_size
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    
    return image


def save_image(image: np.ndarray, path: Union[str, Path], quality: int = 95) -> None:
    """
    Save image to file
    
    Args:
        image: Image array (H, W, 3) or (H, W) for grayscale
        path: Output path
        quality: JPEG quality (1-100)
    """
    path = Path(path)
    ensure_dir(path.parent)
    
    # Convert to PIL Image
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    pil_image = Image.fromarray(image)
    
    # Save with appropriate format
    if path.suffix.lower() in ['.jpg', '.jpeg']:
        pil_image.save(path, 'JPEG', quality=quality, optimize=True)
    elif path.suffix.lower() == '.png':
        pil_image.save(path, 'PNG', optimize=True)
    else:
        pil_image.save(path)


def ensure_dir(path: Union[str, Path]) -> None:
    """Ensure directory exists, create if it doesn't"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def get_image_files(directory: Union[str, Path], extensions: List[str] = None) -> List[Path]:
    """
    Get all image files from directory
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to include
        
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    directory = Path(directory)
    if not directory.exists():
        return []
    
    image_files = []
    for ext in extensions:
        image_files.extend(directory.glob(f'*{ext}'))
        image_files.extend(directory.glob(f'*{ext.upper()}'))
    
    return sorted(image_files)


def pick_random_room(rooms_dir: Union[str, Path], seed: Optional[int] = None) -> Optional[Path]:
    """
    Pick a random room image
    
    Args:
        rooms_dir: Directory containing room images
        seed: Random seed for reproducibility
        
    Returns:
        Path to selected room image
    """
    if seed is not None:
        set_seed(seed)
    
    room_files = get_image_files(rooms_dir)
    if not room_files:
        return None
    
    return random.choice(room_files)


def get_first_room(rooms_dir: Union[str, Path]) -> Optional[Path]:
    """
    Get the first room image (deterministic)
    
    Args:
        rooms_dir: Directory containing room images
        
    Returns:
        Path to first room image
    """
    room_files = get_image_files(rooms_dir)
    return room_files[0] if room_files else None


def get_wallpapers(wallpapers_dir: Union[str, Path], num_wallpapers: int = 10) -> List[Path]:
    """
    Get wallpaper images (first N deterministically)
    
    Args:
        wallpapers_dir: Directory containing wallpaper images
        num_wallpapers: Number of wallpapers to return
        
    Returns:
        List of wallpaper file paths
    """
    wallpaper_files = get_image_files(wallpapers_dir)
    return wallpaper_files[:num_wallpapers]


def create_output_paths(room_name: str, wallpaper_name: str, out_dir: Union[str, Path]) -> dict:
    """
    Create standardized output paths
    
    Args:
        room_name: Name of room (without extension)
        wallpaper_name: Name of wallpaper (without extension)
        out_dir: Base output directory
        
    Returns:
        Dictionary with output paths
    """
    out_dir = Path(out_dir)
    
    return {
        'composite': out_dir / 'composites' / f"{room_name}__{wallpaper_name}.png",
        'wall_mask': out_dir / 'masks' / f"{wallpaper_name}.wall.png",
        'objects_mask': out_dir / 'masks' / f"{wallpaper_name}.objects.png",
        'debug_panel': out_dir / 'debug' / f"{room_name}__{wallpaper_name}.panel.png",
    }
