#!/usr/bin/env python3
"""
Example usage of the wallpaper pipeline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Example usage"""
    print("🎨 Wallpaper Pipeline - Example Usage")
    print("=" * 50)
    
    # Check if we can import basic modules
    try:
        from pipeline.io_utils import set_seed, get_image_files
        print("✓ Basic imports working")
        
        # Set seed for reproducibility
        set_seed(42)
        print("✓ Random seed set")
        
        # Check data directories
        rooms_dir = Path("src/data/rooms")
        wallpapers_dir = Path("src/data/wallpapers")
        
        if rooms_dir.exists():
            room_files = get_image_files(rooms_dir)
            print(f"✓ Found {len(room_files)} room images")
        else:
            print("⚠️  Rooms directory not found")
        
        if wallpapers_dir.exists():
            wallpaper_files = get_image_files(wallpapers_dir)
            print(f"✓ Found {len(wallpaper_files)} wallpaper images")
        else:
            print("⚠️  Wallpapers directory not found")
        
        print("\n📋 To run the full pipeline:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run batch processing:")
        print("   python -m src.scripts.run_batch --num-wallpapers 5")
        print("3. Run sanity check:")
        print("   python -m src.scripts.sanity_check --num-samples 2")
        
        print("\n🎯 Pipeline Features:")
        print("- Semantic segmentation for wall detection")
        print("- Instance segmentation for object masking") 
        print("- Depth estimation for wall refinement")
        print("- Polygonization and homography estimation")
        print("- Illumination transfer for realistic compositing")
        print("- Debug visualization and quality validation")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
