#!/usr/bin/env python3
"""
Setup script for wallpaper pipeline
"""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install requirements"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_installation():
    """Check if installation is working"""
    print("🔍 Checking installation...")
    try:
        import cv2
        import torch
        import numpy as np
        import matplotlib
        print("✅ Core dependencies working!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def test_pipeline():
    """Test pipeline imports"""
    print("🧪 Testing pipeline imports...")
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from pipeline.io_utils import set_seed
        from pipeline.seg_semantic import SemanticSegmenter
        print("✅ Pipeline imports working!")
        return True
    except ImportError as e:
        print(f"❌ Pipeline import error: {e}")
        return False

def main():
    """Main setup function"""
    print("🎨 Wallpaper Pipeline Setup")
    print("=" * 40)
    
    # Install requirements
    if not install_requirements():
        return 1
    
    # Check installation
    if not check_installation():
        return 1
    
    # Test pipeline
    if not test_pipeline():
        return 1
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run example: python3 example_usage.py")
    print("2. Run batch processing: python -m src.scripts.run_batch --num-wallpapers 5")
    print("3. Run sanity check: python -m src.scripts.sanity_check --num-samples 2")
    
    return 0

if __name__ == "__main__":
    exit(main())
