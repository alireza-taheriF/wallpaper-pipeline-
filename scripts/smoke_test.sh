#!/bin/bash
# Smoke test script for macOS/Linux
# Processes 1 room × 2 wallpapers with small images

set -e

echo "Starting wallpaper pipeline smoke test..."

# Check if we're in the right directory
if [ ! -f "src/scripts/run_batch.py" ]; then
    echo "Error: Please run this script from the wallpaper_pipeline root directory"
    exit 1
fi

# Create test directories if they don't exist
mkdir -p src/data/rooms
mkdir -p src/data/wallpapers
mkdir -p src/data/out/smoke_test

# Check if we have test data
if [ ! -d "src/data/rooms" ] || [ -z "$(ls -A src/data/rooms 2>/dev/null)" ]; then
    echo "Warning: No room images found in src/data/rooms"
    echo "Please add at least one room image to run the smoke test"
    exit 1
fi

if [ ! -d "src/data/wallpapers" ] || [ -z "$(ls -A src/data/wallpapers 2>/dev/null)" ]; then
    echo "Warning: No wallpaper images found in src/data/wallpapers"
    echo "Please add at least two wallpaper images to run the smoke test"
    exit 1
fi

# Run smoke test with minimal settings
echo "Running smoke test with 1 room and 2 wallpapers..."
python3 src/scripts/run_batch.py \
    --rooms-dir src/data/rooms \
    --wallpapers-dir src/data/wallpapers \
    --out-dir src/data/out/smoke_test \
    --num-wallpapers 2 \
    --room-pick first \
    --no-depth \
    --device cpu \
    --confidence-threshold 0.3 \
    --save-debug \
    --save-masks \
    --verbose

echo "Smoke test completed successfully!"
echo "Check src/data/out/smoke_test for results"
