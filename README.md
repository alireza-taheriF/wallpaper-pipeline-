# Wallpaper Pipeline — Windows-Optimized (32GB RAM)

Production-grade wallpaper compositor with precise wall detection, object separation, and Windows 32GB RAM optimizations. Automatically composites wallpapers onto room images using state-of-the-art computer vision models.

## Key Features

- **Precise Wall Detection**: ADE20K-trained semantic segmentation (DeepLabV3+ with EfficientNet-B3/B7)
- **Object Segmentation**: Mask R-CNN for indoor object detection and masking
- **Optional Depth Estimation**: Intel DPT-Large for wall refinement and perspective correction
- **Windows Optimizations**: Parallel workers, memory management, caching, and tiling for large images
- **Production Ready**: Comprehensive error handling, debug visualizations, and batch processing

## Quick Start — Windows (Recommended, 32GB RAM)

### Prerequisites
- Python 3.10+ (tested on Windows 10/11)
- 32GB RAM recommended for optimal performance
- CUDA 11.8+ (optional, for GPU acceleration)

### Installation

1. **Clone and install dependencies:**
   ```cmd
   git clone <repository-url>
   cd wallpaper_pipeline
   pip install -r requirements.txt
   ```

2. **Optional CUDA support:**
   ```cmd
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **One-click smoke test:**
   ```cmd
   scripts\smoke_test.bat
   ```

### Golden Path Command (10 wallpapers, depth enabled, Windows-optimized)

```cmd
python -m src.scripts.run_batch_windows ^
  --rooms-dir "src/data/rooms" ^
  --wallpapers-dir "src/data/wallpapers" ^
  --out-dir "src/data/out" ^
  --num-wallpapers 10 ^
  --room-pick first ^
  --use-depth ^
  --windows-optimized ^
  --memory-limit 0.8 ^
  --feather-radius 8 ^
  --device auto ^
  --save-debug --verbose
```

## macOS/Linux (Legacy)

### Quick Start
```bash
# Make smoke test executable
chmod +x scripts/smoke_test.sh

# Run smoke test
./scripts/smoke_test.sh

# Full processing
python3 -m src.scripts.run_batch \
  --rooms-dir src/data/rooms \
  --wallpapers-dir src/data/wallpapers \
  --out-dir src/data/out \
  --num-wallpapers 10 \
  --use-depth \
  --device auto \
  --save-debug --save-masks --verbose
```

### Apple Silicon Notes
- Use `--device mps` for M1/M2 Macs
- Requires PyTorch >= 2.1, torchvision >= 0.16
- MPS limitations: no FP16, smaller batch sizes recommended

## CLI Reference

### `run_batch_windows.py` (Windows-Optimized)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--rooms-dir` | str | `src/data/rooms` | Directory containing room images |
| `--wallpapers-dir` | str | `src/data/wallpapers` | Directory containing wallpaper images |
| `--out-dir` | str | `src/data/out/windows_optimized` | Output directory for results |
| `--num-wallpapers` | int | `10` | Number of wallpapers to process |
| `--room-pick` | str | `random` | Room selection: `first` or `random` |
| `--use-depth` | flag | `False` | Enable depth estimation |
| `--no-depth` | flag | `False` | Disable depth estimation |
| `--windows-optimized` | flag | `True` | Enable Windows 32GB RAM optimizations |
| `--parallel-workers` | int | `None` | Number of parallel workers (auto-detected) |
| `--batch-size` | int | `None` | Batch size (auto-calculated) |
| `--memory-limit` | float | `0.8` | Memory usage limit (0.0-1.0) |
| `--device` | str | `auto` | Processing device: `auto`, `cpu`, `cuda` |
| `--confidence-threshold` | float | `0.3` | Object detection confidence threshold |
| `--feather-radius` | int | `8` | Edge feathering radius |
| `--deterministic` | flag | `False` | Use fixed random seed |
| `--verbose` | flag | `False` | Enable verbose logging |
| `--save-debug` | flag | `False` | Save debug visualizations |

### `run_batch.py` (Cross-Platform)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--rooms-dir` | str | `/Users/.../src/data/rooms` | Directory containing room images |
| `--wallpapers-dir` | str | `/Users/.../src/data/wallpapers` | Directory containing wallpaper images |
| `--out-dir` | str | `/Users/.../src/data/out` | Output directory for results |
| `--num-wallpapers` | int | `10` | Number of wallpapers to process |
| `--room-pick` | str | `first` | Room selection: `first` or `random` |
| `--use-depth` | flag | `True` | Use depth estimation for wall refinement |
| `--no-depth` | flag | `False` | Disable depth estimation |
| `--deterministic` | flag | `True` | Use deterministic processing |
| `--seed` | int | `42` | Random seed for deterministic processing |
| `--device` | str | `auto` | Device: `auto`, `cpu`, `cuda`, `mps` |
| `--confidence-threshold` | float | `0.5` | Confidence threshold for object detection |
| `--save-debug` | flag | `True` | Save debug panels |
| `--save-masks` | flag | `True` | Save intermediate masks |
| `--verbose` | flag | `False` | Enable verbose logging |

### `sanity_check.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--rooms-dir` | str | `/Users/.../src/data/rooms` | Directory containing room images |
| `--output-dir` | str | `/Users/.../src/data/out/sanity` | Output directory for sanity check results |
| `--num-samples` | int | `3` | Number of room samples to check |
| `--device` | str | `auto` | Device: `auto`, `cpu`, `cuda`, `mps` |
| `--verbose` | flag | `False` | Enable verbose logging |

## Models & Weights

### Semantic Segmentation (ADE20K)
- **Model**: DeepLabV3+ (default), UNet, FPN
- **Encoder**: EfficientNet-B3 (default), EfficientNet-B7 (Windows-optimized)
- **Classes**: 150 ADE20K classes with wall/ceiling/floor detection
- **Weights**: Auto-downloaded on first run (internet required)

### Instance Segmentation
- **Model**: `torchvision.models.detection.maskrcnn_resnet50_fpn`
- **Classes**: 80 COCO classes, filtered for indoor objects
- **Weights**: Pre-trained on COCO dataset

### Depth Estimation
- **Model**: `Intel/dpt-large` (Hugging Face)
- **Purpose**: Wall refinement and perspective correction
- **Weights**: Auto-downloaded on first run (internet required)

## Output & Naming

### Directory Structure
```
src/data/out/
├── composites/          # Final wallpaper composites
├── masks/              # Wall and object masks
├── debug/              # Debug visualizations
└── processing_report.json  # Processing statistics
```

### File Naming
- **Composites**: `{room_name}__{wallpaper_name}.png`
- **Wall Masks**: `{wallpaper_name}.wall.png`
- **Object Masks**: `{wallpaper_name}.objects.png`
- **Debug Panels**: `{room_name}__{wallpaper_name}.panel.png`

## Configuration Precedence

**Defaults in code < config.yaml < CLI flags (CLI wins)**

CLI arguments override all other configuration sources.

## Quality Control & Troubleshooting

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **Wall mask too small** | Increase `--confidence-threshold` (0.3-0.7), improve source image quality |
| **Polygonization fallback** | Check wall mask quality, try `--use-depth` for refinement |
| **CUDA OOM** | Use `--device cpu`, reduce `--parallel-workers`, or `--memory-limit 0.6` |
| **Windows path issues** | Use forward slashes: `src/data/rooms`, avoid long paths |
| **No objects detected** | Lower `--confidence-threshold` to 0.1-0.3 |
| **Poor depth estimation** | Try `--no-depth` for faster processing without depth refinement |

### Performance Tuning

**Windows 32GB RAM:**
- `--windows-optimized` (enabled by default)
- `--memory-limit 0.8` (80% of available RAM)
- `--parallel-workers` auto-detected (typically 6-8 workers)
- `--batch-size` auto-calculated based on available memory

**GPU Acceleration:**
- Install CUDA 11.8+ and compatible PyTorch
- Use `--device auto` (automatically selects CUDA if available)
- Monitor GPU memory usage with `nvidia-smi`

## Programmatic Usage

```python
from src.pipeline import WallpaperCompositor
from src.pipeline.io_utils import load_image, save_image

# Initialize compositor
compositor = WallpaperCompositor(
    use_depth=True,
    device='auto',
    confidence_threshold=0.5,
    feather_radius=8,  # Override default of 5 for smoother edges
    windows_optimized=True
)

# Load images
room = load_image("src/data/rooms/room1.jpg")
wallpaper = load_image("src/data/wallpapers/wp1.jpg")

# Process wallpaper
result = compositor.composite_wallpaper(room, wallpaper)

# Save result
if result['success']:
    save_image(result['composite_image'], "src/data/out/composites/room1__wp1.png")
    print(f"Wall area ratio: {result['metadata']['wall_area_ratio']:.1%}")
    print(f"Objects area ratio: {result['metadata']['objects_area_ratio']:.1%}")
else:
    print(f"Processing failed: {result.get('error', 'Unknown error')}")
```

## Benchmarks

Performance depends on:
- **GPU/VRAM**: CUDA acceleration provides 3-5x speedup
- **Image Resolution**: Larger images require more memory and processing time
- **Model Selection**: EfficientNet-B7 (Windows) vs EfficientNet-B3 (default)
- **Parallel Processing**: Windows optimization enables multi-worker processing

Typical processing times (Windows 32GB RAM, RTX 3080):
- 1 wallpaper (1920x1080): ~15-30 seconds
- 10 wallpapers (batch): ~3-5 minutes
- With depth estimation: +50% processing time

## Large-Scale Test Guide (Room Switch Every 50)

For large-scale testing scenarios where you need thousands of composites (e.g., 5000+ outputs), use the automated room-switching scripts that change the room every 50 wallpapers. This approach ensures variety in your dataset while maintaining efficient processing.

### Purpose
- **Variety**: Each room gets exactly 50 wallpapers before switching to the next room
- **Scalability**: Process thousands of composites systematically
- **Organization**: Results are organized by room in separate directories
- **Efficiency**: Leverages Windows optimizations for maximum throughput

### PowerShell Script (Recommended)

```powershell
# Basic usage - processes 5000 total composites (100 rooms × 50 wallpapers each)
.\scripts\per_room_50.ps1

# Custom target - processes 1000 total composites
.\scripts\per_room_50.ps1 -TotalTarget 1000

# Custom parameters
.\scripts\per_room_50.ps1 -TotalTarget 2000 -WallpapersPerRoom 25 -OutputBaseDir "src/data/out/custom_test"
```

**Script Features:**
- Automatically cycles through all available rooms
- Processes exactly 50 wallpapers per room (configurable)
- Uses optimal settings: `--use-depth --windows-optimized --memory-limit 0.8 --device auto --deterministic --save-debug --verbose`
- Creates organized output structure: `src/data/out/per_room_50/<room_name>/`
- Provides detailed logging with timestamps
- Handles errors gracefully and continues processing

### CMD Batch Script (Alternative)

For users who cannot run PowerShell:

```cmd
REM Basic usage
scripts\per_room_50.cmd

REM Custom target
scripts\per_room_50.cmd --total-target 1000

REM Custom parameters
scripts\per_room_50.cmd --total-target 2000 --wallpapers-per-room 25 --output-dir "src/data/out/custom_test"
```

### Output Structure

Results are organized by room in the following structure:
```
src/data/out/per_room_50/
├── room3/
│   ├── composites/          # 50 wallpaper composites for room3
│   ├── masks/              # Wall and object masks
│   ├── debug/              # Debug visualizations
│   └── processing_report.json
├── room4/
│   ├── composites/          # 50 wallpaper composites for room4
│   ├── masks/
│   ├── debug/
│   └── processing_report.json
└── ... (continues for all rooms)
```


### Performance Expectations

For large-scale runs (5000+ composites):
- **Processing time**: ~4-8 hours (depending on hardware)
- **Storage**: ~50-100 GB (with debug images enabled)
- **Memory usage**: Optimized for 32GB RAM systems
- **GPU utilization**: Automatic CUDA detection and usage

### Tips for Large-Scale Testing

1. **Start small**: Test with `--total-target 100` first
2. **Monitor disk space**: Large runs generate significant output
3. **Use SSD storage**: Faster I/O for better performance
4. **Check logs**: Scripts provide detailed progress information
5. **Resume capability**: Scripts can be re-run to continue from where they left off

## License & External Dependencies

- **MIT License** for this project
- **PyTorch/Torchvision**: BSD License
- **Hugging Face Models**: Various licenses (check individual model pages)
- **ADE20K Dataset**: MIT License

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Run `scripts\smoke_test.bat` to verify installation
3. Use `--verbose` flag for detailed logging
4. Check `src/data/out/processing_report.json` for processing statistics
