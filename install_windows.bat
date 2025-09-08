@echo off
REM Windows installation script for Wallpaper Pipeline
REM Optimized for 32GB RAM systems

echo ========================================
echo Wallpaper Pipeline - Windows Installer
echo Optimized for 32GB RAM Systems
echo ========================================
echo.

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo Checking Python version...
python -c "import sys; print(f'Python {sys.version}')"
python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.10+ is required
    echo Current version is too old
    pause
    exit /b 1
)

echo.
echo Python version check passed!
echo.

REM Check available memory
echo Checking system memory...
python -c "import psutil; mem = psutil.virtual_memory(); print(f'Total RAM: {mem.total / (1024**3):.1f} GB'); print(f'Available RAM: {mem.available / (1024**3):.1f} GB')"

echo.
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing PyTorch with CUDA support...
REM Install PyTorch with CUDA support for Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo WARNING: CUDA installation failed, trying CPU-only version...
    pip install torch torchvision torchaudio
    if errorlevel 1 (
        echo ERROR: Failed to install PyTorch
        pause
        exit /b 1
    )
)

echo.
echo Installing other dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Creating data directories...
if not exist "src\data\rooms" mkdir "src\data\rooms"
if not exist "src\data\wallpapers" mkdir "src\data\wallpapers"
if not exist "src\data\out" mkdir "src\data\out"
if not exist "logs" mkdir "logs"

echo.
echo Testing installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo ========================================
echo Installation completed successfully!
echo ========================================
echo.
echo To use the wallpaper pipeline:
echo 1. Activate the virtual environment: venv\Scripts\activate.bat
echo 2. Run the Windows-optimized batch processor:
echo    python -m src.scripts.run_batch_windows --num-wallpapers 5
echo.
echo For more options, run:
echo    python -m src.scripts.run_batch_windows --help
echo.
echo Windows 32GB RAM optimizations are enabled by default.
echo.

pause
