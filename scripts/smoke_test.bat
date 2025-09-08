@echo off
REM Smoke test script for Windows
REM Processes 1 room × 2 wallpapers with small images

echo Starting wallpaper pipeline smoke test...

REM Check if we're in the right directory
if not exist "src\scripts\run_batch.py" (
    echo Error: Please run this script from the wallpaper_pipeline root directory
    exit /b 1
)

REM Create test directories if they don't exist
if not exist "src\data\rooms" mkdir "src\data\rooms"
if not exist "src\data\wallpapers" mkdir "src\data\wallpapers"
if not exist "src\data\out\smoke_test" mkdir "src\data\out\smoke_test"

REM Check if we have test data
dir "src\data\rooms\*" >nul 2>&1
if errorlevel 1 (
    echo Warning: No room images found in src\data\rooms
    echo Please add at least one room image to run the smoke test
    exit /b 1
)

dir "src\data\wallpapers\*" >nul 2>&1
if errorlevel 1 (
    echo Warning: No wallpaper images found in src\data\wallpapers
    echo Please add at least two wallpaper images to run the smoke test
    exit /b 1
)

REM Run smoke test with Windows-optimized settings
echo Running smoke test with 1 room and 2 wallpapers...
python src\scripts\run_batch_windows.py ^
    --rooms-dir src\data\rooms ^
    --wallpapers-dir src\data\wallpapers ^
    --out-dir src\data\out\smoke_test ^
    --num-wallpapers 2 ^
    --room-pick first ^
    --no-depth ^
    --windows-optimized ^
    --device auto ^
    --confidence-threshold 0.3 ^
    --save-debug ^
    --verbose

if errorlevel 1 (
    echo Smoke test failed!
    exit /b 1
)

echo Smoke test completed successfully!
echo Check src\data\out\smoke_test for results
