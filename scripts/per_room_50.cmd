@echo off
REM Large-Scale Test Script: Room Switch Every 50 Wallpapers
REM This script processes 50 wallpapers per room, cycling through all available rooms
REM for large-scale testing (e.g., 5000 total composites = 100 rooms × 50 wallpapers each)

setlocal enabledelayedexpansion

REM Default parameters
set "ROOMS_DIR=src/data/rooms"
set "WALLPAPERS_DIR=src/data/wallpapers"
set "OUTPUT_BASE_DIR=src/data/out/per_room_50"
set "WALLPAPERS_PER_ROOM=50"
set "TOTAL_TARGET=5000"

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :start_processing
if "%~1"=="--help" goto :show_help
if "%~1"=="-h" goto :show_help
if "%~1"=="--rooms-dir" (
    set "ROOMS_DIR=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--wallpapers-dir" (
    set "WALLPAPERS_DIR=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--output-dir" (
    set "OUTPUT_BASE_DIR=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--wallpapers-per-room" (
    set "WALLPAPERS_PER_ROOM=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--total-target" (
    set "TOTAL_TARGET=%~2"
    shift
    shift
    goto :parse_args
)
echo Unknown parameter: %~1
goto :show_help

:show_help
echo Large-Scale Test Script: Room Switch Every 50 Wallpapers
echo.
echo Usage: scripts\per_room_50.cmd [options]
echo.
echo Options:
echo   --rooms-dir ^<path^>        Directory containing room images (default: src/data/rooms)
echo   --wallpapers-dir ^<path^>   Directory containing wallpaper images (default: src/data/wallpapers)
echo   --output-dir ^<path^>       Base output directory (default: src/data/out/per_room_50)
echo   --wallpapers-per-room ^<n^> Number of wallpapers per room (default: 50)
echo   --total-target ^<n^>        Total target composites (default: 5000)
echo   --help                      Show this help message
echo.
echo Example:
echo   scripts\per_room_50.cmd --total-target 1000
echo.
exit /b 0

:start_processing
echo [%date% %time%] [INFO] Starting Large-Scale Test: Room Switch Every %WALLPAPERS_PER_ROOM% Wallpapers
echo [%date% %time%] [INFO] Target: %TOTAL_TARGET% total composites
echo [%date% %time%] [INFO] Wallpapers per room: %WALLPAPERS_PER_ROOM%

REM Validate input directories
if not exist "%ROOMS_DIR%" (
    echo [%date% %time%] [ERROR] Rooms directory not found: %ROOMS_DIR%
    exit /b 1
)

if not exist "%WALLPAPERS_DIR%" (
    echo [%date% %time%] [ERROR] Wallpapers directory not found: %WALLPAPERS_DIR%
    exit /b 1
)

REM Create temporary rooms directory
set "TEMP_ROOMS_DIR=src/data/rooms_tmp"
if exist "%TEMP_ROOMS_DIR%" rmdir /s /q "%TEMP_ROOMS_DIR%"
mkdir "%TEMP_ROOMS_DIR%"
echo [%date% %time%] [SUCCESS] Created temporary directory: %TEMP_ROOMS_DIR%

REM Create base output directory
if not exist "%OUTPUT_BASE_DIR%" mkdir "%OUTPUT_BASE_DIR%"
echo [%date% %time%] [SUCCESS] Created output directory: %OUTPUT_BASE_DIR%

REM Get room files (support common image formats)
set "ROOM_COUNT=0"
for %%f in ("%ROOMS_DIR%\*.jpg" "%ROOMS_DIR%\*.jpeg" "%ROOMS_DIR%\*.png" "%ROOMS_DIR%\*.bmp" "%ROOMS_DIR%\*.tiff") do (
    set /a ROOM_COUNT+=1
    set "ROOM_!ROOM_COUNT!=%%~nf"
    set "ROOM_PATH_!ROOM_COUNT!=%%f"
)

if %ROOM_COUNT%==0 (
    echo [%date% %time%] [ERROR] No room images found in: %ROOMS_DIR%
    exit /b 1
)

echo [%date% %time%] [SUCCESS] Found %ROOM_COUNT% room images

REM Calculate cycles needed
set /a ROOMS_PER_CYCLE=%ROOM_COUNT%
set /a TOTAL_CYCLES=(%TOTAL_TARGET% + %WALLPAPERS_PER_ROOM% * %ROOMS_PER_CYCLE% - 1) / (%WALLPAPERS_PER_ROOM% * %ROOMS_PER_CYCLE%)
set /a ACTUAL_TOTAL=%TOTAL_CYCLES% * %WALLPAPERS_PER_ROOM% * %ROOMS_PER_CYCLE%

echo [%date% %time%] [INFO] Will process %TOTAL_CYCLES% cycles of all %ROOMS_PER_CYCLE% rooms
echo [%date% %time%] [INFO] Expected total composites: %ACTUAL_TOTAL%

set "TOTAL_PROCESSED=0"
set "CYCLE_NUMBER=1"

:cycle_loop
if %TOTAL_PROCESSED% geq %TOTAL_TARGET% goto :cleanup
echo [%date% %time%] [SUCCESS] Starting cycle %CYCLE_NUMBER% of %TOTAL_CYCLES%

set "ROOM_INDEX=1"
:room_loop
if %ROOM_INDEX% gtr %ROOM_COUNT% goto :next_cycle
if %TOTAL_PROCESSED% geq %TOTAL_TARGET% goto :cleanup

REM Get current room info
call set "ROOM_NAME=%%ROOM_!ROOM_INDEX!%%"
call set "ROOM_PATH=%%ROOM_PATH_!ROOM_INDEX!%%"

echo [%date% %time%] [INFO] Processing room: !ROOM_NAME! (Cycle %CYCLE_NUMBER%)

REM Clear temporary rooms directory
del /q "%TEMP_ROOMS_DIR%\*" 2>nul

REM Copy current room to temporary directory
copy "!ROOM_PATH!" "%TEMP_ROOMS_DIR%\" >nul
echo [%date% %time%] [SUCCESS] Copied room to temporary directory

REM Create room-specific output directory
set "ROOM_OUTPUT_DIR=%OUTPUT_BASE_DIR%\!ROOM_NAME!"
if not exist "!ROOM_OUTPUT_DIR!" mkdir "!ROOM_OUTPUT_DIR!"
echo [%date% %time%] [SUCCESS] Created room output directory: !ROOM_OUTPUT_DIR!

REM Run batch processing for this room
echo [%date% %time%] [INFO] Running batch processing with %WALLPAPERS_PER_ROOM% wallpapers...
echo [%date% %time%] [INFO] Command: python -m src.scripts.run_batch_windows --rooms-dir "%TEMP_ROOMS_DIR%" --wallpapers-dir "%WALLPAPERS_DIR%" --out-dir "!ROOM_OUTPUT_DIR!" --num-wallpapers %WALLPAPERS_PER_ROOM% --room-pick first --use-depth --windows-optimized --memory-limit 0.8 --device auto --deterministic --save-debug --verbose

python -m src.scripts.run_batch_windows --rooms-dir "%TEMP_ROOMS_DIR%" --wallpapers-dir "%WALLPAPERS_DIR%" --out-dir "!ROOM_OUTPUT_DIR!" --num-wallpapers %WALLPAPERS_PER_ROOM% --room-pick first --use-depth --windows-optimized --memory-limit 0.8 --device auto --deterministic --save-debug --verbose

if %ERRORLEVEL%==0 (
    set /a ROOM_PROCESSED=%WALLPAPERS_PER_ROOM%
    set /a TOTAL_PROCESSED+=%ROOM_PROCESSED%
    echo [%date% %time%] [SUCCESS] Successfully processed %ROOM_PROCESSED% wallpapers for room: !ROOM_NAME!
    echo [%date% %time%] [INFO] Total processed so far: %TOTAL_PROCESSED% / %TOTAL_TARGET%
) else (
    echo [%date% %time%] [ERROR] Batch processing failed for room: !ROOM_NAME! (Exit code: %ERRORLEVEL%)
)

set /a ROOM_INDEX+=1
goto :room_loop

:next_cycle
set /a CYCLE_NUMBER+=1
goto :cycle_loop

:cleanup
echo [%date% %time%] [SUCCESS] Large-Scale Test Completed!
echo [%date% %time%] [SUCCESS] Total composites generated: %TOTAL_PROCESSED%
echo [%date% %time%] [INFO] Output directory: %OUTPUT_BASE_DIR%

REM Clean up temporary directory
if exist "%TEMP_ROOMS_DIR%" rmdir /s /q "%TEMP_ROOMS_DIR%"
echo [%date% %time%] [SUCCESS] Cleaned up temporary directory

endlocal
exit /b 0
