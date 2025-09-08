# Large-Scale Test Script: Room Switch Every 50 Wallpapers
# This script processes 50 wallpapers per room, cycling through all available rooms
# for large-scale testing (e.g., 5000 total composites = 100 rooms × 50 wallpapers each)

param(
    [string]$RoomsDir = "src/data/rooms",
    [string]$WallpapersDir = "src/data/wallpapers", 
    [string]$OutputBaseDir = "src/data/out/per_room_50",
    [int]$WallpapersPerRoom = 50,
    [int]$TotalTarget = 5000,
    [switch]$Help
)

if ($Help) {
    Write-Host "Large-Scale Test Script: Room Switch Every 50 Wallpapers" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage: .\scripts\per_room_50.ps1 [options]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Cyan
    Write-Host "  -RoomsDir <path>        Directory containing room images (default: src/data/rooms)"
    Write-Host "  -WallpapersDir <path>   Directory containing wallpaper images (default: src/data/wallpapers)"
    Write-Host "  -OutputBaseDir <path>   Base output directory (default: src/data/out/per_room_50)"
    Write-Host "  -WallpapersPerRoom <n>  Number of wallpapers per room (default: 50)"
    Write-Host "  -TotalTarget <n>        Total target composites (default: 5000)"
    Write-Host "  -Help                   Show this help message"
    Write-Host ""
    Write-Host "Example:" -ForegroundColor Yellow
    Write-Host "  .\scripts\per_room_50.ps1 -TotalTarget 1000"
    Write-Host ""
    exit 0
}

# Function to log with timestamp
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $color = switch ($Level) {
        "ERROR" { "Red" }
        "WARNING" { "Yellow" }
        "SUCCESS" { "Green" }
        default { "White" }
    }
    Write-Host "[$timestamp] [$Level] $Message" -ForegroundColor $color
}

# Function to create directory if it doesn't exist
function Ensure-Directory {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
        Write-Log "Created directory: $Path" "SUCCESS"
    }
}

# Function to get room files
function Get-RoomFiles {
    param([string]$RoomsPath)
    $extensions = @("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
    $roomFiles = @()
    
    foreach ($ext in $extensions) {
        $files = Get-ChildItem -Path $RoomsPath -Filter $ext -File -ErrorAction SilentlyContinue
        $roomFiles += $files
    }
    
    return $roomFiles | Sort-Object Name
}

try {
    Write-Log "Starting Large-Scale Test: Room Switch Every $WallpapersPerRoom Wallpapers" "SUCCESS"
    Write-Log "Target: $TotalTarget total composites" "INFO"
    Write-Log "Wallpapers per room: $WallpapersPerRoom" "INFO"
    
    # Validate input directories
    if (-not (Test-Path $RoomsDir)) {
        throw "Rooms directory not found: $RoomsDir"
    }
    
    if (-not (Test-Path $WallpapersDir)) {
        throw "Wallpapers directory not found: $WallpapersDir"
    }
    
    # Get all room files
    $roomFiles = Get-RoomFiles -RoomsPath $RoomsDir
    if ($roomFiles.Count -eq 0) {
        throw "No room images found in: $RoomsDir"
    }
    
    Write-Log "Found $($roomFiles.Count) room images" "SUCCESS"
    
    # Create temporary rooms directory
    $tempRoomsDir = "src/data/rooms_tmp"
    Ensure-Directory -Path $tempRoomsDir
    
    # Create base output directory
    Ensure-Directory -Path $OutputBaseDir
    
    # Calculate how many room cycles we need
    $roomsPerCycle = $roomFiles.Count
    $totalCycles = [Math]::Ceiling($TotalTarget / ($WallpapersPerRoom * $roomsPerCycle))
    $actualTotal = $totalCycles * $WallpapersPerRoom * $roomsPerCycle
    
    Write-Log "Will process $totalCycles cycles of all $roomsPerCycle rooms" "INFO"
    Write-Log "Expected total composites: $actualTotal" "INFO"
    
    $totalProcessed = 0
    $cycleNumber = 1
    
    # Main processing loop
    while ($totalProcessed -lt $TotalTarget) {
        Write-Log "Starting cycle $cycleNumber of $totalCycles" "SUCCESS"
        
        foreach ($roomFile in $roomFiles) {
            if ($totalProcessed -ge $TotalTarget) {
                Write-Log "Reached target of $TotalTarget composites. Stopping." "SUCCESS"
                break
            }
            
            $roomName = $roomFile.BaseName
            $roomOutputDir = Join-Path $OutputBaseDir $roomName
            
            Write-Log "Processing room: $roomName (Cycle $cycleNumber)" "INFO"
            
            try {
                # Clear temporary rooms directory
                Get-ChildItem -Path $tempRoomsDir -File | Remove-Item -Force
                
                # Copy current room to temporary directory
                Copy-Item -Path $roomFile.FullName -Destination $tempRoomsDir -Force
                Write-Log "Copied room to temporary directory" "SUCCESS"
                
                # Create room-specific output directory
                Ensure-Directory -Path $roomOutputDir
                
                # Run batch processing for this room
                $batchArgs = @(
                    "--rooms-dir", $tempRoomsDir
                    "--wallpapers-dir", $WallpapersDir
                    "--out-dir", $roomOutputDir
                    "--num-wallpapers", $WallpapersPerRoom
                    "--room-pick", "first"
                    "--use-depth"
                    "--windows-optimized"
                    "--memory-limit", "0.8"
                    "--device", "auto"
                    "--deterministic"
                    "--save-debug"
                    "--verbose"
                )
                
                Write-Log "Running batch processing with $WallpapersPerRoom wallpapers..." "INFO"
                Write-Log "Command: python -m src.scripts.run_batch_windows $($batchArgs -join ' ')" "INFO"
                
                $process = Start-Process -FilePath "python" -ArgumentList @("-m", "src.scripts.run_batch_windows") + $batchArgs -Wait -PassThru -NoNewWindow
                
                if ($process.ExitCode -eq 0) {
                    $roomProcessed = $WallpapersPerRoom
                    $totalProcessed += $roomProcessed
                    Write-Log "Successfully processed $roomProcessed wallpapers for room: $roomName" "SUCCESS"
                    Write-Log "Total processed so far: $totalProcessed / $TotalTarget" "INFO"
                } else {
                    Write-Log "Batch processing failed for room: $roomName (Exit code: $($process.ExitCode))" "ERROR"
                }
                
            } catch {
                Write-Log "Error processing room $roomName : $($_.Exception.Message)" "ERROR"
            }
        }
        
        $cycleNumber++
    }
    
    Write-Log "Large-Scale Test Completed!" "SUCCESS"
    Write-Log "Total composites generated: $totalProcessed" "SUCCESS"
    Write-Log "Output directory: $OutputBaseDir" "INFO"
    
    # Clean up temporary directory
    if (Test-Path $tempRoomsDir) {
        Remove-Item -Path $tempRoomsDir -Recurse -Force
        Write-Log "Cleaned up temporary directory" "SUCCESS"
    }
    
} catch {
    Write-Log "Fatal error: $($_.Exception.Message)" "ERROR"
    exit 1
}
