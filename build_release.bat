@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1

echo.
echo ============================================================
echo   AI-Trade  ^|  Build Release Package
echo ============================================================
echo.

:: ── Version / timestamp ──────────────────────────────────────
for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set D=%%c%%b%%a
for /f "tokens=1-2 delims=: " %%a in ('time /t') do set T=%%a%%b
set D=%D: =0%
set T=%T: =0%
set STAMP=%D%_%T%
set OUT_DIR=releases
set ZIP_NAME=AI-Trade_%STAMP%.zip

if not exist "%OUT_DIR%\" mkdir "%OUT_DIR%"

:: ── Collect files via PowerShell Compress-Archive ────────────
echo Packaging source files...
powershell -NoProfile -Command ^
  "& { ^
    $src = '%~dp0'; ^
    $dst = '%~dp0%OUT_DIR%\%ZIP_NAME%'; ^
    $exclude = @('venv','__pycache__','.claude','releases','dashboard\node_modules','dashboard\.next','settings.local.json'); ^
    $excludeExt = @('.pyc','.pyo','.db','.log','.pkl','.pt','.joblib'); ^
    $files = Get-ChildItem -Path $src -Recurse -File | Where-Object { ^
        $rel = $_.FullName.Substring($src.Length); ^
        $skip = $false; ^
        foreach ($ex in $exclude) { if ($rel -like ('*' + $ex + '*')) { $skip = $true; break } }; ^
        foreach ($ext in $excludeExt) { if ($_.Extension -eq $ext) { $skip = $true; break } }; ^
        if ($rel -like 'data\*.json') { $skip = $true }; ^
        -not $skip ^
    }; ^
    $tmpDir = Join-Path $env:TEMP 'ai_trade_build'; ^
    if (Test-Path $tmpDir) { Remove-Item $tmpDir -Recurse -Force }; ^
    New-Item -ItemType Directory $tmpDir | Out-Null; ^
    foreach ($f in $files) { ^
        $rel = $f.FullName.Substring($src.Length); ^
        $target = Join-Path $tmpDir $rel; ^
        $targetDir = Split-Path $target; ^
        if (-not (Test-Path $targetDir)) { New-Item -ItemType Directory $targetDir -Force | Out-Null }; ^
        Copy-Item $f.FullName $target ^
    }; ^
    Compress-Archive -Path (Join-Path $tmpDir '*') -DestinationPath $dst -Force; ^
    Remove-Item $tmpDir -Recurse -Force; ^
    Write-Host ('Created: ' + $dst) ^
  }"

if errorlevel 1 (
    echo.
    echo  ERROR: Build failed. Requires PowerShell 5+
    pause & exit /b 1
)

echo.
echo ============================================================
echo   Package ready: %OUT_DIR%\%ZIP_NAME%
echo.
echo   To install on another machine:
echo   1. Copy and extract the zip
echo   2. Run install.bat
echo   3. Open MT5, then run start.bat
echo ============================================================
echo.
pause
