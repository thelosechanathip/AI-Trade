@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1

echo.
echo ============================================================
echo   AI-Trade  ^|  Installer
echo ============================================================
echo.

:: ── 1. Check Python ─────────────────────────────────────────
echo [1/5] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: Python not found.
    echo  Download Python 3.10+ from https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
for /f "tokens=1,2 delims=." %%a in ("!PY_VER!") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)
if !PY_MAJOR! LSS 3 (
    echo  ERROR: Python 3.10+ required. Found: !PY_VER!
    pause & exit /b 1
)
if !PY_MAJOR! EQU 3 if !PY_MINOR! LSS 10 (
    echo  ERROR: Python 3.10+ required. Found: !PY_VER!
    pause & exit /b 1
)
echo  OK — Python !PY_VER!

:: ── 2. Create virtual environment ───────────────────────────
echo.
echo [2/5] Setting up virtual environment...
if not exist "venv\" (
    python -m venv venv
    echo  Created: venv\
) else (
    echo  Already exists: venv\  (skipping)
)

:: ── 3. Upgrade pip ──────────────────────────────────────────
echo.
echo [3/5] Upgrading pip...
venv\Scripts\python.exe -m pip install --upgrade pip --quiet

:: ── 4. Install dependencies ──────────────────────────────────
echo.
echo [4/5] Installing dependencies...
echo  This may take 3-5 minutes on first install.
echo.
venv\Scripts\pip.exe install -r requirements.txt
if errorlevel 1 (
    echo.
    echo  ERROR: Dependency installation failed.
    echo  Check your internet connection and try again.
    pause & exit /b 1
)

:: ── Optional: PyTorch CPU (for LSTM + RL DQN) ───────────────
echo.
set /p TORCH="Install PyTorch (LSTM + RL DQN support)? [y/N]: "
if /i "!TORCH!"=="y" (
    echo  Installing PyTorch CPU...
    venv\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cpu --quiet
    if errorlevel 1 (
        echo  WARNING: PyTorch install failed. LSTM/DQN will use fallback mode.
    ) else (
        echo  PyTorch installed successfully.
    )
)

:: ── 5. Create required directories ──────────────────────────
echo.
echo [5/5] Creating directories...
if not exist "data\"   mkdir data
if not exist "models\" mkdir models
if not exist "logs\"   mkdir logs
if not exist "static\" mkdir static
echo  OK — data\  models\  logs\  static\

:: ── Done ────────────────────────────────────────────────────
echo.
echo ============================================================
echo   Installation complete!
echo ============================================================
echo.
echo   BEFORE FIRST RUN:
echo   1. Open MetaTrader 5 and log in to your account
echo   2. Edit config.yaml — set your broker symbol name
echo      (e.g. XAUUSD, XAUUSDm, XAUUSD.v, etc.)
echo   3. Run: start.bat
echo.
echo   Dashboard will open at: http://localhost:8000
echo ============================================================
echo.
pause
