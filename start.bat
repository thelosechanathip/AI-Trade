@echo off
chcp 65001 >nul 2>&1
title AI-Trade Engine
pushd "%~dp0"

:: ── Check venv ───────────────────────────────────────────────
if not exist "venv\Scripts\python.exe" (
    echo.
    echo  ERROR: Virtual environment not found.
    echo  Run install.bat first, or re-run AI-Trade_Setup.exe
    echo.
    pause
    exit /b 1
)

:: ── Check MT5 ────────────────────────────────────────────────
tasklist /fi "imagename eq terminal64.exe" 2>nul | find /i "terminal64.exe" >nul
if errorlevel 1 (
    echo.
    echo  WARNING: MetaTrader 5 does not appear to be running.
    echo  Open MT5 and log in before continuing.
    echo.
    choice /c YN /m "Continue anyway"
    if errorlevel 2 exit /b 0
)

:: ── Start engine ─────────────────────────────────────────────
echo.
echo ============================================================
echo   AI-Trade Engine  ^|  Starting...
echo   Dashboard: http://localhost:8000
echo   Press Ctrl+C to stop
echo ============================================================
echo.

venv\Scripts\python.exe run.py

if errorlevel 1 (
    echo.
    echo  Engine stopped with an error. Check logs\trading.log for details.
    echo.
    pause
)
popd
