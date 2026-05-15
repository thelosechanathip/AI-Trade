@echo off
setlocal
chcp 65001 >nul 2>&1

echo.
echo ============================================================
echo   AI-Trade  ^|  Build Installer
echo ============================================================
echo.

:: ── Find Inno Setup ─────────────────────────────────────────
set ISCC=
if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" set ISCC=C:\Program Files (x86)\Inno Setup 6\ISCC.exe
if exist "C:\Program Files\Inno Setup 6\ISCC.exe"       set ISCC=C:\Program Files\Inno Setup 6\ISCC.exe
if exist "C:\Program Files (x86)\Inno Setup 5\ISCC.exe" set ISCC=C:\Program Files (x86)\Inno Setup 5\ISCC.exe

if not defined ISCC (
    echo  ERROR: Inno Setup not found.
    echo  Download free from: https://jrsoftware.org/isinfo.php
    pause & exit /b 1
)

:: ── Create releases dir ──────────────────────────────────────
if not exist "releases\" mkdir releases

:: ── Compile ──────────────────────────────────────────────────
echo  Compiling installer...
echo.
"%ISCC%" setup.iss
if errorlevel 1 (
    echo.
    echo  ERROR: Compilation failed. Check setup.iss for errors.
    pause & exit /b 1
)

echo.
echo ============================================================
echo   Done!  releases\AI-Trade_Setup.exe
echo.
echo   Copy this ONE file to another machine and run it.
echo ============================================================
echo.

:: Open releases folder
explorer releases
pause
