@echo off
title Squirrel Notes
color 0A

echo.
echo  =====================================================
echo    Squirrel Notes - AI Meeting Notes (Local)
echo    Because you were definitely paying attention.
echo  =====================================================
echo.

:: ── Check Python is available ────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found. Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

:: ── Install / verify dependencies ────────────────────────────────────────────
echo  Checking Python dependencies...
pip show flask >nul 2>&1
if errorlevel 1 (
    echo  Installing dependencies from requirements.txt...
    echo  (This only happens once — may take a few minutes)
    echo.
    pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo  ERROR: Dependency install failed. Check your internet connection.
        pause
        exit /b 1
    )
)

:: ── Remind user to start Ollama ──────────────────────────────────────────────
echo.
echo  REMINDER: Ollama must be running for summary generation.
echo  If not started yet, open another terminal and run:
echo.
echo      ollama serve
echo.
echo  And make sure your model is pulled, e.g.:
echo.
echo      ollama pull llama3.2
echo.
echo  =====================================================
echo    Opening browser at http://localhost:5000
echo    Press Ctrl+C in this window to stop Squirrel Notes.
echo  =====================================================
echo.

:: ── Small delay then open browser ────────────────────────────────────────────
start "" /B cmd /c "timeout /t 2 >nul && start http://localhost:5000"

:: ── Start the Flask app ──────────────────────────────────────────────────────
python app.py

pause
