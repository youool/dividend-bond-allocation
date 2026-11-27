@echo off
setlocal
cd /d "%~dp0"
title Strategy Board - local web app

set "PY=%~dp0.venv\Scripts\python.exe"
if exist "%PY%" goto run

set "PY=python"
"%PY%" -V >nul 2>&1
if errorlevel 1 goto nopython
"%PY%" -c "import pandas,numpy,matplotlib" >nul 2>&1
if errorlevel 1 goto nodeps

:run
echo ============================================================
echo   Strategy board: drag parameters, backend recomputes
echo   Browser will open automatically at http://127.0.0.1:8777/
echo   Keep this window open. Press Ctrl+C to stop.
echo ============================================================
echo.
"%PY%" src\webapp.py --port 8777
if errorlevel 1 goto fail
exit /b 0

:nopython
echo.
echo [ERROR] Python not found. Install Python 3.10+ with "Add to PATH".
pause
exit /b 1

:nodeps
echo.
echo [ERROR] pandas / numpy / matplotlib missing. Run once:
echo     "%PY%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pause
exit /b 1

:fail
echo.
echo [ERROR] The board exited with an error. Screenshot the lines above.
pause
exit /b 1
