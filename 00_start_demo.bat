@echo off
setlocal
cd /d "%~dp0"
title Strategy Demo - Dividend Low Vol + Treasury

set "PY=%~dp0.venv\Scripts\python.exe"
if exist "%PY%" goto run

set "PY=python"
"%PY%" -V >nul 2>&1
if errorlevel 1 goto nopython
"%PY%" -c "import pandas,numpy,matplotlib" >nul 2>&1
if errorlevel 1 goto nodeps

:run
echo ============================================================
echo   Strategy demo: 40pct dividend low-vol + 60pct treasury
echo   Part 1/2  backtest   (about 7 seconds)
echo   Part 2/2  lookahead  (about 5 seconds)
echo   Do not close this window while it runs.
echo ============================================================
echo.

echo [1/2] Running backtest ...
echo ------------------------------------------------------------
"%PY%" src\run_strategy.py
if errorlevel 1 goto fail
echo ------------------------------------------------------------
echo.

echo [2/2] Checking for look-ahead bias ...
echo ------------------------------------------------------------
"%PY%" src\check_lookahead.py
if errorlevel 1 goto fail
echo ------------------------------------------------------------
echo.

echo ============================================================
echo   DONE - no errors.
echo.
echo   Results are in output\ and charts\ :
echo     output\final_metrics_trade.csv   tradable version
echo     output\final_metrics_long.csv    long sample
echo     charts\14_*.png                  net value / drawdown / position
echo ============================================================
echo.
echo Press any key to open the charts folder, then close the window.
pause > nul
start "" "%~dp0charts"
exit /b 0

:nopython
echo.
echo [ERROR] Python not found.
echo Install Python 3.10+ and tick "Add python.exe to PATH",
echo or run this file from the project folder that contains .venv
pause
exit /b 1

:nodeps
echo.
echo [ERROR] Python found, but pandas / numpy / matplotlib are missing.
echo Run this once, then double-click again:
echo     "%PY%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo.
pause
exit /b 1

:fail
echo.
echo ============================================================
echo   ERROR while running. Take a screenshot of the last lines
echo   and send it to the AI assistant. Keep this window open.
echo ============================================================
pause
exit /b 1
