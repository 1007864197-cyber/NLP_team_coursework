@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto run
where py >nul 2>nul
if errorlevel 1 goto use_python
py -3 -m venv .venv
if errorlevel 1 goto failed
goto install
:use_python
python -m venv .venv
if errorlevel 1 goto failed
:install
".venv\Scripts\python.exe" -m pip install -r requirements.txt -c constraints-model.txt
if errorlevel 1 goto failed
echo ready> ".venv\olive-table-ready.txt"
:run
if not exist ".venv\olive-table-ready.txt" goto install
echo.
echo Olive Table restaurant concierge
echo Open http://127.0.0.1:8765 in your browser.
echo Keep this window open. Press Ctrl+C to stop.
echo.
".venv\Scripts\python.exe" app.py
if errorlevel 1 goto failed
goto end
:failed
echo.
echo Setup or startup did not complete. See the message above.
echo Check that Python 3.11 or newer is installed and that you have
echo an internet connection for the first dependency installation.
pause
:end
endlocal
