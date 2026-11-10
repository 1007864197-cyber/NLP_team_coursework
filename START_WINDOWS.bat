@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "OPENBLAS_NUM_THREADS=1"
set "OMP_NUM_THREADS=1"
set "PIP_CACHE_DIR=%~dp0.venv\pip-cache"

if not exist ".venv\Scripts\python.exe" goto find_python
".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" >nul 2>nul
if errorlevel 1 goto environment_failed
goto check_dependencies

:find_python
rem Validate interpreters, not just the existence of the Windows py launcher.
rem A launcher can be installed even when its registered Python path is stale.
python -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" >nul 2>nul
if errorlevel 1 goto try_py
set "OLIVE_PYTHON_COMMAND=python"
goto create_environment
:try_py
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" >nul 2>nul
if errorlevel 1 goto try_py313
set "OLIVE_PYTHON_COMMAND=py -3"
goto create_environment
:try_py313
py -3.13 -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" >nul 2>nul
if errorlevel 1 goto try_py312
set "OLIVE_PYTHON_COMMAND=py -3.13"
goto create_environment
:try_py312
py -3.12 -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" >nul 2>nul
if errorlevel 1 goto try_py311
set "OLIVE_PYTHON_COMMAND=py -3.12"
goto create_environment
:try_py311
py -3.11 -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" >nul 2>nul
if errorlevel 1 goto python_missing
set "OLIVE_PYTHON_COMMAND=py -3.11"

:create_environment
echo Creating the project Python environment. Please wait...
%OLIVE_PYTHON_COMMAND% -m venv .venv
if errorlevel 1 goto environment_failed

:check_dependencies
if not exist ".venv\olive-table-ready.txt" goto install
".venv\Scripts\python.exe" -c "import flask, sklearn, joblib; assert sklearn.__version__ == '1.9.1'" >nul 2>nul
if not errorlevel 1 goto run
:install
echo Installing project libraries. The first run needs internet access...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt -c constraints-model.txt
if errorlevel 1 goto failed
echo ready> ".venv\olive-table-ready.txt"

:run
echo.
echo Olive Table restaurant concierge
echo Your browser will open when the application is ready.
echo Address: http://127.0.0.1:8765
echo Keep this window open. Press Ctrl+C to stop.
echo.
".venv\Scripts\python.exe" launch.py %*
if errorlevel 1 goto failed
endlocal
exit /b 0

:python_missing
echo No working Python 3.11 or newer was found.
echo Install Python from https://www.python.org/downloads/ and try again.
goto failed
:environment_failed
echo The project Python environment could not be created or started.
echo Check the interpreter path shown above and that this folder is writable.
:failed
echo.
echo Startup did not complete. The error above identifies the failed step.
if /i "%~1"=="--check" exit /b 1
pause
endlocal
exit /b 1
