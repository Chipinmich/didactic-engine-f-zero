@echo off
REM Starts BizHawk with F-Zero + the Lua bridge (if not already running),
REM then runs the standalone seven-ray autopilot.
REM
REM Usage:
REM   start_autopilot.bat                load safe slot 2 and drive for real
REM   start_autopilot.bat --dry-run      print distances/decisions only
REM   start_autopilot.bat --calibrate    tune vision thresholds, no driving
REM
REM Any arguments are passed straight through to autopilot.py.

setlocal

set "PROJECT_DIR=C:\Users\chipw\fzero-ai"
set "BIZHAWK_DIR=C:\Users\chipw\OneDrive\Desktop\BizHawk-2.11.1-win-x64"
set "ROM_PATH=%BIZHAWK_DIR%\F-Zero (USA).sfc"
set "LUA_SCRIPT=%PROJECT_DIR%\lua\bridge.lua"
set "FZERO_LABEL=RAY AUTOPILOT"

cd /d "%PROJECT_DIR%"

tasklist /FI "IMAGENAME eq EmuHawk.exe" 2>NUL | find /I "EmuHawk.exe" >NUL
if errorlevel 1 (
    echo Starting BizHawk with F-Zero and the Lua bridge auto-loaded...
    start "" "%BIZHAWK_DIR%\EmuHawk.exe" "%ROM_PATH%" --lua="%LUA_SCRIPT%"
    echo Waiting for BizHawk to finish loading...
    timeout /t 6 /nobreak >NUL
) else (
    echo BizHawk is already running - not starting a second instance.
    echo Make sure lua\bridge.lua is loaded: Tools -^> Lua Console -^> Script -^> Open Script.
    echo   ^(If --lua= auto-load worked when BizHawk first started, it already is.^)
    echo.
    pause
)

echo.
set "AUTOPILOT_ARGS=%*"
if not defined AUTOPILOT_ARGS set "AUTOPILOT_ARGS=--reset --reset-slot 2"
echo Launching: ray_autopilot.py %AUTOPILOT_ARGS%
echo.

"%PROJECT_DIR%\.venv\Scripts\python.exe" ray_autopilot.py %AUTOPILOT_ARGS%

endlocal
