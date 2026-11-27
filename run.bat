@echo off
REM Starts BizHawk with F-Zero + the Lua bridge (if not already running),
REM then runs one of the Python tools against it.
REM
REM Usage:
REM   run.bat                     check the bridge is healthy (default)
REM   run.bat check               same as above
REM   run.bat train               evolve the direct genetic driving policy
REM   run.bat overnight           fresh GA, DQN, and PPO run (12 instances)
REM   run.bat inherited           continue DQN + PPO, 7 emulators each (no GA)
REM   run.bat drive               run the hand-written autopilot
REM   run.bat calibrate           tune the vision thresholds
REM
REM Any extra arguments are passed straight through, e.g.
REM   run.bat drive --dry-run
REM   run.bat drive --enter-threshold 50
REM
REM NOTE: paths below are expanded with !delayed! expansion inside every
REM parenthesised block. The ROM filename contains brackets -- "F-Zero
REM (USA).sfc" -- and %PLAIN% expansion happens while cmd is still parsing
REM the block, so the ")" would close the block early and kill the script
REM with "unexpected at this time" before any of it runs.

setlocal EnableDelayedExpansion

REM Pause before closing if launched by double-click, so the window doesn't
REM vanish with the output still in it.
set "PAUSE_AT_END="
echo(%cmdcmdline%| "%SystemRoot%\System32\find.exe" /i "%~nx0" >nul 2>&1 && set "PAUSE_AT_END=1"

REM Absolute paths: when this is launched from Git Bash, a bare "find" or
REM "timeout" resolves to Git's Unix versions, which take different flags.
REM That made the EmuHawk check fail and try to launch a second instance.
set "FIND=%SystemRoot%\System32\find.exe"
set "TIMEOUT=%SystemRoot%\System32\timeout.exe"
set "TASKLIST=%SystemRoot%\System32\tasklist.exe"

set "PROJECT_DIR=C:\Users\chipw\fzero-ai"
set "BIZHAWK_DIR=C:\Users\chipw\OneDrive\Desktop\BizHawk-2.11.1-win-x64"
set "ROM_PATH=%BIZHAWK_DIR%\F-Zero (USA).sfc"
set "LUA_SCRIPT=%PROJECT_DIR%\lua\bridge.lua"
set "PYTHON=%PROJECT_DIR%\.venv\Scripts\python.exe"

set "MODE=%~1"
if not defined MODE set "MODE=check"
if not "%~1"=="" shift

REM Collect any remaining arguments to forward to the Python script.
set "ARGS="
:collect_args
if "%~1"=="" goto args_done
set "ARGS=!ARGS! %1"
shift
goto collect_args
:args_done

REM Training moved to train_rl.bat because it owns the clean multi-emulator
REM lifecycle. Keep this old spelling as a forwarding alias, but never start
REM its stale one-emulator PPO path.
if /i "%MODE%"=="train" goto delegate_train
if /i "%MODE%"=="overnight" goto delegate_overnight
if /i "%MODE%"=="inherited" goto delegate_inherited

if /i "%MODE%"=="check" (
    set "SCRIPT=tools\check_bridge.py"
    set "LABEL=Bridge diagnostics"
) else if /i "%MODE%"=="train" (
    set "SCRIPT=training\train.py"
    set "LABEL=PPO training"
) else if /i "%MODE%"=="drive" (
    set "SCRIPT=autopilot.py"
    set "LABEL=Autopilot"
) else if /i "%MODE%"=="calibrate" (
    set "SCRIPT=autopilot.py"
    set "LABEL=Vision calibration"
    set "ARGS= --calibrate!ARGS!"
) else (
    echo [error] Unknown mode: %MODE%
    echo.
    echo This script only understands:  check ^| train ^| overnight ^| inherited ^| drive ^| calibrate
    echo.
    echo If you were after training, use train_rl.bat instead -- that is
    echo where everything lives now, and it takes an emulator count:
    echo.
    echo     train_rl.bat                     12 emulators, 500k timesteps
    echo     train_rl.bat 12 500000           the same, spelled out
    echo     train_rl.bat teacher 8 25        record the autopilot
    echo     train_rl.bat clone ^<run-id^>      behaviour-clone that recording
    echo     train_rl.bat dqn 12 500000       train a DQN instead of PPO
    echo     train_rl.bat evolve 8 30         evolve the controller
    echo     train_rl.bat dashboard           live charts
    echo     train_rl.bat stop                kill every emulator
    echo.
    echo For both current learning systems overnight, use:
    echo.
    echo     run.bat overnight
    goto finish_error
)

cd /d "%PROJECT_DIR%"

if not exist "!PYTHON!" (
    echo [error] Python venv not found at:
    echo     !PYTHON!
    goto finish_error
)

if not exist "!ROM_PATH!" (
    echo [error] ROM not found at:
    echo     !ROM_PATH!
    goto finish_error
)

if not exist "!LUA_SCRIPT!" (
    echo [error] Lua bridge not found at:
    echo     !LUA_SCRIPT!
    goto finish_error
)

"%TASKLIST%" /FI "IMAGENAME eq EmuHawk.exe" 2>NUL | "%FIND%" /I "EmuHawk.exe" >NUL
if errorlevel 1 (
    echo Starting BizHawk with F-Zero and the Lua bridge auto-loaded...
    start "" "!BIZHAWK_DIR!\EmuHawk.exe" "!ROM_PATH!" --lua="!LUA_SCRIPT!"
    echo Waiting for BizHawk to finish loading...
    "%TIMEOUT%" /t 6 /nobreak >NUL
) else (
    echo BizHawk is already running - not starting a second instance.
    echo.
    echo Make sure lua\bridge.lua is loaded AND up to date:
    echo     Tools -^> Lua Console -^> Script -^> Reload
    echo.
    echo The bridge runs in lockstep, so the car only moves while a Python
    echo client is connected. If the script is stale, nothing below will work.
    echo.
    pause
)

if /i "%MODE%"=="train" (
    echo.
    echo Before training, make sure a savestate is saved at the starting line
    echo in slot 1 ^(Shift+F1 in BizHawk^). Every reset reloads that slot; if
    echo it is empty, training will fail with a reset timeout.
    echo.
    pause
)

echo.
echo === %LABEL% ===
echo Launching: %SCRIPT%%ARGS%
echo.

"!PYTHON!" %SCRIPT%%ARGS%
if errorlevel 1 goto finish_error
goto finish

:delegate_train
call "%PROJECT_DIR%\train_rl.bat" evolve-policy!ARGS!
goto finish

:delegate_overnight
cd /d "%PROJECT_DIR%"
if not exist "!PYTHON!" (
    echo [error] Python venv not found at:
    echo     !PYTHON!
    goto finish_error
)
"!PYTHON!" tools\overnight_training.py!ARGS!
if errorlevel 1 goto finish_error
goto finish

:delegate_inherited
cd /d "%PROJECT_DIR%"
if not exist "!PYTHON!" (
    echo [error] Python venv not found at:
    echo     !PYTHON!
    goto finish_error
)
"!PYTHON!" tools\overnight_training.py --inherit!ARGS!
if errorlevel 1 goto finish_error
goto finish

:finish
if defined PAUSE_AT_END (
    echo.
    pause
)
endlocal
exit /b 0

:finish_error
if defined PAUSE_AT_END (
    echo.
    pause
)
endlocal
exit /b 1
