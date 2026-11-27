@echo off
REM Everything that needs emulators. The default is direct genetic policy
REM evolution: acceleration is held, while steering and lean are evolved.
REM
REM Usage:
REM   train_rl.bat                         evolve v6 for 30 generations on 8 emulators
REM   train_rl.bat evolve 8 60             evolve 60 generations on 8 emulators
REM                                            resumes the last completed generation
REM   The current seven-ray rules-v6 run keeps its own checkpoint in
REM   neuro_rays_mlp_v6. It uses 30-second signed-distance trials, a
REM   32-unit hidden layer, emergency-only
REM   intervention, behavior diversity, and a larger population. It starts
REM   separately from the completed linear v3 model but lifts the existing
REM   balanced wall/correction teacher into the new hidden network.
REM   Turn slots: 2 and 4 only until tools\make_turn_states.py has captured
REM   verified checkpoint states. That tool writes models\checkpoint_start_slots.txt;
REM   the next evolve launch then tests every genome from all saved locations.
REM   Slot 3 is excluded on measurement, not
REM   preference -- it starts the machine against a wall, so every episode
REM   from it ends in ~16 steps having gained ~2 checkpoints regardless of
REM   the genome, burning a third of the evaluation budget on a near-
REM   identical crash. Rebuild that savestate before re-adding it.
REM   train_rl.bat controller 8 30         legacy hand-controller parameter search
REM   train_rl.bat ppo 12 500000           legacy PPO, only when explicitly requested
REM   train_rl.bat stop                    just kill every EmuHawk instance
REM
REM Other modes:
REM
REM   train_rl.bat teacher [n] [episodes]  record the autopilot driving
REM   train_rl.bat clone <run-id> [name]   clone it into a PPO policy
REM   train_rl.bat dashboard [run-id]      live charts in a browser
REM   train_rl.bat dqn [n] [timesteps]     train a DQN instead of PPO
REM   train_rl.bat evolve [n] [gens]       evolve the controller's parameters
REM
REM The behaviour-cloning route, end to end:
REM
REM   train_rl.bat teacher 8 25
REM   train_rl.bat clone teacher_20260820_193000 bc_ppo
REM   train_rl.bat 12 500000 --resume --name bc_ppo
REM
REM ---------------------------------------------------------------------
REM WHY THIS FILE DISPATCHES WITH GOTO RATHER THAN if ... ( ... ) BLOCKS
REM
REM It used to use big parenthesised blocks and they broke it twice. cmd
REM counts brackets inside REM and echo lines exactly as it does in code, so
REM a comment or a message that merely MENTIONS a bracket closes the block
REM early. The failure is silent and bizarre: the script fell into the middle
REM of the DQN branch on every no-argument run and launched emulators with no
REM count at all.
REM
REM Labels have no such trap. Each mode is a flat section ending in
REM "goto finish", and a bracket anywhere is now just a character.
REM
REM Two other things that must stay as they are:
REM
REM   the ROM path lives in tools\launch_instances.py, never here. The
REM   filename contains brackets -- "F-Zero (USA).sfc" -- and expanding it
REM   inside a block killed the script before it ran a single line.
REM
REM   sleeps use ping, not timeout.exe. timeout refuses to run when stdin is
REM   redirected, which is what happens when this is launched from a script,
REM   and it leaves a non-zero errorlevel that the next check misreads as a
REM   failure that never happened.
REM ---------------------------------------------------------------------

setlocal EnableDelayedExpansion

set "PROJECT_DIR=C:\Users\chipw\fzero-ai"
set "PYTHON=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "PING=%SystemRoot%\System32\ping.exe"

REM Pause before closing if launched by double-click, so the window does not
REM vanish with the output still in it.
set "PAUSE_AT_END="
echo(%cmdcmdline%| "%SystemRoot%\System32\find.exe" /i "%~nx0" >nul 2>&1 && set "PAUSE_AT_END=1"

REM ---- how many emulators ----------------------------------------------
REM
REM PPO and DQN run each env in its OWN PROCESS via SubprocVecEnv, so they
REM genuinely scale with instance count: torch is pinned to one thread,
REM leaving ~31 of 32 logical cores free, and each EmuHawk uses about a third
REM of one. Twelve costs roughly 13 GB of the machine's 32 GB.
REM
REM The GA and the teacher recorder do NOT scale. They drive every emulator
REM from a single Python process and that process is the ceiling. Benchmarked
REM on 32 identical episodes: 8 emulators 57s, 16 emulators 58s, with total
REM throughput pinned near 290 emulated frames a second either way. Doubling
REM the instances just halves each one's share, so those modes stay at 8.
set "INSTANCES=12"
set "THREADED_INSTANCES=8"
set "RUN_DEFAULT_EXTRA="

REM Long by default. No run has yet gone far enough to show whether the
REM policy learns at all. Checkpoints land in models\ every 20k steps, so a
REM long run is not an all-or-nothing bet.
set "TIMESTEPS=500000"

cd /d "%PROJECT_DIR%"

if not exist "%PYTHON%" goto no_python

set "MODE=%~1"
if not defined MODE goto mode_evolve

if /i "%MODE%"=="stop"      goto mode_stop
if /i "%MODE%"=="ppo"       goto mode_train
if /i "%MODE%"=="clone"     goto mode_clone
if /i "%MODE%"=="dashboard" goto mode_dashboard
if /i "%MODE%"=="dqn"       goto mode_dqn
if /i "%MODE%"=="evolve"        goto mode_evolve
if /i "%MODE%"=="evolve-policy" goto mode_evolve
if /i "%MODE%"=="controller"    goto mode_controller
if /i "%MODE%"=="teacher"        goto mode_teacher

REM Numeric PPO shorthand is intentionally gone. It was too easy for
REM "train_rl.bat 12" to start the old learner by accident after the project
REM had switched to genetic policy evolution. PPO remains available only as
REM the explicit `ppo` mode above.
goto bad_mode

:bad_mode
echo [error] Don't know what "%MODE%" means.
echo.
echo Give either an emulator count or one of the modes:
echo.
echo     train_rl.bat                     evolve the direct steering-and-lean policy
echo     train_rl.bat evolve [n] [gens]   the same, spelled out
echo     train_rl.bat controller [n] [g]  legacy hand-controller parameter search
echo     train_rl.bat ppo [n] [steps]     legacy PPO trainer, explicit only
echo     train_rl.bat dqn [n] [steps]     legacy DQN trainer
echo     train_rl.bat teacher [n] [eps]   record the hand-written autopilot
echo     train_rl.bat dashboard [run-id]  live charts
echo     train_rl.bat stop                kill every emulator
goto finish_error


REM ======================================================================
:mode_stop
echo Stopping all EmuHawk instances...
"%PYTHON%" tools\launch_instances.py --stop
goto finish


REM ======================================================================
REM Cloning and the dashboard touch no emulator, so they skip launch/cleanup.
:mode_clone
if "%~2"=="" goto clone_usage
set "CLONE_NAME=%~3"
if not defined CLONE_NAME set "CLONE_NAME=bc_ppo_fzero"
"%PYTHON%" training\behaviour_cloning.py --run %2 --name !CLONE_NAME!
goto finish

:clone_usage
echo [error] clone needs a run id:  train_rl.bat clone ^<run-id^> [name]
echo.
echo Run ids are the directory names under runs\store\.
goto finish_error


REM ======================================================================
:mode_dashboard
if "%~2"=="" "%PYTHON%" tools\dashboard.py
if not "%~2"=="" "%PYTHON%" tools\dashboard.py --run %2
goto finish


REM ======================================================================
:mode_dqn
set "RUN_N=%~2"
if not defined RUN_N set "RUN_N=%INSTANCES%"
set "RUN_VALUE=%~3"
if not defined RUN_VALUE set "RUN_VALUE=500000"
set "RUN_LABEL=DQN"
set "RUN_SCRIPT=training\train_dqn.py"
set "RUN_ARG=--timesteps"
goto run_with_emulators


REM ======================================================================
:mode_evolve
set "RUN_N=%~2"
if not defined RUN_N set "RUN_N=%THREADED_INSTANCES%"
set "RUN_VALUE=%~3"
if not defined RUN_VALUE set "RUN_VALUE=30"
set "RUN_LABEL=Evolving the direct steering-and-lean policy"
set "RUN_SCRIPT=training\neuroevolution.py"
set "RUN_ARG=--generations"
set "CHECKPOINT_START_SLOTS=2,4"
set "CHECKPOINT_REPEATS="
if exist "models\checkpoint_start_slots.txt" (
    set /p CHECKPOINT_START_SLOTS=<"models\checkpoint_start_slots.txt"
    REM Seven is the maximum valid distributed curriculum: 2,4 plus slots 5-9.
    REM It ensures every saved spot is assessed in the same generation.
    set "CHECKPOINT_REPEATS= --repeats 7"
)
set "RUN_DEFAULT_EXTRA= --hidden 32 --population 48 --out models\neuro_rays_mlp_v11_lineage.npz --no-teacher-seed --correction-run auto --evaluation-seconds 30 --end-on-wall --wall-contact-frames 8 --final-repeats 12 --start-slots %CHECKPOINT_START_SLOTS%%CHECKPOINT_REPEATS%"
goto run_with_emulators


REM ======================================================================
:mode_controller
set "RUN_N=%~2"
if not defined RUN_N set "RUN_N=%THREADED_INSTANCES%"
set "RUN_VALUE=%~3"
if not defined RUN_VALUE set "RUN_VALUE=30"
set "RUN_LABEL=Evolving legacy controller parameters"
set "RUN_SCRIPT=training\genetic.py"
set "RUN_ARG=--generations"
goto run_with_emulators


REM ======================================================================
:mode_teacher
set "RUN_N=%~2"
if not defined RUN_N set "RUN_N=%THREADED_INSTANCES%"
set "RUN_VALUE=%~3"
if not defined RUN_VALUE set "RUN_VALUE=25"
set "RUN_LABEL=Recording the autopilot"
set "RUN_SCRIPT=training\record_teacher.py"
set "RUN_ARG=--episodes"
goto run_with_emulators


REM ======================================================================
REM Shared path for DQN, the GA and the teacher: start N emulators, run one
REM script with --instances N and one extra argument, then clean up.
:run_with_emulators
call :collect_mode_args %*
set "RUN_EXTRA=!RUN_DEFAULT_EXTRA!!RUN_EXTRA!"
"%PYTHON%" training\session_lock.py
if errorlevel 1 goto training_busy
call :start_emulators !RUN_N!
if errorlevel 1 goto launch_failed

echo.
echo === !RUN_LABEL!: !RUN_N! instances, !RUN_ARG! !RUN_VALUE! ===
echo.
"%PYTHON%" !RUN_SCRIPT! --instances !RUN_N! !RUN_ARG! !RUN_VALUE!!RUN_EXTRA!

echo.
echo Stopping emulators...
"%PYTHON%" tools\launch_instances.py --stop
goto finish

:training_busy
echo.
echo Another trainer already controls the emulators. Nothing was stopped or started.
goto finish_error


REM ======================================================================
:mode_train
if not "%~1"=="" set "INSTANCES=%~1"
if not "%~1"=="" shift
if not "%~1"=="" set "TIMESTEPS=%~1"
if not "%~1"=="" shift

REM Collect whatever is left to forward to train.py.
set "ARGS="
:collect_args
if "%~1"=="" goto args_done
set "ARGS=!ARGS! %1"
shift
goto collect_args
:args_done

call :start_emulators !INSTANCES!
if errorlevel 1 goto launch_failed

echo.
echo === Legacy PPO training: !INSTANCES! instances, !TIMESTEPS! timesteps ===
echo Launching: training\train.py --instances !INSTANCES! --timesteps !TIMESTEPS!!ARGS!
echo.
echo Ctrl+C once to stop early -- train.py catches it and saves the model.
echo When cmd then asks to terminate the batch job, answer N, or the
echo emulators are left running.
echo.

"%PYTHON%" training\train.py --instances !INSTANCES! --timesteps !TIMESTEPS!!ARGS!
set "TRAIN_EXIT=!errorlevel!"

echo.
echo Training finished with exit code !TRAIN_EXIT!. Stopping emulators...
"%PYTHON%" tools\launch_instances.py --stop

echo.
echo Model and checkpoints are in models\. Watch or review a run with:
echo     .venv\Scripts\python.exe tools\dashboard.py
echo.
echo Reminder: one run proves nothing here -- identical configs have scored
echo 81 and 66. To compare two configurations, use:
echo     .venv\Scripts\python.exe tools\compare_configs.py --repeats N

if not "!TRAIN_EXIT!"=="0" goto finish_error
goto finish



REM ======================================================================
REM Subroutine: forward arguments after <mode> <instances> <value>.
:collect_mode_args
set "RUN_EXTRA="
shift
shift
shift
:collect_mode_args_loop
if "%~1"=="" exit /b 0
set "RUN_EXTRA=!RUN_EXTRA! %1"
shift
goto collect_mode_args_loop
REM ======================================================================
REM Subroutine: stop anything left over, then start N fresh emulators.
REM
REM Always starts clean. A leftover emulator holds the action/state files and
REM the shared-memory name for its instance number, and may be running a
REM stale copy of lua\bridge.lua -- so a new run would talk to the old bridge
REM and quietly get wrong frames.
:start_emulators
echo Stopping any emulators left over from a previous run...
"%PYTHON%" tools\launch_instances.py --stop

REM taskkill returns before Windows has finished tearing the processes down,
REM and a half-dead emulator still holds its shared-memory name.
"%PING%" -n 3 127.0.0.1 >NUL

echo.
echo === Starting %~1 emulators ===
echo.
"%PYTHON%" tools\launch_instances.py %~1
exit /b %errorlevel%


REM ======================================================================
:launch_failed
echo.
echo [error] Not every emulator came up. Check those BizHawk windows for a
echo         Lua Console error, then try again. Nothing was trained, and the
echo         emulators are still running so you can look at them.
echo         Run "train_rl.bat stop" to close them.
goto finish_error


:no_python
echo [error] Python venv not found at:
echo     %PYTHON%
goto finish_error


:finish
if defined PAUSE_AT_END pause
endlocal
exit /b 0

:finish_error
if defined PAUSE_AT_END pause
endlocal
exit /b 1
