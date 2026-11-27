@echo off
REM Direct genetic policy evolution: steer plus lean, with acceleration held.
REM
REM Usage:
REM   evolve.bat                 40 generations on 8 emulators
REM   evolve.bat 60              60 generations
REM   evolve.bat 30 8 --population 24 --repeats 3
REM   evolve.bat stop            stop all BizHawk instances
REM
REM This is a thin alias so it always shares the launch, cleanup, and no-coast
REM rules in train_rl.bat. The saved policy is models\neuro_best.npz.
if /i "%~1"=="stop" (
    call "%~dp0train_rl.bat" stop
) else (
    call "%~dp0train_rl.bat" evolve-policy %*
)
exit /b %errorlevel%