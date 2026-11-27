@echo off
setlocal
set "BIZHAWK=C:\Users\chipw\OneDrive\Desktop\BizHawk-2.11.1-win-x64\EmuHawk.exe"
set "ROM=C:\Users\chipw\OneDrive\Desktop\BizHawk-2.11.1-win-x64\F-Zero (USA).sfc"
set "LOGGER=C:\Users\chipw\fzero-ai\lua\manual_speed_logger.lua"

echo Starting a separate human-controlled F-Zero speed recording.
echo Do not use a boost. Close this new emulator when the run is finished.
echo Recording: C:\Users\chipw\fzero-ai\runs\manual_speed.csv
echo.
start "F-Zero manual speed recording" "%BIZHAWK%" "%ROM%" --lua="%LOGGER%"
endlocal
