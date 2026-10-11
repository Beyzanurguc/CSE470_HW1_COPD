@echo off
REM Install packages, inspect the dataset and train all models (run once).
cd /d "%~dp0"
python -m pip install -r requirements.txt || goto :err
python inspect_data.py || goto :err
python train.py || goto :err
echo.
echo Done. Start the demo with run_demo.bat
pause
exit /b 0
:err
echo.
echo Something failed - read the message above.
pause
exit /b 1
