@echo off
REM Start the Streamlit demo in the browser.
cd /d "%~dp0"
python -m streamlit run app.py
pause
