@echo off
REM Opens the News Analyst chat app in your browser (keep this window open)
cd /d "%~dp0"
call venv\Scripts\activate
streamlit run app\streamlit_app.py
pause
