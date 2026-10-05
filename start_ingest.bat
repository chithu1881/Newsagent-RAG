@echo off
REM Starts the news ingest service on http://127.0.0.1:8000 (keep this window open)
cd /d "%~dp0"
call venv\Scripts\activate
uvicorn ingest.main:app --host 127.0.0.1 --port 8000
pause
