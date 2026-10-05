@echo off
REM Runs the 3 fetcher agents every few hours + the 07:00 briefing (needs start_ingest.bat running)
cd /d "%~dp0"
call venv\Scripts\activate
python -m agents.orchestrator --schedule
pause
