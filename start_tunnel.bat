@echo off
REM Gives the ingest service a public https address so n8n Cloud can reach it.
REM Start start_ingest.bat first. Copy the https://....trycloudflare.com address printed below
REM into the n8n "Send to Python /ingest" node URL (add /ingest at the end). Keep this window open.
REM The log (with the address) is also saved to tunnel.log in this folder.
cd /d "%~dp0"
if exist tunnel.log del tunnel.log
"C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --no-autoupdate --logfile tunnel.log --url http://127.0.0.1:8000
pause
