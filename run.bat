@echo off
REM Horus one-command start (Windows)
cd /d %~dp0
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
echo Checking Python packages (first time takes a minute) ...
python -m pip install -q -r requirements.txt
python -m horus serve %*
