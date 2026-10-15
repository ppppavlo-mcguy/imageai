@echo off
REM Serve the app locally: http://localhost:3000
REM Run from wherever it was called: the paths below are relative to this file.
cd /d "%~dp0"
node web\server.js
