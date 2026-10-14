@echo off
REM Build the SGM desktop app for Windows and put the download in downloads\.
REM Needs Node.js on PATH. The first run downloads Electron (~120 MB).
cd /d "%~dp0.."
node tools\build_desktop.mjs --platform=win32 %*
echo.
echo Result: desktop\dist\SGM-win32-x64\SGM.exe   and   downloads\sgm-desktop-*.zip
