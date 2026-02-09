@echo off
REM Weekly Screener Automation - Windows Batch Script
REM This script is designed to be called by Windows Task Scheduler

REM Change to script directory
cd /d "%~dp0"

REM Run the screener via uv
uv run python run_weekly.py --verbose

REM Log completion
echo Screener completed at %date% %time% >> logs\scheduler.log

REM Keep window open if run manually (comment out for scheduled task)
REM pause
