@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
python -m streamlit run app.py
if errorlevel 1 (
  echo.
  echo 启动失败，请先运行：python -m pip install -r requirements.txt
  pause
)
