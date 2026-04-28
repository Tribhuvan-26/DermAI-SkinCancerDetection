@echo off
:: ─────────────────────────────────────────────────────────────────────────────
:: start.bat  —  Launch DermAI full stack on Windows
:: ─────────────────────────────────────────────────────────────────────────────
title DermAI - Skin Cancer Detection

echo.
echo   ██████╗ ███████╗██████╗ ███╗   ███╗ █████╗ ██╗
echo   ██╔══██╗██╔════╝██╔══██╗████╗ ████║██╔══██╗██║
echo   ██║  ██║█████╗  ██████╔╝██╔████╔██║███████║██║
echo   ██║  ██║██╔══╝  ██╔══██╗██║╚██╔╝██║██╔══██║██║
echo   ██████╔╝███████╗██║  ██║██║ ╚═╝ ██║██║  ██║██║
echo   ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝
echo.
echo   Skin Cancer Detection — Full Stack Launcher
echo   ──────────────────────────────────────────

IF NOT EXIST models\best_model.pth (
  echo.
  echo   [WARNING] No trained model found at models\best_model.pth
  echo   Run: python train.py --model resnet50 --epochs 30
  echo.
  pause
)

IF NOT EXIST ui\node_modules (
  echo   Installing React dependencies...
  cd ui && npm install && cd ..
)

echo.
echo   Starting FastAPI backend on http://localhost:8000 ...
start "DermAI API" cmd /k "uvicorn api:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 2 /nobreak >nul

echo   Starting React frontend on http://localhost:3000 ...
start "DermAI UI" cmd /k "cd ui && npm run dev"

echo.
echo   ──────────────────────────────────────────
echo   DermAI is running!
echo   UI  -- http://localhost:3000
echo   API -- http://localhost:8000
echo   API docs -- http://localhost:8000/docs
echo   ──────────────────────────────────────────
echo.
pause
