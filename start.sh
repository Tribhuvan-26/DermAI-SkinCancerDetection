#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh  —  Launch DermAI full stack (FastAPI backend + React frontend)
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   chmod +x start.sh
#   ./start.sh
#
# Requires:
#   • Python venv activated (pip install -r requirements.txt)
#   • Node.js ≥18 and npm installed
#   • Trained model at models/best_model.pth
# ─────────────────────────────────────────────────────────────────────────────

set -e

BOLD="\033[1m"
TEAL="\033[0;36m"
GREEN="\033[0;32m"
RED="\033[0;31m"
RESET="\033[0m"

echo -e "${TEAL}${BOLD}"
echo "  ██████╗ ███████╗██████╗ ███╗   ███╗ █████╗ ██╗"
echo "  ██╔══██╗██╔════╝██╔══██╗████╗ ████║██╔══██╗██║"
echo "  ██║  ██║█████╗  ██████╔╝██╔████╔██║███████║██║"
echo "  ██║  ██║██╔══╝  ██╔══██╗██║╚██╔╝██║██╔══██║██║"
echo "  ██████╔╝███████╗██║  ██║██║ ╚═╝ ██║██║  ██║██║"
echo "  ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝"
echo -e "${RESET}"
echo -e "${BOLD}  Skin Cancer Detection — Full Stack Launcher${RESET}"
echo "  ────────────────────────────────────────────"

# ── Check for trained model ──────────────────────────────────────────────────
if [ ! -f "models/best_model.pth" ]; then
  echo -e "\n${RED}  ⚠  No trained model found at models/best_model.pth${RESET}"
  echo -e "  Run training first:\n"
  echo -e "    ${TEAL}python train.py --model resnet50 --epochs 30${RESET}\n"
  read -p "  Continue anyway? (API will start but predictions will fail) [y/N] " yn
  [[ "$yn" != "y" && "$yn" != "Y" ]] && exit 1
fi

# ── Install React deps if needed ─────────────────────────────────────────────
if [ ! -d "ui/node_modules" ]; then
  echo -e "\n${TEAL}  Installing React dependencies...${RESET}"
  cd ui && npm install && cd ..
fi

# ── Launch FastAPI in background ─────────────────────────────────────────────
echo -e "\n${GREEN}  ▶ Starting FastAPI backend on http://localhost:8000${RESET}"
uvicorn api:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!
echo "    PID: $API_PID"

sleep 2

# ── Launch React frontend ─────────────────────────────────────────────────────
echo -e "\n${GREEN}  ▶ Starting React frontend on http://localhost:3000${RESET}"
cd ui && npm run dev &
UI_PID=$!
echo "    PID: $UI_PID"

echo -e "\n  ────────────────────────────────────────────"
echo -e "  ${BOLD}DermAI is running!${RESET}"
echo -e "  • UI  → ${TEAL}http://localhost:3000${RESET}"
echo -e "  • API → ${TEAL}http://localhost:8000${RESET}"
echo -e "  • API docs → ${TEAL}http://localhost:8000/docs${RESET}"
echo -e "  ────────────────────────────────────────────"
echo -e "  Press ${BOLD}Ctrl+C${RESET} to stop both servers\n"

# ── Trap SIGINT to kill both processes cleanly ────────────────────────────────
trap "echo -e '\n  Shutting down...'; kill $API_PID $UI_PID 2>/dev/null; exit 0" SIGINT SIGTERM
wait
