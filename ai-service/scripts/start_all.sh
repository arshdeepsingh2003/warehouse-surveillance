#!/usr/bin/env bash
# scripts/start_all.sh
# ─────────────────────────────────────────────────────────────────────────────
# Start all three services in parallel:
#   1. FastAPI backend          → http://localhost:8000
#   2. AI camera stream service → http://localhost:8001
#   3. React frontend           → http://localhost:5173
#
# Ctrl+C stops all three.
#
# Usage:
#   chmod +x scripts/start_all.sh
#   ./scripts/start_all.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/warehouse-backend"
AI_DIR="$ROOT_DIR/warehouse-ai-service"
FRONTEND_DIR="$ROOT_DIR/warehouse-dashboard"

# Colours
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RESET='\033[0m'

echo ""
echo -e "${CYAN}══════════════════════════════════════════════════${RESET}"
echo -e "${CYAN}  Warehouse AI Surveillance System — Start All    ${RESET}"
echo -e "${CYAN}══════════════════════════════════════════════════${RESET}"
echo ""

# Track child PIDs for clean shutdown
PIDS=()

cleanup() {
  echo ""
  echo -e "${YELLOW}Stopping all services...${RESET}"
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait
  echo -e "${GREEN}All services stopped.${RESET}"
  exit 0
}
trap cleanup INT TERM

# ── 1. FastAPI Backend ────────────────────────────────────────────────────────
echo -e "${GREEN}[1/3] Starting FastAPI backend on :8000${RESET}"
cd "$BACKEND_DIR"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload \
  > /tmp/backend.log 2>&1 &
PIDS+=($!)
echo "      PID ${PIDS[-1]} | logs: /tmp/backend.log"

sleep 2

# ── 2. AI Camera Stream Service ───────────────────────────────────────────────
echo -e "${GREEN}[2/3] Starting AI stream service on :8001${RESET}"
cd "$AI_DIR"
python main.py > /tmp/ai-service.log 2>&1 &
PIDS+=($!)
echo "      PID ${PIDS[-1]} | logs: /tmp/ai-service.log"

sleep 2

# ── 3. React Frontend ─────────────────────────────────────────────────────────
echo -e "${GREEN}[3/3] Starting React dashboard on :5173${RESET}"
cd "$FRONTEND_DIR"
npm run dev > /tmp/frontend.log 2>&1 &
PIDS+=($!)
echo "      PID ${PIDS[-1]} | logs: /tmp/frontend.log"

sleep 3

echo ""
echo -e "${CYAN}══════════════════════════════════════════════════${RESET}"
echo -e "${GREEN}  All services running!${RESET}"
echo ""
echo -e "  ${CYAN}Dashboard${RESET}  → http://localhost:5173"
echo -e "  ${CYAN}API docs${RESET}   → http://localhost:8000/docs"
echo -e "  ${CYAN}Streams${RESET}    → http://localhost:8001/stream/cam-01"
echo -e "  ${CYAN}WS${RESET}         → ws://localhost:8000/ws"
echo ""
echo -e "  ${YELLOW}Press Ctrl+C to stop all services${RESET}"
echo -e "${CYAN}══════════════════════════════════════════════════${RESET}"
echo ""

# Keep script alive
wait
