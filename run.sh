#!/bin/bash
# Runs one job of the Nifty bot.  Usage: run.sh full|alert|backtest|paper [days]
# Called by cron (IST). Same 09:10-16:00 Mon-Fri window as the old GitHub workflow.
APP="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-full}"
DAYS="${2:-365}"
cd "$APP" || exit 1
set -a; . "$APP/.env"; set +a
mkdir -p logs
LOG="logs/$(date +%F).log"
hm=$((10#$(TZ=Asia/Kolkata date +%H%M))); dow=$(TZ=Asia/Kolkata date +%u)
if [ "$MODE" != backtest ] && [ "$MODE" != variants ] && [ "$MODE" != paper ] && [ "$FORCE" != true ]; then
  if [ "$dow" -gt 5 ] || [ "$hm" -lt 910 ] || [ "$hm" -gt 1600 ]; then
    echo "$(date '+%F %T') $MODE: outside market window, skipped" >> "$LOG"; exit 0
  fi
fi
exec 9>"$APP/.lock"
flock -w 300 9 || { echo "$(date '+%F %T') $MODE: another job still running, skipped" >> "$LOG"; exit 1; }
PY="$APP/venv/bin/python"; FF=""; [ "$FORCE" = true ] && FF="--force"
echo "=== $(date '+%F %T') $MODE ===" >> "$LOG"
case "$MODE" in
  full)     "$PY" report.py $FF >> "$LOG" 2>&1 ;;
  alert)    "$PY" ew_alert.py $FF >> "$LOG" 2>&1 ;;
  backtest) "$PY" backtest.py --days "$DAYS" >> "$LOG" 2>&1 ;;
  variants) "$PY" variants.py --days "$DAYS" --adays "${3:-365}" >> "$LOG" 2>&1 ;;
  paper)    "$PY" paper.py    >> "$LOG" 2>&1 ;;
  *) echo "unknown mode $MODE" >> "$LOG"; exit 2 ;;
esac
echo "exit code $?" >> "$LOG"
find logs -name '*.log' -mtime +30 -delete
