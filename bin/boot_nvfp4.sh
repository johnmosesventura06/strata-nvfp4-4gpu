#!/bin/bash
# Boot/restart the NVFP4 lane: kill by resolved PIDs, wait for VRAM release,
# launch detached, readiness = /health only (stdout is block-buffered; NEVER
# grep the log for 'ready:'). Safe to re-run (idempotent).
# Usage: boot_nvfp4.sh [config-name-without-json] [port]   defaults: nvfp4-256k 8190
set -u
CFG="${1:-nvfp4-256k}"
PORT="${2:-8190}"
BASE=/home/ai-host/strata-nvfp4
cd "$BASE/strata" || exit 1

# 1. kill strata server + engine children of THIS config family (bracket greps;
#    the raw pattern string never appears unbracketed in this script's own ps line)
PIDS=$(ps -eo pid,cmd | grep "[s]erve/server.py" | grep -v grep | awk '{print $1}')
EPIDS=$(ps -eo pid,cmd | grep "[b]uild/strata" | awk '{print $1}')
for p in $PIDS $EPIDS; do kill "$p" 2>/dev/null; done
[ -n "${PIDS}${EPIDS}" ] && sleep 3
for p in $(ps -eo pid,cmd | grep "[s]erve/server.py" | awk '{print $1}') \
         $(ps -eo pid,cmd | grep "[b]uild/strata" | awk '{print $1}'); do kill -9 "$p" 2>/dev/null; done

# 2. wait until every card < 200 MiB (the pinned arena releases in 10-40 s;
#    booting early = 'cudaMalloc(...) for the weight arena failed')
for i in $(seq 1 60); do
  BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>200' | wc -l)
  [ "$BUSY" = "0" ] && break
  sleep 5
done
[ "$BUSY" != "0" ] && { echo "FAIL: cards still busy after 300 s"; exit 1; }

# 3. launch detached; log is APPEND-shared - anchor later reads to this marker
MARK="BOOT-$CFG-$(TZ=Asia/Singapore date +%Y%m%d%H%M%S)"
echo "$MARK" > /tmp/nvfp4-boot-marker
setsid nohup python3 serve/server.py --engine strata --config "configs/$CFG.json" \
  --port "$PORT" > "/tmp/$MARK.log" 2>&1 < /dev/null & disown

# 4. readiness = health endpoint only
for i in $(seq 1 90); do
  sleep 5
  if curl -s -m 3 "http://127.0.0.1:$PORT/health" | grep -q '"loaded": true'; then
    echo "READY $MARK (port $PORT, $((i * 5)) s)"
    exit 0
  fi
  if ! pgrep -f "serve/server.py" > /dev/null; then
    echo "DIED $MARK - see /tmp/$MARK.log and strata's own log"; tail -5 "/tmp/$MARK.log"; exit 1
  fi
done
echo "TIMEOUT $MARK after 450 s - health never said loaded"; exit 1
