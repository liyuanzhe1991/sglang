#!/bin/bash
# Runs ON node 0. usage: wait_ready.sh LOG [seconds] [port]. Waits until /health answers 200 or the log shows a failure.
L=$1; T=${2:-600}; P=${3:-30000}; t0=$(date +%s)
while :; do
  c=$(curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:$P/health)
  [ "$c" = 200 ] && { echo "READY after $(( $(date +%s) - t0 )) s"; break; }
  if grep -qE "Traceback|RuntimeError|AssertionError|CUDA out of memory|Killed" $L 2>/dev/null; then echo "FAILED"; grep -nE "Traceback|Error|out of memory|Killed" $L | grep -v "ServerArgs\|server_args=" | head -8 | cut -c1-300; break; fi
  [ $(( $(date +%s) - t0 )) -gt $T ] && { echo "TIMEOUT after $T s"; tail -n 2 $L | cut -c1-300; break; }
  sleep 5
done
