#!/bin/bash
# Banglish Threat Checker — launcher (double-fork, like launch-qdrant.sh).
# Starts gunicorn detached so it survives the calling shell/cron worker.
# Port 7860, binds 0.0.0.0 (reachable on Tailscale IP + Cloudflare tunnel).
cd /home/hatch/workspace/ml-lab-threat/python-app || exit 1
if curl -s --noproxy '*' --max-time 3 http://127.0.0.1:7860/health | grep -q '"ok":true'; then
  echo "already running"
  exit 0
fi
nohup /home/hatch/workspace/ml-lab-threat/.venv/bin/gunicorn \
  --bind 0.0.0.0:7860 --workers 1 --threads 4 --timeout 120 app:app \
  > /home/hatch/workspace/ml-lab-threat/python-app/server.log 2>&1 &
# double-fork: disown and exit immediately; the daemon keeps running.
disown
echo "started pid $!"
