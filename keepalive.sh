#!/bin/bash
# checker-keepalive: restart the Banglish Threat Checker if /health is not 200.
# Same 15m pattern as the other service keepalives. Silent unless it restarts.
if curl -s --noproxy '*' --max-time 5 http://127.0.0.1:7860/health | grep -q '"ok":true'; then
  exit 0
fi
/home/hatch/workspace/ml-lab-threat/python-app/launch.sh
sleep 5
if curl -s --noproxy '*' --max-time 5 http://127.0.0.1:7860/health | grep -q '"ok":true'; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) checker restarted" >> /home/hatch/workspace/ml-lab-threat/python-app/keepalive.log
fi
