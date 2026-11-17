#!/bin/bash
# Banglish Threat Checker — gated auto-retrain + deploy (continuous learning).
# Runs every 30 min via cron `checker-auto-retrain`. Retrains ONLY when the
# inputs changed (learned_words.json / builder / tests / normalizer /
# verifier) — otherwise prints "no changes — skip" and exits 0.
#
# Pipeline: build dataset -> verify/retrain (metrics.json) -> GATES ->
# export model.json -> JS validation -> rebuild checker.html -> page tests
# -> backup deployed .pkl -> copy new files -> restart server -> localhost
# health + live probe of the newest learned word -> upload static checker.
#
# Gates (deploy ONLY if all pass, else old model stays live):
#   variant+demo misses == 0, best_acc >= 0.75, L3 recall >= 0.90,
#   masked_norm >= 0.65, JS validation 0 mismatches, page tests pass.
# On live-probe failure: restore the .bak model + restart (rollback).
set -u
cd /home/hatch/workspace/ml-lab-threat || exit 1
VENV=.venv/bin/python
APP=python-app
LAST=last_retrain.json

fail() { echo "RETRAIN FAILED: $1"; exit 1; }

# --- 1. fingerprint: retrain only on change -------------------------------
FP=$(sha256sum learned_words.json build_slang_dataset.py variant_tests.py \
     textnorm.py verify_jsonl.py 2>/dev/null | sha256sum | cut -d' ' -f1)
if [ -f "$LAST" ]; then
    OLD_FP=$(python3 -c "import json; print(json.load(open('$LAST')).get('fingerprint',''))" 2>/dev/null)
    if [ "$FP" = "$OLD_FP" ]; then
        echo "no changes since last retrain — skip"
        exit 0
    fi
fi
echo "changes detected — retraining (fp ${FP:0:12})"

# --- 2. rebuild dataset ------------------------------------------------------
$VENV build_slang_dataset.py > retrain-build.log 2>&1 \
    || fail "dataset build failed (see retrain-build.log)"
grep -E "^(messages rows|parquet):" retrain-build.log | tail -3

# --- 3. retrain + metrics ----------------------------------------------------
$VENV verify_jsonl.py > retrain-verify.log 2>&1 \
    || fail "retrain failed (see retrain-verify.log)"
[ -f metrics.json ] || fail "metrics.json missing after verify"
$VENV -c "
import json, sys
m = json.load(open('metrics.json'))
print('best:', m['best_name'], 'acc:', m['best_acc'], 'base:', m['base_acc'],
      '| L3 recall:', m['l3_recall_imp'], '| masked:', m['masked_norm'],
      '| variant:', m['variant_total'] - len(m['variant_misses']), '/', m['variant_total'])
gates = []
if m['variant_misses']: gates.append('variant misses: %s' % m['variant_misses'])
if m['best_acc'] < 0.75: gates.append('acc %.3f < 0.75' % m['best_acc'])
if m['l3_recall_imp'] < 0.90: gates.append('L3 recall %.3f < 0.90' % m['l3_recall_imp'])
if m['masked_norm'] < 0.65: gates.append('masked %.3f < 0.65' % m['masked_norm'])
if gates:
    print('GATE FAILED: ' + '; '.join(gates)); sys.exit(1)
print('gates: PASS')
" || fail "gates failed — old model stays live"

# --- 4. JS export + validation + page build ----------------------------------
$VENV export_model.py > retrain-export.log 2>&1 \
    || fail "model export failed"
$VENV gen_validation.py > retrain-genval.log 2>&1 \
    || fail "validation inputs failed"
node validate_js.js > retrain-jsval.log 2>&1 \
    || fail "JS validation script failed"
grep -q "mismatches: 0" retrain-jsval.log \
    || fail "JS port mismatch (see retrain-jsval.log)"
$VENV build_checker_app.py > retrain-html.log 2>&1 \
    || fail "checker.html build failed"
node test_checker_page.js > retrain-pagetest.log 2>&1 \
    || fail "page tests script failed"
grep -q "ALL PAGE TESTS PASSED" retrain-pagetest.log \
    || fail "page tests failed (see retrain-pagetest.log)"
echo "js + page: PASS"

# --- 5. deploy ----------------------------------------------------------------
cp "$APP/banglish_threat_model.pkl" "$APP/banglish_threat_model.pkl.bak"
cp banglish_threat_model.pkl lexicon.json "$APP/"
NEW_LEX=$(python3 -c "import json; print(len(json.load(open('$APP/lexicon.json'))))")
echo "deployed: lexicon_terms=$NEW_LEX"

# restart server: SIGTERM the gunicorn master (found via /proc, never pkill)
python3 - <<'PYEOF'
import os, signal
procs = {}
for pid in [d for d in os.listdir('/proc') if d.isdigit()]:
    try:
        cmd = open(f'/proc/{pid}/cmdline', 'rb').read().decode().replace('\0', ' ')
        ppid = int(open(f'/proc/{pid}/stat').read().split()[3])
    except Exception:
        continue
    if 'gunicorn' in cmd and '7860' in cmd:
        procs[int(pid)] = ppid
masters = [p for p, pp in procs.items() if pp not in procs]
for p in masters:
    os.kill(p, signal.SIGTERM)
print('killed gunicorn masters:', masters)
PYEOF
sleep 3
$APP/launch.sh
sleep 6
HEALTH=$(curl -s --noproxy '*' --max-time 10 http://127.0.0.1:7860/health)
echo "health: $HEALTH"
echo "$HEALTH" | grep -q '"ok":true' || fail "server did not come back after restart"

# live probe: newest learned word must predict its labeled level
PROBE=$($VENV - <<'PYEOF'
import json, urllib.request
words = json.load(open('learned_words.json'))
last = words[-1]
req = urllib.request.Request(
    'http://127.0.0.1:7860/api/check',
    data=json.dumps({"text": last["text"]}).encode(),
    headers={"Content-Type": "application/json"})
import os
os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)
os.environ["no_proxy"] = "*"
d = json.load(urllib.request.urlopen(req, timeout=30))
ok = d["level"] == int(last["level"])
print(("PROBE OK" if ok else "PROBE FAIL") +
      f" :: {last['text']!r} -> L{d['level']} (expected L{last['level']})")
raise SystemExit(0 if ok else 1)
PYEOF
)
echo "$PROBE"
echo "$PROBE" | grep -q "PROBE OK" || {
    echo "live probe failed — rolling back to .bak model"
    cp "$APP/banglish_threat_model.pkl.bak" "$APP/banglish_threat_model.pkl"
    python3 - <<'PYEOF'
import os, signal
procs = {}
for pid in [d for d in os.listdir('/proc') if d.isdigit()]:
    try:
        cmd = open(f'/proc/{pid}/cmdline', 'rb').read().decode().replace('\0', ' ')
        ppid = int(open(f'/proc/{pid}/stat').read().split()[3])
    except Exception:
        continue
    if 'gunicorn' in cmd and '7860' in cmd:
        procs[int(pid)] = ppid
masters = [p for p, pp in procs.items() if pp not in procs]
for p in masters:
    os.kill(p, signal.SIGTERM)
print('killed gunicorn masters for rollback:', masters)
PYEOF
    sleep 3
    $APP/launch.sh
    sleep 6
    curl -s --noproxy '*' --max-time 10 http://127.0.0.1:7860/health
    fail "rolled back — new model failed live probe"
}

# public health is informational only (tunnel may be down independently)
curl -s --max-time 20 https://checker.kalandar.me/health || echo "note: public health unreachable (tunnel?)"

# --- 6. upload static checker + record ----------------------------------------
# remote-storage resolves --path against its own cwd, not this shell's —
# absolute path only (relative "checker.html" failed NotFound 2026-11-18).
UPLOAD=$(/opt/hatch/bin/remote-storage upload-file --path /home/hatch/workspace/ml-lab-threat/checker.html 2>&1)
echo "$UPLOAD" | grep -q '"ok":true' || fail "static upload failed: $UPLOAD"
STATIC_URL=$(echo "$UPLOAD" | $VENV -c "import json,sys; print(json.load(sys.stdin)['url'])")
STATIC_EXP=$(echo "$UPLOAD" | $VENV -c "import json,sys; print(json.load(sys.stdin)['expires_at'])")
python3 - "$FP" "$STATIC_URL" "$STATIC_EXP" <<'PYEOF'
import datetime, json, sys
fp, url, exp = sys.argv[1], sys.argv[2], sys.argv[3]
m = json.load(open('metrics.json'))
rec = {"fingerprint": fp,
       "ts_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
       "learned_words": len(json.load(open('learned_words.json'))),
       "dataset_rows": m["rows"], "levels": m["levels"],
       "best": m["best_name"], "best_acc": m["best_acc"], "base_acc": m["base_acc"],
       "l3_recall": m["l3_recall_imp"], "masked_norm": m["masked_norm"],
       "variant": f"{m['variant_total'] - len(m['variant_misses'])}/{m['variant_total']}",
       "static_url": url, "static_expires": exp}
json.dump(rec, open('last_retrain.json', 'w'), ensure_ascii=False, indent=2)
print("RETRAIN DEPLOYED: acc", m["best_acc"], "| L3 recall", m["l3_recall_imp"],
      "| variant", rec["variant"], "| rows", m["rows"])
print("static:", url, "(expires", exp + ")")
PYEOF
