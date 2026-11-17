#!/usr/bin/env python3
"""Assemble checker.html — one self-contained file with the EXACT trained
model embedded (model.json + lexicon.json + predictor.js, verbatim).
No server, no network: open the file and the checker works.

Verification baked in: asserts the embedded JSON parses back to exactly
the source files, and that predictor.js is embedded verbatim."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "predictor.js"), encoding="utf-8") as f:
    predictor_js = f.read()
with open(os.path.join(HERE, "app_ui.js"), encoding="utf-8") as f:
    app_js = f.read()
with open(os.path.join(HERE, "model.json"), encoding="utf-8") as f:
    model_json = f.read()
with open(os.path.join(HERE, "lexicon.json"), encoding="utf-8") as f:
    lexicon_json = f.read()

# Safety: nothing in the JSON may close the <script> block.
for name, blob in (("model.json", model_json), ("lexicon.json", lexicon_json)):
    assert "</script" not in blob.lower(), f"{name} contains </script"
assert predictor_js.strip().endswith("})(") or "BanglishChecker" in predictor_js

# The embedded JSON must parse back to exactly the source data.
assert json.loads(model_json) == json.loads(
    open(os.path.join(HERE, "model.json"), encoding="utf-8").read())
assert json.loads(lexicon_json) == json.loads(
    open(os.path.join(HERE, "lexicon.json"), encoding="utf-8").read())

HTML = """<!DOCTYPE html>
<html lang="bn">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Banglish Slang &amp; Threat Checker</title>
<style>
  :root { --ink:#1a1a2e; --muted:#6b7280; --line:#e5e7eb; --card:#ffffff; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans Bengali",sans-serif; background:#f6f7f9; color:var(--ink); }
  .wrap { max-width:760px; margin:0 auto; padding:28px 18px 60px; }
  header { margin-bottom:22px; }
  h1 { font-size:26px; margin:0 0 6px; letter-spacing:-.02em; }
  .sub { color:var(--muted); font-size:14px; line-height:1.5; }
  .tabs { display:flex; gap:8px; margin:20px 0 14px; }
  .tab-btn { border:1px solid var(--line); background:#fff; padding:9px 16px; border-radius:999px; font-size:14px; cursor:pointer; }
  .tab-btn.active { background:var(--ink); color:#fff; border-color:var(--ink); }
  .tab-pane { display:none; }
  .tab-pane.active { display:block; }
  textarea { width:100%; min-height:96px; padding:12px 14px; font-size:16px; line-height:1.5; border:1px solid var(--line); border-radius:12px; resize:vertical; font-family:inherit; background:#fff; }
  textarea:focus { outline:2px solid #93c5fd; border-color:#93c5fd; }
  .btn { display:inline-block; margin-top:12px; padding:11px 22px; font-size:15px; font-weight:600; border:none; border-radius:10px; background:var(--ink); color:#fff; cursor:pointer; }
  .btn.secondary { background:#fff; color:var(--ink); border:1px solid var(--line); }
  .examples { display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }
  .example-chip { border:1px solid var(--line); background:#fff; padding:7px 12px; border-radius:999px; font-size:13px; cursor:pointer; }
  .example-chip:hover { border-color:#9ca3af; }
  .examples-label { font-size:13px; color:var(--muted); margin-top:16px; }
  #result { margin-top:20px; }
  .verdict-card { border:2px solid; border-radius:14px; padding:18px; margin-bottom:14px; }
  .verdict-badge { display:inline-block; color:#fff; font-weight:700; font-size:14px; padding:5px 12px; border-radius:999px; }
  .verdict-title { font-size:20px; font-weight:700; margin-top:10px; }
  .verdict-conf { color:#374151; margin-top:4px; font-size:14px; }
  .prob-bars { margin-top:14px; }
  .prob-row { display:flex; align-items:center; gap:10px; margin-top:7px; font-size:13px; }
  .prob-label { width:64px; flex:none; }
  .prob-track { flex:1; height:8px; background:rgba(0,0,0,.08); border-radius:99px; overflow:hidden; }
  .prob-fill { height:100%; border-radius:99px; }
  .prob-val { width:38px; text-align:right; flex:none; }
  .details { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:18px; }
  .details h3 { font-size:14px; margin:18px 0 8px; }
  .details h3:first-child { margin-top:0; }
  .muted { color:var(--muted); font-size:14px; line-height:1.6; }
  .mono { font-family:ui-monospace,Menlo,monospace; font-size:14px; background:#f3f4f6; padding:10px 12px; border-radius:8px; word-break:break-word; }
  .chips { display:flex; flex-wrap:wrap; gap:8px; }
  .chip { padding:6px 11px; border-radius:999px; font-size:13px; font-weight:600; }
  .chip.sev1 { background:#fef3c7; color:#92400e; }
  .chip.sev2 { background:#ffedd5; color:#9a3412; }
  .chip.sev3 { background:#fee2e2; color:#991b1b; }
  .batch-summary { background:#fff; border:1px solid var(--line); border-radius:12px; padding:14px; font-weight:600; margin-bottom:12px; }
  table.batch-table { width:100%; border-collapse:collapse; background:#fff; border:1px solid var(--line); border-radius:12px; overflow:hidden; font-size:14px; }
  .batch-table th, .batch-table td { text-align:left; padding:10px 12px; border-bottom:1px solid var(--line); vertical-align:top; }
  .batch-table tr:last-child td { border-bottom:none; }
  .mini-badge { color:#fff; font-size:12px; font-weight:700; padding:3px 9px; border-radius:999px; white-space:nowrap; }
  footer { margin-top:34px; padding-top:18px; border-top:1px solid var(--line); color:var(--muted); font-size:13px; line-height:1.7; }
  #copyBtn { display:none; }
  @media (max-width:560px){ h1{font-size:22px;} .prob-label{width:56px;} }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Banglish Slang &amp; Threat Checker</h1>
    <div class="sub">ফেসবুক কমেন্টের গালি ও হুমকি চেকার — বাংলিশ + চাটগাঁইয়া। ট্রেনিং করা মডেল (TF-IDF + Logistic Regression) এই পেজের ভেতরেই চলে, কোনো সার্ভার লাগে না। ML Lab প্রজেক্ট, BGC Trust University।</div>
  </header>

  <div class="tabs">
    <button class="tab-btn active" data-tab="tab-single" type="button">একটা চেক করো</button>
    <button class="tab-btn" data-tab="tab-batch" type="button">অনেকগুলো একসাথে</button>
  </div>

  <div id="tab-single" class="tab-pane active">
    <textarea id="inputText" placeholder="এখানে কমেন্ট বা লেখা লিখুন… যেমন: tui ekta harami"></textarea>
    <button id="checkBtn" class="btn" type="button">Check করো</button>
    <button id="copyBtn" class="btn secondary" type="button">রিপোর্ট কপি করো</button>
    <div class="examples-label">উদাহরণ — একটাতে ট্যাপ করলেই চেক হবে:</div>
    <div class="examples" id="examples"></div>
    <div id="result"></div>
  </div>

  <div id="tab-batch" class="tab-pane">
    <textarea id="batchText" placeholder="প্রতি লাইনে একটা করে কমেন্ট লিখুন…"></textarea>
    <button id="batchBtn" class="btn" type="button">সবগুলো চেক করো</button>
    <div id="batchResult" style="margin-top:18px;"></div>
  </div>

  <footer>
    <strong>মডেল:</strong> TF-IDF (word 1–3 + char 2–6) + Logistic Regression, deep learning ছাড়া। ডেটাসেট ৬,৯০৪ রো · লেক্সিকন ৩,২১৪ শব্দ · ১৭ জেলা। Baseline (word-only, no norm) ০.৭২২ → Improved ০.৭৯১ · Level-3 recall ০.৯৭৪।<br>
    <strong>সত্যি সীমা:</strong> বাংলা অক্ষরে লেখা ভালো চেনে না — বাংলিশে (Roman) লিখুন। "কিরে বলদ"-এর মতো বন্ধুর মজা নাকি রাগ, সেটা শুধু লেখা দেখে বোঝা যায় না। এই পেজে যা লিখবেন তা কোথাও পাঠানো হয় না — সব আপনার ব্রাউজারেই হয়।
  </footer>
</div>

<script type="application/json" id="model-data">__MODEL_JSON__</script>
<script type="application/json" id="lexicon-data">__LEXICON_JSON__</script>
<script>
var MODEL_DATA = JSON.parse(document.getElementById("model-data").textContent);
var LEXICON_DATA = JSON.parse(document.getElementById("lexicon-data").textContent);
</script>
<script>
__PREDICTOR_JS__
</script>
<script>
__APP_JS__
</script>
</body>
</html>
"""

html = (HTML.replace("__MODEL_JSON__", model_json)
            .replace("__LEXICON_JSON__", lexicon_json)
            .replace("__PREDICTOR_JS__", predictor_js)
            .replace("__APP_JS__", app_js))

out = os.path.join(HERE, "checker.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
print(f"checker.html: {os.path.getsize(out)} bytes")

# Final self-check: the embedded JSON blocks parse back to the sources.
import re as _re
blocks = _re.findall(
    r'<script type="application/json" id="(?:model|lexicon)-data">(.*?)</script>',
    html, _re.S)
assert len(blocks) == 2
assert json.loads(blocks[0]) == json.loads(model_json)
assert json.loads(blocks[1]) == json.loads(lexicon_json)
assert predictor_js in html and app_js in html
print("embedded JSON == source files: OK | predictor.js + app_ui.js embedded verbatim: OK")
