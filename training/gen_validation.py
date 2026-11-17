#!/usr/bin/env python3
"""Generate validation inputs + Python (sklearn) predictions for the JS port.
Run first, then: node validate_js.js  (compares JS vs these predictions)."""
import json
import os
import random

import joblib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
random.seed(7)

m = joblib.load(os.path.join(HERE, "banglish_threat_model.pkl"))
pipe = m["model"]

# normalize_text now lives in textnorm.py (single source of truth) —
# verify_jsonl.py imports the same copy. predictor.js + app.py + the
# notebook cell are deployment copies; gen_validation.py validates the
# JS port against Python behaviourally, so a drift shows up there.
from textnorm import normalize_text


demo = ["kal class hobe?", "tui ekta pagol", "tui ekta beyadob",
        "toke marbo", "toke m*rbo", "tore dekhbo rastay",
        "tui aai ekbar, tore dekhilom", "basa chini ami, aitesi",
        "tui ekta harami", "tui ekta madarchod", "tui ekta magir fut",
        "h a r a m i", "h4rami", "tui ekta sudanir poya",
        "oda koi jor", "kire bolod", "bhat khaisos", "tore dekhilom",
        "tui ekta bolod", "magir fuya", "kire harami", "kire madarchod",
        "m4d4rch0d", "m.a.d.a.r.c.h.o.d", "tui ekta m*darchod",
        "h a r a m j a d a", "toke m*rbo kalke", "tui ekta b0l0d",
        "ami tomake valobashi", "class kokhon shuru hobe?",
        "tui ekta chudani", "o chudanir poya", "tui ekta dum",
        "shon, tore dekhilom", "tor moto bolod nai", "tui ekdom faltu"]

# GenZ / political variant tests (user-er list, 2026-11-17) — prottek-ta
# target level-er sathe pore compare kora hobe, shudhu JS-vs-Python na.
variants = ["tui ekta dhon", "tui ekta don", "tui ekta dhonchhipi",
            "tui ekta donsipi", "tui ekta criminal", "tui ekta kri1m1nal",
            "ক্রি1ম1নাল", "ক্রিমনাল", "tui ekta razakar", "tui ekta rojokor",
            "tui ekta rojkar", "ros kora", "রস করা", "s3x", "s 3 x",
            "এস থ্রি এক্স", "tui ekta sex", "rajakar tui",
            "তুই একটা রাজাকার", "তুই একটা রজাকর", "তুই একটা রজকার",
            "তুই একটা দন", "তুই একটা মাদারবোর্ড", "tui ekta motherboard",
            "tui ekta gopali", "tui ekta fascist", "tui ekta hybrid",
            "tui ekta akam", "tui ekta baal", "বাল", "tui ekta putki maro",
            "তুই একটা পুটকি মারো", "tui ekta opodartho", "tui ekta lollipop",
            "tui ekta dhandabaji", "tui ekta july jongi"]

edge = ["", "   ", "a", "i", "12345", "!!!", "tui",
        "তুই একটা হারামজাদা", "আমি ভালো আছি", "তোর মারে দেখমু",
        "অডা কই যর?", "কিরে বলদ, ভাত খাইছস?",
        "HARAMI", "TuI EkTa HaRaMi", "haaaaaramiii",
        "tui\tekta\nharami", "tui  ekta   harami",
        "m@darchod", "h4r4mi", "b41nch0d", "kh4nkir fut",
        "tui ekta kuttar baccha", "kuttar baccha tui",
        "tor bapre dekhmu", "tore marbo ami", "tor khobor ase"]

df = pd.read_parquet(os.path.join(HERE, "banglish_threat_dataset.parquet"))
sample = df["text"].sample(n=200, random_state=7).tolist()
inputs = demo + edge + variants + sample
# dedupe, keep order
seen = set()
inputs = [t for t in inputs if not (t in seen or seen.add(t))]

preds = []
for t in inputs:
    c = normalize_text(t)
    if not c:
        preds.append({"level": 0, "probs": [1.0, 0.0, 0.0, 0.0]})
        continue
    proba = pipe.predict_proba([c])[0]
    preds.append({"level": int(proba.argmax()),
                  "probs": [round(float(p), 6) for p in proba]})

with open(os.path.join(HERE, "js_validation_inputs.json"), "w",
          encoding="utf-8") as f:
    json.dump(inputs, f, ensure_ascii=False)
with open(os.path.join(HERE, "js_validation_py_preds.json"), "w",
          encoding="utf-8") as f:
    json.dump(preds, f, ensure_ascii=False)
print(f"inputs: {len(inputs)} -> js_validation_inputs.json")
