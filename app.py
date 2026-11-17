#!/usr/bin/env python3
"""Banglish Threat Checker — Python backend.

The REAL trained sklearn model (banglish_threat_model.pkl) runs here,
server-side. No JS port, no approximation: the same pipeline as
verify_jsonl.py (normalize_text + TF-IDF word+char + LogReg).

Privacy: request text is NEVER stored or logged — it is checked and
forgotten. Only error lines (no text) go to the log.

Routes:
  GET  /            -> checker UI
  GET  /health      -> {"ok": true}
  POST /api/check   {"text": "..."} -> level, probs, clean, matches, naive
  POST /api/batch   {"texts": [...]} -> counts + per-line results
"""
import json
import os
import re

import joblib
from flask import Flask, jsonify, render_template, request

HERE = os.path.dirname(os.path.abspath(__file__))

LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
                      "7": "t", "@": "a", "$": "s"})


def normalize_text(s: str) -> str:
    # verify_jsonl.py-er sathe hubohu eki function — bodle okhaneo bodlate hobe.
    # Bangla okkhor-er majhe ASCII digit = leet (ক্রি1ম1নাল -> ক্রিমনাল).
    s = re.sub(r"(?<=[\u0980-\u09FF])[0-9]+(?=[\u0980-\u09FF])", "", str(s))
    s = str(s).lower().translate(LEET)
    # shobder bhetorer mask char tule fela: m*rbo -> mrbo, kh*n -> khn,
    # চু,দি -> চুদি (comma-o mask char, 2026-11-18: "চু,,, দি")
    s = re.sub(r"(?<=\w)[*#._\-,]+(?=\w)", "", s)
    # baki mask char space diye bodle dao ("চু,,, দি" -> "চু দি")
    s = re.sub(r"[*#_,]+", " ", s)
    # ek-i okkhor 3+ bar -> 2 bar
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


LEVELS = [
    {"n": 0, "label": "Safe", "bn": "স্বাভাবিক — গালি/হুমকি নেই"},
    {"n": 1, "label": "Level 1", "bn": "হালকা — মজা/তাচ্ছিল্য"},
    {"n": 2, "label": "Level 2", "bn": "মাঝারি — অপমানজনক"},
    {"n": 3, "label": "Level 3", "bn": "মারাত্মক — অশ্লীল / হুমকি"},
]

# "Facebook-style" naive keyword filter (baseline) — app_ui.js-er sathe eki list.
NAIVE_WORDS = ["fuck", "shit", "bitch", "bastard", "damn", "asshole",
               "kill", "murder", "die", "bloody", "idiot", "stupid",
               "হারামজাদা", "মাদারচোদ", "বাইনচোদ", "খানকি", "চোদনা", "শুয়োর"]


def naive_filter(raw: str):
    low = str(raw).lower()
    for w in NAIVE_WORDS:
        if w in low:
            return {"verdict": "flagged", "word": w}
    return {"verdict": "safe", "word": None}


def _esc_re(s: str) -> str:
    return re.escape(s)


app = Flask(__name__)

_model = joblib.load(os.path.join(HERE, "banglish_threat_model.pkl"))
_pipe = _model["model"]
with open(os.path.join(HERE, "lexicon.json"), encoding="utf-8") as f:
    _lexicon = json.load(f)
_latin_terms = [r for r in _lexicon if r["s"] == "latin"]
_bangla_terms = [r for r in _lexicon if r["s"] != "latin"]


def find_matches(raw: str, clean: str):
    found, seen = [], set()
    for lt in _latin_terms:
        pat = (r"(^|[^a-z0-9_])" + _esc_re(lt["t"]) + r"([^a-z0-9_]|$)")
        if re.search(pat, clean) and lt["t"] not in seen:
            seen.add(lt["t"])
            found.append({"t": lt["t"], "sev": lt["sev"], "c": lt["c"]})
    for bt in _bangla_terms:
        if bt["t"] in raw and bt["t"] not in seen:
            seen.add(bt["t"])
            found.append({"t": bt["t"], "sev": bt["sev"], "c": bt["c"]})
    found.sort(key=lambda m: -m["sev"])
    return found[:12]


def check_one(raw: str):
    clean = normalize_text(raw)
    if not clean:
        probs = [1.0, 0.0, 0.0, 0.0]
        return {"raw": raw, "level": 0, "probs": probs, "clean": clean,
                "naive": naive_filter(raw), "matches": []}
    probs = [float(v) for v in _pipe.predict_proba([clean])[0]]
    level = max(range(4), key=lambda i: probs[i])
    return {"raw": raw, "level": level, "probs": probs, "clean": clean,
            "naive": naive_filter(raw),
            "matches": find_matches(raw, clean)}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"ok": True, "model": _model.get("best_name", "unknown"),
                    "lexicon_terms": len(_lexicon)})


@app.route("/api/check", methods=["POST"])
def api_check():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", ""))[:2000]
    if not text.strip():
        return jsonify({"error": "empty text"}), 400
    return jsonify(check_one(text))


@app.route("/api/batch", methods=["POST"])
def api_batch():
    data = request.get_json(silent=True) or {}
    texts = data.get("texts", [])
    if not isinstance(texts, list) or not texts:
        return jsonify({"error": "empty texts"}), 400
    texts = [str(t)[:2000] for t in texts[:200] if str(t).strip()]
    results = [check_one(t) for t in texts]
    counts = [0, 0, 0, 0]
    for r in results:
        counts[r["level"]] += 1
    return jsonify({"counts": counts, "total": len(results),
                    "results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
