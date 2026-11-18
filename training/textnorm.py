#!/usr/bin/env python3
"""Single Python source of truth for normalize_text.

Copies live in: build_notebook.py (notebook cell, embedded via
inspect.getsource at build time), predictor.js (JS port — manual sync),
python-app/app.py (standalone deploy copy — manual sync).
gen_validation.py and verify_jsonl.py import from here.
If you change this function, re-run gen_validation.py — it validates the
JS port against Python behaviourally, so a drift shows up there.
"""
import re

LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
                      "7": "t", "@": "a", "$": "s"})


def normalize_text(s: str) -> str:
    # Digits hidden inside Bangla letters are leet (ক্রি1ম1নাল -> ক্রিমনাল).
    # Remove them BEFORE the leet mapping, or 1 becomes i and the scripts mix.
    s = re.sub(r"(?<=[\u0980-\u09FF])[0-9]+(?=[\u0980-\u09FF])", "", str(s))
    s = str(s).lower().translate(LEET)
    # Strip mask characters inside words: m*rbo -> mrbo, kh*n -> khn,
    # চু,দি -> চুদি. Commas count as mask characters (Nov 2026: "চু,,, দি").
    s = re.sub(r"(?<=\w)[*#._\-,]+(?=\w)", "", s)
    # Turn remaining mask characters into spaces ("চু,,, দি" -> "চু দি")
    s = re.sub(r"[*#_,]+", " ", s)
    # Collapse a letter repeated 3+ times down to 2
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
