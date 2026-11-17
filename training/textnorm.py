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
    # Bangla okkhor-er majhe ASCII digit = leet (ক্রি1ম1নাল -> ক্রিমনাল).
    # LEET-er AGE korte hobe, naile 1 -> i hoye mixed-script hoye jay.
    s = re.sub(r"(?<=[\u0980-\u09FF])[0-9]+(?=[\u0980-\u09FF])", "", str(s))
    s = str(s).lower().translate(LEET)
    # shobder bhetorer mask char tule fela: m*rbo -> mrbo, kh*n -> khn,
    # চু,দি -> চুদি. Comma-o mask char (2026-11-18: "চু,,, দি").
    s = re.sub(r"(?<=\w)[*#._\-,]+(?=\w)", "", s)
    # baki mask char space diye bodle dao ("চু,,, দি" -> "চু দি")
    s = re.sub(r"[*#_,]+", " ", s)
    # ek-i okkhor 3+ bar -> 2 bar
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
