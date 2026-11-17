# Training backup — Banglish Threat Checker

This folder is the full training side of the project, kept in the repo so the
work can continue anywhere later (including a future shift to LLM fine-tuning
— the same labeled data is the fine-tuning corpus).

- `banglish_threat_dataset.parquet` — 6,932 labeled messages (levels 0-3),
  Parquet. Sources: TB-OLID (CC BY 4.0), Nirmol v1, LDNOOBW, Meta AI gali
  lists, political-page term sampling, curated Banglish + Chatgaiya rows.
- `banglish_slang_lexicon.parquet` — 3,217 lexicon terms with severity.
- `build_slang_dataset.py` — rebuilds both files from `sources/` + the
  curated sections. `learned_words.json` = words learned from live testing
  (continuous learning) — section 12 of the builder reads it.
- `textnorm.py` — THE single source of truth for normalization (all other
  copies are manually synced).
- `verify_jsonl.py` — retrains (LogReg, word TF-IDF 1-3 + char_wb 2-6),
  writes `metrics.json` + the .pkl; also runs the variant+demo gate.
- `retrain_and_deploy.sh` — gated auto-retrain: rebuild -> retrain -> gates
  (variant 56/56, acc >= 0.75, L3 recall >= 0.90, masked >= 0.65, JS 0
  mismatches, page tests) -> deploy -> live probe -> upload static checker.
  Runs every 30 min via cron `checker-auto-retrain`.
- `METHODOLOGY.md` (Bengali) — full method, limits, and why the model can
  generalize (char n-grams) but cannot invent unseen cultural phrases.
- `research/` (project root, not in this folder yet) — Chittagonian harvests
  (articles, papers, datasets, Facebook groups) feed future learned words.

Live: https://checker.kalandar.me (Flask, gunicorn :7860).
Constraints for the lab report: ML only, NO deep learning. Baseline = the
toy 18-word keyword filter in app_ui.js (demo of the gap) AND the word-only
ML model (0.741) — never conflate the two.
