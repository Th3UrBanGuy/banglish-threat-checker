#!/usr/bin/env python3
"""Local verification of the JSONL dataset — mirrors notebook cells 2,4,5,6:
normalize -> split -> 3 models (improved) -> baseline vs improved ->
masked test. Prints honest numbers; run with the project venv."""
import json
import os
import random
import re

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

HERE = os.path.dirname(os.path.abspath(__file__))
random.seed(42)

from textnorm import LEET, normalize_text  # single source of truth


def mask_word(w: str) -> str:
    if len(w) < 4:
        return w
    i = random.randint(1, len(w) - 2)
    return w[:i] + random.choice(["*", "#"]) + w[i + 1:]


def make_masked(text: str) -> str:
    words = str(text).split()
    cands = [i for i, w in enumerate(words) if len(w) >= 4]
    if not cands:
        return text
    i = random.choice(cands)
    words[i] = mask_word(words[i])
    return " ".join(words)


df = pd.read_parquet(os.path.join(HERE, "banglish_threat_dataset.parquet"))
print("parquet dtypes:\n", df.dtypes)
df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"]).reset_index(drop=True)
df["clean"] = df["text"].apply(normalize_text)
print(f"rows: {len(df)} | levels: {dict(sorted(df['level'].value_counts().items()))}")

LEVEL_NAMES = {0: "0-normal", 1: "1-teasing", 2: "2-harassment", 3: "3-threat"}
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42,
                                     stratify=df["level"])
print(f"train: {len(train_df)} | test: {len(test_df)}")

aug_rows = []
for _, r in train_df[train_df["level"] >= 2].iterrows():
    m = make_masked(r["clean"])
    aug_rows.append({"text": m, "level": int(r["level"]),
                     "dialect": "augmented_masked", "source": "augmentation",
                     "clean": normalize_text(m)})
aug_df = pd.DataFrame(aug_rows)
train_aug = pd.concat([train_df, aug_df], ignore_index=True)
print(f"train + augmented: {len(train_aug)}")

word_tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=8000,
                             sublinear_tf=True, min_df=2)
char_tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6),
                             max_features=12000, sublinear_tf=True, min_df=2)
word_only = TfidfVectorizer(ngram_range=(1, 2), max_features=5000,
                            sublinear_tf=True, min_df=2)


def make_pipe(clf, improved=True):
    feats = (FeatureUnion([("word", word_tfidf), ("char", char_tfidf)])
             if improved else word_only)
    return Pipeline([("tfidf", feats), ("clf", clf)])


models = {
    "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "LinearSVC": LinearSVC(class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=200,
                                           class_weight="balanced", n_jobs=-1),
}
results = {}
for name, clf in models.items():
    pipe = make_pipe(clf, improved=True)
    pipe.fit(train_aug["clean"], train_aug["level"])
    pred = pipe.predict(test_df["clean"])
    acc = accuracy_score(test_df["level"], pred)
    results[name] = (pipe, acc)
    print(f"\n===== {name} — accuracy: {acc:.3f} =====")
    print(classification_report(
        test_df["level"], pred,
        target_names=[LEVEL_NAMES[i] for i in sorted(LEVEL_NAMES)],
        zero_division=0))
    print("confusion matrix (row=asol, col=predict):\n",
          confusion_matrix(test_df["level"], pred))

# Shipped model-er ashol probability (confidence %) dorkar — checker UI
# 4-level prob bar dekhay. LinearSVC-er predict_proba nai; or decision
# score-ke softmax korle confidence vul hobe. Tai LinearSVC ship-er
# candidate-i na — shudhu proba-ala model-er moddhe bachai. LogReg-kei
# ship kori, jodi na onno proba-model 1 point-er (0.01) beshi egiye thake.
best_name = "LogReg"
best_acc = results["LogReg"][1]
for n in results:
    if n in ("LogReg", "LinearSVC"):
        continue
    if results[n][1] > best_acc + 0.01:
        best_name, best_acc = n, results[n][1]
best_pipe = results[best_name][0]
print(f"\n>>> BEST (improved): {best_name} — accuracy {results[best_name][1]:.3f}")

base_pipe = make_pipe(LogisticRegression(max_iter=1000, class_weight="balanced"),
                      improved=False)
base_pipe.fit(train_df["text"], train_df["level"])
base_pred = base_pipe.predict(test_df["text"])
imp_pred = best_pipe.predict(test_df["clean"])
print(f"Baseline (word-only, no norm):  accuracy = {accuracy_score(test_df['level'], base_pred):.3f}")
print(f"Improved (word+char, norm+aug): accuracy = {accuracy_score(test_df['level'], imp_pred):.3f}")
print("baseline L3 recall:", round(recall_score(
    test_df["level"], base_pred, labels=[3], average="macro", zero_division=0), 3))
print("improved L3 recall:", round(recall_score(
    test_df["level"], imp_pred, labels=[3], average="macro", zero_division=0), 3))

masked_test = test_df.copy()
masked_test["masked_text"] = masked_test["clean"].apply(make_masked)
masked_test["masked_clean"] = masked_test["masked_text"].apply(normalize_text)
masked_base = round(accuracy_score(
    masked_test["level"], base_pipe.predict(masked_test["masked_text"])), 3)
masked_no_norm = round(accuracy_score(
    masked_test["level"], best_pipe.predict(masked_test["masked_text"])), 3)
masked_norm = round(accuracy_score(
    masked_test["level"], best_pipe.predict(masked_test["masked_clean"])), 3)
print("\n--- Masked test set accuracy ---")
print("baseline on masked text:        ", masked_base)
print("improved WITHOUT normalization:", masked_no_norm)
print("improved WITH normalization:   ", masked_norm)

# demo predict
def predict(text):
    c = normalize_text(text)
    lvl = int(best_pipe.predict([c])[0])
    print(f"  {text!r} -> Level {lvl} ({LEVEL_NAMES[lvl]})")


print("\n--- demo ---")
for t in ["kal class hobe?", "tui ekta pagol", "tui ekta beyadob",
          "toke marbo", "toke m*rbo", "tore dekhbo rastay",
          "tui aai ekbar, tore dekhilom", "basa chini ami, aitesi"]:
    predict(t)

joblib.dump({"model": best_pipe, "levels": LEVEL_NAMES, "best_name": best_name},
            os.path.join(HERE, "banglish_threat_model.pkl"))
print(f"\nsaved: banglish_threat_model.pkl ({best_name})")

# variant + demo gate (continuous-learning auto-retrain-er jonno) —
# notun shobdo shikhle ei test gulo-tei dhora pore jodi kichu venge jay.
from variant_tests import VARIANT_TESTS, DEMO_TESTS  # noqa: E402

variant_misses = []
for t, exp in VARIANT_TESTS + DEMO_TESTS:
    lv = int(best_pipe.predict([normalize_text(t)])[0])
    if lv != exp:
        variant_misses.append({"text": t, "expected": exp, "got": lv})
print(f"\n--- variant+demo: {len(VARIANT_TESTS) + len(DEMO_TESTS) - len(variant_misses)}/{len(VARIANT_TESTS) + len(DEMO_TESTS)} ---")
for m in variant_misses:
    print("  MISS:", m)

metrics = {
    "best_name": best_name,
    "best_acc": round(results[best_name][1], 3),
    "base_acc": round(accuracy_score(test_df["level"], base_pred), 3),
    "l3_recall_imp": round(recall_score(
        test_df["level"], imp_pred, labels=[3], average="macro",
        zero_division=0), 3),
    "l3_recall_base": round(recall_score(
        test_df["level"], base_pred, labels=[3], average="macro",
        zero_division=0), 3),
    "masked_base": masked_base,
    "masked_no_norm": masked_no_norm,
    "masked_norm": masked_norm,
    "variant_total": len(VARIANT_TESTS) + len(DEMO_TESTS),
    "variant_misses": variant_misses,
    "rows": len(df),
    "levels": {str(k): int(v)
               for k, v in sorted(df["level"].value_counts().items())},
}
with open(os.path.join(HERE, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print("saved: metrics.json")
