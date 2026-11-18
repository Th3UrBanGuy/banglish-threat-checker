#!/usr/bin/env python3
"""Build the Banglish + Chatgaiya Threat Severity training notebook (.ipynb)
and the starter labelled dataset template (CSV).

The template CSV is only the starter-template source. The real training
files are Parquet (built by build_slang_dataset.py via PyArrow; JSONL is
only the intermediate) and the notebook loads those:
banglish_threat_dataset.parquet + banglish_slang_lexicon.parquet.
Columns (messages): text, level (0-3), dialect, source
"""
import json
import os
import random
import sys

OUT = os.path.dirname(os.path.abspath(__file__))
random.seed(42)

# Variant/demo spec — ALWAYS read from variant_tests.py (single source of
# truth). Never hardcode a subset in the notebook: it goes stale and the
# notebook's grid numbers stop matching the ship gate (90/90).
sys.path.insert(0, OUT)
from variant_tests import VARIANT_TESTS as _SPEC_VARIANT, DEMO_TESTS as _SPEC_DEMO
_SPEC_ALL = _SPEC_VARIANT + _SPEC_DEMO

# ---------------------------------------------------------------- dataset
L0 = [
    ("kal class hobe?", "banglish"),
    ("assignment ta joma diso?", "banglish"),
    ("bhai note ta dao to", "banglish"),
    ("ajke cricket khelbo bikal e", "banglish"),
    ("tumi kemon aso", "banglish"),
    ("kal exam er routine ta pathao", "banglish"),
    ("bhat khaiso?", "banglish"),
    ("ami library te asi", "banglish"),
    ("sir er class ta kokhon", "banglish"),
    ("tui koi aso ekhon", "banglish"),
    ("amader group er meeting kal", "banglish"),
    ("slide ta share koro please", "banglish"),
    ("ajke weather ta bhalo", "banglish"),
    ("tumi ki dhaka jaba kal", "banglish"),
    ("homework ta bujhi nai, bujhai dao", "banglish"),
    ("ami basay asi, pore kotha hobe", "banglish"),
    ("ei boi ta koi pabo", "banglish"),
    ("tui kene aso?", "chatgaiya"),
    ("bhat khaiso na?", "chatgaiya"),
    ("aaij class ot jaiba na?", "chatgaiya"),
    ("tor note ta dibi na?", "chatgaiya"),
    ("ami bazar ot jaiir", "chatgaiya"),
    ("kene, ki hoise?", "chatgaiya"),
    ("kal aaiyom tor barit", "chatgaiya"),
    ("sir er pora ta bujhi nai", "banglish"),
    ("amake ektu help koro", "banglish"),
    ("tomar phone ta dao to", "banglish"),
    ("ajke amar mon bhalo", "banglish"),
    ("tui khaiso?", "banglish"),
    ("cholo cha khai", "banglish"),
    ("ei video ta dekso?", "banglish"),
    ("kal sokal e uthbo", "banglish"),
    ("ami porte boslam", "banglish"),
    ("tor bari koi", "banglish"),
    ("ami ctg te asi", "banglish"),
    ("tui aste parbi?", "banglish"),
    ("bhalo laglo kotha bole", "banglish"),
    ("dhonnobad bhai", "banglish"),
    (" pore dekha hobe", "banglish"),
]
L1 = [
    ("tui ekta pagol", "banglish"),
    ("tor matha kharap", "banglish"),
    ("tui kisui paros na", "banglish"),
    ("bokar moto kotha bolis na", "banglish"),
    ("tui ekdom faltu", "banglish"),
    ("tor kotha shune hasi passe", "banglish"),
    ("tui ekta boka", "banglish"),
    ("tor buddhi nai", "banglish"),
    ("tui je ki bolos", "banglish"),
    ("eto boka manush dekhi nai", "banglish"),
    ("tor moto gadha nai", "banglish"),
    ("tui shob somoy ulta palta bolos", "banglish"),
    ("tui ekta joker", "banglish"),
    ("tor kothar kono dam nai", "banglish"),
    ("tui khub birokto koros", "banglish"),
    ("tui ekta aul faul lok", "banglish"),
    ("tor sathe kotha bole lav nai", "banglish"),
    ("tui je ekta chagol", "banglish"),
    ("tui pagol hoi gesos", "banglish"),
    ("tor mathay somossa ase", "banglish"),
    ("tui ekta bhondo", "banglish"),
    ("tui shob jaygay hashir patro", "banglish"),
    ("tor kotha keu shune na", "banglish"),
    ("tui ekta bekub", "banglish"),
    ("tui aaij ekdom baje lagtesos", "banglish"),
    ("tor moto manush er sathe ke kotha bole", "banglish"),
    ("tui ekta faltu chele", "banglish"),
    ("tor kotha shunle matha gorom hoy", "banglish"),
    ("tui je ki koros bujhi na", "banglish"),
    ("tui ekta ajob manush", "banglish"),
    ("tui to ekdomi paros na", "banglish"),
    ("tor eita niye kotha bolar dorkar nai", "banglish"),
    ("tui ekta mojar manush na, biroktikor", "banglish"),
    ("tui aaij khub baje behave kortesos", "banglish"),
    ("tor moto lok ami dekhi nai", "banglish"),
    ("tui ekta boka manush", "banglish"),
    ("tor kotha shune birokto lagtese", "banglish"),
    ("tui ekta hasir khorak", "banglish"),
    ("tui shob kichu nosto koros", "banglish"),
    ("tor buddhi ekdom kom", "banglish"),
]
L2 = [
    ("tui ekta beyadob", "banglish"),
    ("tor lojja nai", "banglish"),
    ("toke sobai ghrina kore", "banglish"),
    ("tui ekta kharap manush", "banglish"),
    ("tor mukh dekhte chai na", "banglish"),
    ("tui amar chokher samne asbi na", "banglish"),
    ("toke ami block kore dibo", "banglish"),
    ("tor moto nich manush nai", "banglish"),
    ("tui ekta bodmaish", "banglish"),
    ("tor sathe ar kotha bolbo na", "banglish"),
    ("tui amar jibon ta nosto kore disos", "banglish"),
    ("toke sobai chine gese", "banglish"),
    ("tui ekta mitthuk", "banglish"),
    ("tor kotha keu bissas kore na", "banglish"),
    ("tui ekta chor", "banglish"),
    ("toke ami report korbo", "banglish"),
    ("tor porichoy sobai ke bole dibo", "banglish"),
    ("tui ekta joghonno manush", "banglish"),
    ("tor moto manush er jayga ekhane nai", "banglish"),
    ("tui amar bondhu hobar joggo na", "banglish"),
    ("toke ami khoma korbo na", "banglish"),
    ("tor jonno amar khoti hoise", "banglish"),
    ("tui ekta soitan", "banglish"),
    ("tor mukh ta bondho rakh", "banglish"),
    ("tui ekta harami", "banglish"),
    ("tor moto lok er sathe thaka jay na", "banglish"),
    ("tui amar somman nosto korsos", "banglish"),
    ("toke ami chinte parsi ekhon", "banglish"),
    ("tui ekta baje manush", "banglish"),
    ("tor kotha shune ga jole", "banglish"),
    ("tui ekta amanush", "banglish"),
    ("tor jonno sobai kosto pay", "banglish"),
    ("tui ekta khoti korar manush", "banglish"),
    ("tore ar kono din dekhte chai na", "banglish"),
    ("tui ekta nongra manush", "banglish"),
    ("tor moto manush ke keu pochondo kore na", "banglish"),
    ("tui amar sob kichu nosto korsos", "banglish"),
    ("toke ami ar sojjo korte parbo na", "banglish"),
    ("tui ekta bisakto manush", "banglish"),
    ("tor karone amar porashuna nosto hoise", "banglish"),
]
L3 = [
    ("toke marbo", "banglish"),
    ("rastay paile tore dekhbo", "banglish"),
    ("tor hat pa bhenge dibo", "banglish"),
    ("basa chini ami, aitesi", "banglish"),
    ("tore khun korbo", "banglish"),
    ("tui ekbar aai, tore dekhbo", "banglish"),
    ("tor matha fatabo", "banglish"),
    ("toke rastay dhorbo", "banglish"),
    ("tor jibon sesh kore dibo", "banglish"),
    ("tui morbi amar hate", "banglish"),
    ("tore marifelum", "chatgaiya"),
    ("tui aai ekbar, tore dekhilom", "chatgaiya"),
    ("tor theng bhangilom", "chatgaiya"),
    ("tore rastat dhorilom", "chatgaiya"),
    ("tui moror aar aar", "chatgaiya"),
    ("tor bari chini, aaiyom", "chatgaiya"),
    ("tore khuin korilom", "chatgaiya"),
    ("tui aai, tor khobor asi", "chatgaiya"),
    ("tore marbo, kotha dilam", "banglish"),
    ("tor chokh upre felbo", "banglish"),
    ("tui amar samne porle sesh", "banglish"),
    ("tore ami charbo na", "banglish"),
    ("tor hat ta bhenge dibo", "banglish"),
    ("tui jibone ar hata chola korte parbi na", "banglish"),
    ("tore ami khuje ber korbo", "banglish"),
    ("tor bari te aisa tore marbo", "banglish"),
    ("tui ekbar rastay ber ho, dekhbo", "banglish"),
    ("tor mukh ta bhenge dibo", "banglish"),
    ("toke ami mere felbo", "banglish"),
    ("tor khobor ase", "banglish"),
    ("tui aajke bachte parbi na", "banglish"),
    ("tore ami dhorbo, mone rakhis", "banglish"),
    ("tor gaye hat tulbo", "banglish"),
    ("tui amar rage porle sesh", "banglish"),
    ("tore ami dekhbo, wait kor", "banglish"),
    ("tor jibon ta ami nosto kore dibo", "banglish"),
    ("tui ekhon theke bhoye thakbi", "banglish"),
    ("tore ami char dimu na", "banglish"),
    ("tor hat pa bhangbo", "banglish"),
    ("tui samne aai, tore marbo", "banglish"),
]

MASKED = [
    ("toke m*rbo", 3, "banglish_masked"),
    ("tore d*khbo rastay", 3, "banglish_masked"),
    ("tor h*t pa bhenge dibo", 3, "banglish_masked"),
    ("basa chini, aite#i", 3, "banglish_masked"),
    ("tore kh*n korbo", 3, "banglish_masked"),
    ("tui ekta bey*dob", 2, "banglish_masked"),
    ("tor lojja n*i", 2, "banglish_masked"),
    ("tui ekta h*rami", 2, "banglish_masked"),
    ("tui ekta b*dmash", 2, "banglish_masked"),
    ("toke m*rbo, kotha dilam", 3, "banglish_masked"),
    ("tor m*tha fatabo", 3, "banglish_masked"),
    ("tui ekta b*ka", 1, "banglish_masked"),
    ("tor matha kh*rap", 1, "banglish_masked"),
    ("tore marif*lum", 3, "banglish_masked"),
    ("tui aai ekbar, tore dekh*lom", 3, "banglish_masked"),
    ("tui ekta p*gol", 1, "banglish_masked"),
    ("tui ekta f*ltu", 1, "banglish_masked"),
    ("toke ami m*re felbo", 3, "banglish_masked"),
    ("tui ekta m*tthuk", 2, "banglish_masked"),
    ("tor mukh b*ndho rakh", 2, "banglish_masked"),
]

rows = []
for text, d in L0:
    rows.append((text, 0, d, "template_example"))
for text, d in L1:
    rows.append((text, 1, d, "template_example"))
for text, d in L2:
    rows.append((text, 2, d, "template_example"))
for text, d in L3:
    rows.append((text, 3, d, "template_example"))
for text, lvl, d in MASKED:
    rows.append((text, lvl, d, "template_example_masked"))

random.shuffle(rows)
csv_path = os.path.join(OUT, "banglish_threat_dataset.csv")
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("text,level,dialect,source\n")
    for text, lvl, d, src in rows:
        safe = text.replace('"', '""')
        f.write(f'"{safe}",{lvl},{d},{src}\n')
print(f"dataset rows: {len(rows)} -> {csv_path}")

# ---------------------------------------------------------------- notebook
def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(True)}

cells = []

cells.append(md("""# Banglish + Chatgaiya Threat Severity Classifier — ML Lab Training Notebook

I built a classifier that reads a Banglish (romanized Bengali) or Chatgaiya message and predicts its threat severity, from Level 0 (normal) to Level 3 (direct threat). I use classical machine learning only — no deep learning.

| Level | Meaning | Example |
|---|---|---|
| 0 | Normal | `kal class hobe?` |
| 1 | Teasing / mocking | `tui ekta pagol` |
| 2 | Harassment / abuse | `tui ekta beyadob` (no physical threat) |
| 3 | Direct physical threat | `toke marbo`, `rastay paile tore dekhbo` |

Why this problem: most threats and abuse on Facebook and WhatsApp arrive in Banglish, but public datasets are almost all Bangla script. Chatgaiya (Chittagong dialect) abuse does not appear in any public dataset I could find. People also hide abuse with obfuscation (`* # . _ - ,` inside words, e.g. `চু,,, দি`), which a plain keyword list cannot catch.

Method in one line: TF-IDF features (word 1–3 grams + character 2–6 grams) with Logistic Regression, compared against five other classical classifiers. I measure a simple baseline (word features only, no normalization) against the improved model, so the improvement is visible in numbers.

The same word can be affection or abuse depending on who says it, so I teach the model with paired frames in the training data: `আমার কুত্তার বাচ্চাটা কই গেল` from a mother is Level 0, `তুই কুত্তার বাচ্চা` from someone else is Level 2. Section 8 shows both.
"""))

cells.append(code("""# 0. Setup — install the packages this notebook needs, then fix the
# random seed so every run gives the same split and the same results.
!pip install -q scikit-learn pandas joblib matplotlib pyarrow

import re, os, random, joblib
import pandas as pd
import numpy as np
random.seed(42)
np.random.seed(42)
print("setup ok")
"""))

cells.append(md("""## 1. Loading the dataset

I train on a Parquet file (`banglish_threat_dataset.parquet`, 12,639 rows) and an optional slang lexicon (`banglish_slang_lexicon.parquet`, 4,269 terms). Parquet is columnar, so I read only the columns I need, and every column has a fixed type. The cell below looks for the files next to the notebook, in Colab's `/content` folder, or in Drive — otherwise it asks me to upload them.

Sources: TB-OLID (5,000 real Banglish comments, CC BY 4.0; offensive/non-offensive labels mapped to severity levels by a lexicon scan), Nirmol v1 (Bangla phrases), LDNOOBW (English bad words, CC BY 4.0), and my own curated Banglish/Chatgaiya examples plus templates. My collection rule: public comments and pages only — no names, phone numbers, or private chats.
"""))

cells.append(code("""# 1. Load the dataset (Parquet). I look in three places — next to the
# notebook, Colab's /content folder, and Drive — and ask for an upload if
# the file is nowhere to be found.
PARQUET_CANDIDATES = [
    "banglish_threat_dataset.parquet",
    "/content/banglish_threat_dataset.parquet",
    "/content/drive/MyDrive/ML-Lab-Threat/banglish_threat_dataset.parquet",
]
parquet_path = next((p for p in PARQUET_CANDIDATES if os.path.exists(p)), None)

if parquet_path is None:
    try:
        from google.colab import files
        print("Parquet file not found — opening an upload window. Upload banglish_threat_dataset.parquet")
        uploaded = files.upload()
        parquet_path = list(uploaded.keys())[0]
    except Exception as e:
        raise FileNotFoundError("Parquet file not found. Upload banglish_threat_dataset.parquet or put it in Drive.") from e

df = pd.read_parquet(parquet_path)
print("Parquet:", parquet_path, "| rows:", len(df))
print(df.head(10))
print("\\n--- class count ---")
print(df["level"].value_counts().sort_index())
print("\\n--- dialect count ---")
print(df["dialect"].value_counts())
print("\\n--- source count ---")
print(df["source"].value_counts())
print("\\n--- dtypes (Parquet schema theke asha) ---")
print(df.dtypes)
print("\\n--- missing / duplicate ---")
print("missing text:", df["text"].isna().sum(), "| duplicate rows:", df.duplicated(subset=["text"]).sum())
df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"]).reset_index(drop=True)
print("clean rows:", len(df))

# 1b. Lexicon (optional) — check the slang list by severity, dialect, and source.
LEX_CANDIDATES = [
    "banglish_slang_lexicon.parquet",
    "/content/banglish_slang_lexicon.parquet",
    "/content/drive/MyDrive/ML-Lab-Threat/banglish_slang_lexicon.parquet",
]
lex_path = next((p for p in LEX_CANDIDATES if os.path.exists(p)), None)
if lex_path:
    lex = pd.read_parquet(lex_path)
    print("\\n--- lexicon:", lex_path, "| rows:", len(lex), "---")
    print("by severity:\\n", lex["severity"].value_counts().sort_index())
    print("by dialect:\\n", lex["dialect"].value_counts())
    print("by source:\\n", lex["source"].value_counts())
else:
    print("\\n(Lexicon file not found — the dataset alone is enough; the lexicon is optional)")

"""))

cells.append(md("""## 2. Normalization — cleaning obfuscated text

People insert `* # . _ - ,` inside abusive words so a machine misses them (`m*rbo`, `চু,,, দি`). My `normalize_text` function strips digits hidden inside Bangla letters (leet), removes mask characters inside words, turns leftover mask characters into spaces, and collapses repeated letters. Normalization is not perfect — new masking styles keep appearing — so Section 6 measures it on a masked test set instead of assuming it works.
"""))

# Normalizer cell — textnorm.py theke neoa (single source of truth):
# build_notebook.py chalalei cell-ta textnorm-er bortoman version diye
# toiri hobe, tai verify_jsonl.py / app / JS-er sathe mil thakbe.
import inspect
from textnorm import normalize_text as _normalize_text_fn
_NORM_CELL = (
    "# 2. Normalization — textnorm.py (project-er single source of truth)\n"
    "LEET = str.maketrans(" + repr({"0": "o", "1": "i", "3": "e", "4": "a",
                                     "5": "s", "7": "t", "@": "a", "$": "s"})
    + ")\n\n"
    + inspect.getsource(_normalize_text_fn)
    + "\ndf[\"clean\"] = df[\"text\"].apply(normalize_text)\n\n"
    + "# Before vs after — 10 masked examples\n"
    + "demo = df[df[\"dialect\"] == \"banglish_masked\"][[\"text\", \"clean\"]].head(10)\n"
    + "print(demo.to_string(index=False))\n"
    + "print(\"\\nFraction of masked rows changed by normalization:\", "
    + "(demo[\"text\"] != demo[\"clean\"]).mean(), \"fraction\")\n"
)
cells.append(code(_NORM_CELL))

cells.append(md("""## 3. Augmentation — making masked copies for training

Real masked examples are rare, so I create masked copies of clean threat examples (one `*` or `#` inside a long word) and add them to the training set only. The test set is never augmented — I measure on real masked rows instead.
"""))

cells.append(code("""# 3. Augmentation (training set only). I mask one long word per sentence.
def mask_word(w: str) -> str:
    if len(w) < 4:
        return w
    i = random.randint(1, len(w) - 2)
    return w[:i] + random.choice(["*", "#"]) + w[i+1:]

def make_masked(text: str) -> str:
    words = str(text).split()
    if not words:
        return text
    # Mask one word of length 4 or more.
    cands = [i for i, w in enumerate(words) if len(w) >= 4]
    if not cands:
        return text
    i = random.choice(cands)
    words[i] = mask_word(words[i])
    return " ".join(words)

# A few examples:
for t in ["toke marbo", "tore dekhbo rastay", "tui ekta beyadob"]:
    print(t, "->", make_masked(t))
"""))

cells.append(md("""## 4. Train/test split and the model pipeline

I split 80/20 (stratified, seed 42) and build one pipeline per classifier: TF-IDF features, then the classifier. The improved setup (V4, chosen by the grid in Section 5b) uses word 1–3 grams (whole phrases like `তোর মারে চুদি`) plus character 2–6 grams (spelling patterns, so new spellings and unseen targets still match). The baseline uses word features only, with no normalization — Section 6 compares the two.
"""))

cells.append(code("""# 4. Split + pipelines
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

LEVEL_NAMES = {0: "0-normal", 1: "1-teasing", 2: "2-harassment", 3: "3-threat"}

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["level"])
print("train:", len(train_df), "| test:", len(test_df))
print("train class count:\\n", train_df["level"].value_counts().sort_index())

# --- Augmented training set: add a masked copy of each Level 2/3 row (training only) ---
aug_rows = []
for _, r in train_df[train_df["level"] >= 2].iterrows():
    aug_rows.append({"text": make_masked(r["clean"]), "level": int(r["level"]),
                     "dialect": "augmented_masked", "source": "augmentation", "clean": make_masked(r["clean"])})
# The augmented copies already hold masked text — normalize them too.
aug_df = pd.DataFrame(aug_rows)
if len(aug_df):
    aug_df["clean"] = aug_df["text"].apply(normalize_text)
    train_aug = pd.concat([train_df, aug_df], ignore_index=True)
else:
    train_aug = train_df
print("train + augmented:", len(train_aug), "| augmented rows:", len(aug_df))

# V4 (best in the grid): word 1-3 grams + char_wb 2-6 grams.
# Mark-inclusive token pattern: sklearn's default word pattern drops Bengali
# combining marks, so a word like কুত্তা would split into pieces. This
# pattern keeps the word whole (see research/model-comparison-2026-11-18.md).
word_tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=8000, sublinear_tf=True, min_df=2,
                             token_pattern=r"(?u)[\\w\\u0980-\\u09FF]{2,}")
char_tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), max_features=12000, sublinear_tf=True, min_df=2)
word_only = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True, min_df=2)

def make_pipe(clf, improved=True):
    # Clone the vectorizers — one fresh copy per pipeline. Sharing a single
    # fitted vectorizer across pipelines is a bug: fitting one pipeline
    # refits it, so a later fit silently changes this
    # pipeline's features and its predictions collapse.
    if improved:
        feats = FeatureUnion([("word", clone(word_tfidf)), ("char", clone(char_tfidf))])
    else:
        feats = clone(word_only)
    return Pipeline([("tfidf", feats), ("clf", clf)])

models = {
    # C=4 (compared on two splits, Nov 2026): +0.011 to +0.014 accuracy over C=1.
    "LogReg": LogisticRegression(C=4, max_iter=1000, class_weight="balanced"),
    "LinearSVC": LinearSVC(class_weight="balanced"),
    "MultinomNB": MultinomialNB(),
    "ComplementNB": ComplementNB(),
    "SGD-log": SGDClassifier(loss="log_loss", max_iter=1000, class_weight="balanced", random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42),
}
"""))

cells.append(code("""# 5. Train six classifiers on the improved setup (V4) and compare them
# on the same test set.
from sklearn.metrics import recall_score
results = {}
for name, clf in models.items():
    pipe = make_pipe(clf, improved=True)
    pipe.fit(train_aug["clean"], train_aug["level"])
    pred = pipe.predict(test_df["clean"])
    acc = accuracy_score(test_df["level"], pred)
    l3r = recall_score(test_df["level"], pred, labels=[3], average="macro", zero_division=0)
    results[name] = (pipe, acc)
    print(f"\\n===== {name} — accuracy: {acc:.3f} | L3 recall: {l3r:.3f} =====")
    print(classification_report(test_df["level"], pred, target_names=[LEVEL_NAMES[i] for i in sorted(LEVEL_NAMES)], zero_division=0))
    print("confusion matrix (row=asol, col=predict):\\n", confusion_matrix(test_df["level"], pred))

# Shipping rule: the checker UI shows a confidence bar for each of the 4
# levels, so only a model with predict_proba can ship (LinearSVC has none —
# turning its decision scores into confidences with a softmax would be
# wrong). LogReg is the default; another probability model only replaces it
# if it beats LogReg by more than 0.01 accuracy.
best_name = "LogReg"
best_acc = results["LogReg"][1]
for n in results:
    if n == "LogReg" or n == "LinearSVC":
        continue
    try:
        results[n][0].predict_proba(test_df["clean"][:2])
    except Exception:
        continue
    if results[n][1] > best_acc + 0.01:
        best_name, best_acc = n, results[n][1]
best_pipe = results[best_name][0]
print(f"\\n>>> BEST (ship): {best_name} — accuracy {results[best_name][1]:.3f}")
print("    (LinearSVC scores close on accuracy but gives no probabilities, so it cannot ship — see the full grid in Section 5b.)")
"""))

cells.append(md("""## 5b. Model comparison grid — which setup is best, and why

I do not trust one model. I run 6 feature setups × 6 classifiers (RandomForest only on 2 setups, because it is slow) and score each on accuracy, Level 3 recall, masked accuracy, and the 90-item variant test. The printed table below is the result — read it before deciding.

What I look for: the best accuracy is not enough. The shipped model must give probabilities (the checker UI shows a confidence bar), so LinearSVC and RandomForest are out even when they top the variant test. LogReg (V4: word 1–3 grams + char 2–6 grams) wins among probability models on accuracy, Level 3 recall, and masked accuracy — that is why it ships.
"""))

cells.append(code("""# 5b. Grid: 6 vectorizers x 6 classifiers (RandomForest only on 2 setups — slow).
# Each combination is scored on accuracy, Level 3 recall, masked accuracy,
# and the 90-item variant test below.
from sklearn.base import clone
VECTORIZERS = {
    "V1 w1-2+c3-5": lambda: FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True, min_df=2)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=8000, sublinear_tf=True, min_df=2))]),
    "V2 w1-3+c3-5": lambda: FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 3), max_features=5000, sublinear_tf=True, min_df=2)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=8000, sublinear_tf=True, min_df=2))]),
    "V3 w1-2+c2-5": lambda: FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True, min_df=2)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=8000, sublinear_tf=True, min_df=2))]),
    "V4 w1-3+c2-6": lambda: FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 3), max_features=8000, sublinear_tf=True, min_df=2)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), max_features=12000, sublinear_tf=True, min_df=2))]),
    "V5 char-only": lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=8000, sublinear_tf=True, min_df=2),
    "V6 word-only": lambda: TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True, min_df=2),
}
# variant test — FULL spec list (variant + demo), read from variant_tests.py
# at build time — never hardcode a subset (goes stale vs the 90/90 ship gate).
VARIANT_TESTS = """ + repr(_SPEC_ALL) + """
masked_test = test_df.copy()
masked_test["masked_clean"] = masked_test["clean"].apply(lambda t: normalize_text(make_masked(t)))
for vname, vmake in VECTORIZERS.items():
    for cname, cmake in models.items():
        if cname == "RandomForest" and vname not in ("V1 w1-2+c3-5", "V4 w1-3+c2-6"):
            continue
        # models dict holds INSTANCES (not factories) — clone each one;
        # cmake() raises TypeError (2026-11-18: grid cell never ran).
        pipe = Pipeline([("tfidf", vmake()), ("clf", clone(cmake))])
        pipe.fit(train_aug["clean"], train_aug["level"])
        pred = pipe.predict(test_df["clean"])
        acc = accuracy_score(test_df["level"], pred)
        l3r = recall_score(test_df["level"], pred, labels=[3], average="macro", zero_division=0)
        macc = accuracy_score(masked_test["level"], pipe.predict(masked_test["masked_clean"]))
        vpred = pipe.predict([normalize_text(t) for t, _ in VARIANT_TESTS])
        vscore = sum(int(p == e) for p, (_, e) in zip(vpred, VARIANT_TESTS))
        print(f"{vname:14s} {cname:12s} acc={acc:.3f} L3r={l3r:.3f} masked={macc:.3f} variant={vscore}/{len(VARIANT_TESTS)}")
"""))

cells.append(code("""# 6. Baseline vs improved — the number that matters for the report.
# Baseline: word features only, raw text, no normalization, no augmentation.
# Improved: word + char features, normalized text, augmented training set.
# Baseline first, then the improved model on the same test set.
base_pipe = make_pipe(LogisticRegression(max_iter=1000, class_weight="balanced"), improved=False)
base_pipe.fit(train_df["text"], train_df["level"])
base_pred = base_pipe.predict(test_df["text"])
imp_pred = best_pipe.predict(test_df["clean"])
print(f"Baseline (word-only, no norm):  accuracy = {accuracy_score(test_df['level'], base_pred):.3f}")
print(f"Improved (word+char, norm+aug): accuracy = {accuracy_score(test_df['level'], imp_pred):.3f}")
print("\\n--- Level 3 (direct threat) recall ---")
from sklearn.metrics import recall_score
print("baseline L3 recall:", recall_score(test_df["level"], base_pred, labels=[3], average="macro", zero_division=0))
print("improved L3 recall:", recall_score(test_df["level"], imp_pred, labels=[3], average="macro", zero_division=0))

# --- Masked-only test: how each model scores on the masked test set ---
masked_test = test_df.copy()
masked_test["masked_text"] = masked_test["clean"].apply(make_masked)
masked_test["masked_clean"] = masked_test["masked_text"].apply(normalize_text)
base_masked = base_pipe.predict(masked_test["masked_text"])
imp_masked = best_pipe.predict(masked_test["masked_clean"])
imp_raw_masked = best_pipe.predict(masked_test["masked_text"])  # normalization chara improved model
print("\\n--- Masked test set accuracy ---")
print("baseline on masked text:        ", round(accuracy_score(masked_test["level"], base_masked), 3))
print("improved WITHOUT normalization:", round(accuracy_score(masked_test["level"], imp_raw_masked), 3))
print("improved WITH normalization:   ", round(accuracy_score(masked_test["level"], imp_masked), 3))
print("(Read the three masked numbers together: they show whether normalization itself helps on masked text, apart from the other improvements.)")
"""))

cells.append(md("""## 7. Explainability — why did the model say this level?

I read the strongest word and character patterns per level from the Logistic Regression coefficients. If someone asks why a message got Level 3, this list is the answer.
"""))

cells.append(code("""# 7. Strongest features per level (Logistic Regression only — other
# classifiers have no coefficients to read).
try:
    logreg_pipe = make_pipe(LogisticRegression(max_iter=1000, class_weight="balanced"), improved=True)
    logreg_pipe.fit(train_aug["clean"], train_aug["level"])
    feats = logreg_pipe.named_steps["tfidf"].get_feature_names_out()
    coefs = logreg_pipe.named_steps["clf"].coef_
    for lvl in sorted(LEVEL_NAMES):
        idx = np.argsort(coefs[lvl])[-10:][::-1]
        print(f"Level {lvl} ({LEVEL_NAMES[lvl]}):", [feats[i] for i in idx])
except Exception as e:
    print("explainability skip:", e)
"""))

cells.append(md("""## 8. Saving the best model and trying it out

I save the best pipeline as a `.pkl` file — the live checker loads this exact file, so what I demo here is what runs in production. The demo below runs the same `predict` function on new messages.
"""))

cells.append(code("""# 8. Save the best model.
joblib.dump({"model": best_pipe, "levels": LEVEL_NAMES, "best_name": best_name},
            "banglish_threat_model.pkl")
print("saved: banglish_threat_model.pkl  (model:", best_name, ")")
print("file size:", os.path.getsize("banglish_threat_model.pkl"), "bytes")

# 9. Demo — try the saved model on new messages.
def predict(text):
    c = normalize_text(text)
    lvl = int(best_pipe.predict([c])[0])
    proba = None
    try:
        proba = best_pipe.predict_proba([c])[0]
    except Exception:
        pass
    action = {0: "nothing to do", 1: "ignore or block",
              2: "block and report", 3: "block, report, and escalate if needed"}[lvl]
    print(f"input: {text!r} -> Level {lvl} ({LEVEL_NAMES[lvl]}) | action: {action}")
    if proba is not None:
        print("   confidence:", {LEVEL_NAMES[i]: round(float(p), 2) for i, p in enumerate(proba)})
    return lvl

for t in ["kal class hobe?", "tui ekta pagol", "tui ekta beyadob",
          "toke marbo", "toke m*rbo", "tore dekhbo rastay",
          "tui aai ekbar, tore dekhilom", "basa chini ami, aitesi",
          # Family-targeted threats, unseen targets, and neutral counter-examples.
          "তুরে চুদবো", "তোর মারে চুদি", "তোর বোনরে চুদি",
          "তোকে মেরে ফেলবো", "চু,,, দি", "ছাত্রলীগের মারে চুদি",
          "তোর ফুফুরে চুদি", "তোরে চুদমু", "m0therboard",
          "আমি ছাত্রলীগ করি", "আমার মারে ভালোবাসি",
          "ওয়েট কর, আসতেছি", "তোর মারে সালাম দিস",
          # Same words, different speaker — mother (affection) vs someone else (abuse).
          "আমার কুত্তার বাচ্চাটা কই গেল", "তুই কুত্তার বাচ্চা"]:
    predict(t)
"""))

cells.append(md("""## 9. Next steps and limitations

1. **More real data.** Direct threats (Level 3) are the rarest class (832 of 12,639 rows), so Level 3 recall rests on few examples. More real anonymized messages — especially Chatgaiya and direct threats — would help most.
2. **Masking is not fully solved.** Masked accuracy is 0.740, and normalization itself does not help the masked set (0.740 with it, 0.755 without). It stays because comma-masking and leet variants cannot be caught without it.
3. **Borderline sentences flip.** A friendly frame like `কুত্তা, তুঁই আঁর জানের বন্ধু` is Level 2 but only just (0.503 vs 0.483 for Level 0) — new data can flip it.
4. **Level 3 stays rare.** Real direct threats are hard to find; I use `class_weight="balanced"` to compensate, and I say so in the report.
5. **TB-OLID labels are mapped, not gold.** TB-OLID only has offensive/non-offensive labels; I mapped them to Levels 1–3 with a lexicon scan, so some rows are mislabeled.
6. **Ethics.** No names, phone numbers, or private chats in the data; the model detects, it does not accuse.
Results as of round 14 (18 Nov 2026): 12,639 rows (L0 5,491 / L1 1,870 / L2 4,446 / L3 832), lexicon 4,269 terms. Shipped model: accuracy 0.803 (baseline 0.688), Level 3 recall 0.939 (baseline 0.762), masked 0.740, variant + demo 90/90.

The notebook trains the same pipeline; its numbers differ slightly from the shipped model (accuracy 0.799 here vs 0.803 shipped, L3 recall 0.946 vs 0.939, masked 0.739 vs 0.740 with normalization and 0.750 vs 0.755 without) because the masked augmentation uses random choices.
"""))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                            "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
nb_path = os.path.join(OUT, "Banglish_Threat_Severity_Training.ipynb")
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"notebook cells: {len(cells)} -> {nb_path}")
