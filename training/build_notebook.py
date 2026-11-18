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

**Goal:** ekta Banglish (romanized Bengali) message pore tar threat severity mapo — Level 0 theke Level 3.

| Level | Mane | Example pattern |
|---|---|---|
| 0 | Normal | `kal class hobe?` |
| 1 | Teasing / mocking | `tui ekta pagol` |
| 2 | Harassment / abuse | `tui ekta beyadob` (sharirik humki nai) |
| 3 | Direct physical threat | `toke marbo`, `rastay paile tore dekhbo` |

**Uniqueness (ma'am-er jonno):**
1. Banglish — ashol humki Facebook/WhatsApp-e beshirbhag Banglish-e ashe, kintu public dataset shob Bangla script-er.
2. Chatgaiya dialect — Chittagong-er bhashar humki/gali kono public dataset-e nai.
3. Obfuscation — gali/humkir majhe `* # . _ - ,` dhukiye AI-ke faki dewar chesta (jemon `চু,,, দি`); amra normalization + augmentation diye seta dhori.
4. Baseline vs improved accuracy — ma'am bolechen, ager kajer upor accuracy baranoi creativity. Tai notebook-e baseline (word TF-IDF only, no normalization) vs improved (char n-gram + normalization + augmentation + Chatgaiya data) alada kore mapa hoy.
5. **"Nearby" generalization** — model shudhu mukhosto shobdo chine na: word n-gram (1–3) diye pasher shobdo (phrase, jemon `তোর মারে চুদি`) ar char n-gram (2–6) diye banan-er mil (`মাদারবোর্ড` ≈ `মাদারচোদ`, `ছাএলীগ` ≈ `ছাত্রলীগ`) dhore — tai training-e nai emon notun gali/target-o dhorte pare (Section 5b-te proman).

**Model:** TF-IDF (word 1–3 gram + char 2–6 gram) + 6-ta classifier-er tulona: Logistic Regression, LinearSVC, MultinomialNB, ComplementNB, SGD (log loss), Random Forest. **No deep learning** — ma'am-er nirdesh.

**Kazi-r JobLens notebook-er moto structure:** load → clean → TF-IDF → model compare → per-class report → best model `.pkl` hishebe save. JobLens-e CV theke 32 job role predict korto; ekhane message theke 4-ta severity level predict kori.

**DUITA TRACK (2026-11-18) — ek model-e 2 kaj na, 2-ta alada procedure:**
- **Track A — word-level (shipped model, Section 1–9):** slang shobdo/phrase theke severity — 9,794 row (lexicon + template + TB-OLID + curated). Eta-i live checker-e chole (checker.kalandar.me). Shobdo chinte parle-i kaj hoy — simple kaj, simple model-i enough.
- **Track B — sentence-level (notun, alada, Section 10):** PURA bakko pore ador/gali/neutral bojha — 21,714 sentence row (Kothon, ChatgaiyyaAlap, ChatgaiyyaBench, ChattogramSent, ONUBAD + Meta AI/Gemini synthetic sentence). Eta Track A-r bodli NA — ek-i shobdo ador-eo bola jay, gali-teo bola jay (`কুত্তার বাচ্চা` — ma bolle ador, onno keu bolle gali); bakko-r context bojhar jonno alada train kora hoyeche. Duitar tulona + keno alada, Section 10-e ache.
"""))

cells.append(code("""# 0. Setup — Colab-e run korle ei cell-ei shob install hobe
# Local-e run korleo kono khoti nai (already installed hole pip skip kore na, tai -q)
!pip install -q scikit-learn pandas joblib matplotlib pyarrow

import re, os, random, joblib
import pandas as pd
import numpy as np
random.seed(42)
np.random.seed(42)
print("setup ok")
"""))

cells.append(md("""## 1. Dataset load koro

Dataset ekhon **Parquet** format-e (`.parquet`) — PyArrow diye banano.
Parquet columnar: shudhu je column dorkar setai pore, baki column
direct skip kore (jemon shudhu `text` lagle `level`/`dialect`/`source`
disk theke porei na). File-er footer-e metadata thake (schema, row group,
min/max stats), ar prottek column-er fixed data type thake
(`text: string`, `level: int8`, `severity: int8`, `auto_translit: bool`) —
tai type niye jhamela nai, kaj soja. JSONL shudhu intermediate build file.

File 2-ta:
- `banglish_threat_dataset.parquet` — training messages (text, level, dialect, source)
- `banglish_slang_lexicon.parquet` — slang/bad-word lexicon (term, script, language,
  dialect, category, severity, source, auto_translit)

**Kotha theke elo:**
- **TB-OLID** (5,000 comment, CC BY 4.0) — ashol Banglish comment, offensive O/N
  label soho. Offensive row-er severity lexicon-scan diye mapa (threat word
  thakle 3, sexual/strong gali thakle 2, mild thakle 1, na mille default 2).
- **Nirmol v1** (1,182 Bangla phrase) — lexicon-er Bangla part + machine-e
  Roman kora Banglish part (`auto_translit=true` lekha thake).
- **LDNOOBW en** (403 English bad word, CC BY 4.0).
- **curated** — nijer hate lekha Banglish + Chatgaiya gali/humki (public
  list-e Chatgaiya nai) + synthetic template sentence.
- ager 179-row template example-o vitore ache (`source=template_example`).

**Colab-e:** 2-ta `.parquet` file-ei notebook-er pashe upload korte hobe
(Files → Upload), athoba Drive mount kore Drive theke load koro.

**Data collection rule (ethics):** public comment/page theke nibe, karo naam /
phone / porichoy rakhbe na, private chat karo permission chara nibe na.
"""))

cells.append(code("""# 1. Load dataset (Parquet) — koyekta jaygay khuje dekhe, jekhane pabe sekhan theke ney
PARQUET_CANDIDATES = [
    "banglish_threat_dataset.parquet",
    "/content/banglish_threat_dataset.parquet",
    "/content/drive/MyDrive/ML-Lab-Threat/banglish_threat_dataset.parquet",
]
parquet_path = next((p for p in PARQUET_CANDIDATES if os.path.exists(p)), None)

if parquet_path is None:
    try:
        from google.colab import files
        print("Parquet khuje pelam na — upload window khulche, banglish_threat_dataset.parquet upload koro")
        uploaded = files.upload()
        parquet_path = list(uploaded.keys())[0]
    except Exception as e:
        raise FileNotFoundError("Parquet pelam na. banglish_threat_dataset.parquet upload koro ba Drive-e rakho.") from e

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

# 1b. Lexicon (thakle) — slang/bad-word list-er hisab dekhe nao
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
    print("\\n(lexicon file pelam na — dataset diyei cholbe, lexicon optional)")

# 1c. Track B — sentence dataset (alada file, alada procedure — Section 10)
SENT_CANDIDATES = [
    "banglish_sentence_dataset.parquet",
    "/content/banglish_sentence_dataset.parquet",
    "/content/drive/MyDrive/ML-Lab-Threat/banglish_sentence_dataset.parquet",
]
sent_path = next((p for p in SENT_CANDIDATES if os.path.exists(p)), None)
if sent_path:
    sdf = pd.read_parquet(sent_path)
    sdf = sdf.dropna(subset=["text"]).drop_duplicates(subset=["text"]).reset_index(drop=True)
    print("\\n--- Track B sentence dataset:", sent_path, "| rows:", len(sdf), "---")
    print(sdf["level"].value_counts().sort_index())
    print("sources:\\n", sdf["source"].value_counts())
else:
    sdf = None
    print("\\n(Track B sentence file pelam na — Track A diyei cholbe; Track B dekhte banglish_sentence_dataset.parquet upload koro)")
"""))

cells.append(md("""## 2. Normalization — obfuscation (masking) porishkar kora

Manush gali/humkir majhe `* # . _ - ,` dhukiye dey jate machine na dhorte pare: `m*rbo`, `kh*n`, `চু,,, দি`.
Amader `normalize_text` ja kore:
1. choto hater okkhor-e ney (Banglish-e case-er mane nai)
2. Bangla okkhor-er majhe ASCII digit thakle tule fele — leet (`ক্রি1ম1নাল` → `ক্রিমনাল`); eta LEET mapping-er AGE korte hoy, naile `1` → `i` hoye mixed-script hoye jay
3. shobder **bhetorer** mask char (`* # . _ - ,`) tule fele — kintu shobdo gulo alada rakhe (`m*rbo` → `mrbo`, `চু,দি` → `চুদি`)
4. baki mask char (comma soho) space diye bodle dey (`চু,,, দি` → `চু দি`)
5. ek-i okkhor 3+ bar thakle 2-te namay (`marbooo` → `marboo`)
6. leet mapping: `0→o`, `1→i`, `3→e`, `4→a`, `5→s`, `7→t`, `@→a`, `$→s`
7. extra space porishkar

**Sotorkota:** normalization 100% perfect na — notun masking style ashtei thakbe. Tai Section 6-e masked test set-e age vs pore alada kore mapbo.
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
    + "# age vs pore — 10-ta example dekhe nao\n"
    + "demo = df[df[\"dialect\"] == \"banglish_masked\"][[\"text\", \"clean\"]].head(10)\n"
    + "print(demo.to_string(index=False))\n"
    + "print(\"\\nMasked row-e normalization bodleche:\", "
    + "(demo[\"text\"] != demo[\"clean\"]).mean(), \"fraction\")\n"
)
cells.append(code(_NORM_CELL))

cells.append(md("""## 3. Augmentation — mask kora version nije banano

Asol masked example kom thakbe — tai clean threat example theke nije masked version banabo (train set-e **shudhu**):
- ekta key word-er majhe `*` dhukano
- `#` dhukano

Ete model masked rup-o chinte shikhbe. **Test set-e augmentation korbo na** — test-e ashol masked row + deterministically mask kora copy diye mapbo (Section 6).
"""))

cells.append(code("""# 3. Augmentation (train-only)
def mask_word(w: str) -> str:
    if len(w) < 4:
        return w
    i = random.randint(1, len(w) - 2)
    return w[:i] + random.choice(["*", "#"]) + w[i+1:]

def make_masked(text: str) -> str:
    words = str(text).split()
    if not words:
        return text
    # lomba word gulor ekta-te mask dao
    cands = [i for i, w in enumerate(words) if len(w) >= 4]
    if not cands:
        return text
    i = random.choice(cands)
    words[i] = mask_word(words[i])
    return " ".join(words)

# example dekho
for t in ["toke marbo", "tore dekhbo rastay", "tui ekta beyadob"]:
    print(t, "->", make_masked(t))
"""))

cells.append(md("""## 4. Train / test split + model pipeline

**Feature (V4 — grid-e best promanito):** word TF-IDF (1–3 gram, 8,000 feature) + character TF-IDF (2–6 gram, `char_wb`, 12,000 feature).
- Word 3-gram keno: phrase dhore — `তোর মারে চুদি` ekta feature hishebe thake, tai shudhu `মারে` dekhe vul kore na.
- Char 2-gram keno: 2-okkhore-r stem (`চু`, `মা`) o dhora pore.
- Char 6-gram keno: lomba variant (`মাদারবোর্ড`, `ছাত্রলীগ`) purota mile.
- Char n-gram keno dorkari: Banglish-e banan ekekjon ekekbhabe lekhe (`dekhbo / dekhmu / deikha`) — char pattern millei model dhorbe, ar ekta okkhor mask thakleo baki pattern thake.

**Model (6-ta tulona):** Logistic Regression, LinearSVC, MultinomialNB, ComplementNB, SGD (log loss), Random Forest — shob `class_weight="balanced"` (Level 3 kom, tai imbalance fix). Grid-er folafol Section 5b-te.

**Baseline vs improved:**
- Baseline = shudhu word TF-IDF, normalization chara, augmentation chara
- Improved = word + char TF-IDF, normalization soho, augmentation soho
"""))

cells.append(code("""# 4. Split + pipelines
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

# --- augmented train set: Level 2/3 row-er masked copy jog koro (train-only) ---
aug_rows = []
for _, r in train_df[train_df["level"] >= 2].iterrows():
    aug_rows.append({"text": make_masked(r["clean"]), "level": int(r["level"]),
                     "dialect": "augmented_masked", "source": "augmentation", "clean": make_masked(r["clean"])})
# augmented copy-r clean field-e already masked text ase; normalize kore nao
aug_df = pd.DataFrame(aug_rows)
if len(aug_df):
    aug_df["clean"] = aug_df["text"].apply(normalize_text)
    train_aug = pd.concat([train_df, aug_df], ignore_index=True)
else:
    train_aug = train_df
print("train + augmented:", len(train_aug), "| augmented rows:", len(aug_df))

# V4 (grid-e best): word 1-3 gram + char_wb 2-6 gram
# Mark-inclusive token pattern (2026-11-18, comparison round 2): sklearn's
# default word pattern drops Bengali combining marks — see verify_jsonl.py.
word_tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=8000, sublinear_tf=True, min_df=2,
                             token_pattern=r"(?u)[\\w\\u0980-\\u09FF]{2,}")
char_tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), max_features=12000, sublinear_tf=True, min_df=2)
word_only = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True, min_df=2)

def make_pipe(clf, improved=True):
    feats = FeatureUnion([("word", word_tfidf), ("char", char_tfidf)]) if improved else word_only
    return Pipeline([("tfidf", feats), ("clf", clf)])

models = {
    # C=4 (2026-11-18, comparison round 2, confirmed rs42+rs7): +0.011..+0.014 acc.
    "LogReg": LogisticRegression(C=4, max_iter=1000, class_weight="balanced"),
    "LinearSVC": LinearSVC(class_weight="balanced"),
    "MultinomNB": MultinomialNB(),
    "ComplementNB": ComplementNB(),
    "SGD-log": SGDClassifier(loss="log_loss", max_iter=1000, class_weight="balanced", random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42),
}
"""))

cells.append(code("""# 5. Train 6 model — improved setup-e (V4), tulona koro
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

# Ship korar niyom: checker UI-te 4-level prob bar dorkar, tai shudhu
# predict_proba-ala model-i ship korte pare (LinearSVC-er seta nai —
# or decision score-ke softmax kore confidence banano jabe na, vul hobe).
# LogReg-i default; onno kono proba-model jodi LogReg-ke 0.01-er beshi
# (1 point) chariye jay, tokhon sheta ship hobe.
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
print("    (LinearSVC accuracy-te kachakachi holeo proba dite pare na, tai ship hoy na — Section 5b-te full grid dekho)")
"""))

# ================================================================ Track B
cells.append(md("""## 10. Track B — sentence-level classification (ALADA procedure)

**Track A ar Track B ek na — tai train-o alada, data-o alada, model-o alada file-e thakbe.**

|  | Track A (word-level) | Track B (sentence-level) |
|---|---|---|
| **Proshno** | Ei shobdo/phrase-ta koto severe? | Ei PURA bakko-ta ador, gali, na normal? |
| **Data** | 9,794 row (lexicon + template + TB-OLID + curated) | 21,714 sentence row (Kothon, ChatgaiyyaAlap, ChatgaiyyaBench, ChattogramSent, ONUBAD + Meta AI/Gemini synthetic) |
| **Keno alada?** | Shobdo chinle-i hoy | Ek-i shobdo ador-eo hoy, gali-teo hoy — context lage |
| **Example** | `চুদি` → L3 | `আমার কুত্তার বাচ্চাটা কই গেল` (ma bolche) → L0, `তুই কুত্তার বাচ্চা` (onno keu) → L2 |

**Track B-r data kotha theke:**
- **Kothon** (7,986 sentence, CC BY 4.0) — Chittagonian everyday sentence
- **ChatgaiyyaAlap** (3,148 sentence + 1,417 dictionary entry, CC BY 4.0)
- **ChatgaiyyaBench** (3,009 vulgarity + 1,582 romanized + 204 lexicon, CC BY 4.0) — silver label (machine-er dewa, vul hote pare) — amader marker-rule diye L0/L1/L2/L3-te mapa hoyeche
- **ChattogramSent v2** (2,703 sentence) — sentiment data, kintu 59% ChatgaiyyaAlap-er duplicate (Vashantor leak — report-e lekha ache)
- **ONUBAD** (605 sentence + 787 word + 113 clause)
- **Gemini Deep Research** (138 entry) + **paired frames** (22 row — ek-i shobder ador vs gali version, hate-banao)
- **Meta AI synthetic** (cholche — 2026-11-18 porjonto 200 sentence: 10-ta shobdo × 10 ador + 10 gali) — **SYNTHETIC, ashol manush-er lekha na** — eta METHODOLOGY-te sposhto kore lekha ache

**Sotorkota (ma'am-er jonno shotti kotha):** sentence data-r 97% L0 (21,112 / 21,714) — L3 matro 28-ta row! Tai accuracy dekhte bhalo lagleo (0.96) ashol kotha macro-F1 ar per-class recall-e. L3 test-e matro 6-ta row — L3 recall (0.50) mane 6-tar moddhe 3-ta dhoreche — choto sample, beshi bhorosha kora jabe na. Sentence data aro barche (Meta AI loop cholche — proti batch-e 5-ta shobdo × 10–15-ta sentence); data barle Track B abar train kore number update korte hobe.
"""))

cells.append(code("""# 10a. Track B — 5-ta classifier, ek-i V4 feature, ek-i niyom (80/20, seed 42)
# Track A-r moto-i: normalization + train-only masked augmentation (L2+ row)
from sklearn.metrics import f1_score, recall_score
if sdf is None:
    print("Track B data nai — banglish_sentence_dataset.parquet upload kore abar chalao")
else:
    sdf["clean"] = sdf["text"].apply(normalize_text)
    s_train, s_test = train_test_split(sdf, test_size=0.2, random_state=42, stratify=sdf["level"])
    print("Track B train:", len(s_train), "| test:", len(s_test),
          "| test L3 n =", int((s_test["level"] == 3).sum()), "(choto sample — L3 recall-e beshi bhorosha koro na)")
    s_aug_rows = []
    for _, r in s_train[s_train["level"] >= 2].iterrows():
        m = make_masked(r["clean"])
        s_aug_rows.append({"text": m, "level": int(r["level"]),
                           "dialect": "augmented_masked", "source": "augmentation",
                           "clean": normalize_text(m)})
    s_train_aug = pd.concat([s_train, pd.DataFrame(s_aug_rows)], ignore_index=True)
    print("train + augmented:", len(s_train_aug))
    s_masked = s_test.copy()
    s_masked["masked_clean"] = s_masked["clean"].apply(lambda t: normalize_text(make_masked(t)))

    s_models = {
        "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "LinearSVC": LinearSVC(class_weight="balanced"),
        "MultinomNB": MultinomialNB(),
        "ComplementNB": ComplementNB(),
        "RandomForest": RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42),
    }
    s_results = {}
    for name, clf in s_models.items():
        pipe = make_pipe(clf, improved=True)
        pipe.fit(s_train_aug["clean"], s_train_aug["level"])
        pred = pipe.predict(s_test["clean"])
        acc = accuracy_score(s_test["level"], pred)
        l3r = recall_score(s_test["level"], pred, labels=[3], average="macro", zero_division=0)
        macc = accuracy_score(s_test["level"], pipe.predict(s_masked["masked_clean"]))
        mf1 = f1_score(s_test["level"], pred, average="macro", zero_division=0)
        s_results[name] = (pipe, acc)
        print(f"\\n===== Track B {name} — acc: {acc:.3f} | L3 recall: {l3r:.3f} | masked: {macc:.3f} | macro-F1: {mf1:.3f} =====")
        print(classification_report(s_test["level"], pred, target_names=[LEVEL_NAMES[i] for i in sorted(LEVEL_NAMES)], zero_division=0))

    # Track B baseline: word-only, no norm
    s_base = make_pipe(LogisticRegression(max_iter=1000, class_weight="balanced"), improved=False)
    s_base.fit(s_train["text"], s_train["level"])
    s_base_pred = s_base.predict(s_test["text"])
    print(f"\\nTrack B baseline (word-only, no norm): acc = {accuracy_score(s_test['level'], s_base_pred):.3f} | "
          f"L3 recall = {recall_score(s_test['level'], s_base_pred, labels=[3], average='macro', zero_division=0):.3f}")
    print("(baseline 0.847 — karon 97% L0, shudhu L0 boleo 0.97 pawa jeto na, kintu L1/L2 pray dhore-i na)")
"""))

cells.append(md("""## 10b. Track B-r folafol — konta best, keno? (2026-11-18, VM-e mapa)

**Accuracy dekhte bhalo (0.93–0.98) — kintu eta dhoka!** Data-r 97% L0, tai shudhu "shob L0" boleo 0.97 accuracy pawa jay. Asol bichar **macro-F1 + per-class recall**-e — L1 (teasing) ar L2 (harassment) koto dhoreche, setai Track B-r kaj.

| Model | Acc | L3 recall | Masked | Macro-F1 | L0 recall | L1 recall | L2 recall | Variant (78) |
|---|---|---|---|---|---|---|---|---|
| **LogReg** | 0.960 | 0.500 | 0.937 | **0.592** | 0.972 | **0.514** | **0.595** | 33/78 |
| LinearSVC | 0.980 | 0.500 | 0.964 | 0.662 | 0.995 | 0.286 | 0.532 | 31/78 |
| MultinomNB | 0.940 | 0.000 | 0.941 | 0.273 | 0.963 | 0.000 | 0.203 | 24/78 |
| ComplementNB | 0.932 | 0.500 | 0.922 | 0.406 | 0.951 | 0.086 | 0.329 | 26/78 |
| RandomForest | 0.980 | 0.333 | 0.979 | 0.621 | 0.997 | 0.286 | 0.456 | 20/78 |
| Baseline (word-only) | 0.847 | 0.333 | — | — | — | — | — | — |

**Track B = Logistic Regression — keno:**
- **Proba dite pare** (LinearSVC/RandomForest-er cheye ei karone-i egiye — checker UI-te confidence bar dorkar; SVC-er proba nai).
- **L1 recall shobcheye bhalo (0.514)** — teasing vs neutral bojha-i Track B-r shobcheye kothin kaj; SVC (0.286) ar RF (0.286) ekhane pray ondho.
- **L2 recall shobcheye bhalo (0.595)** — harassment sentence dhora.
- **Balanced** — macro-F1 proba-model-er moddhe shera (0.592); NB (0.273–0.406) onek pichiye.
- LinearSVC-r macro-F1 (0.662) beshi holeo L1-e 0.286 — accuracy-r pichone lukano durbolota; ar proba na thakay ship kora jay na (Track A-r ek-i niyom).
- **Demo-te shotti kotha:** `কুত্তার বাচ্চা`-r ma-vs-onno-keu parthokko ekhono model shikhe nai (duita-i L2 dey), ar rag-er `ভইঙ্গা` frame L0 dey — paired frame matro 22-ta row bole. Meta AI loop-e ador/gali sentence barche (proti shobde 10+10), shegulo jog hole ei gap kombe. Section 10c-r demo-te eta sposhto dekha jay — lukano hoy nai.

**Variant 33/78 — eta KHARAP na, EXPECTED:** variant test gulo word-level kothin test (ekok shobdo, masked form, notun target) — sentence data diye train kora model ekok shobdo chinbe na, etai shavabik. Eta-i proman kore **Track B Track A-r bodli hote pare na** — duita alada kaj, duita alada model. Track A word chine (78/78), Track B bakko-r context bojhe (L1 0.514, L2 0.595).

**Track A vs Track B — shesh kotha (duita mile):**

> Note: Track A-r number (0.777) holo **word-only split**-er (9,794 row) — jate Track B-r sathe tulona fair hoy. Shipped model Section 5-e **pura 12,376-row parquet**-e train hoy (sentence row shoho) ar 0.781 pay — 0.004 gap-ta sentence row-er contribution, kono vul na.

|  | Track A (word) | Track B (sentence) |
|---|---|---|
| **Chosen model** | **Logistic Regression (V4)** | **Logistic Regression (V4)** |
| **Accuracy** | 0.777 (baseline 0.686) | 0.960 (baseline 0.847) |
| **L3 recall** | 0.954 | 0.500 (test-e L3 matro 6-ta — choto sample) |
| **Masked** | 0.739 | 0.937 |
| **Macro-F1** | 0.804 | 0.592 |
| **Variant** | 78/78 | 33/78 (expected — word test na) |
| **Keno ei model?** | Shobdo chinle-i hoy; proba dey; SVC accuracy-te egiye (0.798) kintu proba dite pare na | Bakko-r context bojhe; L1/L2 recall proba-model-er moddhe shera; SVC accuracy-te egiye kintu L1-e ondho (0.286) |
| **Status** | LIVE (checker.kalandar.me) | RESEARCH — data barche, L3 data kom; production-e jawar age L1/L2/L3 sentence aro lagbe |

**Ek-i algorithm (LogReg) duita track-e — karon algorithm bodlano dorkar pore nai; data ar procedure bodleche.** Ma'am jodi jiggesh koren "alada algorithm keno use koro nai?" — uttor: try korechi (5-ta classifier × 2 track), number-e LogReg-i duita track-e proba-model-er moddhe shera promanito hoyeche.
"""))

cells.append(code("""# 10c. Track B demo — ek-i shobdo, ador vs gali (paired frames)
print("NOTE (shotti kotha): neeche-r demo-te dekhbe — ma-r ador frame "
      "(`আমার কুত্তার বাচ্চাটা...`) ekhono L2 ashe, L0 na; ar rag-er bhoinga "
      "frame (`তুই ভইঙ্গা, যা এখান থেকে`) L0 ashe, L2 na. Karon: paired frame "
      "matro 22-ta row, 21,714 row-er moddhe hariye geche — model speaker "
      "context ekhono shikhe nai. Eta-i Track B-r ashol gap, ar Meta AI loop-e "
      "proti shobde 10 ador + 10 gali sentence ashtese — data barle ei gap "
      "kombe. Demo-ta tai 'kothay model ekhono pare na' dekhano-r jonno-i rakha.")
if sdf is not None:
    s_best = s_results["LogReg"][0]
    def s_predict(text):
        c = normalize_text(text)
        lvl = int(s_best.predict([c])[0])
        proba = s_best.predict_proba([c])[0]
        print(f"{text!r} -> Level {lvl} ({LEVEL_NAMES[lvl]}) | "
              f"proba: {{ {', '.join(f'{LEVEL_NAMES[i]}: {float(p):.2f}' for i, p in enumerate(proba))} }}")
        return lvl
    print("--- ma bolle ador, onno keu bolle gali ---")
    for t in ["আমার কুত্তার বাচ্চাটা কই গেল", "কুত্তার বাচ্চা, আয় ভাত খা",
              "তুই কুত্তার বাচ্চা", "তুই একটা কুত্তার বাচ্চা"]:
        s_predict(t)
    print("\\n--- bhoinga: moja vs rag ---")
    for t in ["আরে ভইঙ্গা, কেমন আছস", "তুই ভইঙ্গা, যা এখান থেকে"]:
        s_predict(t)
    print("\\n--- neutral everyday sentence ---")
    for t in ["আইজ ক্লাসত যাইবা না?", "বাজরত যাইর", "কাল আইয়োম তোর বারিত"]:
        s_predict(t)
    # Track B model save (alada file — Track A-r .pkl-er sathe gulabe na)
    joblib.dump({"model": s_best, "levels": LEVEL_NAMES, "best_name": "LogReg"},
                "banglish_sentence_model.pkl")
    print("\\nsaved: banglish_sentence_model.pkl (Track B, LogReg)")
"""))

cells.append(md("""## 5b. Model comparison grid — konta best, keno?

Ekta model-e bhorosha na kore 6-ta vectorizer setup × 6-ta classifier = 36-ta combination maplam. Prottek-ta 4-ta metric-e mapa hoyeche:
- **accuracy** — test set-e koto percent thik
- **L3 recall** — ashol direct threat-er koto percent dhorte pereche (miss korle bipod, tai eta shobcheye guruttopurno)
- **masked accuracy** — `* #` diye mask kora test set-e koto
- **variant score** — 39-ta hate-banao kothin test: notun gali, comma-masking (`চু,,, দি`), leet, **training-e NAI emon target** (`তোর ফুফুরে চুদি`, `তোরে চুদমু`) ar nirdosh bakko (`আমার মারে ভালোবাসি`) — "nearby" generalization-er ashol proman

**Folafol (2026-11-18, 6,854 row):** LinearSVC ar RandomForest variant-e 39/39 pay, kintu SVC-er predict_proba nai (checker UI-te confidence dekhano jabe na) ar RandomForest-er accuracy onek kom (0.73) + .pkl bishal + JS-e port kora jay na. **LogReg (V4: word 1–3 + char 2–6) — accuracy 0.775, L3 recall 0.961, masked 0.743, variant 39/39** — proba dey, linear (JS-e port kora jay), ar shob metric-e upore. Tai LogReg-i ship hoyeche.
"""))

cells.append(code("""# 5b. Grid: 6 vectorizer x 6 classifier (RandomForest shudhu 2 config-e — slow)
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

cells.append(code("""# 6. Baseline vs Improved — ma'am-er creativity-r proman
# Baseline: word-only, raw text (normalization nai), augmentation nai
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

# --- Masked-only test: mask kora test set-e ke koto dhore ---
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
print("(template data-te ei tin-ta kachakachi — ashol boro data-te abar mapte hobe; normalization shotti kaj korche ki na, ei tulonai bole dibe)")
"""))

cells.append(md("""## 7. Explainability — model keno ei level bollo?

Logistic Regression-er coefficient theke prottek level-er jonno shobcheye strong word/char pattern ber kori. Ma'am jodi jiggesh koren "model keno Level 3 bollo?", tokhon ei list + highlight kora shobdo dekhabe.
"""))

cells.append(code("""# 7. Top features per level (LogReg hole; na hole skip)
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

cells.append(md("""## 8. Best model save koro (.pkl) + demo

JobLens-er moto best model `.pkl` hishebe save hobe — tarpor Streamlit app-e shei file-ta load kore cholbe. Colab theke download kore nao.
"""))

cells.append(code("""# 8. Save best model
joblib.dump({"model": best_pipe, "levels": LEVEL_NAMES, "best_name": best_name},
            "banglish_threat_model.pkl")
print("saved: banglish_threat_model.pkl  (model:", best_name, ")")
print("file size:", os.path.getsize("banglish_threat_model.pkl"), "bytes")

# 9. Demo predict — notun message-e test koro
def predict(text):
    c = normalize_text(text)
    lvl = int(best_pipe.predict([c])[0])
    proba = None
    try:
        proba = best_pipe.predict_proba([c])[0]
    except Exception:
        pass
    action = {0: "kichu korar nai", 1: "ignore / block korte paro",
              2: "block + report koro", 3: "block + report + proyojone kortipokkhoke janao"}[lvl]
    print(f"input: {text!r} -> Level {lvl} ({LEVEL_NAMES[lvl]}) | koroniyo: {action}")
    if proba is not None:
        print("   confidence:", {LEVEL_NAMES[i]: round(float(p), 2) for i, p in enumerate(proba)})
    return lvl

for t in ["kal class hobe?", "tui ekta pagol", "tui ekta beyadob",
          "toke marbo", "toke m*rbo", "tore dekhbo rastay",
          "tui aai ekbar, tore dekhilom", "basa chini ami, aitesi",
          # 2026-11-18 — family-target / threat-verb / unseen / neutral
          "তুরে চুদবো", "তোর মারে চুদি", "তোর বোনরে চুদি",
          "তোকে মেরে ফেলবো", "চু,,, দি", "ছাত্রলীগের মারে চুদি",
          "তোর ফুফুরে চুদি", "তোরে চুদমু", "m0therboard",
          "আমি ছাত্রলীগ করি", "আমার মারে ভালোবাসি",
          "ওয়েট কর, আসতেছি", "তোর মারে সালাম দিস"]:
    predict(t)
"""))

cells.append(md("""## 9. Next steps / limitations (presentation-e shotbhabe bolbe)

1. **Asol data aro barano:** ekhon TB-OLID-er 5,000 ashol Banglish comment
   ache — template-er 179-ta row ar main data na. Tobuo nijer collected
   ~1,000–1,500-ta anonymised message (bishesh kore Chatgaiya + direct
   threat) jog korle Level 3-er recall aro bhalo hobe — sheta-e shotti report.
2. **Masking 100% dhora jay na:** notun masking style ashtei thakbe; normalization-er contribution alada kore mapte hobe (Section 6). Shotyi kotha: masked test set-e normalization-er benefit proman hoy nai (with norm 0.743 vs without norm 0.743) — kintu comma-masking (`চু,,, দি`) ar leet variant test-e normalization chara dhora jeto na.
3. **Borderline bakko:** `ওয়েট কর, আসতেছি`-r moto bakko prosongo chara osposhto — ekbar L3 (0.49 vs L0 0.39, pray tie) hoyechilo, neutral counter-example bariye thik kora hoyeche. Eirokom bakko-te model-er confidence kom thakbe — sheta report-e likhbe.
4. **Level 3 kom thakbe:** ashol direct threat-er example kom pawa jay — class imbalance-er kotha report-e likhbe, `class_weight="balanced"` keno use korecho shetao.
5. **TB-OLID severity mapping-er simaboddhota:** TB-OLID-e shudhu offensive O/N chilo — Level 1/2/3 lexicon-scan diye mapa hoyeche, tai kichu row-er level bhul hote pare. Nijer data-te ashol label-ei bhorosha korbe.
6. **Ethics:** karo naam/phone/private chat permission chara data-te rakhbe na; model-ta shudhu detection-er jonno, karo biruddhe proman hishebe na.
7. **Streamlit demo:** `banglish_threat_model.pkl` load kore ekta choto app banao — text box + button + level + highlighted words + koroniyo.
8. **Dataset (2026-11-18, round 12):** 12,376 row (L0 5,424 / L1 1,777 / L2 4,353 / L3 822), lexicon 4,237 shobdo. Final numbers (Track A): accuracy 0.781 (baseline 0.699), L3 recall 0.969 (baseline 0.832), variant test 78/78. Track B: 21,714 sentence row, LogReg macro-F1 0.592 (Section 10).
9. **Track B (sentence) simaboddhota:** L3 matro 28-ta row (test-e 6-ta) — L3 recall 0.500 choto sample-er number; L1 recall 0.514 — teasing vs neutral ekhono kothin; sentence label-er ekta ongsho silver (ChatgaiyyaBench machine label) + synthetic (Meta AI/Gemini banano bakko) — ashol manush-er label na, METHODOLOGY-te sposhto lekha ache. Data barche (Meta AI loop cholche), barle Track B abar train kore number update korte hobe.
"""))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                            "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
nb_path = os.path.join(OUT, "Banglish_Threat_Severity_Training.ipynb")
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"notebook cells: {len(cells)} -> {nb_path}")
