# Methodology — Banglish Threat Severity Classifier

**প্রজেক্ট:** Banglish + Chatgaiya মেসেজ পড়ে তার threat severity মাপা (Level 0–3)
**তারিখ:** 2026-11-18 (v4 — round 14: mark-inclusive tokenizer + Facebook real data + সুদানির ফুয়া 50/50)
**লেখক:** Alahi Majnur Osama (BGCTUB, CSE 7th sem) — ML Lab প্রজেক্ট
**নিয়ম:** Deep learning নিষেধ (ma'am-er নির্দেশ) — শুধু classical ML

---

## 1. সমস্যা — আমরা কী বানাচ্ছি?

Facebook/WhatsApp-এ হুমকি-গালি বেশিরভাগ **Banglish**-এ (Roman অক্ষরে বাংলা) আসে, আর
সাধারণ keyword filter সেগুলো ধরে না — যেমন `তুই একটা মাদারচোদ` লিখলে filter-এ কিছু
পাওয়া যায় না, কারণ filter শুধু নির্দিষ্ট কিছু ইংরেজি/বাংলা খারাপ শব্দ খোঁজে। আমাদের
সিস্টেম পুরো বাক্য পড়ে severity বলে:

| Level | মানে | উদাহরণ |
|---|---|---|
| 0 | Normal | `kal class hobe?` |
| 1 | Teasing / mocking | `tui ekta pagol` |
| 2 | Harassment / abuse | `tui ekta beyadob` (শারীরিক হুমকি নাই) |
| 3 | Direct threat | `toke marbo`, `তোর মারে চুদি` |

**আসল প্রশ্নটা শুধু "খারাপ শব্দ আছে কি না" না — "হুমকি কি না"।**
`আমার মারে ভালোবাসি`-তে "মারে" আছে, কিন্তু হুমকি নাই → L0। `তোর মারে চুদি`-তে একই
"মারে", কিন্তু sexual threat আছে → L3। পার্থক্য করে **verb আর phrase**, একক শব্দ না।

---

## 2. ডেটা — 12,639 row, 4,269 শব্দ (round 14, 2026-11-18)

Parquet source count (আসল মাপা সংখ্যা):

| উৎস | Row | কী আছে |
|---|---|---|
| TB-OLID (train + test) | 4,838 | আসল Banglish comment (CC BY 4.0); O/N লেবেল lexicon-scan দিয়ে Level 0–3-এ মাপা — কিছু level ভুল হতে পারে, সেটা সীমাবদ্ধতায় লেখা আছে |
| Sentence datasets (Track B data) | 2,582 | Kothon / ChatgaiyyaAlap / ChatgaiyyaBench / ChattogramSent / ONUBAD থেকে নেওয়া sentence — পুরো sentence database 21,714 row, তার মধ্যে 2,582 এই model-এ merge করা (সব L1/L2/L3 + 2,000 L0 sample) |
| Gali-books (OCR) | 921 | আব্দুল মান্নান স্বপনের গালি অভিধান থেকে [চট্ট] headword (headword + ছোট gloss শুধু — passage কপি করা হয় নাই) |
| Learned words | 756 | User (native speaker) যেসব শব্দ/বাক্য দিয়েছে + Gemini interview sentence |
| kit-nlp vulgar lexicon | 630 | Chittagonian vulgar lexicon (HF, Apache-2.0) |
| Harvest round 1 | 538 | Web / paper / Facebook research থেকে everyday + gali term |
| Meta AI gali list (×4 form) | 990 | 17 জেলার গালি — **SYNTHETIC (Meta AI-এর বানানো list, §13 দেখো)** |
| Threat verbs + family-target | 256 | `চুদবো/চুদবি/চুদবে`, `মারে/বোনরে/নানিরে + চুদি`, party-target, kill threat, comma-masking, neutral counter-example |
| Facebook political | 253 | AL/BNP/NCP পেজের কমেন্ট থেকে harvest করা term (কাঁচা কমেন্ট কপি করা হয় নাই — শুধু term + template বাক্য) |
| Synthetic lexicon + template | 340 | Template + masked example (হাতে-বানানো) — **SYNTHETIC** |
| GenZ variant | 178 | Leet / spaced / Bangla-script variant |
| Khoniker round | 55 | `ক্ষণিকের পোলা` + frame row |
| Facebook harvest + video | 51 | গ্রুপ/পেজ থেকে term (নাম/raw কমেন্ট ছাড়া) |
| Facebook groups, real (2026-11-18) | 140 | 11টা public Chittagonian গ্রুপের কমেন্ট থেকে term — শুধু term + ছোট fragment, নাম/raw কমেন্ট নাই |
| Facebook search, real (2026-11-18) | 20 | Public পোস্ট থেকে আসল বাক্য (নাম মুছে) |
| YouTube comments, real (2026-11-18) | 9 | আসল কমেন্ট থেকে বাক্য |
| Meta AI ador/gali sentence | 80 | প্রথম batch-এর 200 sentence-এর মধ্যে 40টা ভুল ador বাদ দিয়ে বাকি — **SYNTHETIC**, §13 দেখো |
| Check-log words | 2 | Live checker-এ আসা নতুন শব্দ |

Level distribution: L0 5,491 / L1 1,870 / L2 4,446 / L3 832। L3 সবচেয়ে কম —
আসল direct threat-এর উদাহরণ কম পাওয়া যায় (class imbalance; `class_weight="balanced"` দিয়ে সামলানো হয়েছে)।

**Lexicon (4,269 term):** Nirmol v1 (1,182 Bangla phrase) + LDNOOBW (403 English bad word) + kit-nlp + curated + user-এর শব্দ — lexicon থেকে dataset row বানানো হয় না, lexicon শুধু severity map + template বানাতে কাজে লাগে।

**Bangla গালির মূল construction:** `[কারো family] + [sexual verb]` — যেমন
"তোর **মারে/বোনরে/নানিরে** চুদি"। তাই family-target **phrase**-কেই শেখানো হয়েছে,
"মারে/বোনরে" একক শব্দকে না (ওগুলো নিজে গালি না)। আর যাতে model শুধু "মারে" দেখেই
L3 না বলে, neutral counter-example দেওয়া হয়েছে: `আমার মারে ভালোবাসি` (L0),
`তোর মারে সালাম দিস` (L0), `আমি ছাত্রলীগ করি` (L0)।

---

## 3. Normalization — লুকানো গালি পরিষ্কার করা

মানুষ গালির মাঝে `* # . _ - ,` ঢুকিয়ে AI-কে ফাঁকি দিতে চায়: `m*rbo`, `kh*n`,
`চু,,, দি`, `ক্রি1ম1নাল`। `normalize_text` যা করে (একই function ৫ জায়গায় —
`textnorm.py` আসল, বাকিগুলো copy; JS port-টা automated validation দিয়ে মেলানো হয়):

1. Bangla অক্ষরের মাঝে ASCII digit তুলে ফেলে — leet (`ক্রি1ম1নাল` → `ক্রিমনাল`); এটা LEET mapping-এর **আগে** করতে হয়, নাহলে `1` → `i` হয়ে mixed-script হয়ে যায়
2. শব্দের **ভেতরের** mask char তুলে ফেলে (`m*rbo` → `mrbo`, `চু,দি` → `চুদি`)
3. বাকি mask char (comma সহ) space দিয়ে বদলে দেয় (`চু,,, দি` → `চু দি`)
4. একই অক্ষর 3+ বার থাকলে 2-তে নামায় (`marbooo` → `marboo`)
5. leet mapping: `0→o`, `1→i`, `3→e`, `4→a`, `5→s`, `7→t`, `@→a`, `$→s`

**সৎ কথা:** masked test set-এ normalization-এর benefit প্রমাণ হয় নাই (round 14: with norm 0.740 vs without norm 0.755 — norm ছাড়া সামান্য বেশি)। কিন্তু comma-masking (`চু,,, দি`) আর leet variant test-এ
normalization ছাড়া ধরা যেত না — তাই রাখা হয়েছে, বাড়িয়ে বলা হয় নাই।

---

## 4. Feature — "Nearby" শব্দ কীভাবে ধরে? (মূল ধারণা)

Model শুধু মুখস্থ শব্দ চেনে না। দুই ধরনের feature একসাথে ব্যবহার হয়:

**ক) Word n-gram (1–3) — পাশাপাশি শব্দের phrase ধরে।**
`তোর মারে চুদি` একটা feature হিসেবে থাকে। তাই `মারে` একা দেখে model L3 বলে না —
"মারে"-র **পাশে কী verb আছে** সেটা দেখে। এটাই "nearby word" বোঝার প্রথম অংশ।

**খ) Char n-gram (2–6, `char_wb`) — বানানের মিল ধরে।**
প্রতিটা শব্দকে অক্ষরের টুকরোয় ভাগ করা হয়। যেমন `মাদারবোর্ড` আর `মাদারচোদ`-এর
মাঝে `মাদার` মিল আছে; `ছাএলীগ` আর `ছাত্রলীগ`-এর মাঝে `ছা`/`লীগ` মিল আছে।
তাই training-এ নাই এমন নতুন বানান/নতুন target-ও ধরা পড়ে — এটাই "nearby" বোঝার
দ্বিতীয় অংশ। প্রমাণ (variant test-এ, training-এ এই target/form ছিল না):
`তোর ফুফুরে চুদি` → L3, `তোর মামিরে চুদবো` → L3, `তোরে চুদমু` → L3,
`তোর বাপরে চুদি` → L3।

দুটো একসাথে (word + char) না হলে হয় phrase বোঝা যায় না (শুধু char), নয়তো নতুন
বানান ধরা যায় না (শুধু word) — grid-এ সেটা সংখ্যায় দেখা গেছে (নিচে V5/V6 দেখো)।

---

## 5. মডেল তুলনা — 6 vectorizer × 6 classifier = 34 combination

একটা model-এ ভরসা না করে পুরো grid চালানো হয়েছে। প্রত্যেক combination ৪টা metric-এ মাপা:

- **Accuracy** — test set-এ কত % ঠিক
- **L3 recall** — আসল direct threat-এর কত % ধরেছে (miss করলে বিপদ, তাই এটা সবচেয়ে গুরুত্বপূর্ণ)
- **Masked accuracy** — `* #` দিয়ে mask করা test set-এ কত
- **Variant score** — ৩৯টা হাতে-বানানো কঠিন test: নতুন গালি, comma-masking, leet, training-এ নাই এমন target, আর নির্দোষ বাক্য (39-এর মধ্যে কয়টা ঠিক)

| Vectorizer | Model | Acc | L3 recall | Masked | Variant |
|---|---|---|---|---|---|
| V1 w1-2+c3-5 | LogReg | 0.790 | 0.948 | 0.758 | 37/39 |
| V1 w1-2+c3-5 | LinearSVC | 0.784 | 0.948 | 0.751 | 38/39 |
| V1 w1-2+c3-5 | MultinomNB | 0.722 | 0.948 | 0.692 | 32/39 |
| V1 w1-2+c3-5 | ComplementNB | 0.736 | 0.942 | 0.711 | 32/39 |
| V1 w1-2+c3-5 | SGD-log | 0.788 | 0.948 | 0.751 | 36/39 |
| V1 w1-2+c3-5 | RandForest | 0.717 | 0.903 | 0.702 | 39/39 |
| V2 w1-3+c3-5 | LogReg | 0.790 | 0.948 | 0.761 | 37/39 |
| V2 w1-3+c3-5 | LinearSVC | 0.782 | 0.948 | 0.752 | 38/39 |
| V2 w1-3+c3-5 | MultinomNB | 0.726 | 0.942 | 0.695 | 32/39 |
| V2 w1-3+c3-5 | ComplementNB | 0.738 | 0.942 | 0.712 | 32/39 |
| V2 w1-3+c3-5 | SGD-log | 0.788 | 0.948 | 0.747 | 36/39 |
| V3 w1-2+c2-5 | LogReg | 0.794 | 0.935 | 0.767 | 37/39 |
| V3 w1-2+c2-5 | LinearSVC | 0.791 | 0.948 | 0.758 | 39/39 |
| V3 w1-2+c2-5 | MultinomNB | 0.728 | 0.909 | 0.698 | 31/39 |
| V3 w1-2+c2-5 | ComplementNB | 0.736 | 0.935 | 0.704 | 32/39 |
| V3 w1-2+c2-5 | SGD-log | 0.798 | 0.942 | 0.759 | 36/39 |
| **V4 w1-3+c2-6** | **LogReg** | **0.796** | **0.948** | **0.768** | **37/39** |
| V4 w1-3+c2-6 | LinearSVC | 0.794 | 0.948 | 0.763 | 39/39 |
| V4 w1-3+c2-6 | MultinomNB | 0.724 | 0.870 | 0.692 | 32/39 |
| V4 w1-3+c2-6 | ComplementNB | 0.739 | 0.935 | 0.704 | 32/39 |
| V4 w1-3+c2-6 | SGD-log | 0.803 | 0.942 | 0.755 | 36/39 |
| V4 w1-3+c2-6 | RandForest | 0.730 | 0.896 | 0.707 | 39/39 |
| V5 char-only | LogReg | 0.769 | 0.955 | 0.749 | 35/39 |
| V5 char-only | LinearSVC | 0.771 | 0.955 | 0.741 | 37/39 |
| V5 char-only | MultinomNB | 0.702 | 0.916 | 0.665 | 32/39 |
| V5 char-only | ComplementNB | 0.710 | 0.929 | 0.697 | 32/39 |
| V5 char-only | SGD-log | 0.772 | 0.955 | 0.736 | 33/39 |
| V6 word-only | LogReg | 0.739 | 0.942 | 0.689 | 32/39 |
| V6 word-only | LinearSVC | 0.732 | 0.935 | 0.674 | 32/39 |
| V6 word-only | MultinomNB | 0.642 | 0.766 | 0.601 | 25/39 |
| V6 word-only | ComplementNB | 0.704 | 0.818 | 0.655 | 25/39 |
| V6 word-only | SGD-log | 0.733 | 0.799 | 0.681 | 25/39 |

(Grid চালানো হয়েছে 6,823 row-এর dataset-এ — round 10-এর grid; চূড়ান্ত model 6,854 row-এ retrain করা। Round 14-এর 32-combination comparison: `research/model-comparison-2026-11-18.md` — V4-mark + LogReg C=4 সেরা (rs42 acc 0.811, rs7 0.791)।)

---

## 6. কেন LogReg (V4) — আর বাকিগুলো কেন না?

**বাছাইয়ের নিয়ম (আগে থেকে ঠিক করা):** checker UI-তে ৪টা level-এর probability bar
দেখাতে হয়, তাই শুধু `predict_proba` দিতে পারে এমন model-ই ship করতে পারে।
LogReg-ই default; অন্য কোনো proba-model LogReg-কে 0.01-এর বেশি ছাড়িয়ে গেলেই
কেবল সেটা ship হবে।

- **LinearSVC (variant 39/39, acc 0.794):** variant-এ সেরা, কিন্তু `predict_proba`
  দেয় না। Decision score-কে softmax করে confidence বানানো যাবে না — সেটা ভুল
  confidence হতো। তাই ship হয় নাই, যদিও accuracy-তে LogReg-এর সমান।
- **RandomForest (variant 39/39, acc 0.717–0.730):** variant-এ সেরা, কিন্তু
  accuracy অনেক কম, L3 recall কম (0.896–0.903), .pkl অনেক বড়, আর JavaScript-এ
  port করা যায় না (linear coefficient নাই)। তাই ship হয় নাই।
- **SGD-log (acc 0.803, সবচেয়ে বেশি):** accuracy-তে সেরা, কিন্তু LogReg-কে
  ছাড়িয়েছে মাত্র 0.007 (0.01-এর কম — নিয়ম অনুযায়ী যথেষ্ট না), L3 recall কম
  (0.942 vs 0.948), variant কম (36/39)। তাই ship হয় নাই।
- **Naive Bayes (MultinomNB/ComplementNB, variant 31–32/39):** সব metric-এ পিছিয়ে —
  NB স্বাধীনতা ধরে নেয় (একটা feature আরেকটার উপর নির্ভর করে না), কিন্তু text-এ
  feature গুলো একে অপরের সাথে জড়িত (যেমন `মারে` + `চুদি` একসাথে এলেই threat)।
- **V5 char-only (variant 35/39):** শুধু char নিলে phrase বোঝা যায় না — `মারে`
  একা দেখে ভুল হয়।
- **V6 word-only (variant 32/39, acc 0.739):** শুধু word নিলে নতুন বানান ধরা যায়
  না — baseline-এর মতোই দুর্বল। এটাই baseline vs improved-এর পার্থক্য প্রমাণ করে।

**V4 কেন V1-এর চেয়ে ভালো:** word 3-gram (`তোর মারে চুদি` পুরো phrase), char
2-gram (2-অক্ষরের stem `চু`), char 6-gram (লম্বা variant) — তিনটাই V1-এ ছিল না।
LogReg V4 বনাম V1: accuracy 0.796 vs 0.790, masked 0.768 vs 0.758।

**চূড়ান্ত model: Logistic Regression C=4, V4-mark (word 1–3 gram, 8,000 + char_wb 2–6 gram, 12,000), `class_weight="balanced"`, multinomial। Word tokenizer mark-inclusive (2026-11-18): sklearn-এর default `\w` বাংলা combining mark ফেলে দিত — `কুত্তা` ভেঙে `বন/রব` হতো; নতুন pattern `[\w\u0980-\u09FF]{2,}` শব্দ আস্ত রাখে। Bangla-subset accuracy +0.034।**

---

## 7. চূড়ান্ত ফলাফল (shipped model, 12,639 row, round 14, 2026-11-18 16:13)

| Metric | Baseline (word-only, no norm) | Improved (V4 + norm + aug) |
|---|---|---|
| Accuracy | 0.688 | **0.803** |
| L3 recall | 0.762 | **0.939** |
| Masked accuracy | 0.648 | **0.740** |
| Variant + demo test (90) | — | **90/90** |

JS port (static checker): Python prediction-এর সাথে **298/298** মিল (0 mismatch)।
Live demo: https://checker.kalandar.me — আসল sklearn model server-এ চলে।

---

## 8. সীমাবদ্ধতা (সৎভাবে)

1. **Masking 100% ধরা যায় না** — নতুন masking style আসতেই থাকবে; masked set-এ normalization-এর benefit প্রমাণ হয় নাই (round 14: 0.740 vs 0.755)।
2. **Borderline বাক্য** — `ওয়েট কর, আসতেছি`-র মতো বাক্য প্রসঙ্গ ছাড়া অস্পষ্ট; একবার L3 হয়েছিল (0.49 vs L0 0.39 — প্রায় tie), neutral counter-example বাড়িয়ে ঠিক করা হয়েছে। এমন বাক্যে model-এর confidence কম থাকবে।
3. **Bangla-script দুর্বল ছিল — round 14-এ ঠিক হয়েছে** — mark-inclusive tokenizer (2026-11-18): Bangla-subset accuracy +0.034, variant 90/90। Training data এখনো Roman-first, তাই বাংলা input-এ কিছু ক্ষেত্রে এখনো সাবধানে যাচাই করা দরকার।
4. **`s 3 x`** — এখনো miss (L1, 0.39 confidence); single-letter split-এর সীমা।
5. **TB-OLID label mapping** — O/N থেকে Level 1/2/3 lexicon-scan দিয়ে মাপা, কিছু row-এর level ভুল হতে পারে; L0 vs L2 boundary সবচেয়ে দুর্বল (L2 recall ~0.61–0.65)।
6. **Tone/context** — `kire bolod` বন্ধুর সাথে মজা করে বললে L0, রাগ করে বললে L1 — model speaker context জানে না; এটা data-র সীমা, model-এর না।
7. **"Facebook-এর আসল system" না** — presentation-এ কখনো দাবি করা যাবে না; এটা একটা সাধারণ keyword filter (baseline)-এর বিপরীতে আমাদের moderation checker।

---

## 9. কীভাবে চালাবে (reproduce)

```bash
cd ~/workspace/ml-lab-threat
.venv/bin/python build_slang_dataset.py   # Parquet rebuild (12,639 row, round 14)
.venv/bin/python compare_models.py       # 34-combination grid
.venv/bin/python verify_jsonl.py         # retrain + .pkl save
.venv/bin/python export_model.py         # model.json (JS port)
.venv/bin/python gen_validation.py && node validate_js.js   # JS == Python check
.venv/bin/python build_checker_app.py    # checker.html
.venv/bin/python build_notebook.py       # Colab notebook rebuild
```

---

## 10. 2026-11-18 — "ক্ষণিকের পোলা" round: model কি আসলেই শেখে?

User-er প্রশ্ন: model কি training data-r বাইরে নিজে শিখতে পারে, নাকি শুধু দেওয়া
data filter করে? সৎ উত্তর — **দুটোই, সীমা সহ:**

- **শেখে (generalize করে):** char n-gram বানানের মিল ধরে — `মাদারবোর্ড` ≈
  `মাদারচোদ`, `ছাএলীগ` ≈ `ছাত্রলীগ`। Word n-gram phrase ধরে — `তোর ফুফুরে
  চুদি` training-এ না থাকলেও L3 ধরে, কারণ `তোর X চুদি` pattern-ta শেখা আছে।
- **শেখে না (জানতে পারে না):** একদম নতুন cultural phrase, যেখানে কোনো শব্দই
  known slur না। `ক্ষণিকের পোলা`-র প্রমাণ: live test-এ near-tie দিয়েছিল —
  L3 0.32 vs L0 0.26 (মানে আন্দাজে বলেছে, জানে না)। কারণ "ক্ষণিকের" কোথাও
  ছিল না, আর "পোলা" নিজে neutral (ছেলে)। গালিটা combination-er cultural
  meaning-এ — "এক মুহূর্তের মহিলার ছেলে"। কোনো ML model-ই এমন meaning
  data ছাড়া আবিষ্কার করতে পারে না। তাই user (native speaker) যেসব শব্দ
  দেয়, সেগুলোই data — retrain করলেই model শেখে।
- **Fix:** lexicon +8 term, dataset +50 row (`ক্ষণিকের পোলা/পোয়া/ফুয়া`,
  Roman form, masked form; neutral counter-example: `আমার পোলা স্কুলে যায়`,
  `পোলাটা ভালো ছেলে` — যাতে "পোলা" একা L2 না হয়)। ফল: `ক্ষণিকের পোলা` →
  **L2 (0.88)**, `khoniker pola` → L2 (0.89), neutral L0 ঠিক আছে।
  Variant+demo এখন **56/56** (48 variant + 8 demo)। Dataset **6,904 row**,
  lexicon **3,214 term**, accuracy **0.791** (baseline 0.722), L3 recall
  **0.974**, masked 0.759।

## 11. দুইটা "baseline" — গুলিয়ে ফেললে চলবে না

- **Checker page-r নিচের baseline:** একটা toy keyword filter — মাত্র **১৮টা
  শব্দের** list (১২টা English: fuck/shit/bitch/... + ৬টা বাংলা অক্ষরের:
  হারামজাদা/মাদারচোদ/বাইনচোদ/খানকি/চোদনা/শুয়োর)। Plain substring match
  করে। Banglish শব্দ একটাও list-এ নাই — তাই `ক্ষণিকের পোলা` তো বটেই,
  `তোর মারে চুদি`-র মতো স্পষ্ট L3-ও এটা "Safe" বলে। এটাই দেখানোর point:
  সাধারণ filter Banglish ধরে না। কিন্তু এটা শুধু demo — প্রমাণ না।
- **Methodology-r baseline:** আসল ML model, শুধু word feature (no char,
  no normalization) — accuracy **0.688**। আমাদের improved model **0.803** (round 14)।
  এই সংখ্যাটাই আসল প্রমাণ — toy box-টা না।

Notebook (Colab): https://colab.research.google.com/drive/1XE8JhfPbvXGD0FOeVWbFbKI0_iUpMUMB
Live checker: https://checker.kalandar.me
Code: https://github.com/Th3UrBanGuy/banglish-threat-checker

---

## 12. Track B — sentence-level classification (আলাদা procedure, 2026-11-18)

**Track A আর Track B এক কাজ না — তাই train-ও আলাদা, data-ও আলাদা, model-ও আলাদা।**

|  | Track A (word-level) | Track B (sentence-level) |
|---|---|---|
| **প্রশ্ন** | এই শব্দ/phrase-টা কত severe? | এই পুরো বাক্যটা আদর, গালি, না normal? |
| **ডেটা** | 9,794 row (parquet থেকে sentence source বাদ দিয়ে) | 21,714 sentence row |
| **কেন আলাদা?** | শব্দ চিনলেই হয় | একই শব্দ আদরেও হয়, গালিতেও হয় — context লাগে |
| **উদাহরণ** | `চুদি` → L3 | `আমার কুত্তার বাচ্চাটা কই গেল` (মা বলছে) → L0, `তুই কুত্তার বাচ্চা` (অন্য কেউ) → L2 |

**Track B-র ডেটা (21,714 row):** Kothon 7,986 / ChatgaiyyaAlap 3,148 + dict 1,417 / ChatgaiyyaBench vulgarity 3,009 + romanized 1,582 + lexicon 204 / ChattogramSent 2,703 / ONUBAD sentence 605 + word 787 + clause 113 / Gemini Deep Research 138 / paired frames 22। Level: L0 21,112 / L1 177 / L2 397 / L3 28 — **97% L0, L3 মাত্র 28টা।**

**তুলনা (একই V4 feature, একই 80/20 split, seed 42 — VM-এ মাপা):**

| Model | Acc | L3 recall | Masked | Macro-F1 | L0 rec | L1 rec | L2 rec | Variant (78) |
|---|---|---|---|---|---|---|---|---|
| **LogReg** | 0.960 | 0.500 | 0.937 | **0.592** | 0.972 | **0.514** | **0.595** | 33/78 |
| LinearSVC | 0.980 | 0.500 | 0.964 | 0.662 | 0.995 | 0.286 | 0.532 | 31/78 |
| MultinomNB | 0.940 | 0.000 | 0.941 | 0.273 | 0.963 | 0.000 | 0.203 | 24/78 |
| ComplementNB | 0.932 | 0.500 | 0.922 | 0.406 | 0.951 | 0.086 | 0.329 | 26/78 |
| RandomForest | 0.980 | 0.333 | 0.979 | 0.621 | 0.997 | 0.286 | 0.456 | 20/78 |
| Baseline (word-only) | 0.847 | 0.333 | — | — | — | — | — | — |

**Accuracy দেখতে ভালো (0.93–0.98) — কিন্তু এটা ধোঁকা:** 97% L0 বলে শুধু "সব L0" বললেও 0.97 accuracy পাওয়া যায়। আসল বিচার macro-F1 + per-class recall-এ।

**Track B = Logistic Regression — কেন:**
- **Proba দিতে পারে** — checker UI-তে confidence bar দরকার; LinearSVC-র proba নাই (Track A-র একই নিয়ম)।
- **L1 recall সবচেয়ে ভালো (0.514)** — teasing vs neutral বোঝাই Track B-র সবচেয়ে কঠিন কাজ; SVC (0.286) আর RF (0.286) এখানে প্রায় অন্ধ।
- **L2 recall সবচেয়ে ভালো (0.595)** — harassment sentence ধরা।
- **Balanced** — macro-F1 proba-model-এর মধ্যে সেরা (0.592); NB (0.273–0.406) অনেক পিছিয়ে।
- LinearSVC-র macro-F1 (0.662) বেশি হলেও L1-এ 0.286 — accuracy-র পেছনে লুকানো দুর্বলতা।

**Variant 33/78 — এটা খারাপ না, EXPECTED:** variant test গুলো word-level কঠিন test (একক শব্দ, masked form, নতুন target) — sentence data দিয়ে train করা model একক শব্দ চিনবে না, এটাই স্বাভাবিক। এটাই প্রমাণ করে **Track B Track A-র বদলি হতে পারে না** — দুইটা আলাদা কাজ, দুইটা আলাদা model।

**Track B-র সীমা (সৎভাবে):**
1. L3 মাত্র 28টা row — test-এ 6টা; L3 recall 0.500 মানে 6-টার মধ্যে 3টা — ছোট sample, বেশি ভরসা করা যাবে না।
2. L1 recall 0.514 — teasing vs neutral এখনো কঠিন; L1/L2/L3 sentence আরও লাগবে।
3. Sentence label-এর একটা অংশ silver (ChatgaiyyaBench machine label — ভুল হতে পারে) + synthetic (§13)।
4. ChattogramSent 59% ChatgaiyyaAlap-এর duplicate (Vashantor leak) — দুই dataset মিলিয়ে unique sentence কম।
5. **Status: RESEARCH — production-এ যায় নাই।** Sentence data বাড়ছে (Meta AI loop চলছে — প্রতি batch-এ 5টা শব্দ × 10–15টা sentence); data বাড়লে Track B আবার train করে number update করতে হবে।

**দুই track মিলে — শেষ কথা:** একই algorithm (LogReg) দুইটা track-এ — কারণ algorithm বদলানো দরকার পড়ে নাই; data আর procedure বদলেছে। 5টা classifier × 2 track try করে number-এ LogReg-ই দুইটা track-এ proba-model-এর মধ্যে সেরা প্রমাণিত হয়েছে।

---

## 13. SYNTHETIC DATA — যা আসল মানুষের লেখা না (স্পষ্ট ঘোষণা)

এই প্রজেক্টের ডেটার একটা অংশ **synthetic** — মানে AI (Meta AI / Gemini) বা template দিয়ে বানানো, আসল মানুষের লেখা comment/message না। Presentation-এ এটা লুকানো যাবে না — নিচে পুরো হিসাব:

| Synthetic উৎস | কী আছে | কোথায় |
|---|---|---|
| Meta AI gali list (×4 form: Roman / spaced / leet / dot) | 17 জেলার গালি list — Meta AI-এর বানানো (2026-11-18 03:11, 1,800+ line CSV) | Dataset 990 row (`meta_ai`, `meta_ai_friendly`, `meta_ai_spaced`, `meta_ai_leet` source) |
| Meta AI ador/gali sentence | প্রতি শব্দে 10 ador + 10 gali sentence (2026-11-18 থেকে চলছে — প্রথম batch 200 sentence: 10টা শব্দ) | `research/meta-ai-ador-gali-2026-11-18.csv` — Track B-র বাড়তি data |
| Gemini interview sentence | 25টা শব্দ নিয়ে Gemini-র বানানো example sentence (2026-11-18) | `learned_words.json`-এ (+78 row) |
| Gemini Deep Research entry | 122 entry (27 gali, 20 context-dependent, 75 everyday) — Gemini-র research summary থেকে | Sentence database 138 row |
| Template / synthetic lexicon / masked example | হাতে-লেখা template + augmentation-এর masked copy | Dataset ~340 row + train-time augmentation |
| TB-OLID level mapping | O/N label থেকে lexicon-scan দিয়ে Level 1/2/3 মাপা — আসল label না, অনুমান | §8-এর সীমা 5 দেখো |

**কেন synthetic রাখা হয়েছে, আর কী সাবধানতা:**
- আসল Chatgaiya gali/threat sentence public-এ প্রায় নাই — না থাকলে model শিখবে কী দিয়ে? Synthetic দিয়ে শুরু করে আসল data দিয়ে যাচাই করাই একমাত্র পথ ছিল।
- Synthetic row কখনো একা training-এর ভরসা না — প্রতিটা concept-এ neutral counter-example + variant test দিয়ে যাচাই করা হয় (যেমন `ক্ষণিকের পোলা` round)।
- **কোনো synthetic row-কে "আসল মানুষের লেখা" বলে দাবি করা যাবে না** — report/presentation-এ এই section-টাই তার প্রমাণ।
- **kit-nlp DOI সংশোধন (2026-11-18):** Gemini Deep Research report-এ দেওয়া Mendeley DOI 10.17632/k769s4vk5z.2 ভুল — ওটা "BD-Dialect" dataset। আসল lexicon: Hugging Face `kit-nlp/Vulgar_Lexicon_of_Chittagonian_Dialect_of_Bangla_or_Bengali` (DOI 10.57967/hf/1296, Apache-2.0; Mahmud, Ptaszynski, Masui, Applied Sciences 13(21) 11875, 2023)।

---

## 14. Round 13 → 14 — যা বদলেছে (2026-11-18 বিকেল)

**Live model এখন round 14** (deploy 2026-11-18 16:13; 15:15-এর টিক VM recycle-এ থেমে গিয়েছিল, recovery করে আপলোড শেষ করা হয়েছে)।

1. **Tokenizer fix (সবচেয়ে বড় পরিবর্তন):** sklearn-এর default word pattern `\w` বাংলা combining mark ফেলে দিত — `কুত্তা, তুঁই আঁর জানের বন্ধু` থেকে শুধু `বন/রব` থাকতো, model বাক্যটা দেখতেই পেত না। নতুন mark-inclusive pattern `[\w\u0980-\u09FF]{2,}` + LogReg C=4। দুইটা আলাদা split-এ যাচাই: rs42 acc 0.797→0.811, rs7 0.780→0.791 — দুইটাতেই +0.01-এর বেশি, তাই ship করা হয়েছে। Bangla-subset accuracy +0.034।
2. **Facebook real data:** 11টা public Chittagonian গ্রুপ থেকে 41টা term (140 row) + Facebook search-এর 20টা আসল বাক্য — শুধু term/fragment, নাম/raw কমেন্ট নাই। এই merge-এর পরেই একটা borderline test ফেল করেছিল (`কুত্তা, তুঁই আঁর জানের বন্ধু` L0 হয়ে গিয়েছিল — কারণ 2টা আসল L0 বাক্যে একই friendly frame ছিল), তারপর retrain-এ সেটা আবার L2 হয়েছে — তবে margin মাত্র 0.02, এটা সীমাবদ্ধতায় লেখা আছে।
3. **সুদানির ফুয়া — 50/50 (user-এর verdict, 2026-11-18 15:01):** খালি `সুদানির ফুয়া` এখন L1 (teasing), রাগ করে `তুই সুদানির ফুয়া` বললে L2। ভইঙ্গার মতোই — context-এ বদলায়। Live probe: bare L1 (0.783), angry frame L2 (0.503)।
4. **সংখ্যা (round 14, live-verify করা):** accuracy 0.803 (baseline 0.688), L3 recall 0.939, masked 0.740, variant+demo 90/90, dataset 12,639 row, lexicon 4,269 term।
5. **Live probe (2026-11-18 16:14):** `কুত্তা, তুঁই আঁর জানের বন্ধু` → L2 (0.503 vs L0 0.483), `সুদানির ফুয়া` → L1 (0.783), `তুই সুদানির ফুয়া` → L2 (0.503)।
6. **Static checker (round 14):** https://muse.ai/files/1282720808262570/1129095986501478/ygscx8mn154hgvm9oasbnwmh/checker.html — মেয়াদ 2026-11-20 16:13 +06 পর্যন্ত।
