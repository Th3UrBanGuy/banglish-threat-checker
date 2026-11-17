# Methodology — Banglish Threat Severity Classifier

**প্রজেক্ট:** Banglish + Chatgaiya মেসেজ পড়ে তার threat severity মাপা (Level 0–3)
**তারিখ:** 2026-11-18 (v3 — family-target + threat-verb enhancement)
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

## 2. ডেটা — 6,854 row, 3,206 শব্দ

| উৎস | Row | কী আছে |
|---|---|---|
| TB-OLID | 5,000 | আসল Banglish comment (CC BY 4.0); O/N লেবেল lexicon-scan দিয়ে Level 0–3-এ মাপা — কিছু level ভুল হতে পারে, সেটা সীমাবদ্ধতায় লেখা আছে |
| Nirmol v1 | 1,182 | বাংলা phrase (auto-transliterate করে Roman-ও বানানো) |
| LDNOOBW | 403 | ইংরেজি খারাপ শব্দ |
| Meta AI gali list | 302 | 17 জেলার গালি (user-এর আনা list) |
| Facebook research | 201 | পোস্ট/ব্লগ/গ্রুপ থেকে আসল Banglish slang |
| Facebook political | 60-odd | AL/BNP/NCP পেজের কমেন্ট থেকে harvest করা term (কাঁচা কমেন্ট কপি করা হয় নাই — শুধু term + template বাক্য) |
| Curated + GenZ variant | ~400 | template, masked, leet, spaced variant |
| **Threat verbs + family-target (2026-11-18)** | **218** | `চুদবো/চুদবি/চুদবে`, `মারে/বোনরে/নানিরে + চুদি`, party-target, kill threat, comma-masking, neutral counter-example |

Level distribution: L0 2,698 / L1 599 / L2 2,786 / L3 771। L3 সবচেয়ে কম —
আসল direct threat-এর উদাহরণ কম পাওয়া যায় (class imbalance; `class_weight="balanced"` দিয়ে সামলানো হয়েছে)।

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

**সৎ কথা:** masked test set-এ normalization-এর benefit প্রমাণ হয় নাই (with norm 0.743
vs without norm 0.743)। কিন্তু comma-masking (`চু,,, দি`) আর leet variant test-এ
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

(Grid চালানো হয়েছে 6,823 row-এর dataset-এ; চূড়ান্ত model 6,854 row-এ retrain করা।)

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

**চূড়ান্ত model: Logistic Regression, V4 (word 1–3 gram, 8,000 + char_wb 2–6 gram, 12,000), `class_weight="balanced"`, multinomial।**

---

## 7. চূড়ান্ত ফলাফল (shipped model, 6,854 row)

| Metric | Baseline (word-only, no norm) | Improved (V4 + norm + aug) |
|---|---|---|
| Accuracy | 0.705 | **0.775** |
| L3 recall | 0.929 | **0.961** |
| Masked accuracy | 0.664 | **0.743** |
| Variant test (39) | — | **39/39** |
| Demo (8) | — | **8/8** |

JS port (static checker): Python prediction-এর সাথে **296/296** মিল (0 mismatch)।
Live demo: https://checker.kalandar.me — আসল sklearn model server-এ চলে।

---

## 8. সীমাবদ্ধতা (সৎভাবে)

1. **Masking 100% ধরা যায় না** — নতুন masking style আসতেই থাকবে; masked set-এ normalization-এর benefit প্রমাণ হয় নাই (0.743 vs 0.743)।
2. **Borderline বাক্য** — `ওয়েট কর, আসতেছি`-র মতো বাক্য প্রসঙ্গ ছাড়া অস্পষ্ট; একবার L3 হয়েছিল (0.49 vs L0 0.39 — প্রায় tie), neutral counter-example বাড়িয়ে ঠিক করা হয়েছে। এমন বাক্যে model-এর confidence কম থাকবে।
3. **Bangla-script দুর্বল** — training data Roman-first, তাই বাংলা অক্ষরের input-এ কিছু ক্ষেত্রে level কম আসে (যেমন `তুই একটা হারামজাদা` → L1, আসল L3)। পরের ধাপ: mark-inclusive tokenizer।
4. **`s 3 x`** — এখনো miss (L1, 0.39 confidence); single-letter split-এর সীমা।
5. **TB-OLID label mapping** — O/N থেকে Level 1/2/3 lexicon-scan দিয়ে মাপা, কিছু row-এর level ভুল হতে পারে; L0 vs L2 boundary সবচেয়ে দুর্বল (L2 recall ~0.61–0.65)।
6. **Tone/context** — `kire bolod` বন্ধুর সাথে মজা করে বললে L0, রাগ করে বললে L1 — model speaker context জানে না; এটা data-র সীমা, model-এর না।
7. **"Facebook-এর আসল system" না** — presentation-এ কখনো দাবি করা যাবে না; এটা একটা সাধারণ keyword filter (baseline)-এর বিপরীতে আমাদের moderation checker।

---

## 9. কীভাবে চালাবে (reproduce)

```bash
cd ~/workspace/ml-lab-threat
.venv/bin/python build_slang_dataset.py   # Parquet rebuild (6,854 row)
.venv/bin/python compare_models.py       # 34-combination grid
.venv/bin/python verify_jsonl.py         # retrain + .pkl save
.venv/bin/python export_model.py         # model.json (JS port)
.venv/bin/python gen_validation.py && node validate_js.js   # JS == Python check
.venv/bin/python build_checker_app.py    # checker.html
.venv/bin/python build_notebook.py       # Colab notebook rebuild
```

Notebook (Colab): https://colab.research.google.com/drive/1XE8JhfPbvXGD0FOeVWbFbKI0_iUpMUMB
Live checker: https://checker.kalandar.me
Code: https://github.com/Th3UrBanGuy/banglish-threat-checker
