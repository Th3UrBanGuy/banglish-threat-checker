# Banglish + Chatgaiya Threat Severity Classifier — ML Lab

Project picked 2026-11-17. No deep learning (ma'am-er nirdesh).
Classical ML: TF-IDF (word 1-2 + char_wb 3-5) + Logistic Regression / LinearSVC / Random Forest.

## Files (Parquet — JSONL/CSV ar training file na)
- `Banglish_Threat_Severity_Training.ipynb` — Colab training notebook (JobLens-style structure, Parquet load kore)
- `banglish_threat_dataset.parquet` — training messages, **6,904 rows** (text: string, level: int8, dialect: string, source: string)
- `banglish_slang_lexicon.parquet` — slang/bad-word lexicon, **3,214 rows** (term, script, language, dialect, district: string, category, severity: int8, source, auto_translit: bool)
- `banglish_threat_model.pkl` — best model (RandomForest/LogReg ~0.75, ashol data-te train kora)
- `build_notebook.py` — notebook + 179-row template CSV generator (template source; notebook bodlate hole ei file bodle rebuild koro)
- `build_slang_dataset.py` — dataset builder: template CSV + TB-OLID + Nirmol + LDNOOBW + curated -> JSONL (intermediate) -> Parquet (PyArrow, snappy)
- `verify_jsonl.py` — local verification (notebook-er pipeline-er moto kore chalai, Parquet pore)
- `banglish_threat_dataset.jsonl` / `banglish_slang_lexicon.jsonl` — shudhu intermediate build file; training-e Parquet use hoy
- `banglish_threat_dataset.csv` — sudhu 179-row template source; training-e ar use hoy na
- `sources/` — downloaded public data: `nirmol/` (Nirmol v1), `tbolid/` (TB-OLID train+test), `ldnoobw_en.txt`, `ldnoobw_hi.txt`, `facebook-slang-research.md` (Facebook theke khuje pawa slang + Chatgaiya glossary, sources soho)

## Parquet keno (CSV/JSONL-er bodle)
- **Columnar:** shudhu je column dorkar setai pore, baki column direct skip kore (`pq.read_table(path, columns=["text"])`)
- **Footer metadata:** schema, row group, min/max stats file-er sheshe thake
- **Fixed data type:** prottek column-er type schema-te lekha (`level: int8`, `auto_translit: bool`) — type niye jhamela nai
- **Choto:** dataset 271 KB (JSONL 732 KB), lexicon 33 KB — snappy compression

## Kotha theke elo (sources + license)
- **TB-OLID** (LanguageTechnologyLab, CC BY 4.0) — 5,000 ashol Banglish comment, offensive O/N label. Severity lexicon-scan diye mapa (threat word thakle 3, sexual/strong gali 2, mild 1, na mille default 2) — tai kichu row-er level bhul hote pare.
- **Nirmol v1** (Sigmakib2/Nirmol) — 1,182 Bangla-script offensive phrase; lexicon-er Bangla part. License repo-te lekha nai — cite kore training-er jonno use.
- **LDNOOBW en** (CC BY 4.0) — 403 English bad word/phrase.
- **curated** — nijer hate lekha Banglish + Chatgaiya gali/humki (public list-e Chatgaiya nai) + synthetic template sentence (~161 row).
- Nirmol phrase theke machine-e Roman (Banglish) kora hoyeche (`auto_translit=true`); transliteration phonetic, perfect na — char n-gram-er jonno jothesto.

## Facebook slang research (2026-11-17, his order)
Facebook + web ghure khuje pawa slang lexicon-e jog kora hoise — lexicon
**2,640 → 2,846 rows** (`facebook_research` 201 + `facebook_chatgaiya_glossary` 6).
Full list + source links: `sources/facebook-slang-research.md`.
- **Bangla gali:** Shafiul Sami-r FB post "হালকা বাংলা গালি" (15-ta),
  Grammar Academy blog (2026-10-27, ~100-ta categorized), Australia
  Unwrapped-er ranked list, Aminul Jewel-er carousel, "Collecting Fav Gali"
  group post-er comments (asol Banglish: Gu kha, Shawaa, Madarcod, **Mc**,
  **Bnc**, K.. Pola — abbreviation + dotted masking).
- **Notun obfuscation style pawa gelo:** letter-spacing ("H e d a",
  "M a d a r c h u d") ar abbreviation ("Mc", "Bnc") — ekhonkar
  normalizer-e dhora hoy na; next step hishebe notebook-e lekha ache.
- **Chatgaiya:** dedicated gali/slang list **kothao nai** (web + Facebook).
  Shahriar Khaled-er "চাটগাইয়া টু বাংলা" glossary theke shudhu 3-ta
  jhograr shobdo pelam (খইজ্জা, জান্টা, গিরাখাডা); baki 40-ta rojkar
  shobdo research note-e. Chatgaiya slang-er data shudhu tumi (native
  speaker) dite parba — etai project-er uniqueness-er proman.
- **Video transcription (2026-11-17):** "চিটাগাং গালি" video 4-ta download
  kore Whisper-small diye transcribe korlam — 2-ta music-backed reel-e
  kono porishkar kotha nai (Whisper hallucinate kore), Facebook-er nijer
  transcript-i valo chilo. Harvest: **সুদানির পোয়া** (video caption),
  **চুদানির পোয়া** + **ডুম** (comments). **"মাগীর ফ্যান" CORRECTION:**
  rant video-r transcript-e "মাগীর ফ্যান" shunechilam — user-er Meta AI
  list-e ashol form **"মাগির ফুত" (Chattogram) / "মাগির ফুয়া" (Sylhet)**,
  tai transcript-er vul form-ta lexicon theke muche thik form bosano
  hoise. Demo: sudanir/chudanir poya → Level 2 thik; "dum" eka → Level 1
  (miss, karon ek shobdo ambiguous). Full story:
  `sources/facebook-slang-research.md` section 5.

## Meta AI list (2026-11-17, user-er anano — 4-ta table/list)
User Meta AI theke jela-wise gali list koriye anlo — `sources/meta-ai-gali-lists.csv`
(477 row, **157 distinct word, 17 jela**) theke Parquet-e dhukano hoise.
Lexicon **2,854 → 3,154**, dataset **5,186 → 6,174**.
- **Lexicon:** prottek shobdo Bangla + auto-Roman, source `meta_ai`,
  `district` column-e sob jela (harami = 17 jela, bolod = 12 jela).
  Category: Level 3 + sexual stem → sexual, Level 3 baki → profanity,
  Level 1/2 → insult.
- **Sentence (Meta AI-er formula):** prottek gali x 3 template
  ("tui ekta X", "X koi jas", "ei X edike ai") = 465 row.
- **Spaced + leet augmentation (Meta AI-er 4-version suggestion):**
  "h a r a m i" (157 row, source `meta_ai_spaced`) + "h4rami" (153 row,
  source `meta_ai_leet`). Demo: h4rami → L2 thik; "h a r a m i" gali
  hishebe dhora pore kintu Level 3 dey (L2 howar kotha — level ek dhap
  beshi, known miss).
- **Friendly / non-gali context (Meta AI-er suggestion):** "অডা কই যর?",
  "কিরে বলদ" — eki shobdo bondhu-ke moja kore bolle gali NA (Level 0).
  Halka (L1) shobdo diye 3-ta friendly template + Level 2/3 shobdo diye
  "kire X" counter-example (jate "kire" nijei L0 signal na hoye jay).
  **Partial success:** "oda koi jor" → L0 thik, "bolod koi jor" → L0 thik,
  "tui ekta bolod" → L1 thik — kintu **"kire bolod" → L1** (L0 howar
  kotha) — "kire" frame-ta unstable, run-e run-e bodlay. Sotti kotha:
  text-only model-e tone/relationship dekha jay na, tai "kire bolod"
  bondhu-r moja na rag — eta model-er pokkhe nishchit kora somvob na.
  Eta project-er **limit** hishebe report-e likhte hobe, na je aro
  friendly udahoron jog korle puropuri thik hobe (aro jog korle valo hobe,
  kintu guarantee nai).

## Local verification (2026-11-17, ashol data, Parquet pore, test n=1,321) — HONEST numbers
- LogReg **0.791** accuracy (V4: word 1–3 gram + char_wb 2–6 gram; LinearSVC tie, kintu predict_proba nai bole ship kora hoy nai; checker UI-te confidence % dorkar)
- Baseline (word-only, no norm) 0.722 -> Improved (word+char, norm+aug) **0.791**
- Level-3 recall: baseline 0.981 -> improved **0.974**
- Masked test: baseline 0.699, improved w/o norm **0.739**, improved WITH norm 0.734
  -> normalization-er lav EKHONO proman hoy NAI (w/o norm ektu age); ashol boro data-te abar mapte hobe.
- Levels: {0: 2651, 1: 599, 2: 2784, 3: 571} — L1 ekhono choto (Meta AI list-e L1 shobdo onek, kintu sentence kom).
- Weak spot: Level 0 vs Level 2 boundary (L2 recall 0.61-0.65; TB-OLID-er O->default-2 mapping noisy bole L0/L2 guliye jay). Level-er definition aro sharp + nijer ashol labeled data lagbe.
- Demo predict 14/16 correct — 2 miss: "kire bolod" → L1 (L0 howar kotha, tone-dependent — limit), "h a r a m i" → L3 (L2 howar kotha, gali hishebe dhora poreche, level ek dhap beshi). Baki sob thik: harami L2, madarchod L3, magir fut/fuya L3, sudanir poya L2, h4rami L2, tore dekhilom L3.

## GenZ / political variant robustness (2026-11-17 ~23:16, user-er order — "prediction kore powerful output ano")
User-er point: lokjon gali lukiye lekhe — ধ/দ bodol ("দন"), vowel sign drop ("রাজাকার" → "রজাকর"), Bangla leet ("ক্রি1ম1নাল"), acronym mock ("দনসিপি"), letter-spelling ("এস থ্রি এক্স" = S3X). Model-ke na dekha variant-o dhorte hobe.
- **Normalizer fix:** Bangla okkhor-er majhe ASCII digit tule fela (ক্রি1ম1নাল → ক্রিমনাল) — LEET-er AGE, naile 1→i hoye mixed-script ("ক্রিiমiনাল") hoto. 4-ta jaygay eki function (verify_jsonl.py, notebook, predictor.js, gen_validation.py) — eksathe bodlate hoy.
- **Dataset:** political/GenZ harvest (AL/BNP/NCP page comment, 60-ta pore, raw copy na — shudhu term) + user-er variant list → lexicon **3,154 → 3,192** (+38, source `facebook_political`), dataset **6,174 → 6,605** (+431: 3 Roman template + spaced + leet + 7 Bangla template + bare row + vowel-drop/consonant-swap variant + Bangla leet row).
- **Level correction (audit-e dhora):** Meta AI "ধোন"-ke L3 diyechilo, kintu Nirmol lexicon (sev 2) + TB-OLID majority (28 L2 vs 8 L3) bole L2 — 10-ta row override kora hoise (`LEVEL_OVERRIDES`).
- **JS port bug (validation-e dhora):** predictor.js-er word tokenizer-e `\p{M}` chilo — Python `\w`-te combining mark pore na, tai Bangla shobdo Python-e fragment ("একটা" → "এক") hoy, JS-e puro shobdo — 16-ta prob diff + 1 level miss. Fix: `\p{M}` bad + `\b` lookaround — ekhon **296/296, 0 mismatch, 0 prob diff** (V4 config-e).
- **Variant test 39/39 (2026-11-18):** family-target (`তোর মারে/বোনরে/নানিরে চুদি`), threat verb (`তুরে চুদবো`), comma-masking (`চু,,, দি`), kill threat (`তোকে মেরে ফেলবো`), UNSEEN target (`তোর ফুফুরে চুদি`, `তোরে চুদমু` — training-e chilo na) ar neutral (`আমার মারে ভালোবাসি`, `ওয়েট কর, আসতেছি`) — sob thik. `"s 3 x"` ekhono miss (L1, 0.39 — single-letter spaced form-এর সীমা)।
- **Sotorkota:** training data ekhono Roman-first — Bangla-script input-e model durbol (jemon "তুই একটা হারামজাদা" → L1, ashol L3). Tokenizer bodle (mark-soho token) Bangla accuracy barano jabe — porer step, alada kaaj.

## Checker web app (2026-11-17, his order — model diye web app + host koro)
- `checker.html` — **ek file-e pura app**: trained LogReg model-ta (TF-IDF vocab/idf + coefficients) JavaScript-e port kore embed kora — kono Python server lage na, browser-ei chole, offline-o chole. Single check (verdict + confidence + dhora pora shobdo + normalized lekha + naive-filter tulona + report copy) + batch check (table + CSV download) + 8-ta example button.
- **Sotti kotha (hosting):** TinkerHost/Google Apps Script-e Python sklearn chole na (PHP/Apps Script host) — tai model-ta JavaScript-e port korte hoise; eki weight, eki model, shudhu runtime bodleche. Static file bole TinkerHost-eo cholbe (htdocs-e copy korlei hobe) — permanent link chaile porer step.
- **Verify kora (notun, ashol):** JS port Python model-er sathe **298/298 input-e 100% mileche** (level + probability) — `validate_js.js`. Page test 10/10 (single L3/Safe/masked, example chip, batch counts). Live checker: https://checker.kalandar.me (Python server, আসল model) — ar static `checker.html` (Muse storage, 48h): notun upload **expires 2026-11-20 02:16 +06**, je keu link pele khulte parbe. File-ta chat-eo attach kora — offline-e jekono somoy khola jay, kono expiry nai.
- Files: `export_model.py` (pkl -> model.json + lexicon.json), `predictor.js` (TF-IDF + LogReg JS port), `app_ui.js`, `build_checker_app.py` (checker.html banay), `gen_validation.py` + `validate_js.js`, `test_checker_page.js`.
1. **Notun notebook-ta abar upload koro** (Colab-er ager copy-ta purono, CSV version) — File > Upload notebook
2. 2-ta `.parquet` file-ei upload koro: `banglish_threat_dataset.parquet` + `banglish_slang_lexicon.parquet` (ager CSV/JSONL session file ar lagbe na — notun notebook Parquet khuje; setup cell-e `pyarrow` install kora ache)
3. Runtime > Run all
4. Asol data jog korle Parquet-e row jog koro (`build_slang_dataset.py` chalale abar banabe) — code bodlate hobe na

## Next steps
1. Nijer collected ~1,000-1,500 anonymised Banglish + Chatgaiya message jog koro (bishesh kore Level 3 + Chatgaiya) — Parquet-e row jog korlei hobe
2. Notebook abar chalao, ashol accuracy report nao
3. `.pkl` diye Streamlit demo banao

## 2026-11-18 — Family-target + threat-verb enhancement (v3)
- **User-er report:** `তুরে চুদবো ... মারে চু,,, দি ...` model L2 dichchilo (L3 howa uchit chilo)। Diagnosis: (1) `চুদবো/চুদবি/চুদবে` future-tense threat verb lexicon-ei chilo na; (2) `চুদি/চোদ` L2 chilo (insult hishebe thik, kintu "তোরে চুদবো" / "মারে চুদি" direct sexual threat → L3); (3) comma-masking (`চু,,, দি`) normalizer-e dhora porto na; (4) Bangla-script kill threat (`তোকে মেরে ফেলবো`)-er kono template row chilo na।
- **Data:** +249 row → dataset **6,854** (L3 571→771), lexicon **3,206** (+14 threat-verb term, L3)। Family-target phrase (`মারে/বোনরে/নানিরে + চুদি`), party-target, kill threat, comma-masked row, ar neutral counter-example (`আমার মারে ভালোবাসি`, `তোর মারে সালাম দিস`, `ওয়েট কর, আসতেছি` — "ওয়েট কর"/"তোর মারে" jate nije L3 signal na hoy)।
- **Normalizer:** comma-o ekhon mask char (`চু,দি` → `চুদি`, `চু,,, দি` → `চু দি`); `textnorm.py` ekhon Python-side single source of truth (verify/gen_validation import kore; notebook cell `inspect.getsource` diye toiri hoy)।
- **Model grid (34 combination):** 6 vectorizer × 6 classifier — accuracy + L3 recall + masked + variant (39 test)। **V4 (word 1–3 + char 2–6) + LogReg** ship hoyeche: acc 0.775, L3 recall 0.961, variant **39/39**। LinearSVC/RandomForest variant-e 39/39 peleo ship hoy nai — SVC-er predict_proba nai, RF-er accuracy kom (0.73) + JS-e port kora jay na। Purno table + keno: `METHODOLOGY.md` (Bengali)।
- **Live verify:** user-er text → **L3 (0.97)**; unseen `তোর ফুফুরে চুদি` → L3 (0.98); neutral `আমার মারে ভালোবাসি` → L0 (0.89)। Notebook Colab-e update kora (eki link)।
- **Infra note:** VM recycle-er pore tunnel stack + checker server duitai down chilo (sob hostname 530) — edge relay + 3 replica + checker haate restart kore live kora hoyeche।

## 2026-11-18 — "ক্ষণিকের পোলা" round (user-er test)
- **User-er proshno:** model ki training data-r baire nije shikhte pare, naki shudhu deoa data filter kore? Live test: `ক্ষণিকের পোলা` → near-tie L3 0.32 vs L0 0.26 (mane ashole chinto na — "ক্ষণিকের" kothao chilo na, "পোলা" nije neutral)। **Shot uttor:** kachakachi banan/pattern shikhe (char n-gram: মাদারবোর্ড ≈ মাদারচোদ; word n-gram: ফুফুরে unseen target L3), kintu ekdom notun cultural phrase-er meaning data chara abishkar korte pare na — tai user-er deoa notun shobdoi data।
- **Fix:** lexicon +8 term, dataset +50 row (Bangla + Roman + masked form; neutral counter-example: `আমার পোলা স্কুলে যায়`, `পোলাটা ভালো ছেলে` — "পোলা" eka jate L2 na hoy)। Fol: `ক্ষণিকের পোলা` → **L2 (0.88)**, `khoniker pola` → L2 (0.89)। Variant+demo **56/56** (48+8)। Dataset **6,904**, lexicon **3,214**, accuracy **0.791** (baseline 0.722), L3 recall 0.974, masked 0.759 (with norm 0.759 vs without 0.749 — ebar chhoto benefit dekha geche)।
- **Baseline niye user-er proshno (page-r nicher section):** oita ekta toy keyword filter — matro 18-ta shobder list (12 English + 6 Bangla-script), Banglish shobdo ektai nai — tai `তোর মারে চুদি`-keo "Safe" bole। Eta demo, proman na; ashol proman holo accuracy 0.722 → 0.791। Purno bekhya: `METHODOLOGY.md` §10–11।
