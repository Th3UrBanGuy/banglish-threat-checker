#!/usr/bin/env python3
"""Build the slang / bad-words lexicon + training messages for the
Banglish Threat Severity project — as JSONL (one JSON object per line),
NOT csv. JSONL mane prottek line ekta row, ar line-by-line pore prottek
row alada kore check/validate kora jay.

Sources (shob public, download kore sources/ folder-e rakha):
  - Nirmol v1 (Sigmakib2/Nirmol) — 1,182-ta Bangla-script offensive phrase.
    License repo-te lekha nai; cite kore use korbo, ML training-er jonno
    author nijei download link diyeche.
  - TB-OLID (LanguageTechnologyLab/TB-OLID, CC BY 4.0) — 5,000-ta
    transliterated Banglish comment, offensive O/N label soho. Eta ashol
    training data — template-er bodole ekhan theke ashol row ashbe.
  - LDNOOBW en (CC BY 4.0) — 403-ta English bad word/phrase.
  - curated — nijer hate lekha Banglish + Chatgaiya gali/humki
    (public list-e Chatgaiya nai, tai nije likhte hoyeche).

Outputs:
  - banglish_slang_lexicon.jsonl — ek line = ekta term.
    keys: term, script, language, dialect, category, severity, source
    (+ auto_translit=true jokhon Bangla theke machine-e Roman kora)
  - banglish_threat_dataset.jsonl — ek line = ekta training message.
    keys: text, level, dialect, source
    (ager 179-row template CSV + TB-OLID 5,000 + synthetic ~150)

Severity = project-er level: 1 teasing, 2 harassment/abuse, 3 direct threat.
Lexicon-e level 0 nai — normal word lexicon-er kaj na.
"""
import csv
import json
import os
import random
import re
import unicodedata

OUT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(OUT, "sources")
random.seed(42)

# ------------------------------------------------------------------ helpers
def norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


# ------------------------------------------------- Bangla -> Roman (Banglish)
# Shadharon phonetic map. Perfect na — kintu char n-gram-er jonno jothesto,
# ar prottek row-te auto_translit=true lekha thake, jate bojha jay eta
# machine-e banano.
_BN_CONS = {
    "ক": "k", "খ": "kh", "গ": "g", "ঘ": "gh", "ঙ": "ng",
    "চ": "ch", "ছ": "ch", "জ": "j", "ঝ": "jh", "ঞ": "n",
    "ট": "t", "ঠ": "th", "ড": "d", "ঢ": "dh", "ণ": "n",
    "ত": "t", "থ": "th", "দ": "d", "ধ": "dh", "ন": "n",
    "প": "p", "ফ": "f", "ব": "b", "ভ": "bh", "ম": "m",
    "য": "j", "র": "r", "ল": "l", "শ": "sh", "ষ": "sh", "স": "s", "হ": "h",
    "ড়": "r", "ঢ়": "rh", "য়": "y", "ৎ": "t",
}
_BN_VOW = {
    "অ": "o", "আ": "a", "ই": "i", "ঈ": "i", "উ": "u", "ঊ": "u",
    "ঋ": "ri", "এ": "e", "ঐ": "oi", "ও": "o", "ঔ": "ou",
}
_BN_SIGN = {
    "া": "a", "ি": "i", "ী": "i", "ু": "u", "ূ": "u", "ৃ": "ri",
    "ে": "e", "ৈ": "oi", "ো": "o", "ৌ": "ou",
    "ং": "ng", "ঃ": "h", "ঁ": "n", "্": "",
}
_BN_DIGIT = {"০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
             "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9"}


def bn_to_roman(text: str) -> str:
    """Bangla -> Roman, inherent-'o' niyom soho.
    - Banjonborner por vowel sign thakle seta; ্ thakle kichu na.
    - Banjonborner por arekta banjonborno: 'o' pabe sudhu jokhon porer
      banjonborno-ta vowel ney na (vowel sign nai) — 'morbi' vs 'kolom'.
    - Shesher banjonborno 'o' pabe sudhu ্-er pore ('shobdo').
    - NFC age kore ni, jate ড+় -> ড়, য+় -> য় hoye jay.
    Perfect na, kintu ager version-er 'mrbi'/'kndmer' dhoroner bhul ar nai.
    """
    text = unicodedata.normalize("NFC", text)
    out = []
    n = len(text)

    def is_cons(ch):
        return ch in _BN_CONS

    def is_vowel_sign(ch):
        return ch in _BN_SIGN and ch != "্"

    i = 0
    while i < n:
        ch = text[i]
        if ch == " ":
            out.append(" ")
            i += 1
            continue
        if ch == "্":
            i += 1
            continue
        if ch == "়":  # nukta — জ় -> জ, ফ় -> ফ; alada kono awaj na
            i += 1
            continue
        if text[i:i + 2] == "ক্ষ":
            out.append("kh")
            i += 2
            continue
        if ch in _BN_VOW:
            out.append(_BN_VOW[ch])
            i += 1
            continue
        if ch in _BN_SIGN and ch != "্":
            out.append(_BN_SIGN[ch])
            i += 1
            continue
        if ch in _BN_DIGIT:
            out.append(_BN_DIGIT[ch])
            i += 1
            continue
        if is_cons(ch):
            # য-ফলা / ব-ফলা / র-ফলা: ager-ta ্ chilo
            prev_virama = i > 0 and text[i - 1] == "্"
            if prev_virama and ch == "য":
                out.append("y")
                i += 1
                continue
            if prev_virama and ch == "ব":
                out.append("w")
                i += 1
                continue
            nxt = text[i + 1] if i + 1 < n else ""
            if is_vowel_sign(nxt):
                out.append(_BN_CONS[ch] + _BN_SIGN[nxt])
                i += 2
                continue
            if nxt == "্":
                out.append(_BN_CONS[ch])
                i += 1
                continue
            if is_cons(nxt):
                # porer banjonborno-ta vowel nile (vowel sign thakle) ami coda
                # ('o' pai na); na nile ami onset ('o' pai)
                after = text[i + 2] if i + 2 < n else ""
                if is_vowel_sign(after):
                    out.append(_BN_CONS[ch])
                else:
                    out.append(_BN_CONS[ch] + "o")
                i += 1
                continue
            # sheshe / space-er age: 'o' sudhu ্-er pore
            if prev_virama:
                out.append(_BN_CONS[ch] + "o")
            else:
                out.append(_BN_CONS[ch])
            i += 1
            continue
        # onno character (English/punctuation/Bangla range-er bairer kichu)
        # Bangla range-er kichu roye gele seta drop na kore rekhe dao, pore dekha jabe
        out.append(ch)
        i += 1
    return "".join(out)


# ------------------------------------------------------- category / severity
THREAT_STEMS_BN = ["মারব", "মারি", "খুন", "মেরে ফেল", "হাত পা ভাঙ", "গলা কাট",
                    "মাথা ফাট", "চোখ উপড়", "ঠ্যাং ভাঙ", "জানে মার", "মেরে দেব"]
THREAT_STEMS_RO = ["marbo", "marifelum", "khun", "mere fel", "bhenge", "bhangilom",
                    "fatabo", "gola kat", "chokh upre", "toke mar", "tore mar",
                    "morbi", "rastay paile", "dekhilom", "tor khobor"]
SEXUAL_STEMS = ["চোদ", "চুদ", "মাগী", "মাগি", "খানকি", "বাইনচোদ", "মাদারচোদ",
                "লুচ্চা", "বেশ্যা", "রান্ডি", "চোদা", "চুদা", "chod", "chud",
                "magi", "khanki", "bainchod", "madarchod", "luchcha", "bessha",
                "randi", "motherchod", "fuck", "bitch", "cunt", "dick", "pussy",
                "slut", "whore"]
STRONG_STEMS = ["হারামি", "বেয়াদব", "কুত্তা", "শুয়োর", "জানোয়ার", "জারজ",
                "বদমাইশ", "বদমাশ", "harami", "beyadob", "kutta", "suyor",
                "janoyar", "jaroj", "bodmaish", "bodmash", "bastard",
                "asshole", "son of a bitch"]
MILD_STEMS = ["পাগল", "বোকা", "ফালতু", "গাধা", "বলদ", "ছাগল", "বেকুব",
              "আবাল", "ভণ্ড", "ভন্ড", "pagol", "boka", "faltu", "gadha",
              "bolod", "chagol", "bekub", "abal", "bhondo", "idiot", "stupid",
              "dumb", "fool", "joker"]


def classify_term(term: str):
    """term -> (category, severity). Threat age check, tarpor sexual,
    strong, mild. Kichu na mille default profanity/2."""
    t = norm_key(term)
    for s in THREAT_STEMS_BN + THREAT_STEMS_RO:
        if s in t:
            return "threat", 3
    for s in SEXUAL_STEMS:
        if s in t:
            return "sexual", 2
    for s in STRONG_STEMS:
        if s in t:
            return "insult", 2
    for s in MILD_STEMS:
        if s in t:
            return "insult", 1
    return "profanity", 2


# ------------------------------------------------- Facebook research (2026-11-17)
# Facebook + web theke khuje pawa Bangla slang — Shafiul Sami-r "হালকা বাংলা
# গালি" post, Grammar Academy blog (2026-10-27), Australia Unwrapped-er
# Bangladeshi swear words list, Aminul Jewel-er vocabulary carousel,
# ar "Collecting Fav Gali" group post-er comments theke.
# (term, category, severity) — Bangla script; Roman version bn_to_roman
# diye auto banano hoy (Nirmol-er moto), source=facebook_research.
FACEBOOK_BN = [
    # mild (1) — boka/nirbuddhita, halka teasing
    ("গাধা", "insult", 1), ("বোকা", "insult", 1), ("গবেট", "insult", 1),
    ("আহাম্মক", "insult", 1), ("মাথামোটা", "insult", 1),
    ("হদ্দবোকা", "insult", 1), ("গণ্ডমূর্খ", "insult", 1),
    ("মূর্খ", "insult", 1), ("নির্বোধ", "insult", 1),
    ("অপদার্থ", "insult", 1), ("আকাট", "insult", 1),
    ("হাবাগোবা", "insult", 1), ("উজবুক", "insult", 1),
    ("বুদ্ধুরাম", "insult", 1), ("ফাজিল", "insult", 1),
    ("পাজি", "insult", 1), ("দুষ্টু", "insult", 1),
    ("হতচ্ছাড়া", "insult", 1), ("বেকুব", "insult", 1),
    ("ক্ষ্যাপা", "insult", 1), ("বাউণ্ডুলে", "insult", 1),
    ("বাচাল", "insult", 1), ("মুখরা", "insult", 1),
    ("পণ্ডিতমূর্খ", "insult", 1), ("খিটখিটে", "insult", 1),
    ("শালা", "insult", 1), ("ফালতু", "insult", 1),
    ("ঘন্টা", "insult", 1), ("জাহাজ", "insult", 1),
    ("হিংসুটে", "insult", 1), ("কুচুটে", "insult", 1),
    ("কুটিল", "insult", 1), ("বদখত", "insult", 1),
    ("নাকাল", "insult", 1),
    # strong (2) — osovvota, protarona, ohongkar, rag
    ("অভদ্র", "insult", 2), ("অসভ্য", "insult", 2),
    ("ইতর", "insult", 2), ("বদমাশ", "insult", 2),
    ("বখাটে", "insult", 2), ("দুশ্চরিত্র", "insult", 2),
    ("নচ্ছার", "insult", 2), ("বেয়াদব", "insult", 2),
    ("নির্লজ্জ", "insult", 2), ("বেহায়া", "insult", 2),
    ("উচ্ছৃঙ্খল", "insult", 2), ("নীচ", "insult", 2),
    ("জঘন্য", "insult", 2), ("কুলাঙ্গার", "insult", 2),
    ("ইতরামি", "insult", 2), ("মিথ্যাবাদী", "insult", 2),
    ("প্রতারক", "insult", 2), ("ধোঁকাবাজ", "insult", 2),
    ("জোচ্চোর", "insult", 2), ("ঠকবাজ", "insult", 2),
    ("ভণ্ড", "insult", 2), ("কপট", "insult", 2),
    ("বিশ্বাসঘাতক", "insult", 2), ("বেঈমান", "insult", 2),
    ("ধাপ্পাবাজ", "insult", 2), ("জালিয়াত", "insult", 2),
    ("প্রবঞ্চক", "insult", 2), ("ধূর্ত", "insult", 2),
    ("দুইমুখো", "insult", 2), ("স্বার্থপর", "insult", 2),
    ("অলস", "insult", 2), ("অকর্মণ্য", "insult", 2),
    ("ফাঁকিবাজ", "insult", 2), ("দায়িত্বজ্ঞানহীন", "insult", 2),
    ("গাফিল", "insult", 2), ("ঢিলেঢালা", "insult", 2),
    ("কাজচোর", "insult", 2), ("বেহিসাবি", "insult", 2),
    ("উদাসীন", "insult", 2), ("অযোগ্য", "insult", 2),
    ("অহংকারী", "insult", 2), ("দাম্ভিক", "insult", 2),
    ("আত্মম্ভরী", "insult", 2), ("দেমাগি", "insult", 2),
    ("উদ্ধত", "insult", 2), ("শয়তান", "insult", 2),
    ("বদ", "insult", 2), ("জ্বালাতনকারী", "insult", 2),
    ("অসহ্য", "insult", 2), ("বিরক্তিকর", "insult", 2),
    ("শয়তানের হাড্ডি", "insult", 2), ("গাধার বাচ্চা", "insult", 2),
    ("গাধামি", "insult", 2), ("বুদ্ধিহীন", "insult", 2),
    ("চরিত্রহীন", "insult", 2), ("ভণ্ডামি", "insult", 2),
    ("পাপিষ্ঠ", "insult", 2), ("অবাধ্য", "insult", 2),
    ("ঘুষখোর", "insult", 2), ("বজ্জাত", "insult", 2),
    ("খচ্চর", "insult", 2), ("চামার", "insult", 2),
    # sexual / strong (2)
    ("মাগীবাজ", "sexual", 2), ("হারামজাদা", "sexual", 2),
    ("বোকাচোদা", "sexual", 2), ("গাধামার", "sexual", 2),
    ("হারামখোর", "sexual", 2), ("লম্পট", "sexual", 2),
    ("লুচ্চা", "sexual", 2),
]

# Facebook comments-e manush ashole jevabe lekhe (Banglish, abbreviation,
# dotted masking) — "Collecting Fav Gali" post-er comments theke.
# (term, dialect, category, severity)
FACEBOOK_RO = [
    ("gu kha", "banglish", "sexual", 2),
    ("ghu kha", "banglish", "sexual", 2),
    ("shawaa", "banglish", "sexual", 2),
    ("sawyaaa", "banglish", "sexual", 2),
    ("dhur shauwa", "banglish", "sexual", 2),
    ("madarcod", "banglish", "sexual", 2),
    ("madarchud", "banglish", "sexual", 2),
    ("mc", "banglish", "sexual", 2),
    ("bnc", "banglish", "sexual", 2),
    ("baccah suor", "banglish", "insult", 2),
    ("bal", "banglish", "sexual", 2),
    ("bal siru", "banglish", "sexual", 2),
    ("beadob", "banglish", "insult", 2),
    ("heda", "banglish", "sexual", 2),
    ("k.. pola", "banglish_masked", "sexual", 2),
    ("ma... r pola", "banglish_masked", "sexual", 2),
]

# Chatgaiya — Facebook-er "চাটগাইয়া টু বাংলা ভাষা শিক্ষা কোর্স" post
# (Shahriar Khaled, 2026-10-27) theke. Puro post-ta rojkar shobdo-r
# glossary; slang/ঝগড়া-r sathe jorito 3-ta ekhane, bakigulo research
# note-e (sources/facebook-slang-research.md).
# (bangla term, category, severity)
CHATGAITYA_FB = [
    ("খইজ্জা", "insult", 1),   # ঝগড়া
    ("জান্টা", "insult", 1),    # ঝগড়াটে
    ("গিরাখাডা", "insult", 2),  # পকেটমার
]

# ------------------------------------------------- Facebook video research (2026-11-17)
# "চিটাগাং গালি" video ghure transcript theke pawa Chatgaiya gali.
# (roman, bangla, category, severity)
# - sudanir poya: মিস্টার চাটগাঁইয়া video-r caption theke
#   ("বয়ানী কিয়রে সুদানির পোয়া ডাক্কি না?")
# - chudanir poya: oi video-r comments theke ("অ চুদানির পোয়া ডুম")
# - dum: oi comments thekei (ডুম — Chatgaiya caste insult)
# - magir fan: BAD — Mabrur rant video-r transcript-e "মাগীর ফ্যান" shunechilam,
#   kintu user-er Meta AI list-e ashol form "মাগির ফুত" (Chattogram) /
#   "মাগির ফুয়া" (Sylhet) — transcript vul shunechilo. Meta AI import-e
#   (niche) thik form-ta ache, tai ekhan theke muche fela holo.
VIDEO_CHATGAITYA = [
    ("sudanir poya", "সুদানির পোয়া", "insult", 2),
    ("chudanir poya", "চুদানির পোয়া", "sexual", 2),
    ("dum", "ডুম", "insult", 2),
]

# ------------------------------------------------------- curated Banglish/Chatgaiya
# Public list-e Chatgaiya nai, ar Banglish threat phrase-o kom — tai nijer
# hate lekha. Egulo real Chatgaiya/Banglish-e babohrito kotha.
CURATED = [
    # (term, dialect, category, severity)
    ("tui ekta pagol", "banglish", "insult", 1),
    ("tui ekta boka", "banglish", "insult", 1),
    ("tui ekta faltu", "banglish", "insult", 1),
    ("tui ekta gadha", "banglish", "insult", 1),
    ("tui ekta bolod", "banglish", "insult", 1),
    ("tui ekta abal", "banglish", "insult", 1),
    ("tui ekta bhondo", "banglish", "insult", 1),
    ("tui ekta chagol", "banglish", "insult", 1),
    ("tui ekta bekub", "banglish", "insult", 1),
    ("tor matha kharap", "banglish", "insult", 1),
    ("tui ekta beyadob", "banglish", "insult", 2),
    ("tui ekta harami", "banglish", "insult", 2),
    ("tui ekta bodmaish", "banglish", "insult", 2),
    ("tui ekta mitthuk", "banglish", "insult", 2),
    ("tui ekta chor", "banglish", "insult", 2),
    ("tui ekta soitan", "banglish", "insult", 2),
    ("tui ekta amanush", "banglish", "insult", 2),
    ("tui ekta janoyar", "banglish", "insult", 2),
    ("kuttar baccha", "banglish", "insult", 2),
    ("suyor er baccha", "banglish", "insult", 2),
    ("bainchod", "banglish", "sexual", 2),
    ("madarchod", "banglish", "sexual", 2),
    ("khankir pola", "banglish", "sexual", 2),
    ("chudir bhai", "banglish", "sexual", 2),
    ("motherchod", "banglish", "sexual", 2),
    ("luchcha", "banglish", "sexual", 2),
    ("randi", "banglish", "sexual", 2),
    ("toke marbo", "banglish", "threat", 3),
    ("tore marbo", "banglish", "threat", 3),
    ("toke mere felbo", "banglish", "threat", 3),
    ("tore khun korbo", "banglish", "threat", 3),
    ("tor hat pa bhenge dibo", "banglish", "threat", 3),
    ("tor matha fatabo", "banglish", "threat", 3),
    ("tor gola katbo", "banglish", "threat", 3),
    ("tor chokh upre felbo", "banglish", "threat", 3),
    ("tor bari chini, aitesi", "banglish", "threat", 3),
    ("tui morbi amar hate", "banglish", "threat", 3),
    ("rastay paile tore dekhbo", "banglish", "threat", 3),
    ("tore ami charbo na", "banglish", "threat", 3),
    ("tor khobor ase", "banglish", "threat", 3),
    ("tui aai ekbar, tore dekhbo", "banglish", "threat", 3),
    # Chatgaiya — public dataset-e nai, tai curated
    ("tore marifelum", "chatgaiya", "threat", 3),
    ("tore dekhilom", "chatgaiya", "threat", 3),
    ("tor theng bhangilom", "chatgaiya", "threat", 3),
    ("tore rastat dhorilom", "chatgaiya", "threat", 3),
    ("tore khuin korilom", "chatgaiya", "threat", 3),
    ("tor bari chini, aaiyom", "chatgaiya", "threat", 3),
    ("tui moror", "chatgaiya", "threat", 3),
    ("tor khobor asi", "chatgaiya", "threat", 3),
    ("tui aai, tor khobor asi", "chatgaiya", "threat", 3),
    ("tui ekta pagol", "chatgaiya", "insult", 1),
    ("tui ekta boka", "chatgaiya", "insult", 1),
    ("tui ekta harami", "chatgaiya", "insult", 2),
    ("tui ekta beyadob", "chatgaiya", "insult", 2),
]

# ------------------------------------------------- political / Gen Z harvest
# 2026-11-17: Awami League / BNP / NCP Facebook page-er comment theke
# harvest (proti page 1 post x 20 comment = 60 comment) + user-er
# variant list (dhon/don, criminal leet, razakar variant, s3x).
# (roman, bangla, category, severity). Raw comment kokhono copy kora hoy
# nai — shudhu shobdo + template bakko (anonymized).
POLITICAL_GENZ = [
    ("razakar", "রাজাকার", "insult", 2),
    ("rojokor", "রজাকর", "insult", 2),
    ("rojkar", "রজকার", "insult", 2),
    ("rojokor", "রজকর", "insult", 2),
    ("ros kora", "রস করা", "insult", 2),
    ("dhon", "ধন", "sexual", 2),
    ("don", "দন", "sexual", 2),
    ("dhonchhipi", "ধনছিপি", "sexual", 2),
    ("donsipi", "দনসিপি", "sexual", 2),
    ("motherboard", "মাদারবোর্ড", "insult", 2),
    ("criminal", "ক্রিমিনাল", "insult", 1),
    ("baal", "বাল", "insult", 2),
    ("gopali", "গোপালি", "insult", 1),
    ("fascist", "ফ্যাসিস্ট", "insult", 1),
    ("hybrid", "হাইব্রিড", "insult", 1),
    ("lollipop", "ললি পপ", "insult", 1),
    ("akam", "আকাম", "insult", 1),
    ("dhandabaji", "ধান্দাবাজি", "insult", 1),
    ("opodartho", "অপদার্থ", "insult", 2),
    ("july jongi", "জুলাই জঙ্গি", "insult", 2),
    ("putki maro", "পুটকি মারো", "sexual", 3),
    ("s3x", "এস থ্রি এক্স", "sexual", 2),
]

# ------------------------------------------------------------------ lexicon
lexicon = []
seen = set()


def add_lex(term, script, language, dialect, category, severity, source,
            auto_translit=False, district=None):
    term = norm_key(term)
    if not term or len(term) < 2:
        return
    key = (term, script, dialect)
    if key in seen:
        return
    seen.add(key)
    row = {"term": term, "script": script, "language": language,
           "dialect": dialect, "category": category, "severity": severity,
           "source": source}
    if auto_translit:
        row["auto_translit"] = True
    if district:
        row["district"] = district
    lexicon.append(row)


# 1. Nirmol — Bangla script phrases + auto Roman
nirmol = json.load(open(os.path.join(SRC, "nirmol", "nirmol.json"), encoding="utf-8"))
for phrase in nirmol:
    cat, sev = classify_term(phrase)
    add_lex(phrase, "bangla", "bn", "standard", cat, sev, "nirmol_v1")
    roman = bn_to_roman(phrase)
    if roman != norm_key(phrase):
        cat_r, sev_r = classify_term(roman)
        add_lex(roman, "latin", "bn", "banglish", cat_r, sev_r,
                "nirmol_v1", auto_translit=True)

# 2. LDNOOBW English
with open(os.path.join(SRC, "ldnoobw_en.txt"), encoding="utf-8") as f:
    for line in f:
        w = norm_key(line)
        if not w or w.isdigit():
            continue
        cat, sev = classify_term(w)
        add_lex(w, "latin", "en", "standard", cat, sev, "ldnoobw_en")

# 3. Facebook research — Bangla row + auto Roman row (Nirmol-er moto)
for term, cat, sev in FACEBOOK_BN:
    add_lex(term, "bangla", "bn", "standard", cat, sev, "facebook_research")
    roman = bn_to_roman(term)
    if roman != norm_key(term):
        add_lex(roman, "latin", "bn", "banglish", cat, sev,
                "facebook_research", auto_translit=True)

# 4. Facebook comments-er Banglish (Roman-only) + dotted masking
for term, dialect, cat, sev in FACEBOOK_RO:
    add_lex(term, "latin", "bn", dialect, cat, sev, "facebook_research")

# 5. Chatgaiya — Facebook glossary post theke
for term, cat, sev in CHATGAITYA_FB:
    add_lex(term, "bangla", "bn", "chatgaiya", cat, sev,
            "facebook_chatgaiya_glossary")
    roman = bn_to_roman(term)
    if roman != norm_key(term):
        add_lex(roman, "latin", "bn", "chatgaiya", cat, sev,
                "facebook_chatgaiya_glossary", auto_translit=True)

# 6. Chatgaiya — Facebook video transcript/comments theke
for roman, term, cat, sev in VIDEO_CHATGAITYA:
    add_lex(term, "bangla", "bn", "chatgaiya", cat, sev,
            "facebook_video_research")
    add_lex(roman, "latin", "bn", "chatgaiya", cat, sev,
            "facebook_video_research")

# 7. Meta AI — user-er anano district-wise gali list (2026-11-17)
# sources/meta-ai-gali-lists.csv: (word, district, level) — 477 row, 157
# distinct word, 17 jela. District-ke dialect-e map: Chattogram-area
# (Chattogram/Cox's Bazar/Rangamati/Bandarban/Khagrachari) -> chatgaiya,
# baki jela -> banglish (Bangla row-e "standard" na — district-i dialect-er
# kaj kore). Ashol jela "district" column-e thake (Parquet schema-e jog).
# Category: Level 3 + sexual stem thakle "sexual", Level 3 baki "profanity",
# Level 1/2 "insult" (Meta AI-er level-i severity).
CHATGAITYA_DISTRICTS = {"Chattogram", "CoxsBazar", "Rangamati",
                        "Bandarban", "Khagrachari"}
SEXUAL_STEMS_META = ("চোদ", "চুদ", "ছুদ", "মাদার", "বাইন", "গান্ডু",
                     "ভোদা", "হোগা", "পোন্দ", "ধোন", "মাগি", "খানকি",
                     "বেশ্যা", "রেন্ডি", "নটি")
import csv as _csv
meta_rows = []
with open(os.path.join(SRC, "meta-ai-gali-lists.csv"),
          encoding="utf-8") as f:
    for row in _csv.DictReader((ln for ln in f if not ln.startswith("#"))):
        meta_rows.append((row["word"], row["district"], int(row["level"])))
# Eki shobdo onek jelay thake (harami = 17 jela) — lexicon-e ek row per
# (term, dialect) rakhi, district column-e SOB jela comma diye jora thake,
# jate prothom jela-ta matro dekhe vul na hoy.
META_BY_WORD = {}
for word, district, lvl in meta_rows:
    d = META_BY_WORD.setdefault(word, {"level": lvl, "districts": set()})
    d["districts"].add(district)
META_TERMS = {}  # roman -> (level, dialect) — message/sentence bananor jonno
for word, info in META_BY_WORD.items():
    lvl = info["level"]
    district = ",".join(sorted(info["districts"]))
    is_chat = bool(info["districts"] & CHATGAITYA_DISTRICTS)
    dialect_bn = "chatgaiya" if is_chat else "standard"
    dialect_ro = "chatgaiya" if is_chat else "banglish"
    if lvl == 3 and any(s in word for s in SEXUAL_STEMS_META):
        cat = "sexual"
    elif lvl == 3:
        cat = "profanity"
    else:
        cat = "insult"
    add_lex(word, "bangla", "bn", dialect_bn, cat, lvl, "meta_ai",
            district=district)
    roman = bn_to_roman(word)
    if roman != norm_key(word):
        add_lex(roman, "latin", "bn", dialect_ro, cat, lvl, "meta_ai",
                auto_translit=True, district=district)
    if roman not in META_TERMS:
        META_TERMS[roman] = (lvl, dialect_ro)

# 8. Curated Banglish / Chatgaiya
for term, dialect, cat, sev in CURATED:
    add_lex(term, "latin", "bn", dialect, cat, sev, "curated")

# 9. Political / GenZ harvest — Facebook comment + user-er variant list
for roman, bn, cat, sev in POLITICAL_GENZ:
    add_lex(bn, "bangla", "bn", "standard", cat, sev, "facebook_political")
    add_lex(roman, "latin", "bn", "banglish", cat, sev, "facebook_political")

# 10. Sexual-threat verbs (2026-11-18) — future-tense threat form gulo
# L3 "threat" (explicit add — classify_term "চুদ" stem dekhe L2 dito).
for v in ["চুদবো", "চুদবি", "চুদবে", "চোদবো", "চোদবি"]:
    add_lex(v, "bangla", "bn", "standard", "threat", 3,
            "threat_verbs_20261118")
for v in ["chudbo", "chudbi", "chudbe", "chodbo", "chodbi"]:
    add_lex(v, "latin", "bn", "banglish", "threat", 3,
            "threat_verbs_20261118")
for t in ["মারে চুদি", "মারে চোদি"]:
    add_lex(t, "bangla", "bn", "standard", "threat", 3,
            "threat_verbs_20261118")
for t in ["mare chudi", "mare chodi"]:
    add_lex(t, "latin", "bn", "banglish", "threat", 3,
            "threat_verbs_20261118")

# 11. Chatgaiya insult phrase "ক্ষণিকের পোলা" (2026-11-18, user-er test)
# "ক্ষণিকের" (ek muhurter mohila) + "পোলা" (chhele) — mane "beshyar
# chhele"-r moto parentage gali. Ekok shobdo gali NA: "ক্ষণিকের" kothao
# nai, "পোলা" nije neutral — insult-ta combination-er cultural meaning-e.
# Tai phrase hishebei L2 "insult" (L3 NA — kono threat verb nai;
# "জারজ পোলা"/"ফইন্নির পোলা"-r moto L2).
for t in ["ক্ষণিকের পোলা", "ক্ষণিকের পোয়া", "খনিকের পোলা", "ক্ষণিকের ফুয়া"]:
    add_lex(t, "bangla", "bn", "chatgaiya", "insult", 2,
            "chatgaiya_khoniker_20261118")
for t in ["khoniker pola", "khoniker poa", "khoniker fua", "khoniker polay"]:
    add_lex(t, "latin", "bn", "chatgaiya", "insult", 2,
            "chatgaiya_khoniker_20261118")

# 12. Learned words (continuous learning, 2026-11-18) — user-er test +
# review theke label kora notun shobdo. learned_words.json =
# [{"text", "level", "dialect", "category"}]. Script auto-detect (Bangla
# okkhor thakle bangla). add_lex nije dedupe kore (term, script, dialect).
def _learned_script(t):
    return "bangla" if re.search(r"[\u0980-\u09FF]", t) else "latin"


LEARNED = []
try:
    LEARNED = json.load(open(os.path.join(OUT, "learned_words.json"),
                             encoding="utf-8"))
except Exception:
    LEARNED = []
for e in LEARNED:
    t = str(e.get("text", "")).strip()
    if not t:
        continue
    add_lex(t, _learned_script(t), "bn", e.get("dialect", "banglish"),
            e.get("category", "insult"), int(e.get("level", 2)),
            "learned_words")

# 7. Masked variants (obfuscation) — severity>=2 Banglish term-er ekta
#    masked copy: majher ekta okkhor * diye bodle dao
masked_rows = []


def mask_term(term: str) -> str:
    words = term.split()
    cands = [i for i, w in enumerate(words) if len(w) >= 4]
    if not cands:
        return term
    i = random.choice(cands)
    w = words[i]
    j = random.randint(1, len(w) - 2)
    words[i] = w[:j] + "*" + w[j + 1:]
    return " ".join(words)


for row in list(lexicon):
    if row["script"] == "latin" and row["dialect"] in ("banglish", "chatgaiya") \
            and row["severity"] >= 2:
        m = mask_term(row["term"])
        if m != row["term"]:
            masked_rows.append((m, row))
for m, row in masked_rows[:400]:
    add_lex(m, "latin", "bn", "banglish_masked", row["category"],
            row["severity"], "generated_masked")

random.shuffle(lexicon)
lex_path = os.path.join(OUT, "banglish_slang_lexicon.jsonl")
with open(lex_path, "w", encoding="utf-8") as f:
    for row in lexicon:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"lexicon rows: {len(lexicon)} -> {lex_path}")

# ------------------------------------------------------------------ messages
# Lexicon scan diye TB-OLID offensive row-er severity mapo
THREAT_SCAN = ["marbo", "marifelum", "khun", "mere fel", "bhenge", "bhangilom",
               "fatabo", "gola kat", "chokh upre", "toke mar", "tore mar",
               "morbi", "rastay paile", "dekhilom"]
SEXUAL_SCAN = ["chod", "chud", "magi", "khanki", "bainchod", "madarchod",
               "luchcha", "bessha", "randi", "motherchod", "fuck", "bitch",
               "kutta", "suyor", "harami", "beyadob", "jaroj"]
MILD_SCAN = ["pagol", "boka", "faltu", "gadha", "bolod", "chagol", "bekub",
             "abal", "idiot", "stupid"]


def tbolid_level(text: str, offensive: str) -> int:
    if offensive != "O":
        return 0
    t = norm_key(text)
    for s in THREAT_SCAN:
        if s in t:
            return 3
    for s in SEXUAL_SCAN:
        if s in t:
            return 2
    for s in MILD_SCAN:
        if s in t:
            return 1
    return 2  # offensive kintu kono lexicon term mileni — default harassment


messages = []
seen_msg = set()


def add_msg(text, level, dialect, source):
    text = re.sub(r"\s+", " ", str(text).strip())
    if not text or len(text) < 2:
        return
    if "?" * 3 in text:  # mojibake / broken encoding row
        return
    key = norm_key(text)
    if key in seen_msg:
        return
    seen_msg.add(key)
    messages.append({"text": text, "level": int(level),
                     "dialect": dialect, "source": source})


# 1. ager 179-row template (CSV theke convert — CSV ar training file na)
csv_path = os.path.join(OUT, "banglish_threat_dataset.csv")
with open(csv_path, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        add_msg(r["text"], int(r["level"]), r["dialect"], r["source"])

# 2. TB-OLID — ashol Banglish data (5,000)
for name in ("train", "test"):
    data = json.load(open(os.path.join(SRC, "tbolid", f"{name}.json"), encoding="utf-8"))
    for item in data:
        text = item.get("text", "")
        lvl = tbolid_level(text, item.get("offensive_gold", "N"))
        add_msg(text, lvl, "banglish", f"tbolid_{name}")

# 3. Synthetic — curated threat/insult term diye template sentence
#    TB-OLID-e Level 3 (direct threat) ar Level 1 (mild teasing) kom, tai
#    synthetic diye ektu balance kori. Term nijei full phrase hole wrapper
#    lagbe na; shudhu prefix diye variety barai.
THREAT_WRAPPERS = ["{t}", "shon, {t}", "tui shon, {t}", "mone rakhis, {t}",
                   "tui ekbar aai, {t}", "kalke {t}"]
MILD_WORDS = ["pagol", "boka", "faltu", "gadha", "bolod", "chagol", "bekub",
              "abal", "bhondo", "idiot", "stupid"]
MILD_TEMPLATES = ["tui ekta {w}", "tui to {w}", "tor moto {w} nai",
                  "tui ekdom {w}"]
syn_count = 0
for term, dialect, cat, sev in CURATED:
    if sev == 3:
        for tpl in THREAT_WRAPPERS:
            add_msg(tpl.format(t=term), 3, dialect, "synthetic_lexicon")
            syn_count += 1
for w in MILD_WORDS:
    for tpl in MILD_TEMPLATES:
        add_msg(tpl.format(w=w), 1, "banglish", "synthetic_lexicon")
        syn_count += 1
# Facebook video theke pawa Chatgaiya gali — Level 2 (harassment/insult,
# direct physical threat na) hishebe message row
for roman, _bn, _cat, _sev in VIDEO_CHATGAITYA:
    add_msg(f"tui ekta {roman}", 2, "chatgaiya", "facebook_video")
    add_msg(f"o {roman}", 2, "chatgaiya", "facebook_video")
    syn_count += 2

# Meta AI terms theke sentence (Meta AI-er formula: 1 gali x 3 template)
# + spaced/leet augmentation (Meta AI-er 4-version suggestion)
META_SENT_TEMPLATES = ["tui ekta {r}", "{r} koi jas", "ei {r} edike ai"]
LEET_MAP = str.maketrans({"a": "4", "o": "0", "e": "3", "i": "1", "s": "5"})
for roman, (lvl, dialect) in META_TERMS.items():
    for tpl in META_SENT_TEMPLATES:
        add_msg(tpl.format(r=roman), lvl, dialect, "meta_ai")
        syn_count += 1
    # spaced variant: "h a r a m i" — filter fanki dewar style; model jate chine
    spaced = " ".join(" ".join(w) for w in roman.split())
    add_msg(spaced, lvl, "banglish_masked", "meta_ai_spaced")
    syn_count += 1
    # leet variant: "h4rami" — normalizer-e dhora pore, tobu surface form shikhao
    leet = roman.translate(LEET_MAP)
    if leet != roman:
        add_msg(leet, lvl, dialect, "meta_ai_leet")
        syn_count += 1
# 9. Political/GenZ terms theke sentence + variant augmentation.
#    User-er point: model-ke VARIANT predict korte hobe — na dekha banan-o
#    dhorte hobe. Tai prottek term-er: 3 template + spaced + leet +
#    Bangla confusion variant (vowel-sign drop / consonant swap) + Bangla
#    leet (ি -> 1) row. Shob row eki level-e.
GENZ_TERMS = {}
BN_VOWEL_SIGNS = set("ািীুূেোৌ")
BN_CONFUSE = {"ধ": "দ", "দ": "ধ", "ঢ": "ড", "ড": "ঢ", "শ": "স", "স": "শ",
              "ষ": "স", "ন": "ণ", "ণ": "ন", "য": "য়", "য়": "য",
              "ঈ": "ই", "ঊ": "উ", "ং": "ঙ", "ঙ": "ং"}


def bn_variants(term, cap=4):
    out = []
    seen_v = set()
    for i, ch in enumerate(term):
        if ch in BN_VOWEL_SIGNS:
            v = term[:i] + term[i + 1:]  # vowel sign drop: রাজাকার -> রজাকার
            if v and v != term and v not in seen_v:
                seen_v.add(v)
                out.append(v)
        if ch in BN_CONFUSE:
            v = term[:i] + BN_CONFUSE[ch] + term[i + 1:]
            if v != term and v not in seen_v:
                seen_v.add(v)
                out.append(v)
        if len(out) >= cap:
            break
    return out


def bn_leet_variants(term, cap=2):
    out = []
    for i, ch in enumerate(term):
        if ch in ("ি", "ী"):
            out.append(term[:i] + "1" + term[i + 1:])
        if len(out) >= cap:
            break
    return out


BN_SENT_TEMPLATES = ["তুই একটা {b}", "{b} কই যাস", "এই {b} এদিকে আয়",
                     "ওই {b} টা", "শোন {b}", "{b} রে", "আরে {b}"]
for roman, bn, cat, sev in POLITICAL_GENZ:
    GENZ_TERMS[roman] = (sev, "banglish")
    for tpl in META_SENT_TEMPLATES:
        add_msg(tpl.format(r=roman), sev, "banglish", "facebook_political")
        syn_count += 1
    spaced = " ".join(" ".join(w) for w in roman.split())
    add_msg(spaced, sev, "banglish_masked", "genz_spaced")
    syn_count += 1
    leet = roman.translate(LEET_MAP)
    if leet != roman:
        add_msg(leet, sev, "banglish", "genz_leet")
        syn_count += 1
    # Bangla-script row AGE 1-ta chilo — min_df=2 ar 80/20 split-e sheta
    # train-e porto na, tai Bangla input shob L0/L1 dito. 3 template dile
    # token-er df>=3 hoy, train-e pore. (Training data age Roman-only chilo.)
    # Bare row: ekok shobdo input ("বাল", "don")-o jate chine.
    add_msg(roman, sev, "banglish", "facebook_political")
    syn_count += 1
    add_msg(bn, sev, "banglish", "facebook_political")
    syn_count += 1
    for tpl in BN_SENT_TEMPLATES:
        add_msg(tpl.format(b=bn), sev, "banglish", "facebook_political")
        syn_count += 1
    for v in bn_variants(bn):
        add_msg(f"তুই একটা {v}", sev, "banglish", "genz_bn_variant")
        add_msg(f"{v} কই যাস", sev, "banglish", "genz_bn_variant")
        syn_count += 2
    for v in bn_leet_variants(bn):
        add_msg(v, sev, "banglish", "genz_bn_variant")
        add_msg(f"তুই একটা {v}", sev, "banglish", "genz_bn_variant")
        syn_count += 2
# ------------------------------------------------- sexual-threat verbs +
# family-target patterns (2026-11-18, user-er report)
# User-er example text ("তুরে চুদবো ... মারে চু,,, দি") model L2 dichchilo,
# ashol-e L3 (direct sexual threat). Karon: (1) "চুদবো/চুদবি/চুদবে"
# future-tense threat verb lexicon-ei chilo na; (2) "চুদি/চোদ"
# classify_term-e sexual->L2 (insult hishebe thik, kintu "তোরে চুদবো" /
# "মারে চুদি" holo DIRECT THREAT -> L3); (3) "চু,,, দি" comma-masking
# normalizer-e dhora porto na; (4) Bangla-script kill threat
# ("তোকে মেরে ফেলবো")-er kono template row chilo na (Roman-first data).
# Bangla gali-r mul construction: [target-er family] + [sexual verb] —
# "তোর মারে/বোনরে/নানিরে চুদি" — tai family-target PHRASE-kei shikhate hoy,
# "মারে/বোনরে" ekok shobdo ke NA (ogulo nije gali na: "আমার মারে
# ভালোবাসি" L0 thakte hobe — niche neutral counter-example ache).
# Euphemism: "motherboard" (mother+chod-er bodle) — GenZ-e gali hishebei
# use hoy, tai L2 (POLITICAL_GENZ-e age thekei ache).
THREAT_VERBS_BN = ["চুদবো", "চুদবি", "চুদবে", "চোদবো", "চোদবি"]
THREAT_VERBS_RO = ["chudbo", "chudbi", "chudbe", "chodbo", "chodbi"]
FAMILY_BN = ["মারে", "বোনরে", "নানিরে", "মায়ের", "বোনের", "নানীর",
             "খালারে", "চাচিরে", "দাদিরে"]
FAMILY_RO = ["mare", "bonre", "nanire", "mayer", "boner", "nanir",
             "khalare", "chachire", "dadire"]
PARTY_TGT_BN = ["ছাত্রলীগের", "ছাএলীগের", "এনসিপির", "আওয়ামী লীগের",
                "বিএনপির", "জামাতের", "টোকাইর", "টোকাইয়ার"]
PARTY_TGT_RO = ["chatroleaguer", "ncpr", "awami leaguer", "bnpr",
                "jamater", "tokair"]
KILL_BN = ["তোকে মেরে ফেলবো", "তোরে মেরে ফেলবো", "তুরে মেরে ফেলবো",
           "তোকে মেরে ফেলব", "তোকে খুন করবো", "তোরে খুন করবো",
           "তোকে মারবো", "তোরে মারবো", "তুরে মারবো", "তোকে গলা কাটবো",
           "তোর হাত পা ভেঙে দেবো", "তোরে দেখে নেবো"]
KILL_RO = ["toke mere felbo", "tore mere felbo", "toke khun korbo",
           "tore khun korbo", "toke marbo", "toke gola katbo",
           "tor hat pa bhenge dibo", "tore dekhe nebo"]
MASKED_COMMA_L3 = ["চু,,, দি", "চু,,দি", "চু,দি", "মারে চু,,, দি",
                   "তুর মারে চু,,, দি", "তোর মারে চু,,, দি",
                   "chu,,,di", "chud,,,bo", "তুরে চু,,,দবো"]
COMPOSITE_L3 = ["তুরে চুদবো ওয়েট কর আর কয়দিন", "তুর টোকাইর মারে চুদি",
                "ওয়েট কর আর কয়দিন তোরে চুদবো",
                "তোর মারে চুদি এনসিপির মারে চুদি"]
NEUTRAL_L0 = ["আমি ছাত্রলীগ করি", "ছাত্রলীগের সমাবেশ কাল",
              "এনসিপি ভালো দল", "আওয়ামী লীগের মিছিল হবে",
              "বিএনপির নেতা আসবে", "এটা শেখ হাসিনার বাংলাদেশ",
              "আমি এনসিপি করি", "আমার মাকে ভালোবাসি",
              "বোনকে ভালোবাসি", "আমার মা ভালো মানুষ",
              "ওয়েট কর, আসতেছি", "আমার মারে ভালোবাসি",
              "তোর মারে সালাম দিস", "মারে জিজ্ঞেস কর",
              "আমার মারে ডাক দিস", "বোনরে ডাক দিস",
              # 2026-11-18 fix: "ওয়েট কর" + "তোর মারে" threat template-e
              # eto bar ashay model ei phrase-kei L3 signal dhore felechilo —
              # nirdosh bebohar-o L3 hoye jachchilo ("ওয়েট কর, আসতেছি" L3,
              # "তোর মারে সালাম দিস" L3). Verb-i parthokko — tai ei
              # counter-example gulo dorkar.
              "একটু ওয়েট কর", "ওয়েট করো প্লিজ", "ওয়েট কর, আসছি",
              "আমি আসতেছি", "বাসায় আসতেছি", "একটু পরে আসতেছি",
              "মারে সালাম দিও", "তোর মাকে সালাম দিস",
              "আমার মারে সালাম দিস", "সালাম দিস", "আসসালামু আলাইকুম",
              # 2026-11-18 round 2: "তোর মারে সালাম দিস" tokhono L3
              # (0.52) — "তোর মারে" bigram-ta L3 row-te eto bar j
              # model verb-ta na dekhe prefix-kei signal dhoreche. Tai
              # "তোর মারে + nirdosh verb"-er aro boichitro dorkar.
              "তোর মারে দেখলাম", "মারে বলিস", "তোর মারে জিজ্ঞেস করিস",
              "মারে ফোন দিস", "তোর মারে ডাক দিস", "তোর মারে কেমন আছে",
              "মারে ভালো রেখো", "তোর মারে খবর নিস",
              "শেখ হাসিনার বাংলাদেশ", "বাংলাদেশ আমার দেশ",
              "আমি বাংলাদেশকে ভালোবাসি", "ঢাকা বাংলাদেশের রাজধানী",
              # "এটা শেখ হাসিনার বাংলাদেশ" borderline flipper — L3 variant
              # test-er full text-er tail-e thakay ("...এটা শেখ হাসিনার
              # বাংলাদেশ") "বাংলাদেশ"-er char n-gram L3 signal pay; abar
              # L2-oo hoy (retrain gate-e dhora 2026-11-18, 2 bar). Tai
              # ei exact sentence-er aro nirdosh frame dorkar.
              "এটা শেখ হাসিনার বাংলাদেশ আমার দেশ",
              "আমি শেখ হাসিনার বাংলাদেশে থাকি",
              "শেখ হাসিনার বাংলাদেশ সুন্দর দেশ",
              "এটা আমার বাংলাদেশ", "এটা আমাদের বাংলাদেশ",
              "বাংলাদেশ স্বাধীন দেশ", "আমি বাংলাদেশের নাগরিক",
              # 2026-11-18 round 3: "ওয়েট কর, আসতেছি" abar L3-te firlo
              # (0.49 vs L0 0.39 — near-tie). Threat-side-e "ওয়েট কর"
              # ~20 row, neutral-side-e matro ~7 — tai aro neutral frame.
              "ওয়েট কর, খাইতেছি", "ওয়েট কর, পড়তেছি",
              "ওয়েট কর, যাইতেছি", "ওয়েট কর ভাই", "ওয়েট করো একটু",
              "আসতেছি ভাই", "আমি আসতেছি একটু পরে", "তুমি ওয়েট করো"]
# Lexicon: threat verb form gulo L3 "threat" — lexicon section-e (10)
# agei add kora ache; ekhane shudhu message row banano hoy.
for v in THREAT_VERBS_BN:
    for tpl in ["তুরে {v}", "তোরে {v}", "তোকে {v}", "{v} তোরে",
                "ওয়েট কর {v}", "আর কয়দিন {v}"]:
        add_msg(tpl.format(v=v), 3, "banglish", "threat_verbs_20261118")
        syn_count += 1
    add_msg(v, 3, "banglish", "threat_verbs_20261118")
    syn_count += 1
for v in THREAT_VERBS_RO:
    for tpl in ["tore {v}", "ture {v}", "toke {v}", "{v} tore",
                "wait kor {v}"]:
        add_msg(tpl.format(v=v), 3, "banglish", "threat_verbs_20261118")
        syn_count += 1
    add_msg(v, 3, "banglish", "threat_verbs_20261118")
    syn_count += 1
for f in FAMILY_BN:
    for tpl in ["তোর {f} চুদি", "তুর {f} চুদি", "তোর {f} চুদবো",
                "{f} চুদি"]:
        add_msg(tpl.format(f=f), 3, "banglish", "threat_verbs_20261118")
        syn_count += 1
for f in FAMILY_RO:
    for tpl in ["tor {f} chudi", "tur {f} chudi", "tor {f} chudbo"]:
        add_msg(tpl.format(f=f), 3, "banglish", "threat_verbs_20261118")
        syn_count += 1
for p in PARTY_TGT_BN:
    for tpl in ["{p} মারে চুদি", "তুর {p} মারে চুদি", "{p} মারে চুদবো",
                "ওয়েট কর {p} মারে চুদি"]:
        add_msg(tpl.format(p=p), 3, "banglish", "threat_verbs_20261118")
        syn_count += 1
for p in PARTY_TGT_RO:
    for tpl in ["{p} mare chudi", "tor {p} mare chudi"]:
        add_msg(tpl.format(p=p), 3, "banglish", "threat_verbs_20261118")
        syn_count += 1
for t in KILL_BN + KILL_RO + MASKED_COMMA_L3 + COMPOSITE_L3:
    add_msg(t, 3, "banglish", "threat_verbs_20261118")
    syn_count += 1
for t in ["তুই একটা মাদারবোর্ড", "m0therboard", "mother board"]:
    add_msg(t, 2, "banglish", "threat_verbs_20261118")
    syn_count += 1
for t in NEUTRAL_L0:
    add_msg(t, 0, "banglish", "threat_verbs_20261118")
    syn_count += 1
# Chatgaiya "ক্ষণিকের পোলা" (2026-11-18, user-er test — model near-tie
# L3 0.32 vs L0 0.26 dichchilo, mane ashole chinto na). L2 (parentage
# insult, threat verb nai). "পোলা" nije neutral — tai L0 counter-example
# dorkar, naile prottek "পোলা" L2 hoye jabe.
KHONIKER_BN = ["ক্ষণিকের পোলা", "ক্ষণিকের পোয়া", "খনিকের পোলা", "ক্ষণিকের ফুয়া"]
KHONIKER_RO = ["khoniker pola", "khoniker poa", "khoniker fua", "khoniker polay"]
for t in KHONIKER_BN:
    for tpl in ["তুই একটা {t}", "তুই {t}", "ওই {t}", "{t} কই যাস"]:
        add_msg(tpl.format(t=t), 2, "chatgaiya", "chatgaiya_khoniker_20261118")
        syn_count += 1
    add_msg(t, 2, "chatgaiya", "chatgaiya_khoniker_20261118")
    syn_count += 1
for t in KHONIKER_RO:
    for tpl in ["tui ekta {t}", "tui {t}", "oi {t}", "{t} koi jas"]:
        add_msg(tpl.format(t=t), 2, "chatgaiya", "chatgaiya_khoniker_20261118")
        syn_count += 1
    add_msg(t, 2, "chatgaiya", "chatgaiya_khoniker_20261118")
    syn_count += 1
for t in ["kh*niker pola", "ক্ষণিকের পো*লা"]:
    add_msg(t, 2, "chatgaiya", "chatgaiya_khoniker_20261118")
    syn_count += 1
for t in ["পোলাটা ভালো ছেলে", "আমার পোলা স্কুলে যায়", "পোলা কই যাস",
          "oi pola", "pola ta bhalo", "amar pola school e jay",
          "পোলাটা ভাত খাইছে", "chhele ta bhalo",
          # "oi pola" neutral counterweight: L2 template "oi {gali} pola"
          # eto bar ashay bare "oi pola"-o L2 hoye jachchilo (gate-e dhora
          # poreche 2026-11-18) — "oi + pola" nirdosh frame-e aro row dorkar.
          "oi pola edike ai", "oi pola koi jas", "oi pola ta dek",
          "oi pola school-e jabi", "oi pola, bhat khaisos"]:
    add_msg(t, 0, "chatgaiya", "chatgaiya_khoniker_20261118")
    syn_count += 1
# Friendly / non-gali context (Meta AI-er suggestion): eki shobdo bondhu-ke
# moja kore bolle gali NA — Level 0 row. Model-e context shikhanor ekmatro
# upay sentence-soho data; speaker context amader feature-e nai, tai eta
# shudhu partial fix — README-te limit hishebe lekha ache.
FRIENDLY_L0 = [
    ("oda koi jor", "chatgaiya"),   # অডা কই যর?
    ("kire bolod", "chatgaiya"),    # কিরে বলদ, ভাত খাইছস?
    ("bhat khaisos", "chatgaiya"),  # ভাত খাইছস?
    ("bondhu", "banglish"),         # বন্ধু
    ("bhalo", "banglish"),          # ভালো
]
for text, dialect in FRIENDLY_L0:
    add_msg(text, 0, dialect, "meta_ai")
    syn_count += 1
# Halka (Level 1) shobdo bondhu-ke moja kore bolle L0 — aro beshi udahoron
# lagbe, naile model context shikhe na (prothom 5-ta row-e "oda koi jor"
# L1-i dichchilo). Tai prottek Level-1 Meta AI shobdo diye 3-ta friendly
# template: "kire X", "oi X", "X koi jor". Level 2/3 shobdo ekhane NAI —
# "kire harami" kokhonoi L0 na, model jate seta na shikhe.
FRIENDLY_TEMPLATES = ["kire {r}", "oi {r}", "{r} koi jor"]
for roman, (lvl, dialect) in META_TERMS.items():
    if lvl != 1:
        continue
    for tpl in FRIENDLY_TEMPLATES:
        add_msg(tpl.format(r=roman), 0, dialect, "meta_ai_friendly")
        syn_count += 1
# Counter-example (na dile "kire" shobdo-ta nijei L0 signal hoye jay):
# "kire harami" -> L2, "kire madarchod" -> L3 — friendly prefix strong
# gali-ke normal banay na. Tai Level 2/3 shobdo diyeo "kire X" row banai,
# kintu tar nijer level-e.
for roman, (lvl, dialect) in META_TERMS.items():
    if lvl < 2:
        continue
    add_msg(f"kire {roman}", lvl, dialect, "meta_ai_friendly")
    syn_count += 1

# Level correction — ager source-er vul label thik kora. "ধোন/ধন"-ke Meta AI
# L3 diyechilo, kintu Nirmol lexicon (sev 2) + TB-OLID-er beshirbhag
# (28 L2 vs 8 L3) bole eta L2 (harassment/insult, direct threat na).
# Dedup-e pore add kora L2 row hariye jeto, tai sheshe exact-text override.
LEVEL_OVERRIDES = {
    "tui ekta dhon": 2, "ei dhon edike ai": 2, "dhon koi jas": 2,
    "kire dhon": 2, "kire dhoner bachcha": 2,
    "ei dhoner bachcha edike ai": 2, "dhoner bachcha koi jas": 2,
    "tui ekta dhoner bachcha": 2, "d h o n": 2, "dh0n": 2,
}
n_fixed = 0
for row in messages:
    k = norm_key(row["text"])
    if k in LEVEL_OVERRIDES and row["level"] != LEVEL_OVERRIDES[k]:
        row["level"] = LEVEL_OVERRIDES[k]
        n_fixed += 1
print(f"level overrides applied: {n_fixed}")

# Learned words — message row (bare text + template, level onujayi).
# L0: shudhu bare text (neutral-ke amplify kora jabe na).
for e in LEARNED:
    t = str(e.get("text", "")).strip()
    if not t:
        continue
    lvl = int(e.get("level", 2))
    dia = e.get("dialect", "banglish")
    add_msg(t, lvl, dia, "learned_words")
    syn_count += 1
    if lvl >= 2:
        if _learned_script(t) == "bangla":
            for tpl in [f"তুই একটা {t}", f"ওই {t}", f"{t} কই যাস"]:
                add_msg(tpl, lvl, dia, "learned_words")
                syn_count += 1
        else:
            for tpl in [f"tui ekta {t}", f"oi {t}", f"{t} koi jas"]:
                add_msg(tpl, lvl, dia, "learned_words")
                syn_count += 1

random.shuffle(messages)
msg_path = os.path.join(OUT, "banglish_threat_dataset.jsonl")
with open(msg_path, "w", encoding="utf-8") as f:
    for row in messages:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"messages rows: {len(messages)} (synthetic tries: {syn_count}) -> {msg_path}")

# ------------------------------------------------------------------ validate
# JSONL-er point tai: prottek line alada kore pora + check kora jay.
from collections import Counter


def validate_jsonl(path, required):
    bad = 0
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  BAD LINE {i}: {e}")
                bad += 1
                continue
            missing = [k for k in required if k not in row]
            if missing:
                print(f"  LINE {i} missing {missing}")
                bad += 1
                continue
            rows.append(row)
    return rows, bad


print("\n--- validate lexicon ---")
lex_rows, bad = validate_jsonl(
    lex_path, ["term", "script", "language", "dialect", "category", "severity", "source"])
print("rows:", len(lex_rows), "| bad lines:", bad)
print("by severity:", dict(Counter(r["severity"] for r in lex_rows)))
print("by dialect:", dict(Counter(r["dialect"] for r in lex_rows)))
print("by source:", dict(Counter(r["source"] for r in lex_rows)))
print("by category:", dict(Counter(r["category"] for r in lex_rows)))

print("\n--- validate messages ---")
msg_rows, bad = validate_jsonl(
    msg_path, ["text", "level", "dialect", "source"])
print("rows:", len(msg_rows), "| bad lines:", bad)
print("by level:", dict(sorted(Counter(r["level"] for r in msg_rows).items())))
print("by dialect:", dict(Counter(r["dialect"] for r in msg_rows)))
print("by source:", dict(Counter(r["source"] for r in msg_rows)))

# ------------------------------------------------------------------ parquet
# JSONL shudhu intermediate — ashol training file Parquet (.parquet).
# Parquet columnar: shudhu je column dorkar setai pore (baki column skip),
# footer-e metadata (schema, row group, min/max stats) thake, ar prottek
# column-er fixed data type thake (string / int8 / bool) — tai kaj soja.
import pyarrow as pa
import pyarrow.parquet as pq

DATASET_SCHEMA = pa.schema([
    ("text", pa.string()),
    ("level", pa.int8()),
    ("dialect", pa.string()),
    ("source", pa.string()),
])
LEXICON_SCHEMA = pa.schema([
    ("term", pa.string()),
    ("script", pa.string()),
    ("language", pa.string()),
    ("dialect", pa.string()),
    ("district", pa.string()),
    ("category", pa.string()),
    ("severity", pa.int8()),
    ("source", pa.string()),
    ("auto_translit", pa.bool_()),
])


def jsonl_to_parquet(jsonl_path, parquet_path, schema):
    rows = [json.loads(line) for line in open(jsonl_path, encoding="utf-8")
            if line.strip()]
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, parquet_path, compression="snappy")
    return parquet_path


ds_pq = jsonl_to_parquet(msg_path,
                         os.path.join(OUT, "banglish_threat_dataset.parquet"),
                         DATASET_SCHEMA)
lx_pq = jsonl_to_parquet(lex_path,
                         os.path.join(OUT, "banglish_slang_lexicon.parquet"),
                         LEXICON_SCHEMA)
print(f"\nparquet: {ds_pq} + {lx_pq}")
for p in (ds_pq, lx_pq):
    md = pq.read_metadata(p)
    print(f"  {os.path.basename(p)}: rows={md.num_rows}, "
          f"row_groups={md.num_row_groups}, size={os.path.getsize(p)} bytes")
    print(f"    schema: {pq.read_schema(p)}")
