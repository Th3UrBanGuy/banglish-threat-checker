#!/usr/bin/env python3
"""Export the trained LogReg pipeline to JSON so the checker web app can
run the EXACT same model (same TF-IDF vocab/idf, same coefficients) in
plain JavaScript — no Python server needed.

Output: model.json + lexicon.json in the same folder.
Validation: predictor.js (Node) must reproduce Python predictions 100%
on a held-out sentence list — see validate_js.py."""
import json
import os

import joblib
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))

m = joblib.load(os.path.join(HERE, "banglish_threat_model.pkl"))
pipe = m["model"]
fu = pipe.named_steps["tfidf"]
wv, cv = fu.transformer_list[0][1], fu.transformer_list[1][1]
clf = pipe.named_steps["clf"]

def features_in_index_order(vec):
    inv = {i: f for f, i in vec.vocabulary_.items()}
    return [inv[i] for i in range(len(inv))]

model = {
    "meta": {
        "best_name": m["best_name"],
        "classes": [int(c) for c in clf.classes_],
        "levels": {str(k): v for k, v in m["levels"].items()},
        "word_ngram_range": list(wv.ngram_range),
        "word_max_features": wv.max_features,
        "char_ngram_range": list(cv.ngram_range),
        "char_max_features": cv.max_features,
        "sublinear_tf": True,
        "note": "TF-IDF: tf=1+ln(count), idf=ln((1+n)/(1+df))+1, l2 norm per "
                "sub-vector, word+char concatenated. LogReg multinomial, "
                "predict=argmax(X.coef^T+intercept), proba=softmax.",
    },
    "word_features": features_in_index_order(wv),
    "word_idf": [round(float(v), 6) for v in wv.idf_],
    "char_features": features_in_index_order(cv),
    "char_idf": [round(float(v), 6) for v in cv.idf_],
    "coef": [[round(float(v), 6) for v in row] for row in clf.coef_],
    "intercept": [round(float(v), 6) for v in clf.intercept_],
}
with open(os.path.join(HERE, "model.json"), "w", encoding="utf-8") as f:
    json.dump(model, f, ensure_ascii=False, separators=(",", ":"))
print("model.json:",
      os.path.getsize(os.path.join(HERE, "model.json")), "bytes |",
      "word feats:", len(model["word_features"]),
      "| char feats:", len(model["char_features"]),
      "| coef:", len(model["coef"]), "x", len(model["coef"][0]))

# Lexicon for highlighting matched slang words in the checker UI.
lex = pq.read_table(
    os.path.join(HERE, "banglish_slang_lexicon.parquet"),
    columns=["term", "script", "category", "severity", "dialect"]).to_pylist()
lex_out = [{"t": r["term"], "s": r["script"], "c": r["category"],
            "sev": int(r["severity"]), "d": r["dialect"]} for r in lex]
with open(os.path.join(HERE, "lexicon.json"), "w", encoding="utf-8") as f:
    json.dump(lex_out, f, ensure_ascii=False, separators=(",", ":"))
print("lexicon.json:",
      os.path.getsize(os.path.join(HERE, "lexicon.json")), "bytes |",
      "terms:", len(lex_out))
