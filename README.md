# Banglish Threat Checker — Python backend

ফেসবুক কমেন্টের গালি ও হুমকি চেকার — বাংলিশ + চাটগাঁইয়া। ML Lab প্রজেক্ট, BGC Trust University (7th semester CSE)।

**মডেল:** TF-IDF (word 1–2 + char 3–5) + Logistic Regression, deep learning ছাড়া। ডেটাসেট ৬,৬০৫ রো, লেক্সিকন ৩,১৯২ শব্দ, ১৭ জেলা। Baseline (word-only, no norm) ০.৭১৮ → Improved ০.৭৫৮, Level-3 recall ০.৯৪৭।

**কী করে:** একটা কমেন্ট লিখলে Level 0–3 (Safe / হালকা / মাঝারি / মারাত্মক) বলে দেয়, সাথে confidence, ধরা পড়া শব্দ, normalized লেখা, আর সাধারণ keyword ফিল্টারের সাথে তুলনা — যেটা বাংলিশ গালি ধরতে পারে না। Batch ট্যাবে অনেকগুলো কমেন্ট একসাথে চেক করে CSV ডাউনলোড করা যায়।

**Run:** `pip install -r requirements.txt && python app.py` (port 7860) বা `gunicorn --bind 0.0.0.0:$PORT app:app`।

**Privacy:** লেখা শুধু চেক করার জন্য সার্ভারে পাঠানো হয় — সংরক্ষণ করা হয় না, লগ-এ লেখা থাকে না।

**Deploy:** Render free web service (`render.yaml` — Blueprint deploy) বা Hugging Face Spaces (Docker SDK — PRO লাগে, free-তে হয় না)।
