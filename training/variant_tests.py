#!/usr/bin/env python3
"""Fixed test lists shared by compare_models.py and the notebook.
VARIANT_TESTS: (text, expected level). UNSEEN = training data-te ei
exact target/form NAI — "nearby" generalization-er proman.
"""


VARIANT_TESTS = [
    # purono GenZ/political (age theke)
    ("ক্রি1ম1নাল", 1), ("তুই একটা রাজাকার", 2), ("রস করা", 2),
    ("এস থ্রি এক্স", 2), ("তুই একটা দন", 2), ("বাল", 2),
    ("তুই একটা পুটকি মারো", 3), ("s3x", 2), ("তুই একটা মাদারবোর্ড", 2),
    ("tui ekta motherboard", 2),
    # user-er report (2026-11-18)
    ("তুরে চুদবো ছাএলীগে ওয়েট কর আর কয়দিন তুর টোকাইর মারে চু,,, দি "
     "এনসিপির মারে চু,,, দি এটা টোকাইয়ার দেশ না এটা শেখ হাসিনার বাংলাদেশ", 3),
    ("তুরে চুদবো", 3), ("তোর মারে চুদি", 3), ("তোর বোনরে চুদি", 3),
    ("তোর নানিরে চুদবো", 3), ("তোকে মেরে ফেলবো", 3),
    ("তোকে খুন করবো", 3), ("তোকে মারবো", 3),
    ("চু,,, দি", 3), ("মারে চু,,, দি", 3),
    ("ছাত্রলীগের মারে চুদি", 3), ("এনসিপির মারে চুদি", 3),
    ("ছাএলীগের মারে চুদি", 3), ("তুর টোকাইর মারে চুদি", 3),
    ("tor mare chudi", 3), ("tore chudbo", 3),
    # UNSEEN generalization — training-e ei target/form nai
    ("তোর ফুফুরে চুদি", 3), ("তোর মামিরে চুদবো", 3),
    ("তোর বাপরে চুদি", 3), ("তোরে চুদমু", 3),
    ("tor fufure chudi", 3), ("m0therboard", 2),
    # neutral — family/party word thakleo threat NA
    ("আমি ছাত্রলীগ করি", 0), ("আমার মারে ভালোবাসি", 0),
    ("এটা শেখ হাসিনার বাংলাদেশ", 0), ("আমার মাকে ভালোবাসি", 0),
    ("ওয়েট কর, আসতেছি", 0), ("তোর মারে সালাম দিস", 0),
    ("এনসিপি ভালো দল", 0),
    # user-er test 2 (2026-11-18) — "ক্ষণিকের পোলা" Chatgaiya gali
    ("ক্ষণিকের পোলা", 2), ("তুই একটা ক্ষণিকের পোলা", 2),
    ("তুই একটা ক্ষণিকের পোয়া", 2), ("khoniker pola", 2),
    ("tui ekta khoniker pola", 2), ("খনিকের পোলা", 2),
    # neutral — "পোলা" nije gali NA
    ("আমার পোলা স্কুলে যায়", 0), ("পোলাটা ভালো ছেলে", 0),
    ("oi pola", 0),
]


DEMO_TESTS = [("kal class hobe?", 0), ("tui ekta pagol", 1),
              ("tui ekta beyadob", 2), ("toke marbo", 3),
              ("toke m*rbo", 3), ("tore dekhbo rastay", 3),
              ("tui aai ekbar, tore dekhilom", 3),
              ("basa chini ami, aitesi", 3)]


