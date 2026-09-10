"""
Candidate vocabulary for the synthetic reasoning corpora — *proposals*, not the
final lexicon.

Nothing in this file is trusted. Every entry here is a candidate that must survive
`validate_lexicon.py`, which measures how often it actually occurs in that language's
own pretraining corpus and how many tokens the language's own tokenizer spends on it.
The validated survivors are written to `lexicon_<language>.json`, and only that file is
read by the generator.

Why this indirection exists
---------------------------
The brief requires the finetuning data to be in "natural phrasing for that language, not
English prompts with foreign names". The author is not a fluent speaker of Sindhi, so
asserting naturalness on authority would be dishonest. Corpus attestation replaces that
judgement with a measurement: a construction that occurs tens of thousands of times in
618M tokens of real Sindhi is natural *by evidence*, whatever the author believes.

The same standard is applied to Hindi rather than exempting it, because the author's
intuitions there are also worth checking — and they did not survive contact with the
data. `वज़न` (with nukta) drew 76 occurrences against 2,433 for `वजन`; the intuitive
spelling was the wrong one.

Grammatical gender
------------------
Both languages inflect adjectives for the gender of the noun they describe. Every entity
therefore carries an explicit gender, and every adjective is listed in both forms, so the
generator never has to guess. Frames that avoid agreement entirely (built on the
invariant "more"/"less" quantifiers) are preferred as the backbone of the dataset,
with agreeing frames used for template variety only once both forms are attested.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Entities. `m` = masculine, `f` = feminine (grammatical gender, which for the
# person names coincides with conventional given-name gender in both languages).
# ─────────────────────────────────────────────────────────────────────────────

HINDI_PEOPLE = [
    ("राम", "m"), ("श्याम", "m"), ("मोहन", "m"), ("सोहन", "m"), ("अनिल", "m"),
    ("सुनील", "m"), ("रवि", "m"), ("अमित", "m"), ("विजय", "m"), ("राजेश", "m"),
    ("दीपक", "m"), ("मनोज", "m"), ("संजय", "m"), ("अजय", "m"), ("विकास", "m"),
    ("राहुल", "m"), ("रोहित", "m"), ("सुरेश", "m"), ("महेश", "m"), ("रमेश", "m"),
    ("नितिन", "m"), ("गौरव", "m"), ("हरीश", "m"), ("नरेश", "m"), ("दिनेश", "m"),
    ("संदीप", "m"), ("प्रदीप", "m"), ("अशोक", "m"), ("अरुण", "m"), ("वरुण", "m"),
    ("करण", "m"), ("तरुण", "m"), ("मुकेश", "m"), ("योगेश", "m"), ("कमल", "m"),
    ("मनीष", "m"), ("प्रवीण", "m"), ("जितेंद्र", "m"), ("अनुज", "m"), ("हर्ष", "m"),
    ("गीता", "f"), ("सीता", "f"), ("प्रिया", "f"), ("नेहा", "f"), ("पूजा", "f"),
    ("कविता", "f"), ("किरण", "f"), ("अंजलि", "f"), ("स्वाति", "f"), ("मीना", "f"),
    ("आशा", "f"), ("रेखा", "f"), ("सुनीता", "f"), ("ममता", "f"), ("सरिता", "f"),
    ("वंदना", "f"), ("अर्चना", "f"), ("ज्योति", "f"), ("सपना", "f"), ("रचना", "f"),
    ("माया", "f"), ("उषा", "f"), ("लता", "f"), ("पुष्पा", "f"), ("शारदा", "f"),
    ("नीलम", "f"), ("शिखा", "f"), ("अनीता", "f"), ("सविता", "f"), ("बबीता", "f"),
]

SINDHI_PEOPLE = [
    ("احمد", "m"), ("علي", "m"), ("حسن", "m"), ("حسين", "m"), ("سليم", "m"),
    ("ڪريم", "m"), ("رحيم", "m"), ("نديم", "m"), ("اسلم", "m"), ("اڪرم", "m"),
    ("اقبال", "m"), ("انور", "m"), ("اختر", "m"), ("بشير", "m"), ("جميل", "m"),
    ("حامد", "m"), ("خالد", "m"), ("رشيد", "m"), ("زاهد", "m"), ("سعيد", "m"),
    ("شاهد", "m"), ("صادق", "m"), ("طارق", "m"), ("عابد", "m"), ("فريد", "m"),
    ("قاسم", "m"), ("مجيد", "m"), ("ناصر", "m"), ("وحيد", "m"), ("يوسف", "m"),
    ("منصور", "m"), ("مقبول", "m"), ("نثار", "m"), ("امير", "m"), ("بشر", "m"),
    ("الطاف", "m"), ("اشرف", "m"), ("ايوب", "m"), ("بلال", "m"), ("توفيق", "m"),
    ("فاطمه", "f"), ("زينب", "f"), ("عائشه", "f"), ("خديجه", "f"), ("رقيه", "f"),
    ("صفيه", "f"), ("نصرت", "f"), ("شبانه", "f"), ("ثمينه", "f"), ("رابعه", "f"),
    ("سميه", "f"), ("حليمه", "f"), ("آمنه", "f"), ("مريم", "f"), ("سارا", "f"),
    ("هاجره", "f"), ("زبيده", "f"), ("رضيه", "f"), ("نازيه", "f"), ("شميم", "f"),
    ("پروين", "f"), ("نسيم", "f"), ("رخسانه", "f"), ("فرزانه", "f"), ("شگفته", "f"),
    ("نگهت", "f"), ("عذرا", "f"), ("ناهيد", "f"),
]

HINDI_PLACES = [
    ("दिल्ली", "f"), ("मुंबई", "f"), ("कोलकाता", "m"), ("चेन्नई", "m"), ("जयपुर", "m"),
    ("लखनऊ", "m"), ("पटना", "m"), ("भोपाल", "m"), ("इंदौर", "m"), ("कानपुर", "m"),
    ("नागपुर", "m"), ("सूरत", "m"), ("रांची", "f"), ("शिमला", "m"), ("अजमेर", "m"),
    ("ग्वालियर", "m"), ("वाराणसी", "f"), ("अमृतसर", "m"), ("जोधपुर", "m"), ("उदयपुर", "m"),
]

SINDHI_PLACES = [
    ("ڪراچي", "f"), ("حيدرآباد", "m"), ("سکر", "m"), ("لاڙڪاڻو", "m"), ("ميرپورخاص", "m"),
    ("نوابشاهه", "m"), ("ٺٽو", "m"), ("دادو", "m"), ("جيڪب آباد", "m"), ("شڪارپور", "m"),
    ("خيرپور", "m"), ("بدين", "m"), ("سانگهڙ", "m"), ("عمرڪوٽ", "m"), ("ٽنڊوآدم", "m"),
    ("مٺي", "f"), ("نوشهرو فيروز", "m"), ("قمبر", "m"), ("مورو", "m"), ("سيوهڻ", "m"),
]

HINDI_OBJECTS = [
    ("किताब", "f"), ("कुर्सी", "f"), ("मेज", "f"), ("बैग", "m"), ("कमरा", "m"),
    ("मकान", "m"), ("बगीचा", "m"), ("पेड़", "m"), ("डिब्बा", "m"), ("बोतल", "f"),
    ("गाड़ी", "f"), ("साइकिल", "f"), ("रस्सी", "f"), ("पत्थर", "m"), ("थैला", "m"),
]

SINDHI_OBJECTS = [
    ("ڪتاب", "m"), ("ڪرسي", "f"), ("ميز", "f"), ("ٿيلهو", "m"), ("ڪمرو", "m"),
    ("گهر", "m"), ("باغ", "m"), ("وڻ", "m"), ("دٻو", "m"), ("بوتل", "f"),
    ("گاڏي", "f"), ("سائيڪل", "f"), ("رسي", "f"), ("پٿر", "m"), ("دري", "f"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Attributes. Each carries the noun, its unit, and the comparative adjective in
# masculine and feminine form. `applies_to` restricts an attribute to the entity
# kinds it makes sense for — a book has no age, a person has no price.
# ─────────────────────────────────────────────────────────────────────────────

HINDI_ATTRIBUTES = {
    "age":    {"noun": "उम्र",  "poss": "की", "noun_gender": "f", "unit": "साल",
               "adj_more_m": "बड़ा", "adj_more_f": "बड़ी",
               "adj_less_m": "छोटा", "adj_less_f": "छोटी", "applies_to": ["person"]},
    "height": {"noun": "लंबाई", "poss": "की", "noun_gender": "f", "unit": "सेमी",   "applies_to": ["person", "object"]},
    "weight": {"noun": "वजन",   "poss": "का", "noun_gender": "m", "unit": "किलो",   "applies_to": ["person", "object"]},
    "price":  {"noun": "कीमत",  "poss": "की", "noun_gender": "f", "unit": "रुपये",  "applies_to": ["object"]},
    # Added after the first generation attempt found the validated place names unusable:
    # dropping `distance` had left no attribute that applies to a place, so a fifth of
    # the entity pool could never be selected.
    "population": {"noun": "आबादी", "poss": "की", "noun_gender": "f", "unit": "लाख",
                   "applies_to": ["place"]},
}

SINDHI_ATTRIBUTES = {
    "age":    {"noun": "عمر",   "poss": "جي", "noun_gender": "f", "unit": "سال",
               "adj_more_m": "وڏو",  "adj_more_f": "وڏي",
               "adj_less_m": "ننڍو", "adj_less_f": "ننڍي", "applies_to": ["person"]},
    "height": {"noun": "ڊيگهه", "poss": "جي", "noun_gender": "f", "unit": "سينٽي ميٽر", "applies_to": ["person", "object"]},
    "weight": {"noun": "وزن",   "poss": "جو", "noun_gender": "m", "unit": "ڪلو",        "applies_to": ["person", "object"]},
    "price":  {"noun": "قيمت",  "poss": "جي", "noun_gender": "f", "unit": "رپيا",       "applies_to": ["object"]},
    # Sindhi writes this unit as لک (25,849) not لاک (782) — the same kind of spelling
    # check that caught वज़न/वजन and مفاصلو/فاصلو.
    "population": {"noun": "آبادي", "poss": "جي", "noun_gender": "f", "unit": "لک",
                   "applies_to": ["place"]},
}

# ─────────────────────────────────────────────────────────────────────────────
# Construction fragments.
#
# These are the compositional atoms the generator glues together, deliberately kept
# MINIMAL. An earlier version validated whole clauses ("से बड़ा है", "کان وڏو آهي")
# and both came back rare — 23 and 29 occurrences — while their cores ("से बड़ा",
# "کان وڏو") drew 1,044 and 738. The clause was over-specified, not unnatural: real
# sentences rarely put the copula immediately after the comparative. Validating atoms
# and composing them is both a truer test and a more flexible generator.
#
# The backbone is the QUANTIFIER comparative ("X's age is MORE than Y's age"), not the
# adjective comparative ("X is BIGGER than Y). The corpus is emphatic about this:
#
#     से ज्यादा  3,676   |   کان وڌيڪ  6,930      quantifier
#     से बड़ा    1,044   |   کان وڏو     738      adjective, generic
#     से लंबा       28   |   کان ڊگهو     17      adjective, attribute-specific
#     से महंगा      47   |   کان ڳرو       3      adjective, attribute-specific
#
# Attribute-specific adjective comparatives are effectively absent from both corpora.
# A dataset built on them would read as translated English, which is precisely what the
# brief forbids. The quantifier construction carries a second advantage: وڌيڪ, گهٽ,
# ज्यादा and कम are all gender-INVARIANT, so the backbone templates cannot produce an
# agreement error at all.
#
# The adjective comparative survives for `age` only, where it is genuinely idiomatic
# (सबसे बड़ा 899, سڀ کان وڏو 521). That gives the paraphrase pool used to build the
# held-out-template test split, so the split is grounded in a real register difference
# rather than in artificial rewording.
# ─────────────────────────────────────────────────────────────────────────────

HINDI_FRAGMENTS = {
    "than":        "से",        "copula":     "है",
    "quant_more":  "ज्यादा",    "quant_less": "कम",
    "sup_prefix":  "सबसे",
    "who":         "कौन",       "who_poss_f": "किसकी",  "who_poss_m": "किसका",
    "equal":       "बराबर",     "and":        "और",       "or": "या",
    "q_marker":    "प्रश्न",    "a_marker":   "उत्तर",
    "cmp_more_generic": "से ज्यादा",  "cmp_less_generic": "से कम",
    "sup_more_generic": "सबसे ज्यादा", "sup_less_generic": "सबसे कम",
    "adj_sup_more": "सबसे बड़ा", "adj_sup_less": "सबसे छोटा",
}

SINDHI_FRAGMENTS = {
    "than":        "کان",       "copula":     "آهي",
    "quant_more":  "وڌيڪ",      "quant_less": "گهٽ",
    "sup_prefix":  "سڀ کان",
    "who":         "ڪير",       "who_poss_f": "ڪنهن جي", "who_poss_m": "ڪنهن جو",
    "equal":       "برابر",     "and":        "۽",        "or": "يا",
    "q_marker":    "سوال",      "a_marker":   "جواب",
    "cmp_more_generic": "کان وڌيڪ",  "cmp_less_generic": "کان گهٽ",
    "sup_more_generic": "سڀ کان وڌيڪ", "sup_less_generic": "سڀ کان گهٽ",
    "adj_sup_more": "سڀ کان وڏو", "adj_sup_less": "سڀ کان ننڍو",
}

HINDI_PUNCT = {"question": "?", "comma": ",", "full_stop": "।"}
SINDHI_PUNCT = {"question": "؟", "comma": "،", "full_stop": "."}

CANDIDATES = {
    "hindi": {
        "people": HINDI_PEOPLE, "places": HINDI_PLACES, "objects": HINDI_OBJECTS,
        "attributes": HINDI_ATTRIBUTES, "fragments": HINDI_FRAGMENTS, "punct": HINDI_PUNCT,
    },
    "sindhi": {
        "people": SINDHI_PEOPLE, "places": SINDHI_PLACES, "objects": SINDHI_OBJECTS,
        "attributes": SINDHI_ATTRIBUTES, "fragments": SINDHI_FRAGMENTS, "punct": SINDHI_PUNCT,
    },
}
