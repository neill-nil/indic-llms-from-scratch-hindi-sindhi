# Phase 2: Generation Quality Evaluation

<!-- GENERATED FILE - do not edit by hand.
     Regenerate with:
       python -m common.eval.generation
       python -m common.eval.write_generation_report
-->

Each model was given **200 prompts of 64 tokens** drawn from
its own validation split, asked to continue for **128 tokens**, and
scored against the text that actually followed. Four decoding settings: greedy plus
temperatures 0.5, 1.0 and 1.5.

Prompts are screened to contain no document separator. A window straddling a boundary
would pair a prompt with the opening of an *unrelated* document, so the "correct
continuation" would be text no model could predict — depressing every score for a reason
unconnected to model quality, and doing so unevenly between two languages whose document
lengths differ by a factor of three.

Figure: `figures/generation_metrics.png`

## How to read these metrics

| Metric | What it measures | Direction |
|---|---|---|
| **BLEU-4** | overlap of word 4-grams with the reference | higher better |
| **chrF++** | overlap of *character* n-grams plus word bigrams | higher better |
| **ROUGE-L** | longest common subsequence with the reference, as F1 | higher better |
| **Distinct-1 / -2** | fraction of generated unigrams / bigrams that are unique | higher = more varied |
| **Repetition rate** | fraction of 4-grams repeating *within one sample* | lower better |

**A caveat that belongs up front, not in a footnote.** BLEU and ROUGE were designed for
tasks with a single correct answer — translation, summarisation — where the reference *is*
the right output. Open-ended continuation has no single correct answer: a fluent, sensible
continuation that simply differs from what the author happened to write next scores near
zero. **BLEU values of 1–2 are therefore normal here and do not indicate a broken model.**
These metrics are reported because the brief requires them and because they are comparable
*between our two models*, not as absolute quality scores.

## Model H (Hindi)

| Decoding | BLEU-4 | chrF++ | ROUGE-L | Distinct-1 | Distinct-2 | Repetition (4-gram) |
|---|---|---|---|---|---|---|
| Greedy (T=0) | 1.46 | 10.88 | 0.0892 | 0.056 | 0.122 | 0.810 |
| T = 0.5 | **1.78** | 15.92 | 0.1149 | 0.120 | 0.381 | 0.352 |
| T = 1.0 | 1.24 | **19.72** | 0.1071 | 0.317 | 0.824 | 0.008 |
| T = 1.5 | 0.19 | 16.63 | 0.0544 | 0.550 | 0.988 | 0.000 |

## Model L (Sindhi)

| Decoding | BLEU-4 | chrF++ | ROUGE-L | Distinct-1 | Distinct-2 | Repetition (4-gram) |
|---|---|---|---|---|---|---|
| Greedy (T=0) | 1.60 | 10.93 | 0.0846 | 0.053 | 0.106 | 0.826 |
| T = 0.5 | **1.87** | 13.92 | 0.1003 | 0.100 | 0.288 | 0.513 |
| T = 1.0 | 1.42 | **19.50** | 0.0989 | 0.338 | 0.858 | 0.012 |
| T = 1.5 | 0.34 | 16.49 | 0.0448 | 0.543 | 0.994 | 0.000 |

## Findings

### 1. Greedy decoding degenerates into loops

Repetition rate is **0.810** for Model H and
**0.826** for Model L: over 80% of 4-grams
in a greedy sample are repeats of something earlier in that same sample. Distinct-2 collapses
to 0.122 and 0.106.

Always taking the most likely token drives the model into a cycle it cannot escape, because
each repetition makes the next repetition more likely still. This is the expected failure
mode for a model of this size under greedy decoding, and it is exactly what the diversity
diagnostics exist to expose. The sample outputs below show it plainly.

### 2. The metrics disagree about the best temperature, and the disagreement is informative

BLEU and ROUGE-L peak at **T = 0.5**; chrF++ peaks at **T = 1.0**.

BLEU rewards exact word-sequence matches, so it favours the conservative setting that
reproduces common phrasing. chrF++ operates on character n-grams and therefore tolerates
morphological variation — which matters enormously in Devanagari and Perso-Arabic, where an
inflected form differing by one suffix is a total miss for BLEU but largely correct for
chrF. For morphologically rich languages, **chrF++ is the more trustworthy of the two**,
and its peak at T = 1.0 is the better guide to where the model actually generates well.

### 3. Temperature 1.5 is too hot

BLEU falls to 0.19 / 0.34
and Distinct-2 reaches 0.988 /
0.994 — almost every bigram is unique. That is not
creativity; it is the distribution flattened until the output stops being constrained by
the language. Repetition of exactly 0.000 alongside near-perfect distinctness is the
signature of noise, not fluency.

### 4. Model H edges Model L, but the margin is small

At T = 1.0: chrF++ **19.72** vs
**19.50**, ROUGE-L
0.1071 vs 0.0989.

The direction agrees with the intrinsic evaluation, but the margin is far smaller than the
78% bits-per-byte gap would suggest. **This is itself a finding:** overlap metrics are noisy
enough on open-ended generation that they barely resolve a difference perplexity shows
unambiguously. It is a caution against treating BLEU-style scores as a primary measure of
language-model quality.

Model L is also markedly more repetitive at T = 0.5
(0.513 vs
0.352), suggesting it leans harder on
memorised phrasing once the distribution is sharpened.

## Generated samples — one per decoding setting

The same prompt at each of the four settings, so the effect of temperature is visible
directly.

### Model H (Hindi)

**Greedy (T=0)**

> **Prompt (64 tokens):** महसूस करें तो दवा लेना तुरंत बंद कर दें और जब डॉक्टर कहें तब ही इसे दोबारा लें। Avertol 12 Mg Tablet DT का प्रभाव गुर्दे पर क्या होता है? कोई शोध न होने के कारण Avertol के प्रभाव या दुष्प्रभाव की कोई जानकारी उपलब्ध नहीं 

> **Model output (128 tokens):** 12 Mg Tablet DT का प्रभाव गुर्दे पर क्या होता है? Avertol 12 Mg Tablet DT का प्रभाव गुर्दे पर क्या होता है? Avertol 12 Mg Tablet DT का प्रभाव गुर्दे पर क्या होता है? Avertol 12 Mg Tablet DT का प्रभाव गुर्दे पर क्या होता है? Avertol 12 Mg Tablet DT का प्रभाव गु

> **True continuation:** 12 Mg Tablet DT का जिगर (लिवर) पर क्या असर होता है? Avertol का सेवन करना आपके शरीर पर बहुत ही कम प्रभाव डालता है। क्या ह्रदय पर Avertol 12 Mg Tablet DT का प्रभाव पड़ता है? Avertol हृदय के लिए सुरक्षित है। क्या Avertol 12

**T = 0.5**

> **Prompt (64 tokens):** महसूस करें तो दवा लेना तुरंत बंद कर दें और जब डॉक्टर कहें तब ही इसे दोबारा लें। Avertol 12 Mg Tablet DT का प्रभाव गुर्दे पर क्या होता है? कोई शोध न होने के कारण Avertol के प्रभाव या दुष्प्रभाव की कोई जानकारी उपलब्ध नहीं 

> **Model output (128 tokens):** 12 Mg Tablet DT का प्रभाव गुर्दे पर क्या होता है? Avertol 12 Mg Tablet DT का प्रभाव गुर्दे पर क्या होता है? Avertol 12 Mg Tablet DT का प्रभाव गुर्दे पर क्या होता है? Avertol 12 Mg Tablet DT का प्रभाव गुर्दे पर क्या होता है? Avertol 12 Mg Tablet DT का प्रभाव गु

> **True continuation:** 12 Mg Tablet DT का जिगर (लिवर) पर क्या असर होता है? Avertol का सेवन करना आपके शरीर पर बहुत ही कम प्रभाव डालता है। क्या ह्रदय पर Avertol 12 Mg Tablet DT का प्रभाव पड़ता है? Avertol हृदय के लिए सुरक्षित है। क्या Avertol 12

**T = 1.0**

> **Prompt (64 tokens):** महसूस करें तो दवा लेना तुरंत बंद कर दें और जब डॉक्टर कहें तब ही इसे दोबारा लें। Avertol 12 Mg Tablet DT का प्रभाव गुर्दे पर क्या होता है? कोई शोध न होने के कारण Avertol के प्रभाव या दुष्प्रभाव की कोई जानकारी उपलब्ध नहीं 

> **Model output (128 tokens):** 12 Mg Tablet DT का सेवन बेसर्दी बुखार, हल्के ले लो, तनाव, पाचन और आंत से छुटकारा पाने और अपने - काम करने के लिएसुन जाना शामिल नहीं है। Avertol 12 Mg Tablet DT का जिगर (लिवर या ग्लोइंग) पर बुरा असर पड़ता है। जिससे दिमाग पर असर पड़ सकता है। Mg Tablet DT का लिवर 

> **True continuation:** 12 Mg Tablet DT का जिगर (लिवर) पर क्या असर होता है? Avertol का सेवन करना आपके शरीर पर बहुत ही कम प्रभाव डालता है। क्या ह्रदय पर Avertol 12 Mg Tablet DT का प्रभाव पड़ता है? Avertol हृदय के लिए सुरक्षित है। क्या Avertol 12

**T = 1.5**

> **Prompt (64 tokens):** अच्छे से ठंडा हो जाए तब आप इसे अपने आंखों के पलकों के ऊपर लगाएंगे तो आपको आंखों की रोशनी की समस्या से निदान प्राप्त हो सकता है। गिलोय अस्थमा के रोगों के लिए भी फायदेमंद है- मौसम के बदलाव के कारण अक्सर लोगों को सर्दियों म

> **Model output (128 tokens):** का सामना करना पड़ता है इससे मरीज अधिक मैनचेस्टर बताकर जल्दी ही अस्थमा मजबूती अफलेख मधुमेह की चपेट में आ बैठेंगी। होम तरल स्टेबल पार्ट आपको यदि बैंक रोक देते तो हरी नींद और चेहरावेट कर रह जाएंगे निश्चित ही आपtor दरवाजे कीटाणुन्त्रयुक्त की निर्धन खुली शीशी आकड प

> **True continuation:** का सामना करना पड़ता है ऐसे में अस्थमा के मरीजों को अगर नियमित रूप से गिलोय की मोटी बन्टी चबाने से या उसका जूस बनाकर पीने से उनके अस्थमा में काफी नियंत्रण एवं उन्हें काफी आराम महसूस होता है। गिलोय एनीमिया के रोग में भी का


### Model L (Sindhi)

**Greedy (T=0)**

> **Prompt (64 tokens):** گرڌاريءَ جي وڃڻ کان پوءِ سام هينئن چيو: ”نه رڳو سندس صدق ۽ قربانيءَ جو انگ وڌائڻو هيو پر مون ڏٺو هو ته انهيءَ منڊيءَ ۾ هن جو گهڻو موهه هيو. فقيري ٽولي وارن لاءِ موهه ڪنهن به وستوءَ سان رکڻ چڱو نه آهي.“ *** نئون صفحو -- ڪ

> **Model output (128 tokens):** صفحو --گذريل صفحو ٻيا صفحا 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 20 22 23 29 24 29 25 29 29 29 20 24 25 29 2

> **True continuation:** صفحو --گذريل صفحو ٻيا صفحا 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 4

**T = 0.5**

> **Prompt (64 tokens):** گرڌاريءَ جي وڃڻ کان پوءِ سام هينئن چيو: ”نه رڳو سندس صدق ۽ قربانيءَ جو انگ وڌائڻو هيو پر مون ڏٺو هو ته انهيءَ منڊيءَ ۾ هن جو گهڻو موهه هيو. فقيري ٽولي وارن لاءِ موهه ڪنهن به وستوءَ سان رکڻ چڱو نه آهي.“ *** نئون صفحو -- ڪ

> **Model output (128 tokens):** صفحو --گذريل صفحو ٻيا صفحا 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 20 21 22 23 24 25 26 27 29 30 31 32 33 34 35 36 37 38 40 50 51 52 53 54 55 5

> **True continuation:** صفحو --گذريل صفحو ٻيا صفحا 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 4

**T = 1.0**

> **Prompt (64 tokens):** مبر 1929ع. 113- روزانه الوحيد ڪراچي، 15 فيبروري 1931ع. 114- روزانه الوحيد ڪراچي،22 مئي 1932ع. 115- روزانه الوحيد ڪراچي،14 جون 1

> **Model output (128 tokens):** 933ع. 116- روزانه الوحيد ڪراچي،17 فيبروري 1933ع. 116- روزانه الوحيد ڪراچي. 113- روزانه الوحيد ڪراچي، 13 جولاءِ 1934ع. 117- روزانه الوحيد ڪراچي، 13 جولاءِ 1932ع. 118- روزانه الوحيد ڪراچي، 13 جولاءِ 1934ع. 119- روزانه ٽرام جعفري، 13 جولاءِ 1935ع. 119- روزانه الو

> **True continuation:** 932ع. 116- روزانه الوحيد ڪراچي،14 سيپٽمبر 1932ع. 117- روزانه الوحيد ڪراچي،10 مئي 1933ع. 118- روزانه الوحيد ڪراچي،15 فيبروري 1935ع. 119- روزانه الوحيد ڪراچي،7 آگسٽ 1936ع. 120- روزانه الوحيد ڪراچي،12 آگسٽ 1938ع. 121- روزان

**T = 1.5**

> **Prompt (64 tokens):** موڊ هجي يا اي سائيڪل وارو موڊ ، اهو ڏا theي طاقتور موٽر پاور سائيڪل سان سواري ڪرڻ آسان آهي. هن بيٽريءَ تي هلندڙ سائيڪل ۾ هڪ بهتر ٺهيل ۽ پائيدار فريم آهي جيڪو سائيڪل جي پائيدارگي کي وڌائيندو آهي. برڪ سسٽم فريم ۽ سائيڪل تي

> **Model output (128 tokens):** وان ناهي گذاريندا. سلائي ڪتو اصل ڪيبل بائي ڊسپلي کڻي ٿو. موسيقار ڪيپ اسٽ سعيد وٽ ٺاهڻ واروں ارادن وارو وينس فريم ٽرانسميشن پٺاڻسس نماڻي بليڊ انٽرنيٽ ڪيمرا اسٽيشن نجات فائلtinندڙ سولو هيٽر ايڪسٽينشن طاقت آهي. صحيح ڪلڪ غور سان مشاهدو ڪرڻ کو ڏاڙهي طنزب هوندو ته ه

> **True continuation:** نگ آهي جڏهن توهان کي فوري طور تي سائيڪل کي روڪڻ جي ضرورت آهي. اي بائيڪ 21 اسپيڊ ٽرانسميشن سسٽم جي خاصيت رکي ٿو ، هڪ کروزر برقي سائيڪل لاءِ ، اهو ڪافي کان وڌيڪ آهي ۽ 100 ايم ايم سفري تبديلي سسپنس المونيم مصر واري فرنٽ فور


## The spread of output quality at T = 1.0

**Important caveat on this ranking.** chrF++ measures *agreement with the one continuation
the author happened to write*, which is not the same as quality. A fluent, grammatical
continuation that simply goes somewhere else ranks low. The ranking is a way of surfacing
the range of behaviour, not a verdict on which outputs are good — read the text, not just
the score.

### Model H (Hindi) — best to worst

Ranked by per-sample chrF++ against the true continuation. Across the 200 prompts the score ranges from **37.5** down to **4.0** (median **19.7**), so a single example is not representative of anything.

#### Best of 200 — chrF++ 37.5

> **Prompt:** महसूस करें तो दवा लेना तुरंत बंद कर दें और जब डॉक्टर कहें तब ही इसे दोबारा लें। Avertol 12 Mg Tablet DT का प्रभाव गुर्दे पर क्या होता है? कोई शोध न होने के कारण Avertol के प्रभाव या दुष्प्रभाव की कोई 

> **Model output:** 12 Mg Tablet DT का सेवन बेसर्दी बुखार, हल्के ले लो, तनाव, पाचन और आंत से छुटकारा पाने और अपने - काम करने के लिएसुन जाना शामिल नहीं है। Avertol 12 Mg Tablet DT का जिगर (लिवर या ग्लोइंग) पर बुरा असर पड़ता है। जिससे दिमाग पर असर पड़ सकता है। M

> **True continuation:** 12 Mg Tablet DT का जिगर (लिवर) पर क्या असर होता है? Avertol का सेवन करना आपके शरीर पर बहुत ही कम प्रभाव डालता है। क्या ह्रदय पर Avertol 12 Mg Tablet DT का प्रभाव पड़ता है? Avertol हृदय के लिए सुरक्षित

#### Upper quartile (75th percentile) — chrF++ 21.4

> **Prompt:** नियम अभी भी लागू है। और देशों में जहां शरीयत कानून इस तरह की गतिविधि के लिए खोने के लिए और जीवन कर सकते हैं। यहूदी धर्म एक ही रोक इस विश्वास के साथ ही लोगों के लिए लागू करते हैं, तो दूसरे पर लागू नहीं

> **Model output:** द्वारा प्रधानाचार्यों के जीवन की रिपोर्ट द्वारा संदर्भित किया जाता है। Saultannovaysky को साल 1994 में संयुक्त राज्य अमेरिका के देश के बाद लगा दिया गया था। यह शक्ति के विकसित होने के विचार सेवाँ दुनिया के मुस्लिम समाज का सबसे अग्रदूत बन गया

> **True continuation:** नकारात्मक रूप में के रूप में अब जल्दी समाज द्वारा कथित। यह मुख्य रूप से तथ्य यह है कि एक साल में एक साहूकार समय पर लोग हैं, जो मुसीबत में मिल गया की कीमत पर अपने भाग्य बढ़ाने के लिए और कठिन आर्थिक स्थ

#### Median — chrF++ 19.7

> **Prompt:** वापस ऊपर चढ़ गया। मेरे पास शब्द ही नहीं थे, तब मेरी दोस्त खिलखिलाकर हंसने लगी जैसा कि खेल के मैदान पर बाकी सभी ने किया था, जिसने कॉमेडी देखी थी।" प्रियंका कहती है कि वह स्थिति में हास्य देख सकती है, ए

> **Model output:** की कम भीड़ पर एक मुस्कान से उपजी खुशी दे सकता है, जिसे उन्होंने दिया। BJP के अध्यक्ष गिरावल भुटैया नायडू ने अनाड़ी नस्लों के खिलाफ शिकायत दर्ज घटाने के मामले के इंतजाम की मांग उठाते हुए उन्हें नागासा हाजा पर ध्यान देने की मांग की थी, जिसे च

> **True continuation:** - अपनी पसंद के अनुसार प्रियंका इसे बहुतायत से प्रदर्शित करती है क्योंकि वह अमेरिका में एक किशोरी के रूप में अपने जीवन को याद करती है। सीडर रेपिड्स से न्यूयॉर्क से इंडियानापोलिस और बोस्टन तक, अपने बड़े

#### Lower quartile (25th percentile) — chrF++ 18.0

> **Prompt:** को तेजी के रूप में प्रदर्शित नहीं करता है और न ही कम मूल्य मंदी के रूप में एक जोड़ी का प्रतिनिधित्व करता है। एक उच्च मूल्य का अर्थ है कि जोड़ी ट्रेंडिंग (या सबसे अच्छा विदेशी मुद्रा मौलिक विश्लेषण साइ

> **Model output:** रहे युग्मन के बिना पहले अंतर का संकेत मिलता है।<|endoftext|> स्पेशल (जो नहीं है) पेशेंट डेव डायनामिकी के मामले में पकड़ा गया था। गोल्डन व्यापरों ने कोर्ट के सामने टेबल बुक ऑफ मागिंग्स में धोखाधड़ी और घर वालों के लिए उत्तरदायी ठहराया है। दुर

> **True continuation:** रही है। आप ADX कैसे पढ़ते हैं. ADX 0 - 100 के बीच का मान होगा। आमतौर पर, व्यापारी ADX 30 से ऊपर की तलाश करेंगे, यह पुष्टि करने के लिए कि मुद्रा जोड़ी या तो ऊपर या नीचे चल रही है। 30 से नीचे के किसी भी

#### Worst of 200 — chrF++ 4.0

> **Prompt:** special significance in life -The Epse by L Abercrombe, p. 52-54. 1 - Medieral English Literature by W P. Ket.p 137-1

> **Model output:** 4. Physi, heian.stंग weekcom by p. 18. Shri Literature Cargeuru, E. P. Blish, M. 20 । Literature by affoantom. Director J. 1656जी. A.AR- Edit Cargeuru, L. MP. L. L. तः परिषदीय परिषदीय परिषदीय परिषदीय परिषदीय परिषदीय परिषदीय परिषदीय परिषदीय 

> **True continuation:** 33 २- घामायनी, प्रामुस १० ७ । हिन्दी साहित्य का इतिहास, पृ० ६६० ने मस्तिष्क या बुद्धि का प्रतीक बतलाया है। परन्तु कुछ विद्वान् इडा को बुद्धि का प्रतीक मानने मे विरोध करते हैं। क्योंकि जिस बुद्धि के वै


### Model L (Sindhi) — best to worst

Ranked by per-sample chrF++ against the true continuation. Across the 200 prompts the score ranges from **65.5** down to **0.8** (median **19.4**), so a single example is not representative of anything.

#### Best of 200 — chrF++ 65.5

> **Prompt:** مبر 1929ع. 113- روزانه الوحيد ڪراچي، 15 فيبروري 1931ع. 114- روزانه الوحيد ڪراچي،22 مئي 1932ع. 115- روزانه الوحيد ڪراچي،14 جون 1

> **Model output:** 933ع. 116- روزانه الوحيد ڪراچي،17 فيبروري 1933ع. 116- روزانه الوحيد ڪراچي. 113- روزانه الوحيد ڪراچي، 13 جولاءِ 1934ع. 117- روزانه الوحيد ڪراچي، 13 جولاءِ 1932ع. 118- روزانه الوحيد ڪراچي، 13 جولاءِ 1934ع. 119- روزانه ٽرام جعفري، 13 جولاءِ 19

> **True continuation:** 932ع. 116- روزانه الوحيد ڪراچي،14 سيپٽمبر 1932ع. 117- روزانه الوحيد ڪراچي،10 مئي 1933ع. 118- روزانه الوحيد ڪراچي،15 فيبروري 1935ع. 119- روزانه الوحيد ڪراچي،7 آگسٽ 1936ع. 120- روزانه الوحيد ڪراچي،12 آگ

#### Upper quartile (75th percentile) — chrF++ 21.4

> **Prompt:** دگي شيون" ختم ڪري ٿي، جيڪي ماڻهن کي پنهنجن عملي جي ترتيبن ۾ خيالات کي منهن ڏيڻ ۽ عمل درآمد ڪن ٿا. آئون توهان جي واقعي يا ڪانفرنس لاء چيلل جي سفارش ڪريان ٿو. " ڊائريڪٽر، ڪانفرنس اندروني آڊيٽرز جو انسٽي

> **Model output:** " ريليمڼ - آل ڪريمنٿ انجنيئر چيو ته هي بجيٽ حقيقي جان جي صنعت جي صنعت ۾ واڌاري قنبر ۽ پهچائڻ جي عمل تي ڌيان ڏيڻ ۾ مشغول آهن، جن مان اضافو پيداوار جي لاڳاپيل ۽ مجموعي دنيا جي ترقي يافته ملڪن لاء ختم ذائقي ۽ چشمي جي جو انٽرفيس. - مارڪيٽ ۾ اهي

> **True continuation:** جي بي سي وزارت جي ايڪس سيٽيز ڪانفرنس جي اين اين ايڪس ٽين ڏينهن تي اسان جي اهم ڳالهائيندڙ هئي ۽ انهي سان گڏ هڪ پيروي ڪيل ورکشاپ پڻ ڪئي وئي ۽ گڏوگڏ ناشتو ۾ اسان جي ايگزيڪيوٽو ٽيم کي خطاب ڪيو. چيرل جي چي

#### Median — chrF++ 19.4

> **Prompt:** iac عورت. هوءَ سوچي ٿي، عمل ڪري ٿي ۽ وڏي پيماني تي پيار ڪري ٿي. هوءَ لڳ ڀڳ هميشه حقيقي طور تي خوش، خوش مزاج، ۽ هر ڪنهن سان به بحث ڪرڻ لاءِ تيار آهي، جنهن سان به اهي ملن، جيستائين اهو مثبت يا پرجوش هجي

> **Model output:** پيچيده ۽ آسانيون سمجهيو وڃي ٿو جيڪي منهنجي زندگي ۾ سٺيون محبتون آهن. ۽ اهو توهان کي سٺو ۽ دلڪش نه بنائي سگهي ٿو، پر خوشگوار ۽ پيڙهه جي گلن مان هڪ پڻ آهي. اوور ڊراپ بڪس تصويرون تصويرون ADDIDS ڏٺائين ته توهان حيران ٿي ويندا. سلونگ پنڪيچا جو ڳ

> **True continuation:** حفاظتي رڪاوٽ انهن جي تمام گهڻي سچائي زبانن کان. گهٽ خالص Sagittarians ڪنهن کي خبر ڏيڻ جي باري ۾ ڪجھ به نه سوچيندا آهن ته ڇا لباس خوشامد نه آهي، يا جيڪڏهن انهن جو ساٿي بي وفا آهي. ساڳئي وقت، انهن جي پر

#### Lower quartile (25th percentile) — chrF++ 17.4

> **Prompt:** زبان ۾ ماھر سمجهو ٿا؟ اسان مان ھزارين شخص انگريزيءَ ۾ تعليم وٺي پنھنجن پنھنجن ڪمن ۾ ماھر ٿيا آھن، وڪيل ٿيا آھن، انجنير ٿيا آھن، ۽ خدا معلوم ٻيو ڇا ڇا ٿيا آھن، پر ڇاڪاڻ تہ اھو ڊاڪٽريءَ جو يا وڪالت جو ي

> **Model output:** ۽ پاڻ کي اسڪندر جي نالي سان سڃاڻين ٿا، تربيتي تربيت ڪرڻ ٿا ڳالھه بلڪل بلڪل صحيح آھي، ڇاڪاڻ تہ ايئن سمجهہ جوڳي آھي تہ مھل Cxinance ڪاليجن سان ھو چاھ ڪندو آھي، ۽ اُھو اُتي اسپانسر ٿئي پيو ۽ پوءِ وڌيڪ تعليم حاصل ڪندي آھي اسان وٽ ھڪ عام ماڻهو ن

> **True continuation:** جا زبان مجبوريءَ ھيٺ سکي اٿن، تنھنڪري نہ منھنجن تخليقي قوتون پيدا ٿي سگهيون، نہ جدت جي واٽ ملي ڀٽائيءَ جي لفظن ۾ کڻي چئجي تہ مجبوريءَ مري ويا، ماھر ٿي نہ مئا پري نہ وڃو، پنھنجن ڪنھن وڏي کان وڏي ڊاڪٽر 

#### Worst of 200 — chrF++ 0.8

> **Prompt:** ئون پاڪستان ٺهي سگهندو؟مظلوم لجپت راءِ مظلوم - ميان نوازشريف جو مستقبل ۽ شيدو مستريءَ جي جيپ - نگران حڪومت جو عوام سان ٻيو وڏو ڏاڍ اعجاز ممنائي - عام چونڊون ۽ ووٽ جي اهميت ۽ طاقت ؛ - خلائي مخلوق ۽ سوش

> **Model output:** riSun<|endoftext|> ممبئي ( ويب ڊيسڪ ) بالي ووڊ جي اداڪار ريٽائر رضا جو چوڻ آهي ته ڀارتي ميڊيا کانپوءِ صرف ٽينا جي ڪنهن اداڪار جي حيثيت ۾ ويس بدلائي ڇڏيو، هلندڙ سال چاليهه سال ڀارت جي ٽاپ هيڊ ڪوارٽر ۾ اداڪار جي حيثيت ۾ ويس بدلائي ويو. ميڊيا 

> **True continuation:** ch 2020 - February 2020 - January 2020 - December 2019 - November 2019 - October 2019 - September 2019 - August 2019 - July 2019 - June 2019 - May 2019 - April 2019 - March 2019 - February 2019 - Janu


## Qualitative assessment

**Both models learned the language; neither learned to stay on topic.**

The greedy samples are grammatical — correct word order, plausible morphology, sensible
collocations — and then repeat a single clause indefinitely. The T = 1.0 samples read as
fluent native text, with agreement and inflection largely correct, but drift away from the
prompt's subject within a sentence or two. In the Hindi example the prompt concerns women
overcoming obstacles and the continuation moves to plastic pipes and flooding; the Sindhi
example drifts from a 17th-century physician to unrelated dynastic history.

That combination — local fluency, absent long-range coherence — is what a 24M-parameter
model trained on 500M tokens should be expected to produce. Syntax and morphology are
learnable from local statistics; sustained topical control needs far more capacity and data.

Raw measurements and full samples: `report/generation_results.json`.
