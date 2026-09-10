# Phase 3 — Reasoning task design

The specification for the two synthetic reasoning corpora, fixed **before** any data was
generated or any model finetuned, so that the evaluation splits could not be shaped by
early results.

## Where this document sits

Phase 3's report **is** the final report — the brief asks it to consolidate the whole
project rather than adding a separate one. This file is a companion to it, on the same
hub-and-spoke pattern Phase 2 used (`phase2.md` carrying the argument,
`context_length_decision.md` carrying one decision in full).

The division of labour is deliberate, and each document is written to stay on its own
side of it:

| Document | Owns | Does not contain |
|---|---|---|
| `phase3.md` (final report) | The argument, headline results, the brief's four questions, cross-phase synthesis | Full evidence tables, template schemas, rejected candidates |
| **this file** | The *specification* — task families, split rules, balance controls — and why the design is shaped this way | Results, model behaviour, anything measured after data existed |
| `phase3_lexicon_validation.md` | Generated evidence: every candidate, its corpus count, its fertility, accept/reject | Any hand-written prose |
| `<language>/finetune/data/dataset_statistics.json` | Measured distributions and audit outcomes | Interpretation |

The rule that keeps them from duplicating: **this file explains why the design is what it
is; the report explains what was found and what it means.** Where the report needs a
design fact it states it in a sentence and links here, rather than reproducing the table.

There is also a reason this document exists *separately* rather than as a section of the
report. It was written and committed **before any data was generated or any model
finetuned**, so the test splits demonstrably were not shaped by early results. Folding it
into a report written afterwards would destroy that evidence; its commit history is part
of what it asserts.

Companion document: [`phase3_lexicon_validation.md`](phase3_lexicon_validation.md) — the
occurrence counts backing every word and construction used here.

---

## 1. What the brief requires

> Focus on **comparative and basic reasoning**, especially transitive / ordering
> comparisons. […] Text must be in the target language's script and natural phrasing for
> that language (not English prompts with foreign names only). Include train / validation
> / test splits; report size, template variety, and how you avoid train–test leakage.

Three obligations follow: the reasoning must be genuinely comparative and transitive, the
language must be idiomatic rather than translated, and the splits must be defensible.
Sections 2, 3 and 4 address these in turn.

---

## 2. Naturalness: how it is established, and what is *not* claimed

The author is not a fluent speaker of Sindhi. Asserting that the generated Sindhi is
natural would therefore be an assertion without grounds. The corpus replaces that
judgement with a measurement, at three clearly separated levels of confidence.

### 2.1 Atoms — measured

Every word and construction the generator can emit was counted in that language's **own
pretraining corpus** before being admitted. The full evidence table is in
`phase3_lexicon_validation.md`. This is direct evidence.

**It changed the design substantially.** The intended backbone was the adjective
comparative — "X is *taller* than Y". The corpus rejected it:

| construction | Hindi | Sindhi | |
|---|---:|---:|---|
| quantifier: `से ज्यादा` / `کان وڌيڪ` | **3,676** | **6,930** | *adopted* |
| adjective, generic: `से बड़ा` / `کان وڏو` | 1,044 | 738 | variety only |
| adjective, height: `से लंबा` / `کان ڊگهو` | 28 | 17 | rejected |
| adjective, weight: `से भारी` / `کان ڳرو` | 109 | **3** | rejected |
| superlative, weight: `सबसे भारी` / `سڀ کان ڳرو` | 5 | **1** | rejected |

Attribute-specific adjective comparatives are effectively **absent from both corpora**. A
dataset built on them would have read as translated English — the exact failure the brief
names. The design therefore uses the **quantifier comparative** ("X's weight is *more*
than Y's weight"), which is the dominant form in both languages.

That choice carries a second benefit. `ज्यादा`, `कम`, `وڌيڪ` and `گهٽ` are
**gender-invariant**, where the adjectives inflect (`बड़ा`/`बड़ी`, `وڏو`/`وڏي`). The
backbone templates therefore cannot produce an agreement error at all, because there is
no agreement to get wrong.

Two smaller corrections came from the same process, both cases where the author's first
choice was simply wrong:

| intended | occurrences | used instead | occurrences |
|---|---:|---|---:|
| `वज़न` (with nukta) | 76 | `वजन` | 2,433 |
| `सेंटीमीटर` | 85 | `सेमी` | 477 |
| `مفاصلو` | 82 | `فاصلو` | 263 |

### 2.2 Composition — argued, not measured

Whole rendered sentences are **not** corpus-attested, and cannot be. Measured directly:

| string | Hindi | Sindhi |
|---|---:|---:|
| `किसकी उम्र` / `ڪنهن جي عمر` | 0 | 1 |
| `सबसे ज्यादा उम्र` / `سڀ کان وڌيڪ عمر` | 1 | 0 |

This is **base-rate collapse, not evidence of unnaturalness**. A specific four-word
combination of independently common constructions will essentially never recur verbatim in
60M characters; that is a property of language, not a defect in the phrase. It is the
reason the validator scores atoms rather than clauses — an earlier version scored whole
clauses and rejected `साल है` (93) and `سال آهي` (73), both entirely ordinary.

**The honest statement is therefore:** the *pieces* are attested by measurement; their
*composition* is the author's construction, argued from the grammar of the pieces but not
independently verified. This is the weakest link in the Sindhi half of Phase 3 and is
recorded as such rather than glossed over.

### 2.3 Format scaffolding — not a naturalness claim

The `प्रश्न:` / `उत्तर:` and `سوال:` / `جواب:` markers that delimit prompt from answer are
**deliberate structure introduced for finetuning**, not a claim about either language. The
marker words are attested (`प्रश्न` 1,726, `उत्तर` 7,498, `سوال` 5,217, `جواب` 8,152), but
the colon convention is ours, and the model learns it during finetuning. Reported here so
it is not mistaken for a corpus finding.

---

## 3. Task families

Five families, rendered independently in each language from that language's own lexicon.
No text is translated between them; the two corpora share task *structure* only.

| # | Family | Tests | Answer type |
|---|---|---|---|
| A | Two-entity numeric comparison | reading two values, comparing | entity name |
| B | Three-entity superlative | selecting an extremum from three | entity name |
| C | **Transitive chain**, no numbers | composing `A>B`, `B>C` | entity name |
| D | Endpoint relation from a chain | inferring the `A`–`C` relation | relation word |
| E | Equality / tie | recognising that no entity wins | relation word |

### Worked examples

**Family A** — Hindi

> प्रश्न: राम की उम्र 25 साल है। श्याम की उम्र 30 साल है। किसकी उम्र ज्यादा है?
> उत्तर: श्याम

**Family C** — Hindi (transitive, no numbers given)

> प्रश्न: राम की उम्र श्याम से ज्यादा है। श्याम की उम्र मोहन से ज्यादा है। सबसे ज्यादा उम्र किसकी है?
> उत्तर: राम

**Family C** — Sindhi

> سوال: احمد جي عمر علي کان وڌيڪ آهي. علي جي عمر حسن کان وڌيڪ آهي. سڀ کان وڌيڪ عمر ڪنهن جي آهي؟
> جواب: احمد

**Family E** — Sindhi (tie)

> سوال: احمد جي عمر 25 سال آهي. علي جي عمر 25 سال آهي. ڪنهن جي عمر وڌيڪ آهي؟
> جواب: برابر

Family E exists to defeat a specific degenerate strategy. Without ties, a model can score
well by always naming *some* entity; it never has to represent equality. With ties present
the "always guess a name" policy is capped, and the model must actually compare.

---

## 4. Splits and leakage control

### 4.1 Four test splits

Each isolates a different way the model could appear to reason without reasoning.

| Split | Held out | Defeats |
|---|---|---|
| `test_iid` | nothing | — (in-distribution ceiling) |
| `test_entity` | **entity names** | memorising name→answer associations |
| `test_template` | **surface phrasings** | memorising a fixed word order |
| `test_hop` | **chain length** (train 2-hop → test 3-hop) | pattern-matching instead of composing |

`test_hop` is the strongest probe: a model that has genuinely learned transitivity should
extend to a longer chain, while one that has memorised the shape of a two-step problem
should not. **A low score here is an expected and reportable result at 24M parameters, not
a pipeline failure.**

### 4.2 Entity partition is structural, not audited

The train / held-out entity partition is fixed in the committed lexicon files
(`lexicon_hindi.json`, `lexicon_sindhi.json`), where every entity carries an explicit
`split` field, rather than being decided inside the generator. The generator is handed two
disjoint pools and has no path to cross them, so leakage is impossible by construction
rather than detected afterwards. A reader can verify the partition by opening the JSON.

The split is **stratified by gender** (25% withheld):

| | people | places | objects |
|---|---|---|---|
| Hindi | 52 train / 18 held-out | 15 / 5 | 10 / 4 |
| Sindhi | 47 train / 16 held-out | 15 / 5 | 11 / 4 |

Stratification matters: an all-male held-out pool would confound "unseen name" with
"unseen agreement pattern" and corrupt the measurement `test_entity` exists to make.

### 4.3 Template partition

The `age` attribute retains the adjective comparative (`सबसे बड़ा` 899, `سڀ کان وڏو` 521)
alongside the quantifier form. This gives a **genuine register difference**, attested in
both corpora, from which the held-out paraphrase templates are drawn — rather than
artificial reworderings invented to fill a split.

### 4.4 Balance controls

Applied at generation time so no post-hoc filtering is needed:

- **Answer labels balanced** across classes, so majority-class guessing scores at chance.
- **Answer position uniform** over mention order, so "always name the last entity" fails.
- **Numeric gap varied**, recorded per example, so accuracy can be reported against margin
  — near-ties should be measurably harder if the model is genuinely comparing.

---

## 5. What is deliberately not done

- **Chain-of-thought rendering** is not in the primary dataset. It is a candidate
  extension, kept separate so the headline result is not confounded by two changes at once.
- **Distractor attributes** (mentioning height when asking about weight) are not included
  in v1. Worth adding if in-distribution accuracy saturates.
- **The composed sentences are not verified by a native speaker.** See §2.2. This is the
  principal limitation of the Sindhi half of this phase.
