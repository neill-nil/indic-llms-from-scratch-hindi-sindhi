"""
Validate the candidate reasoning vocabulary against each language's own corpus.

Turns "is this natural Sindhi?" — a question the author cannot answer honestly — into
"how often does this occur in 618M tokens of real Sindhi?", which the corpus answers.

Two measurements per candidate:

  occurrences   raw count in the held-out validation split. Validation is used rather
                than train because it is 40x smaller and drawn from the same
                distribution, so it gives the same verdict in seconds rather than
                minutes. Candidates that score zero there are re-checked against the
                full training split before being rejected, so a rare-but-real word is
                not thrown away on sampling noise alone.

  fertility     how many tokens that language's own Phase 1 tokenizer spends on the
                string. This matters independently of frequency: an entity name that
                shatters into eight rare subwords costs the model far more to read than
                one that is a single token, and a reasoning dataset built from such
                names would confound "cannot reason" with "cannot read the name".

Usage:
    python -m common.reasoning.validate_lexicon                    # both languages
    python -m common.reasoning.validate_lexicon --language sindhi
    python -m common.reasoning.validate_lexicon --scan_train       # slower, exhaustive
"""

import argparse
import json
import logging
import random
from pathlib import Path

import sentencepiece as spm

from common.reasoning.candidates import CANDIDATES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("validate_lexicon")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Acceptance thresholds. Entity names are held to a lower bar than constructions
# because a name only needs to be *readable* by the model, whereas a grammatical
# frame is the thing carrying the claim that the text is natural — if a frame is
# rare, the sentences built on it are not idiomatic and the whole corpus is suspect.
MIN_COUNT_ENTITY = 20
MAX_FERTILITY_ENTITY = 6

# Constructions are scored against a length-dependent bar. A flat threshold is wrong
# because base rate falls steeply with phrase length: a single word competes with the
# whole vocabulary, a three-word phrase with the space of all trigrams. The first run
# of this script used a flat 100 and rejected "साल है" (93) and "سال آهي" (73) —
# both perfectly ordinary — while a threshold that would have admitted them would have
# admitted almost any single word. The bar below is set per word count instead.
# The single-word bar was initially 100 and is now 50. Recording the change and its
# effect rather than quietly retuning it: the first setting was chosen for CONTENT
# words, and it excluded किसकी (99) and किसका (77) — closed-class interrogatives that
# are unquestionably part of Hindi. Function words have a different base rate from
# content words, and the bar exists to catch constructions that are not part of the
# language, not to rank words by frequency. At 50 occurrences in 59M characters of
# held-out text (~1 per million), a word is attested by any reasonable standard.
#
# The change is narrow and auditable: across both languages the only candidates that
# cross from reject to accept are those two Hindi interrogatives. It was made to admit
# a specific pair the author judged wrongly excluded, which is exactly the kind of
# motivated adjustment worth flagging in the open.
MIN_COUNT_BY_WORDS = {1: 50, 2: 40, 3: 20}
MIN_COUNT_LONG_PHRASE = 10

# Candidates landing within this factor of their threshold are re-counted against the
# full training split, which is ~40x larger, instead of being decided on the smaller
# sample. Applied symmetrically: near-misses can be admitted and near-passes rejected.
BORDERLINE_FACTOR = 2.0

# Fraction of each entity pool withheld from training, to build the `test_entity`
# split that measures whether the model learned the relation or just the names.
HELDOUT_ENTITY_FRACTION = 0.25
ENTITY_SPLIT_SEED = 20260910


def fragment_threshold(surface: str) -> int:
    """Minimum occurrences required of a construction, by its length in words."""
    return MIN_COUNT_BY_WORDS.get(len(surface.split()), MIN_COUNT_LONG_PHRASE)

# Reading the training split in chunks rather than whole: Hindi's is 4.7 GB and would
# not fit comfortably in RAM alongside the tokenizer and the page cache.
TRAIN_CHUNK_BYTES = 256 * 1024 * 1024


def load_val_text(language: str) -> str:
    """Read a language's validation split into memory (~110-140 MB)."""
    path = PROJECT_ROOT / language / "data" / "preprocessed" / "val.txt"
    return path.read_text(encoding="utf-8")


def scan_train(language: str, needles: list[str]) -> dict[str, int]:
    """
    Count a handful of strings across the full training split.

    Only called for candidates that scored zero on validation, so `needles` is small
    and a single streaming pass is affordable. An overlap margin is carried between
    chunks so a match straddling a boundary is not lost.
    """
    if not needles:
        return {}
    path = PROJECT_ROOT / language / "data" / "preprocessed" / "train.txt"
    counts = {n: 0 for n in needles}
    margin = max(len(n) for n in needles)
    tail = ""
    logger.info("scanning %s (%.1f GB) for %d zero-count candidates",
                path.name, path.stat().st_size / 1e9, len(needles))
    with open(path, encoding="utf-8") as f:
        while chunk := f.read(TRAIN_CHUNK_BYTES):
            buf = tail + chunk
            for n in needles:
                counts[n] += buf.count(n)
            tail = buf[-margin:]
    return counts


def validate(language: str, scan_train_split: bool) -> dict:
    """Measure every candidate for one language and partition into accepted/rejected."""
    cand = CANDIDATES[language]
    sp = spm.SentencePieceProcessor(
        model_file=str(PROJECT_ROOT / language / "tokenizer"
                       / f"{language}_unigram_16000.model"))

    logger.info("%s: loading validation split", language)
    text = load_val_text(language)
    logger.info("%s: %.0fM characters", language, len(text) / 1e6)

    # Every distinct string we need a count for, gathered before scanning so the text
    # is traversed once per candidate rather than once per lookup site.
    items: list[tuple[str, str, str]] = []          # (category, key, surface)
    for kind in ("people", "places", "objects"):
        for name, gender in cand[kind]:
            items.append((kind, f"{name}|{gender}", name))
    for attr, spec in cand["attributes"].items():
        for field, surface in spec.items():
            if field == "applies_to":
                continue
            items.append(("attribute", f"{attr}.{field}", surface))
        # The possessive phrase is validated as a COMPOSED unit, not as two words.
        # Both languages inflect the possessive for the gender of the possessed noun
        # ("की उम्र" but "का वजन"; "جي عمر" but "جو وزن"), so checking `poss` and
        # `noun` separately would pass even if they were wrongly paired. The phrase
        # is what the generator actually emits, so the phrase is what gets measured.
        items.append(("attribute", f"{attr}.poss_phrase",
                      f"{spec['poss']} {spec['noun']}"))
    for key, frag in cand["fragments"].items():
        items.append(("fragment", key, frag))
    for key, mark in cand["punct"].items():
        items.append(("punct", key, mark))

    counts = {surface: text.count(surface) for _, _, surface in items}

    def bar(category: str, surface: str) -> int:
        """The validation-split occurrence threshold this candidate must clear."""
        if category in ("people", "places", "objects"):
            return MIN_COUNT_ENTITY
        if category == "punct":
            return 1
        return fragment_threshold(surface)

    # Escalate the uncertain calls to the full training split rather than deciding them
    # on the validation sample. Two kinds of candidate are uncertain: those that scored
    # zero (possibly real but rare) and those sitting within BORDERLINE_FACTOR of their
    # threshold, where the accept/reject flip is sampling noise rather than evidence.
    # "किसकी" motivated this: it scored 99 against a bar of 100 and would have been
    # thrown out over a single occurrence.
    #
    # Escalating *both* directions of the boundary matters. Re-checking only the near
    # misses would be a ratchet that can admit but never exclude — it would tune the
    # procedure toward the answer the author wanted. Candidates just ABOVE the bar are
    # re-checked on the same rule and can fail there.
    train_counts: dict[str, int] = {}
    if scan_train_split:
        uncertain = sorted({
            surface for category, _, surface in items
            if counts[surface] == 0
            or counts[surface] < bar(category, surface) * BORDERLINE_FACTOR
        })
        train_counts = scan_train(language, uncertain)

    # Thresholds are calibrated on the validation split, so a train count has to be
    # scaled back to a val-equivalent before it can be compared against them. The two
    # splits come from one shuffled pool, so a simple size ratio is the right scale.
    #
    # Both sides of that ratio must be measured in the SAME unit. An earlier version
    # divided val CHARACTERS by train BYTES; Devanagari and Perso-Arabic average about
    # 2.4 UTF-8 bytes per character, so every train-derived count came out 2.4x too
    # small and the escalation rejected words that are plainly common. Bytes on both
    # sides is the cheap correct choice — the train split is far too large to count
    # characters without decoding all 4.9 GB of it.
    val_bytes = (PROJECT_ROOT / language / "data" / "preprocessed"
                 / "val.txt").stat().st_size
    train_bytes = (PROJECT_ROOT / language / "data" / "preprocessed"
                   / "train.txt").stat().st_size if train_counts else 0
    scale = val_bytes / train_bytes if train_bytes else 1.0

    records = []
    for category, key, surface in items:
        pieces = sp.encode(surface, out_type=str)
        n = counts[surface]
        threshold = bar(category, surface)
        source = "val"
        effective = float(n)
        if surface in train_counts:
            effective = train_counts[surface] * scale
            source = "train"
        ok = effective >= threshold
        if category in ("people", "places", "objects"):
            ok = ok and len(pieces) <= MAX_FERTILITY_ENTITY
        records.append({
            "category": category, "key": key, "surface": surface,
            "occurrences": n, "train_occurrences": train_counts.get(surface),
            "val_equivalent": round(effective, 1), "threshold": threshold,
            "decided_on": source,
            "fertility": len(pieces), "pieces": pieces,
            "accepted": bool(ok),
        })

    return {"language": language, "val_chars": len(text),
            "val_bytes": val_bytes, "train_bytes": train_bytes, "records": records}


def build_lexicon(result: dict) -> dict:
    """Assemble the accepted survivors into the structure the generator consumes."""
    language = result["language"]
    cand = CANDIDATES[language]
    ok = {r["key"] for r in result["records"] if r["accepted"]}

    # The train / held-out entity partition is decided HERE, in the committed lexicon,
    # rather than inside the generator. Two reasons. It makes leakage structurally
    # impossible instead of merely audited afterwards: the generator is handed two
    # disjoint pools and never sees a reason to cross them. And it makes the partition
    # an inspectable artifact — a reader can open the JSON and check which names were
    # withheld, instead of having to re-run a script and trust its seed.
    #
    # The split is stratified by gender so the held-out pool cannot end up all-male,
    # which would turn "unseen name" into "unseen name AND unseen agreement pattern"
    # and confound the generalisation measurement it exists to make.
    rng = random.Random(ENTITY_SPLIT_SEED)
    entities = {}
    for kind in ("people", "places", "objects"):
        survivors = [{"name": n, "gender": g}
                     for n, g in cand[kind] if f"{n}|{g}" in ok]
        by_gender: dict[str, list] = {}
        for e in survivors:
            by_gender.setdefault(e["gender"], []).append(e)
        for group in by_gender.values():
            rng.shuffle(group)
            n_held = max(1, round(len(group) * HELDOUT_ENTITY_FRACTION))
            for i, e in enumerate(group):
                e["split"] = "heldout" if i < n_held else "train"
        entities[kind] = sorted(survivors, key=lambda e: (e["split"], e["name"]))
        held = sum(1 for e in entities[kind] if e["split"] == "heldout")
        logger.info("%s: %s -> %d train / %d held-out",
                    language, kind, len(entities[kind]) - held, held)

    attributes = {}
    for attr, spec in cand["attributes"].items():
        # `noun`, `unit` and the composed possessive are load-bearing: without any one
        # of them the attribute cannot be rendered at all, so a failure drops it.
        required = ["noun", "unit", "poss_phrase"]
        missing = [f for f in required if f"{attr}.{f}" not in ok]
        if missing:
            logger.warning("%s: dropping attribute %r — unattested %s",
                           language, attr, missing)
            continue
        # The adjective forms are optional variety, used only for the paraphrase
        # templates. An attribute keeps working without them; it just contributes
        # fewer surface forms.
        kept = {k: v for k, v in spec.items()
                if k in ("noun", "poss", "unit", "applies_to") or f"{attr}.{k}" in ok}
        dropped = [k for k in spec if k not in kept]
        if dropped:
            logger.info("%s: attribute %r keeps core, drops variety forms %s",
                        language, attr, dropped)
        attributes[attr] = kept

    return {
        "language": language,
        "entities": entities,
        "attributes": attributes,
        "fragments": {k: v for k, v in cand["fragments"].items() if k in ok},
        "punct": {k: v for k, v in cand["punct"].items() if k in ok},
    }


def enforce_parity(lexicons: dict[str, dict]) -> None:
    """
    Reduce both lexicons to the template inventory available in BOTH languages.

    Phase 3's headline claim is a comparison between Model H and Model L, and Phase 2
    earned that comparison by holding everything else equal — identical architecture,
    identical token budget, identical optimisation. A finetuning set that gave one model
    more surface variety than the other would quietly undo that: any measured gap would
    be partly an artifact of the dataset rather than of the languages.

    Validation left exactly one asymmetry — Hindi retained the adjective superlative
    `सबसे छोटा` (val 36, train-confirmed) while Sindhi's `سڀ کان ننڍو` fell below the
    bar (val 11). Small, but it points the wrong way, so the richer language is trimmed
    to the poorer one rather than the reverse.

    The cost is real and one-directional: Model H's data is slightly less varied than it
    could have been. That is the correct trade — a controlled comparison is worth more
    than one extra Hindi phrasing, and the loss is recorded here rather than absorbed
    silently.
    """
    if len(lexicons) < 2:
        logger.warning("parity not enforced: needs both languages in one run")
        return

    shared = set.intersection(*(set(l["fragments"]) for l in lexicons.values()))
    for language, lex in lexicons.items():
        dropped = sorted(set(lex["fragments"]) - shared)
        for key in dropped:
            logger.warning("%s: dropping fragment %r (%r) — no counterpart in the "
                           "other language; keeping the two datasets parallel",
                           language, key, lex["fragments"][key])
        lex["fragments"] = {k: v for k, v in lex["fragments"].items() if k in shared}

    # The same rule applies to the per-attribute adjective variety forms, which are the
    # other place an asymmetry could enter.
    for attr in set().union(*(set(l["attributes"]) for l in lexicons.values())):
        present = [l["attributes"].get(attr) for l in lexicons.values()]
        if any(spec is None for spec in present):
            for language, lex in lexicons.items():
                if lex["attributes"].pop(attr, None) is not None:
                    logger.warning("%s: dropping attribute %r — absent in the other "
                                   "language", language, attr)
            continue
        shared_fields = set.intersection(
            *({k for k in spec if k.startswith("adj_")} for spec in present))
        for language, lex in lexicons.items():
            spec = lex["attributes"][attr]
            for k in sorted({k for k in spec if k.startswith("adj_")} - shared_fields):
                spec.pop(k)
                logger.warning("%s: dropping %s.%s — no counterpart", language, attr, k)

    lex_any = next(iter(lexicons.values()))
    logger.info("parity enforced: %d fragments, %d attributes in both languages",
                len(lex_any["fragments"]), len(lex_any["attributes"]))


def write_report(results: list[dict], path: Path) -> None:
    """Write the human-readable evidence table backing every acceptance decision."""
    lines = [
        "# Phase 3 — Reasoning lexicon validation",
        "",
        "<!-- GENERATED FILE - do not edit by hand.",
        "     Regenerate with: python -m common.reasoning.validate_lexicon --scan_train",
        "-->",
        "",
        "Every word and construction used to build the synthetic reasoning corpora was",
        "checked against that language's **own pretraining corpus** before use. This",
        "replaces an authorial claim that the text is natural with a measurement of how",
        "often it actually occurs in real text of that language.",
        "",
        "`occurrences` counts the held-out **validation** split (Hindi 141 MB, Sindhi",
        "112 MB). The training splits are ~40x larger, so a count of *n* here implies",
        "roughly *40n* in training. `fertility` is how many tokens that language's own",
        "16K Unigram tokenizer spends on the string.",
        "",
        f"Thresholds: entities need >= {MIN_COUNT_ENTITY} occurrences and <= "
        f"{MAX_FERTILITY_ENTITY} tokens. Constructions are scored on a length-dependent",
        "bar — " + ", ".join(f"{w} word: {c}" for w, c in MIN_COUNT_BY_WORDS.items())
        + f", longer: {MIN_COUNT_LONG_PHRASE} — because base rate falls steeply with",
        "phrase length and a flat threshold rejects ordinary multi-word phrases.",
        "",
    ]
    for r in results:
        recs = r["records"]
        acc = sum(1 for x in recs if x["accepted"])
        lines += [f"## {r['language'].title()}", "",
                  f"{acc}/{len(recs)} candidates accepted.", ""]
        for category in ("people", "places", "objects", "attribute", "fragment", "punct"):
            sub = [x for x in recs if x["category"] == category]
            if not sub:
                continue
            lines += [f"### {category}", "",
                      "| surface | key | val | train | bar | tokens | accepted |",
                      "|---|---|---:|---:|---:|---:|:---:|"]
            for x in sorted(sub, key=lambda v: -v["occurrences"]):
                mark = "yes" if x["accepted"] else "**NO**"
                tr = (f"{x['train_occurrences']:,}"
                      if x.get("train_occurrences") is not None else "—")
                lines.append(f"| `{x['surface']}` | {x['key']} | {x['occurrences']:,} "
                             f"| {tr} | {x['threshold']} | {x['fertility']} | {mark} |")
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote %s", path)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--language", choices=["hindi", "sindhi"], help="default: both")
    p.add_argument("--scan_train", action="store_true",
                   help="re-check zero-count candidates against the full training split")
    args = p.parse_args()

    languages = [args.language] if args.language else ["hindi", "sindhi"]
    results = [validate(lang, args.scan_train) for lang in languages]

    out_dir = Path(__file__).resolve().parent
    lexicons = {r["language"]: build_lexicon(r) for r in results}
    enforce_parity(lexicons)

    for r in results:
        lex = lexicons[r["language"]]
        path = out_dir / f"lexicon_{r['language']}.json"
        path.write_text(json.dumps(lex, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        n_ent = sum(len(v) for v in lex["entities"].values())
        logger.info("%s: %d entities, %d attributes -> %s",
                    r["language"], n_ent, len(lex["attributes"]), path.name)

    report = PROJECT_ROOT / "report" / "phase3_lexicon_validation.md"
    write_report(results, report)

    for r in results:
        rejected = [x for x in r["records"] if not x["accepted"]]
        if rejected:
            logger.warning("%s: %d rejected", r["language"], len(rejected))
            for x in rejected:
                logger.warning("    %-22s %-18s n=%-7d fert=%d",
                               x["surface"], x["key"], x["occurrences"], x["fertility"])


if __name__ == "__main__":
    main()
