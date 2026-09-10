"""
Validate hand-edited reasoning templates before they are used to build a dataset.

This is the safety net for human-authored templates. `templates_<language>.json` is meant
to be edited by hand, and a template that is malformed, uses an unknown placeholder, or
introduces a word that does not occur in that language's corpus would otherwise surface
only as a silently corrupted dataset — or worse, as a plausible-looking dataset that is
subtly not the language.

Four checks:

  schema        every required field present; `family`, `question_type` and `split`
                drawn from the permitted sets; ids unique.
  placeholders  every {placeholder} is one the generator knows how to fill, and is legal
                for that template's family and position. A typo like {atr} would
                otherwise raise deep inside generation with an unhelpful KeyError.
  literals      every word of literal text — anything outside a placeholder — is checked
                against the language's own pretraining corpus, on the same standard the
                lexicon itself had to meet. This is what stops a hand-written template
                from quietly reintroducing vocabulary that the corpus does not support.
  render        each template is rendered with real lexicon values and printed, so the
                author can read the actual sentences rather than the format string.

Usage:
    python -m common.reasoning.check_templates                    # both languages
    python -m common.reasoning.check_templates --language hindi
"""

import argparse
import json
import logging
import random
import re
from pathlib import Path

from common.reasoning.generate import (HERE, Renderer, applicable_attributes,
                                       sample_values, VALUE_RANGES)
from common.reasoning.validate_lexicon import load_val_text, fragment_threshold

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger("check_templates")

PLACEHOLDER = re.compile(r"\{(\w+)\}")

# Placeholders the generator can fill, by where they are legal.
SLOT_PLACEHOLDERS = {
    "attr", "poss", "unit", "cop", "than", "more", "less",
    "sup_more", "sup_less", "who", "who_poss", "or", "equal", "fs", "qm",
}
FACT_ONLY = {"e", "v", "ea", "eb"}
QUESTION_ONLY = {"efirst", "elast"}

REQUIRED = {"id", "family", "question_type", "fact", "question", "split"}
FAMILIES = {"value_compare", "chain"}
QUESTION_TYPES = {"argmax", "argmin", "relation"}
SPLITS = {"train", "heldout"}


def check_schema(templates: list[dict]) -> list[str]:
    """Structural validity: required fields, permitted values, unique ids."""
    errors, seen_ids = [], set()
    for i, t in enumerate(templates):
        where = t.get("id", f"index {i}")
        for field in sorted(REQUIRED - set(t)):
            errors.append(f"{where}: missing required field {field!r}")
        if t.get("family") not in FAMILIES:
            errors.append(f"{where}: family must be one of {sorted(FAMILIES)}")
        if t.get("question_type") not in QUESTION_TYPES:
            errors.append(f"{where}: question_type must be one of {sorted(QUESTION_TYPES)}")
        if t.get("split") not in SPLITS:
            errors.append(f"{where}: split must be one of {sorted(SPLITS)}")
        if t.get("id") in seen_ids:
            errors.append(f"{where}: duplicate id")
        seen_ids.add(t.get("id"))
        if t.get("question_type") == "relation" and t.get("family") != "chain":
            errors.append(f"{where}: relation questions need a chain to have endpoints")
    return errors


def check_placeholders(templates: list[dict]) -> list[str]:
    """Every {placeholder} must be fillable, and legal where it appears."""
    errors = []
    for t in templates:
        where = t.get("id", "?")
        for field in ("fact", "fact_reverse", "question"):
            if field not in t:
                continue
            is_question = field == "question"
            allowed = SLOT_PLACEHOLDERS | (QUESTION_ONLY if is_question else FACT_ONLY)
            for name in PLACEHOLDER.findall(t[field]):
                if name in SLOT_PLACEHOLDERS:
                    continue
                if name not in allowed:
                    hint = ""
                    if name in FACT_ONLY:
                        hint = " — entity placeholders belong in `fact`, not `question`"
                    elif name in QUESTION_ONLY:
                        hint = " — endpoint placeholders belong in `question`, not `fact`"
                    errors.append(f"{where}.{field}: unknown placeholder {{{name}}}{hint}")

        if t.get("family") == "chain":
            if not {"ea", "eb"} <= set(PLACEHOLDER.findall(t.get("fact", ""))):
                errors.append(f"{where}: a chain `fact` must use both {{ea}} and {{eb}}")
            if t.get("question_type") == "relation":
                if not {"efirst", "elast"} <= set(PLACEHOLDER.findall(t["question"])):
                    errors.append(f"{where}: a relation `question` must use "
                                  f"{{efirst}} and {{elast}}")
        elif t.get("family") == "value_compare":
            if not {"e", "v"} <= set(PLACEHOLDER.findall(t.get("fact", ""))):
                errors.append(f"{where}: a value_compare `fact` must use {{e}} and {{v}}")
            if "fact_reverse" in t:
                errors.append(f"{where}: fact_reverse applies to chains only")
    return errors


def check_literals(templates: list[dict], text: str) -> list[str]:
    """
    Any literal word in a template must be attested in that language's corpus.

    Templates are mostly placeholders, so this usually finds nothing — which is the
    point. It exists so that a hand-written template introducing a new word cannot
    bypass the corpus validation that every lexicon entry had to pass.
    """
    warnings = []
    for t in templates:
        for field in ("fact", "fact_reverse", "question"):
            if field not in t:
                continue
            literal = PLACEHOLDER.sub(" ", t[field])
            for word in literal.split():
                word = word.strip("।.,?؟،!:\"'()")
                if not word or word.isascii():
                    continue
                n = text.count(word)
                if n < fragment_threshold(word):
                    warnings.append(
                        f"{t.get('id','?')}.{field}: literal {word!r} occurs {n:,} times "
                        f"in the validation split — below the bar of "
                        f"{fragment_threshold(word)}. Add it to candidates.py and "
                        f"re-validate, or use an existing placeholder.")
    return warnings


def render_samples(language: str, templates: list[dict], seed: int = 0) -> list[str]:
    """Render each template with real values so the author can read the output."""
    lexicon = json.loads((HERE / f"lexicon_{language}.json").read_text(encoding="utf-8"))
    rend = Renderer(lexicon)
    rng = random.Random(seed)
    out = []
    for t in templates:
        kinds = [k for k, v in lexicon["entities"].items()
                 if v and applicable_attributes(lexicon, k)]
        kind = rng.choice(kinds)
        attr_key = rng.choice(applicable_attributes(lexicon, kind))
        slots = rend.base_slots(attr_key)
        pool = [e for e in lexicon["entities"][kind] if e["split"] == "train"]
        ents = rng.sample(pool, 3)
        lo, hi = VALUE_RANGES[attr_key]

        if t["family"] == "value_compare":
            # Distinct values, exactly as the real generator draws them. Using
            # independent randint here produced previews with accidental duplicate
            # values, which made an argmax question look ambiguous in the sample even
            # though the dataset itself never contains one.
            values = sample_values(attr_key, len(ents), tie=False, rng=rng)
            facts = [t["fact"].format(e=e["name"], v=v, **slots)
                     for e, v in zip(ents, values)]
            question = t["question"].format(**slots)
        else:
            facts = [t["fact"].format(ea=ents[i]["name"], eb=ents[i + 1]["name"], **slots)
                     for i in range(2)]
            if "fact_reverse" in t:
                facts[1] = t["fact_reverse"].format(
                    ea=ents[1]["name"], eb=ents[2]["name"], **slots)
            question = t["question"].format(
                efirst=ents[0]["name"], elast=ents[-1]["name"], **slots)

        frag = lexicon["fragments"]
        body = " ".join(facts) + " " + question
        out.append(f"[{t['id']}] ({t['split']}, {t.get('author','generated')})\n"
                   f"  {frag['q_marker']}: {body}\n  {frag['a_marker']}: ...")
    return out


def check_language(language: str, show_samples: bool) -> int:
    doc = json.loads((HERE / f"templates_{language}.json").read_text(encoding="utf-8"))
    templates = doc["templates"]
    print(f"\n{'=' * 72}\n{language.upper()}: {len(templates)} templates\n{'=' * 72}")

    authors: dict[str, int] = {}
    for t in templates:
        authors[t.get("author", "generated")] = authors.get(t.get("author", "generated"), 0) + 1
    print("provenance:", ", ".join(f"{k}={v}" for k, v in sorted(authors.items())))
    for split in sorted(SPLITS):
        n = sum(1 for t in templates if t.get("split") == split)
        print(f"  split {split:8} {n}")

    errors = check_schema(templates) + check_placeholders(templates)
    if errors:
        print(f"\n{len(errors)} ERROR(S) — the dataset cannot be built:")
        for e in errors:
            print(f"  ✗ {e}")
        return len(errors)

    print("\nschema and placeholders: OK")
    print("checking literal words against the corpus...")
    warnings = check_literals(templates, load_val_text(language))
    if warnings:
        print(f"\n{len(warnings)} WARNING(S):")
        for w in warnings:
            print(f"  ! {w}")
    else:
        print("literals: OK — no unattested words introduced")

    if show_samples:
        print("\nrendered samples:\n")
        for s in render_samples(language, templates):
            print(s + "\n")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--language", choices=["hindi", "sindhi"], help="default: both")
    p.add_argument("--no_samples", action="store_true", help="skip the rendered examples")
    args = p.parse_args()

    failures = sum(check_language(lang, not args.no_samples)
                   for lang in ([args.language] if args.language else ["hindi", "sindhi"]))
    if failures:
        raise SystemExit(1)
    print("\nAll template checks passed.")


if __name__ == "__main__":
    main()
