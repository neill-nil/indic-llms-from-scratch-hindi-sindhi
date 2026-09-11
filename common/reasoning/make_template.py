"""
Turn a plain sentence into a reasoning template.

Writing a template by hand means knowing fifteen placeholder names and which ones are
legal where. That is a mechanical chore, and it is not the part a human is needed for.
The part a human is needed for is writing a sentence that sounds natural in the language.

So: write the sentence with real words. This converts it.

    python -m common.reasoning.make_template --language hindi \\
        --fact     "राम की उम्र 25 साल है।" \\
        --question "सबसे ज्यादा उम्र किसकी है?" \\
        --type argmax --id H-VC-user1

It prints a finished template object, ready to paste into templates_hindi.json, plus a
preview of the sentences it will produce. Anything it could not recognise is reported
loudly rather than silently left as literal text, because a literal word that should have
been a placeholder is exactly the bug that produces a dataset where every question is
about the same attribute.

Longest-match-first is used deliberately: `सबसे ज्यादा` must win over `सबसे` + `ज्यादा`,
and `से` must not match inside `सबसे`. Matching runs over whitespace tokens rather than
raw characters so that cannot happen.
"""

import argparse
import json
import logging
import re
from pathlib import Path

from common.reasoning.generate import HERE

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger("make_template")


def build_lookup(lexicon: dict, is_question: bool, family: str) -> dict[tuple, str]:
    """
    Map surface phrases to placeholder text, keyed by tuples of whitespace tokens.

    Multi-word entries exist where a single placeholder covers several words
    (`सबसे ज्यादा` is one `{sup_more}`) and where a natural phrase maps to a *sequence*
    of placeholders (`से ज्यादा` is `{than} {more}`, because that is how the existing
    templates spell it).
    """
    frag, attrs = lexicon["fragments"], lexicon["attributes"]
    table: dict[tuple, str] = {}

    def add(surface: str, placeholder: str) -> None:
        if surface:
            table[tuple(surface.split())] = placeholder

    # Comparative phrases first — longest-match handles precedence, but these are the
    # ones where getting it wrong is silent.
    add(frag.get("sup_more_generic"), "{sup_more}")
    add(frag.get("sup_less_generic"), "{sup_less}")
    add(frag.get("cmp_more_generic"), "{than} {more}")
    add(frag.get("cmp_less_generic"), "{than} {less}")

    add(frag.get("than"), "{than}")
    add(frag.get("copula"), "{cop}")
    add(frag.get("quant_more"), "{more}")
    add(frag.get("quant_less"), "{less}")
    add(frag.get("who"), "{who}")
    add(frag.get("who_poss_f"), "{who_poss}")
    add(frag.get("who_poss_m"), "{who_poss}")
    add(frag.get("equal"), "{equal}")
    add(frag.get("or"), "{or}")

    # Attribute nouns, their possessives and their units all collapse onto the same three
    # placeholders — which is the point. One template then covers every attribute.
    for spec in attrs.values():
        add(spec["noun"], "{attr}")
        add(spec["poss"], "{poss}")
        add(spec["unit"], "{unit}")

    return table


def entity_names(lexicon: dict) -> set[str]:
    return {e["name"] for kind in lexicon["entities"] for e in lexicon["entities"][kind]}


def convert(sentence: str, lexicon: dict, is_question: bool, family: str,
            question_type: str) -> tuple[str, list[str]]:
    """
    Replace every recognised word with its placeholder.

    Returns the template string and a list of words left unrecognised.
    """
    table = build_lookup(lexicon, is_question, family)
    names = entity_names(lexicon)
    punct = lexicon["punct"]
    trailing = {punct["full_stop"]: "{fs}", punct["question"]: "{qm}",
                punct["comma"]: "{comma}"}

    # Split off trailing punctuation so "है।" is seen as the word "है" plus a full stop.
    tokens, tails = [], []
    for raw in sentence.split():
        tail = ""
        while raw and raw[-1] in trailing:
            tail = trailing[raw[-1]] + tail
            raw = raw[:-1]
        tokens.append(raw)
        tails.append(tail)

    max_len = max(len(k) for k in table)
    out: list[str] = []
    unknown: list[str] = []
    entity_slots = (["{efirst}", "{elast}"] if is_question and question_type == "relation"
                    else ["{ea}", "{eb}"] if family == "chain"
                    else ["{e}"])
    seen_entities = 0

    i = 0
    while i < len(tokens):
        # Longest phrase match starting here.
        for span in range(min(max_len, len(tokens) - i), 0, -1):
            key = tuple(tokens[i:i + span])
            if key in table:
                out.append(table[key] + tails[i + span - 1])
                i += span
                break
        else:
            tok = tokens[i]
            if tok in names:
                slot = entity_slots[min(seen_entities, len(entity_slots) - 1)]
                out.append(slot + tails[i])
                seen_entities += 1
            elif re.fullmatch(r"\d+", tok):
                out.append("{v}" + tails[i])
            elif not tok:
                if tails[i]:
                    out.append(tails[i])
            else:
                unknown.append(tok)
                out.append(tok + tails[i])
            i += 1

    return " ".join(out).replace(" {fs}", "{fs}").replace(" {qm}", "{qm}"), unknown


# Fixed fillers for the --list view. Every existing template is rendered with the SAME
# names and the SAME attribute, so the only thing that differs between two lines is the
# word order — which is the whole point of reading the list.
LIST_FILLERS = {
    "hindi":  {"names": ["राम", "श्याम", "मोहन"], "attr": "age"},
    "sindhi": {"names": ["احمد", "علي", "حسن"], "attr": "age"},
}


def render_plain(language: str, template: dict) -> str:
    """
    Render one template as an ordinary sentence, with no placeholders visible.

    The template files are unreadable to anyone who does not know the placeholder
    vocabulary, which makes "do not duplicate an existing word order" an impossible
    instruction to follow. This renders them back into the language so the existing
    inventory can actually be read.
    """
    from common.reasoning.generate import Renderer

    lexicon = json.loads(
        (HERE / f"lexicon_{language}.json").read_text(encoding="utf-8"))
    fill = LIST_FILLERS[language]
    slots = Renderer(lexicon).base_slots(fill["attr"])
    a, b, c = fill["names"]

    if template["family"] == "value_compare":
        facts = [template["fact"].format(e=n, v=v, **slots)
                 for n, v in zip((a, b), (25, 30))]
    else:
        facts = [template["fact"].format(ea=a, eb=b, **slots),
                 template["fact"].format(ea=b, eb=c, **slots)]

    if template["question_type"] == "relation":
        question = template["question"].format(efirst=a, elast=c, **slots)
    else:
        question = template["question"].format(**slots)

    frag = lexicon["fragments"]
    return f"{frag['q_marker']}: {' '.join(facts)} {question}"


def list_existing(language: str) -> None:
    """Print the current template inventory as readable sentences."""
    doc = json.loads(
        (HERE / f"templates_{language}.json").read_text(encoding="utf-8"))
    templates = doc["templates"]

    print(f"\n{'=' * 74}")
    print(f"{language.upper()} — {len(templates)} templates already written.")
    print("Every line below uses the SAME names and the SAME attribute, so the only")
    print("difference between them is WORD ORDER. Write something not on this list.")
    print("=" * 74)

    for group, label in (("value_compare", "WITH NUMBERS GIVEN"),
                         ("chain", "NO NUMBERS — relations only")):
        rows = [t for t in templates if t["family"] == group]
        if not rows:
            continue
        print(f"\n--- {label} ---")
        for t in rows:
            asks = {"argmax": "asks: which is MOST",
                    "argmin": "asks: which is LEAST",
                    "relation": "asks: more or less"}[t["question_type"]]
            tag = "  [HELD OUT]" if t["split"] == "heldout" else ""
            who = t.get("author", "generated")
            print(f"\n  {render_plain(language, t)}")
            print(f"     ({asks}; by {who}{tag})")

    n_user = sum(1 for t in templates if t.get("author") == "user")
    print(f"\n{'=' * 74}")
    print(f"Yours so far: {n_user} of {len(templates)}. "
          f"Held out: {sum(1 for t in templates if t['split'] == 'heldout')}.")
    print("=" * 74)


def find_duplicate(language: str, fact: str, question: str) -> dict | None:
    """Return an existing template with the same fact and question, if any."""
    doc = json.loads(
        (HERE / f"templates_{language}.json").read_text(encoding="utf-8"))
    for t in doc["templates"]:
        if t.get("fact") == fact and t.get("question") == question:
            return t
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--language", required=True, choices=["hindi", "sindhi"])
    p.add_argument("--list", action="store_true",
                   help="show the templates that already exist, as plain sentences")
    p.add_argument("--fact", help="one statement sentence, with real words")
    p.add_argument("--question", help="the question sentence")
    p.add_argument("--type", dest="question_type",
                   choices=["argmax", "argmin", "relation"])
    p.add_argument("--family", default=None, choices=["value_compare", "chain"],
                   help="default: inferred — a fact containing a number is value_compare")
    p.add_argument("--id", default=None, help="template id; default is auto-generated")
    p.add_argument("--split", default="train", choices=["train", "heldout"])
    args = p.parse_args()

    if args.list:
        list_existing(args.language)
        return
    missing = [f"--{n}" for n, v in (("fact", args.fact), ("question", args.question),
                                     ("type", args.question_type)) if not v]
    if missing:
        p.error(f"{', '.join(missing)} required (or pass --list to see what exists)")

    lexicon = json.loads(
        (HERE / f"lexicon_{args.language}.json").read_text(encoding="utf-8"))

    # A statement that quotes a number is giving a value; one that does not is asserting
    # a relation between two entities.
    family = args.family or ("value_compare" if re.search(r"\d", args.fact) else "chain")
    if args.question_type == "relation":
        family = "chain"

    fact, unk_f = convert(args.fact, lexicon, False, family, args.question_type)
    question, unk_q = convert(args.question, lexicon, True, family, args.question_type)

    tid = args.id or f"{args.language[0].upper()}-{'VC' if family == 'value_compare' else 'CH'}-{args.question_type}-user"
    template = {"id": tid, "family": family, "question_type": args.question_type,
                "fact": fact, "question": question,
                "split": args.split, "author": "user"}
    if family == "chain":
        frag, attrs = lexicon["fragments"], lexicon["attributes"]
        template["fact_reverse"] = ("{eb} {poss} {attr} {ea} {than} {less} "
                                    "{cop}" + "{fs}")

    print("\n" + "=" * 70)
    print("PASTE THIS into the \"templates\" list of "
          f"common/reasoning/templates_{args.language}.json")
    print("=" * 70)
    print(json.dumps(template, indent=2, ensure_ascii=False))

    existing = find_duplicate(args.language, fact, question)
    if existing:
        print("\n" + "!" * 70)
        print("THIS IS A DUPLICATE — do not paste it.")
        print(f"It produces exactly the same sentences as {existing['id']!r}, which")
        print(f"already exists (written by {existing.get('author', 'generated')}):")
        print(f"\n   {render_plain(args.language, existing)}")
        print("\nDifferent names or numbers do not make a different template — only a")
        print("different WORD ORDER does. Run with --list to see what is already taken.")
        print("!" * 70)
        raise SystemExit(1)

    unknown = unk_f + unk_q
    if unknown:
        print("\n" + "-" * 70)
        print("KEPT AS LITERAL TEXT (every example will contain these exact words):")
        for w in sorted(set(unknown)):
            print(f"   {w}")
        print("\nThis is usually FINE — grammatical words like postpositions are meant")
        print("to stay constant. Two things to confirm:")
        print("  1. None of these is a name, number or attribute. Those must vary, and")
        print("     if one is listed here the lexicon does not know it.")
        print("  2. check_templates (step 2) will verify each one actually occurs in the")
        print("     corpus. If it flags one, the word needs validating before use.")
        print("-" * 70)
    else:
        print("\nAll words recognised — nothing left hard-coded.")

    print("\nNext:")
    print(f"  1. paste the object above into templates_{args.language}.json")
    print(f"  2. python -m common.reasoning.check_templates --language {args.language}")
    print(f"  3. python -m common.reasoning.generate")


if __name__ == "__main__":
    main()
