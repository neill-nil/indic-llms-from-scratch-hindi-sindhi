"""
Generate the synthetic comparative-reasoning corpora.

Programmatic and seeded, so the datasets are reproducible from this script alone — the
brief requires the generation scripts to be released for exactly that reason.

    python -m common.reasoning.generate                    # both languages
    python -m common.reasoning.generate --language sindhi

Design decisions that are not obvious from the code
---------------------------------------------------

*Chain links are presented in shuffled order.* A chain example states A>B, B>C, ... If
those links were always emitted in order, the answer to "which is largest" would always
be the first entity named, and a model could score perfectly by copying the first name it
sees without representing the relation at all. Shuffling the presentation forces the
model to assemble the ordering from links that arrive out of sequence, which is the part
of the task that is actually transitive reasoning.

*Ties are deliberately injected.* Without them a model can do well by always naming some
entity; it never has to represent equality. A fixed fraction of value comparisons are
generated with the top value shared, and their answer is the equality word.

*Values are drawn to control the margin.* The gap between the best and second-best value
is recorded on every example, so accuracy can be reported against margin. A model that
has genuinely learned to compare should degrade on near-ties; one that has learned a
surface heuristic may not.

*The task shape is identical across the two languages.* Only the surface strings differ.
Phase 3's claim is a comparison between Model H and Model L, so the datasets must differ
in language and nothing else.
"""

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("generate")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent

# Plausible ranges per attribute, so the numbers read as real quantities rather than as
# arbitrary integers. Values are integers in both scripts' ordinary ASCII digits, which
# is how both corpora write numbers (verified in the Phase 2 generation samples).
VALUE_RANGES = {
    "age":        (5, 80),
    "height":     (100, 200),
    "weight":     (1, 100),
    "price":      (10, 5000),
    "population": (1, 300),
}

# `applies_to` names entity kinds in the singular; the lexicon groups them in the plural.
# An earlier version bridged the two with rstrip("s"), which silently mapped "people" to
# "people" and never matched "person" — so every person entity was rejected as having no
# applicable attribute. An explicit table cannot fail that way.
KIND_TO_APPLIES = {"people": "person", "places": "place", "objects": "object"}

# How many entities a problem mentions. The held-out `test_hop` split uses strictly more
# than anything seen in training, which is the compositional generalisation probe.
TRAIN_SIZES = (2, 3)
HOP_TEST_SIZES = (4, 5)

TIE_FRACTION = 0.15

SPLIT_PLAN = {
    #  name             n_examples  entity_pool  template_pool  sizes
    "train":          (40_000, "train",   "train",   TRAIN_SIZES),
    "val":            ( 2_000, "train",   "train",   TRAIN_SIZES),
    "test_iid":       ( 2_000, "train",   "train",   TRAIN_SIZES),
    "test_entity":    ( 2_000, "heldout", "train",   TRAIN_SIZES),
    "test_template":  ( 2_000, "train",   "heldout", TRAIN_SIZES),
    "test_hop":       ( 2_000, "train",   "train",   HOP_TEST_SIZES),
}


class Renderer:
    """Resolves a template's placeholders against one language's validated lexicon."""

    def __init__(self, lexicon: dict):
        self.lex = lexicon
        self.frag = lexicon["fragments"]
        self.punct = lexicon["punct"]

    def base_slots(self, attr_key: str) -> dict:
        """Placeholder values fixed by the attribute, independent of the entities."""
        attr = self.lex["attributes"][attr_key]
        # The interrogative possessive agrees with the ATTRIBUTE noun's gender
        # ("किसकी उम्र" but "किसका वजन"), not with the entity being asked about. Getting
        # this from the entity would produce agreement errors on three of four attributes.
        who_poss = self.frag[f"who_poss_{attr['noun_gender']}"]
        return {
            "attr": attr["noun"], "poss": attr["poss"], "unit": attr["unit"],
            "cop": self.frag["copula"], "than": self.frag["than"],
            "more": self.frag["quant_more"], "less": self.frag["quant_less"],
            "sup_more": self.frag["sup_more_generic"],
            "sup_less": self.frag["sup_less_generic"],
            "who": self.frag["who"], "who_poss": who_poss,
            "or": self.frag["or"], "equal": self.frag["equal"],
            "fs": self.punct["full_stop"], "qm": self.punct["question"],
        }


# Entity mentions inside a rendered example are ALWAYS followed by a postposition — a
# possessive, the comparison particle, or `में`/`۾` — so they take the oblique form where
# the corpus says a name inflects. The answer is a bare name governed by nothing, so it
# keeps the citation form. See resolve_oblique_forms in validate_lexicon.py.
def in_text(ent: dict) -> str:
    """The form of an entity name as it appears inside a sentence."""
    return ent.get("oblique", ent["name"])


def format_entity_list(names: list[str], lexicon: dict) -> str:
    """
    Join entity names the way the language does: "राम और श्याम", "राम, श्याम और मोहन".

    This fills the `{entities}` placeholder, which lets a question name its candidates
    ("of Ram and Shyam, whose age is greater?") instead of only referring back to them
    implicitly. It is the most natural-sounding question shape available and none of the
    original templates could express it, because entity names were only ever substituted
    into `fact`.

    The names are listed in the order they were MENTIONED in the facts. Listing them in
    the underlying ranking order would leak the answer outright.
    """
    if len(names) == 1:
        return names[0]
    joiner = f" {lexicon['fragments']['and']} "
    if len(names) == 2:
        return joiner.join(names)
    comma = lexicon["punct"]["comma"] + " "
    return comma.join(names[:-1]) + joiner + names[-1]


def applicable_attributes(lexicon: dict, entity_kind: str) -> list[str]:
    """Attributes that make sense for this kind of entity — a book has no age."""
    want = KIND_TO_APPLIES[entity_kind]
    return [k for k, spec in lexicon["attributes"].items()
            if want in spec["applies_to"]]


def sample_values(attr_key: str, n: int, tie: bool, rng: random.Random) -> list[int]:
    """
    Draw n distinct values for an attribute, optionally forcing a tie at the extreme.

    Distinctness matters everywhere except the deliberate tie: an accidental duplicate
    would make an argmax question ambiguous, and the example would be scored against an
    answer that is not uniquely correct.
    """
    lo, hi = VALUE_RANGES[attr_key]
    values = rng.sample(range(lo, hi + 1), n)
    if tie:
        # Duplicate the maximum onto a second entity so the correct answer becomes the
        # equality word rather than a name.
        top = max(values)
        others = [i for i, v in enumerate(values) if v != top]
        values[rng.choice(others)] = top
    return values


def make_value_example(lexicon: dict, rend: Renderer, template: dict,
                       entities: list[dict], attr_key: str, tie: bool,
                       rng: random.Random) -> dict:
    """One 'here are the values, which is biggest' problem."""
    n = len(entities)
    values = sample_values(attr_key, n, tie, rng)
    slots = rend.base_slots(attr_key)

    facts = [template["fact"].format(e=in_text(ent), v=v, **slots)
             for ent, v in zip(entities, values)]

    want_max = template["question_type"] == "argmax"
    target = max(values) if want_max else min(values)
    winners = [ent for ent, v in zip(entities, values) if v == target]
    answer = winners[0]["name"] if len(winners) == 1 else slots["equal"]

    ordered = sorted(values, reverse=want_max)
    margin = abs(ordered[0] - ordered[1]) if n > 1 else 0

    return {
        "facts": facts,
        "question": template["question"].format(
            entities=format_entity_list([in_text(e) for e in entities], lexicon), **slots),
        "answer": answer,
        "family": "value_compare",
        "question_type": template["question_type"],
        "n_entities": n, "attribute": attr_key, "values": values,
        "margin": margin, "is_tie": len(winners) > 1,
        "answer_position": (None if len(winners) > 1
                            else entities.index(winners[0])),
    }


def make_chain_example(lexicon: dict, rend: Renderer, template: dict,
                       entities: list[dict], attr_key: str,
                       rng: random.Random) -> dict:
    """One 'A > B, B > C, so which is largest' problem, with links shuffled."""
    order = list(entities)
    rng.shuffle(order)                       # the true ranking, largest first
    slots = rend.base_slots(attr_key)

    links = [(order[i], order[i + 1]) for i in range(len(order) - 1)]
    presentation = list(links)
    rng.shuffle(presentation)                # see module docstring

    # Each link is stated from a randomly chosen end: "A > B" or the equivalent "B < A".
    # Shuffling link ORDER alone was not enough — within a link the larger entity was
    # still always named first, so the winner of an argmax question was systematically
    # the earliest-mentioned name. Measured on the first build, that put 58.6% of
    # `test_template` answers at position 0, and made two-entity chains solvable by
    # copying the first name outright.
    facts = []
    for a, b in presentation:
        if "fact_reverse" in template and rng.random() < 0.5:
            facts.append(template["fact_reverse"].format(
                ea=in_text(a), eb=in_text(b), **slots))
        else:
            facts.append(template["fact"].format(
                ea=in_text(a), eb=in_text(b), **slots))

    # First-mention order, read off the rendered text. Needed by every branch below:
    # by `{entities}` so the question lists names in the order the reader met them, and
    # by `answer_position` so the balance statistic measures what a positional shortcut
    # would actually key on. A reversed link names its two entities in the opposite order
    # from `presentation`, so the presentation list is not a reliable guide. Names are
    # searched space-padded so one name cannot match inside another.
    padded = " " + " ".join(facts) + " "
    mentioned = sorted((in_text(ent) for ent in order),
                       key=lambda nm: padded.find(f" {nm} "))
    entity_list = format_entity_list(mentioned, lexicon)

    qtype = template["question_type"]
    if qtype == "argmax":
        answer = order[0]["name"]
    elif qtype == "argmin":
        answer = order[-1]["name"]
    else:                                     # relation between the two endpoints
        # The question names the endpoints in a random order, so "more" and "less" are
        # equally likely answers. Always asking about (largest, smallest) in that order
        # would make "more" correct every time.
        first, last = (order[0], order[-1]) if rng.random() < 0.5 else (order[-1], order[0])
        answer = slots["more"] if first is order[0] else slots["less"]
        question = template["question"].format(
            efirst=in_text(first), elast=in_text(last), entities=entity_list, **slots)
        return {
            "facts": facts, "question": question, "answer": answer,
            "family": "chain", "question_type": qtype,
            "n_entities": len(order), "n_hops": len(links), "attribute": attr_key,
            "margin": None, "is_tie": False, "answer_position": None,
        }

    winner = order[0] if qtype == "argmax" else order[-1]

    return {
        "facts": facts,
        "question": template["question"].format(entities=entity_list, **slots),
        "answer": answer, "family": "chain", "question_type": qtype,
        "n_entities": len(order), "n_hops": len(links), "attribute": attr_key,
        "margin": None, "is_tie": False,
        "answer_position": mentioned.index(in_text(winner)),
    }


def render_example(lexicon: dict, example: dict) -> dict:
    """Wrap facts and question into the prompt/answer pair the trainer consumes."""
    frag = lexicon["fragments"]
    body = " ".join(example["facts"]) + " " + example["question"]
    prompt = f"{frag['q_marker']}: {body}\n{frag['a_marker']}:"
    example["prompt"] = prompt
    example["text"] = f"{prompt} {example['answer']}"
    return example


def generate_split(lexicon: dict, templates: list[dict], name: str, n: int,
                   entity_pool: str, template_pool: str, sizes: tuple,
                   rng: random.Random, seen: set[str]) -> list[dict]:
    """
    Build one split under its own entity and template restrictions.

    `seen` carries every example text emitted so far, across ALL splits, and any repeat
    is discarded. This makes train/val/test disjointness a property of construction
    rather than something checked afterwards.

    It is needed because the problem space is finite and much smaller than it looks. A
    first version relied on the audit alone and found 323 of 2,000 validation examples
    already present verbatim in train — roughly the birthday-collision rate for 42,000
    draws from an effective space of ~250k, which is what a handful of templates over
    ~50 entities and a bounded value range actually spans. Those collisions are not a
    bug in the sampler; they are what sampling with replacement does. Deduplicating
    globally is the fix, and the residual duplicate rate is reported as a statistic
    because it says something real about how much variety the design supports.
    """
    rend = Renderer(lexicon)
    pool_templates = [t for t in templates if t["split"] == template_pool]
    if not pool_templates:
        raise ValueError(f"no templates in pool {template_pool!r}")

    # Only kinds that have at least one applicable attribute AND enough entities in the
    # requested pool. Filtering here rather than retrying inside the loop means an
    # unusable kind is a visible warning at startup, not an invisible infinite retry.
    kinds = [k for k, v in lexicon["entities"].items()
             if v and applicable_attributes(lexicon, k)]
    for k, v in lexicon["entities"].items():
        if v and not applicable_attributes(lexicon, k):
            logger.warning("no attribute applies to %r — those entities are unused", k)
    if not kinds:
        raise ValueError("no usable entity kinds")
    out = []
    attempts = 0
    # A generous cap. Hitting it means the requested split is a large fraction of the
    # space that its pool can express, which is a design problem to surface loudly
    # rather than a condition to spin on.
    max_attempts = n * 200
    while len(out) < n:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"{name}: only {len(out)}/{n} unique examples after {attempts:,} "
                f"attempts. The {entity_pool} entity pool and {template_pool} template "
                f"pool cannot express enough distinct problems; add templates or "
                f"entities, or reduce the split size.")
        template = rng.choice(pool_templates)
        kind = rng.choice(kinds)
        available = [e for e in lexicon["entities"][kind] if e["split"] == entity_pool]
        size = rng.choice(sizes)
        if len(available) < size:
            continue
        attr_key = rng.choice(applicable_attributes(lexicon, kind))
        entities = rng.sample(available, size)

        # A two-entity "chain" has a single link and requires no composition at all —
        # the answer is stated outright. Chains therefore start at three entities.
        if template["family"] == "chain" and size < 3:
            continue

        if template["family"] == "value_compare":
            ex = make_value_example(lexicon, rend, template, entities, attr_key,
                                    tie=rng.random() < TIE_FRACTION, rng=rng)
        else:
            ex = make_chain_example(lexicon, rend, template, entities, attr_key, rng)

        ex["template_id"] = template["id"]
        ex["template_author"] = template.get("author", "generated")
        ex["entity_kind"] = kind
        ex["split"] = name
        ex = render_example(lexicon, ex)
        if ex["text"] in seen:
            continue
        seen.add(ex["text"])
        out.append(ex)
    logger.info("%s: %-14s %6d examples (%.1f%% of draws were duplicates)",
                lexicon["language"], name, len(out),
                100 * (attempts - len(out)) / max(1, attempts))
    return out


def audit(splits: dict[str, list[dict]], lexicon: dict) -> dict:
    """
    Check the guarantees the design claims, rather than assuming the code delivered them.

    Every one of these has a way of silently failing: a shuffle that does not shuffle, a
    pool filter with an off-by-one, a template that slips into the wrong split. They are
    asserted here so a violation stops the build instead of quietly weakening a result.
    """
    report = {}
    train_texts = {e["text"] for e in splits["train"]}

    for name, rows in splits.items():
        if name == "train":
            continue
        overlap = sum(1 for e in rows if e["text"] in train_texts)
        report[f"{name}.exact_overlap_with_train"] = overlap
        assert overlap == 0, f"{name}: {overlap} examples appear verbatim in train"

    heldout_names = set()
    for kind in lexicon["entities"]:
        for e in lexicon["entities"][kind]:
            if e["split"] == "heldout":
                # Both forms: a held-out name that inflects appears in text as its
                # oblique form, so checking the citation form alone would miss it.
                heldout_names.add(e["name"])
                heldout_names.add(e.get("oblique", e["name"]))
    leaked = [e for e in splits["train"]
              if any(f" {h} " in f" {e['text']} " for h in heldout_names)]
    report["train.heldout_entity_leaks"] = len(leaked)
    assert not leaked, f"{len(leaked)} train examples use a held-out entity name"

    train_templates = {e["template_id"] for e in splits["train"]}
    tt_templates = {e["template_id"] for e in splits["test_template"]}
    report["test_template.shared_templates_with_train"] = len(train_templates & tt_templates)
    assert not (train_templates & tt_templates), "test_template reuses a train template"

    train_hops = {e.get("n_entities") for e in splits["train"]}
    hop_hops = {e.get("n_entities") for e in splits["test_hop"]}
    report["test_hop.shared_sizes_with_train"] = len(train_hops & hop_hops)
    assert not (train_hops & hop_hops), "test_hop reuses a training problem size"

    return report


def summarise(rows: list[dict]) -> dict:
    """Per-split distribution stats, used to verify the balance controls actually held."""
    answers = Counter(e["answer"] for e in rows)
    positions = Counter(e["answer_position"] for e in rows
                        if e["answer_position"] is not None)
    total_pos = sum(positions.values()) or 1
    return {
        "n": len(rows),
        "families": dict(Counter(e["family"] for e in rows)),
        "question_types": dict(Counter(e["question_type"] for e in rows)),
        "attributes": dict(Counter(e["attribute"] for e in rows)),
        "template_authors": dict(Counter(e["template_author"] for e in rows)),
        "tie_rate": round(sum(e["is_tie"] for e in rows) / max(1, len(rows)), 4),
        "distinct_answers": len(answers),
        "most_common_answer_share": round(answers.most_common(1)[0][1] / max(1, len(rows)), 4),
        "answer_position_distribution": {
            str(k): round(v / total_pos, 3) for k, v in sorted(positions.items())},
        "mean_prompt_chars": round(sum(len(e["prompt"]) for e in rows) / max(1, len(rows)), 1),
    }


def build(language: str, seed: int) -> dict:
    lexicon = json.loads((HERE / f"lexicon_{language}.json").read_text(encoding="utf-8"))
    tpl_doc = json.loads((HERE / f"templates_{language}.json").read_text(encoding="utf-8"))
    templates = tpl_doc["templates"]

    rng = random.Random(seed)
    splits = {}
    seen: set[str] = set()
    for name, (n, ent_pool, tpl_pool, sizes) in SPLIT_PLAN.items():
        splits[name] = generate_split(lexicon, templates, name, n,
                                      ent_pool, tpl_pool, sizes, rng, seen)

    checks = audit(splits, lexicon)
    logger.info("%s: leakage audit passed (%s)", language, checks)

    out_dir = PROJECT_ROOT / language / "finetune" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for e in rows:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    stats = {
        "language": language, "seed": seed,
        "n_templates": len(templates),
        "n_templates_by_author": dict(Counter(t.get("author", "generated")
                                              for t in templates)),
        "audit": checks,
        "splits": {name: summarise(rows) for name, rows in splits.items()},
    }
    (out_dir / "dataset_statistics.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("%s: wrote %s", language, out_dir)
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--language", choices=["hindi", "sindhi"], help="default: both")
    p.add_argument("--seed", type=int, default=20260910)
    args = p.parse_args()

    for language in ([args.language] if args.language else ["hindi", "sindhi"]):
        stats = build(language, args.seed)
        for name, s in stats["splits"].items():
            logger.info("  %-14s n=%-6d ties=%.2f  top-answer share=%.3f  pos=%s",
                        name, s["n"], s["tie_rate"], s["most_common_answer_share"],
                        s["answer_position_distribution"])


if __name__ == "__main__":
    main()
