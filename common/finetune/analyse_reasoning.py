"""
Error analysis for the reasoning evaluation.

Computation only — the report is written from the JSON this produces, so the numbers can
be recomputed and checked without re-running any model.

    python -m common.finetune.analyse_reasoning

Reads every `<language>/eval/reasoning_<condition>_<shots>shot.jsonl` and joins each
prediction back to its example by split index.

What an error is, more precisely
--------------------------------
Accuracy alone says how often a model is wrong, not how. A wrong answer is classified as:

  off_answer        not a possible answer at all — neither a name from the question nor
                    a relation word. A format failure, not a reasoning failure.
  wrong_entity      a name from the question, but the wrong one.
    opposite_extreme  ...specifically the entity at the OTHER extreme: the smallest when
                      asked for the largest. The signature of having learned the
                      comparison but attached it to the wrong direction word.
  missed_tie        the values were tied but the model named an entity.
  false_tie         the model answered "equal" when one entity was the clear winner.
  relation_flip     asked "more or less?", gave the opposite.

The split matters because each points at a different defect. A model with many
`off_answer` errors has not learned the task format; one with many `opposite_extreme`
errors has learned to find extremes but not which one is being asked for.

Margin
------
For numeric comparisons the gap between the best and second-best value is binned
*relative to the attribute's range* — a 20-rupee gap on a 5,000-rupee scale is a near-tie,
while 20 years on an 80-year scale is not. A model that genuinely compares should degrade
on near-ties; a surface heuristic may not care.
"""

import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

from common.reasoning.generate import VALUE_RANGES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("analyse_reasoning")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LANGUAGES = ["hindi", "sindhi"]
SPLITS = ["test_iid", "test_entity", "test_template", "test_hop"]

MARGIN_BINS = [(0.0, 0.02, "<2%"), (0.02, 0.05, "2-5%"), (0.05, 0.15, "5-15%"),
               (0.15, 0.40, "15-40%"), (0.40, 1.01, ">40%")]


def load_lexicon(language: str) -> dict:
    return json.loads((PROJECT_ROOT / "common" / "reasoning" / f"lexicon_{language}.json")
                      .read_text(encoding="utf-8"))


def load_split(language: str, split: str) -> list[dict]:
    path = PROJECT_ROOT / language / "finetune" / "data" / f"{split}.jsonl"
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def entities_in_order(example: dict, lexicon: dict) -> list[str]:
    """
    Citation forms of the entities in an example, in order of first mention.

    Entities appear in text in their oblique form where one exists, but answers use the
    citation form, so both must be matched and the citation form returned. Names are
    matched space-padded so that one name cannot be found inside another.
    """
    text = " " + " ".join(example["facts"]) + " "
    found = []
    for kind in lexicon["entities"].values():
        for ent in kind:
            forms = {ent["name"], ent.get("oblique", ent["name"])}
            positions = [text.find(f" {f} ") for f in forms if f" {f} " in text]
            if positions:
                found.append((min(positions), ent["name"]))
    return [name for _, name in sorted(found)]


def classify(record: dict, example: dict, lexicon: dict) -> str | None:
    """Return the error category for a wrong prediction, or None if it was correct."""
    if record["correct"]:
        return None
    frag = lexicon["fragments"]
    pred, gold = record["predicted"], record["gold"]
    equal, more, less = frag["equal"], frag["quant_more"], frag["quant_less"]
    names = entities_in_order(example, lexicon)

    if record["question_type"] == "relation":
        return "relation_flip" if {pred, gold} == {more, less} else "off_answer"
    if gold == equal:
        return "missed_tie" if pred in names else "off_answer"
    if pred == equal:
        return "false_tie"
    if pred not in names:
        return "off_answer"

    # A name from the question, but not the right one. Is it the opposite extreme?
    if example["family"] == "value_compare" and len(names) == len(example["values"]):
        vals = dict(zip(names, example["values"]))
        opposite = min(vals, key=vals.get) if record["question_type"] == "argmax" \
            else max(vals, key=vals.get)
        if pred == opposite:
            return "opposite_extreme"
    return "wrong_entity"


def margin_bin(example: dict) -> str | None:
    if example["family"] != "value_compare" or example["is_tie"] or example["margin"] is None:
        return None
    lo, hi = VALUE_RANGES[example["attribute"]]
    rel = example["margin"] / (hi - lo)
    return next(label for a, b, label in MARGIN_BINS if a <= rel < b)


def chance_baselines(language: str, split: str, lexicon: dict) -> dict:
    """
    What a model scores if it knows the answer FORMAT but not the task.

    This, not zero, is the floor finetuned accuracy has to be read against. The
    pretrained models score exactly 0 because they never produce an answer at all; but a
    model that merely learns to answer with *one of the names in the question* scores
    about 36% by guessing, since most problems have two or three candidates. Measured
    during the learning-rate probe, every run sat at 33-35% after 200 steps — the format
    had been learned and the reasoning had not. Reporting accuracy against a zero floor
    would credit format learning as reasoning.

    The positional strategies test the balance controls from Chunk B: if always naming the
    first- or last-mentioned entity beat random guessing, a model could exploit position.
    """
    data = load_split(language, split)
    tot = {"random_name": 0.0, "random_name_or_equal": 0.0,
           "always_first_mentioned": 0.0, "always_last_mentioned": 0.0}
    for ex in data:
        if ex["question_type"] == "relation":
            tot["random_name"] += 0.5
            tot["random_name_or_equal"] += 0.5
            continue
        names = entities_in_order(ex, lexicon)
        n = len(names)
        hit = ex["answer"] in names
        tot["random_name"] += (1 / n) if hit else 0.0
        tot["random_name_or_equal"] += (1 / (n + 1)) if ex["family"] == "value_compare" \
            else ((1 / n) if hit else 0.0)
        tot["always_first_mentioned"] += float(names[0] == ex["answer"])
        tot["always_last_mentioned"] += float(names[-1] == ex["answer"])
    return {k: round(v / len(data), 4) for k, v in tot.items()}


def rate(values: list[bool]) -> dict:
    return {"n": len(values), "accuracy": round(sum(values) / len(values), 4) if values else None}


def controlled_in_distribution(records: list[dict], data: list[dict], lexicon: dict) -> dict:
    """
    Breakdowns on test_iid alone, with the confounds removed.

    The pooled breakdowns in `analyse` mix all four test splits, and two of them proved
    misleading when first read:

      * Pooling includes test_entity, whose failure to produce unseen names depresses
        every figure it touches.
      * "By template author" compared user templates — all numeric comparisons — against
        generated ones that include the much harder chains. The 10-point gap it showed
        was task mix; controlled for family it shrinks to 1-4 points.
      * Raw counts of which mention position a wrong answer came from favour "middle"
        simply because larger problems have more middle positions. Only a rate per
        available slot says anything about bias.

    These are the numbers to report; the pooled ones are kept for completeness.
    """
    pairs = [(r, data[r["index"]]) for r in records if r["split"] == "test_iid"]

    margin = defaultdict(list)
    task = defaultdict(list)
    author = defaultdict(list)
    slot = defaultdict(lambda: [0, 0])
    frag = lexicon["fragments"]
    relation_confusion = Counter()

    for r, ex in pairs:
        b = margin_bin(ex)
        if b:
            margin[b].append(r["correct"])
        if ex["family"] == "value_compare":
            task["numeric, tie" if ex["is_tie"] else "numeric, no tie"].append(r["correct"])
            if not ex["is_tie"]:
                author[r["template_author"]].append(r["correct"])
        else:
            task[f"chain, {ex['question_type']}"].append(r["correct"])
        if ex["question_type"] == "relation":
            relation_confusion[f"gold={r['gold']} predicted={r['predicted']}"] += 1

        names = entities_in_order(ex, lexicon)
        if r["gold"] in names:
            for i, nm in enumerate(names):
                if nm == r["gold"]:
                    continue
                key = "first" if i == 0 else "last" if i == len(names) - 1 else "middle"
                slot[key][1] += 1
                if not r["correct"] and nm == r["predicted"]:
                    slot[key][0] += 1

    return {
        "split": "test_iid",
        "accuracy_by_margin": {lab: rate(margin[lab]) for *_, lab in MARGIN_BINS if margin[lab]},
        "accuracy_by_task": {k: rate(v) for k, v in sorted(task.items())},
        "accuracy_by_template_author_numeric_no_tie": {k: rate(v) for k, v in author.items()},
        "wrong_pick_rate_per_slot": {k: round(a / b, 4) for k, (a, b) in slot.items() if b},
        "relation_confusion": dict(relation_confusion),
    }


def analyse(language: str, condition_file: Path, seed: int = 0) -> dict:
    lexicon = load_lexicon(language)
    records = [json.loads(line) for line in condition_file.open(encoding="utf-8")]
    splits = {s: load_split(language, s) for s in {r["split"] for r in records}}

    by_split = defaultdict(list)
    by_margin = defaultdict(list)
    by_hops = defaultdict(list)
    by_author = defaultdict(list)
    by_template = defaultdict(list)
    errors = Counter()
    errors_by_split = defaultdict(Counter)
    wrong_entity_position = Counter()
    successes, failures = [], []

    for r in records:
        ex = splits[r["split"]][r["index"]]
        assert ex["answer"] == r["gold"], \
            f"record/example mismatch at {r['split']}[{r['index']}]: the dataset changed " \
            f"since this evaluation was run — regenerate predictions"
        ok = r["correct"]
        by_split[r["split"]].append(ok)
        by_author[r["template_author"]].append(ok)
        by_template[r["template_id"]].append(ok)
        by_hops[f"{ex['family']}:{ex['n_entities']}"].append(ok)
        b = margin_bin(ex)
        if b:
            by_margin[b].append(ok)

        cat = classify(r, ex, lexicon)
        if cat:
            errors[cat] += 1
            errors_by_split[r["split"]][cat] += 1
            if cat in ("wrong_entity", "opposite_extreme"):
                names = entities_in_order(ex, lexicon)
                if r["predicted"] in names:
                    pos = names.index(r["predicted"])
                    wrong_entity_position["first" if pos == 0 else
                                          "last" if pos == len(names) - 1 else "middle"] += 1
            failures.append((r, ex, cat))
        else:
            successes.append((r, ex, None))

    def sample(pool, k):
        rng = random.Random(seed)
        picked = rng.sample(pool, min(k, len(pool)))
        return [{"split": r["split"], "template_id": r["template_id"],
                 "text": ex["prompt"], "gold": r["gold"], "predicted": r["predicted"],
                 "error": cat} for r, ex, cat in picked]

    n_wrong = sum(errors.values())
    return {
        "language": language,
        "file": str(condition_file.relative_to(PROJECT_ROOT)),
        "n": len(records),
        "accuracy_by_split": {s: rate(by_split[s]) for s in SPLITS if s in by_split},
        "chance_by_split": {s: chance_baselines(language, s, lexicon)
                            for s in SPLITS if s in by_split},
        "accuracy_by_margin": {label: rate(by_margin[label])
                               for _, _, label in MARGIN_BINS if label in by_margin},
        "accuracy_by_family_and_size": {k: rate(v) for k, v in sorted(by_hops.items())},
        # POOLED and confounded by task family — see controlled_in_distribution.
        "accuracy_by_template_author_POOLED": {k: rate(v) for k, v in by_author.items()},
        "accuracy_by_template": {k: rate(v) for k, v in sorted(by_template.items())},
        "errors": {"total": n_wrong,
                   "by_category": {k: {"n": v, "share": round(v / n_wrong, 4)}
                                   for k, v in errors.most_common()},
                   "by_split": {s: dict(c) for s, c in errors_by_split.items()}},
        # Raw counts, not rates — larger problems have more middle slots. See
        # controlled_in_distribution for the normalised figure.
        "wrong_entity_mention_position_RAW_COUNTS": dict(wrong_entity_position),
        "controlled_in_distribution": (
            controlled_in_distribution(records, splits["test_iid"], lexicon)
            if "test_iid" in splits else None),
        "qualitative": {"successes": sample(successes, 6), "failures": sample(failures, 10)},
    }


def main():
    out = {}
    for language in LANGUAGES:
        for f in sorted((PROJECT_ROOT / language / "eval").glob("reasoning_*shot.jsonl")):
            key = f"{language}/{f.stem.removeprefix('reasoning_')}"
            out[key] = analyse(language, f)
            a = out[key]
            accs = "  ".join(
                f"{s.replace('test_', '')} {v['accuracy']:.3f} "
                f"(chance {a['chance_by_split'][s]['random_name']:.3f})"
                for s, v in a["accuracy_by_split"].items())
            logger.info("%-28s n=%-5d %s", key, a["n"], accs)
            if a["errors"]["total"]:
                cats = ", ".join(f"{k} {v['share']:.0%}"
                                 for k, v in a["errors"]["by_category"].items())
                logger.info("%-28s errors: %s", "", cats)

    path = PROJECT_ROOT / "report" / "reasoning_analysis.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
