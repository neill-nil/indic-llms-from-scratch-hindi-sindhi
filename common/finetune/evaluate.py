"""
Reasoning evaluation: greedy-decode an answer and score it by exact match.

This is the measurement Phase 3 reports, and it is deliberately built *before* any model
is finetuned. A metric written after seeing results is a metric that has been tuned to
them, however carefully one tries not to.

    # pretrained baselines (this is the point of running it now)
    python -m common.finetune.evaluate --checkpoint pretrained --shots 0
    python -m common.finetune.evaluate --checkpoint pretrained --shots 4

    # after finetuning
    python -m common.finetune.evaluate --checkpoint finetuned

Why exact match rather than an overlap metric
---------------------------------------------
The answer is a single entity name or a single relation word, and it is either the
correct one or it is not. There is no partial credit to award and no paraphrase to
tolerate, so the overlap metrics that Phase 2 needed for open-ended continuation would
add noise here rather than information.

What "few-shot" is doing here
-----------------------------
A pretrained model has never seen the `प्रश्न:`/`उत्तर:` format and has no reason to stop
after emitting one name. Scoring it zero-shot measures the format barrier as much as the
reasoning. The 4-shot condition prepends worked examples so the format is demonstrated,
which separates "cannot do the task" from "does not know what is being asked". Both are
reported, because the gap between them is itself the finding.

The shot examples are drawn from the TRAIN split, never from the split being scored, and
are fixed by seed so every model sees the same ones.
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch

from common.finetune.dataset import DOC_SEPARATOR, load_reasoning_split
from common.model.config import GPTConfig
from common.model.gpt import GPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("finetune.evaluate")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

TEST_SPLITS = ["test_iid", "test_entity", "test_template", "test_hop"]

# Answers are 1-5 tokens. 12 is generous enough that a correct answer is never cut off,
# and small enough that a rambling pretrained model does not cost minutes per batch.
MAX_NEW_TOKENS = 12


def resolve_checkpoint(language: str, which: str) -> Path:
    """`pretrained` and `finetuned` are the two named runs; anything else is a path."""
    if which == "pretrained":
        return PROJECT_ROOT / language / "train" / "run" / "best.pt"
    if which == "finetuned_fullseq":
        return PROJECT_ROOT / language / "finetune" / "run_fullseq" / "best.pt"
    if which == "finetuned":
        return PROJECT_ROOT / language / "finetune" / "run" / "best.pt"
    return Path(which)


def load_model(path: Path, device: str) -> GPT:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def build_shot_prefix(language: str, n_shots: int, seed: int = 0) -> str:
    """
    Worked examples prepended to every prompt, drawn from the train split.

    Separated by the document separator the model already knows from pretraining, so the
    boundary between examples is marked by a token whose meaning it has actually learned
    rather than by an arbitrary convention introduced at test time.
    """
    if n_shots == 0:
        return ""
    train = load_reasoning_split(language, "train", limit=2000)
    rng = random.Random(seed)
    picked = rng.sample(train.examples, n_shots)
    return "".join(f"{ex['text']}{DOC_SEPARATOR}" for ex in picked)


def normalise(text: str) -> str:
    """
    Reduce a decoded answer to the span that should be compared.

    The model emits the answer and then, ideally, the document separator. A pretrained
    model usually keeps going instead, so the answer is taken as everything up to the
    first separator or line break, stripped of whitespace and trailing punctuation.
    Cutting at the newline matters: without it a fluent continuation onto a second
    sentence would be judged wrong even when its first word was right.
    """
    for stop in (DOC_SEPARATOR, "\n"):
        if stop in text:
            text = text.split(stop)[0]
    return text.strip().strip("।.?؟،,!:")


@torch.no_grad()
def evaluate_split(model: GPT, language: str, split: str, shot_prefix: str,
                   device: str, batch_size: int, limit: Optional[int]) -> dict:
    """Greedy-decode every example in one split and score exact match."""
    data = load_reasoning_split(language, split, limit=limit)
    sp = data.sp
    eos_id = sp.piece_to_id(DOC_SEPARATOR)
    prefix_ids = sp.encode(shot_prefix) if shot_prefix else []

    # Group by EXACT encoded length so a batch needs no padding at all.
    #
    # Padding a generation batch is not safe for this model. Right padding puts the pad
    # tokens after the prompt, so generation would continue from a pad rather than from
    # the real final token. Left padding avoids that but breaks two other things: the
    # model uses LEARNED ABSOLUTE position embeddings, so shifting a prompt rightward
    # gives every token a position it was never trained at, and under a causal mask the
    # real tokens can attend back to the pads. Equal-length buckets sidestep all of it —
    # the prompts are short and their lengths cluster tightly, so buckets are large.
    buckets: dict[int, list] = defaultdict(list)
    for i, ex in enumerate(data.examples):
        ids = prefix_ids + sp.encode(ex["prompt"])
        buckets[len(ids)].append(({**ex, "_index": i}, ids))

    records = []
    for length, items in sorted(buckets.items()):
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            idx = torch.tensor([ids for _, ids in chunk], dtype=torch.long, device=device)

            out = model.generate(idx, max_new_tokens=MAX_NEW_TOKENS,
                                 temperature=0.0, eos_id=eos_id)
            for (ex, _), row in zip(chunk, out[:, length:].tolist()):
                prediction = normalise(sp.decode(row))
                records.append({
                    "predicted": prediction,
                    "gold": ex["answer"],
                    "correct": prediction == ex["answer"],
                    "family": ex["family"],
                    "question_type": ex["question_type"],
                    "n_entities": ex["n_entities"],
                    "template_id": ex["template_id"],
                    "template_author": ex["template_author"],
                    "is_tie": ex["is_tie"],
                    "margin": ex.get("margin"),
                    "index": ex["_index"],
                })

    # Restore split order: bucketing by length scrambled it.
    records.sort(key=lambda r: r["index"])
    return summarise(split, records), records


def summarise(split: str, records: list[dict]) -> dict:
    """Overall accuracy plus the breakdowns the report needs."""
    n = len(records)
    acc = sum(r["correct"] for r in records) / max(1, n)

    def by(key):
        groups = defaultdict(list)
        for r in records:
            groups[r[key]].append(r["correct"])
        return {str(k): {"n": len(v), "accuracy": round(sum(v) / len(v), 4)}
                for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}

    # How often the model produced a string that is not even a candidate answer — a
    # different failure from picking the wrong candidate, and the one that dominates
    # before finetuning.
    valid = {r["gold"] for r in records}
    off_vocab = sum(1 for r in records if r["predicted"] not in valid) / max(1, n)

    return {
        "split": split,
        "n": n,
        "accuracy": round(acc, 4),
        "off_answer_rate": round(off_vocab, 4),
        "by_family": by("family"),
        "by_question_type": by("question_type"),
        "by_n_entities": by("n_entities"),
        "by_template_author": by("template_author"),
        "tie_accuracy": round(
            sum(r["correct"] for r in records if r["is_tie"])
            / max(1, sum(1 for r in records if r["is_tie"])), 4),
        "examples": records[:8],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--language", choices=["hindi", "sindhi"], help="default: both")
    p.add_argument("--checkpoint", default="finetuned",
                   help="'pretrained', 'finetuned', or a path")
    p.add_argument("--shots", type=int, default=0,
                   help="in-context examples prepended to every prompt")
    p.add_argument("--splits", nargs="+", default=TEST_SPLITS)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None,
                   help="evaluate only the first N examples of each split")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    languages = [args.language] if args.language else ["hindi", "sindhi"]
    results = []
    for language in languages:
        path = resolve_checkpoint(language, args.checkpoint)
        if not path.exists():
            logger.error("%s not found — skipping %s", path, language)
            continue
        model = load_model(path, args.device)
        prefix = build_shot_prefix(language, args.shots, args.seed)
        logger.info("%s | %s | %d-shot", language, path, args.shots)

        all_records: list[dict] = []
        # Relative to the repository, so results files do not carry this machine's home
        # directory around and mean the same thing on a grader's checkout.
        rel = path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path
        entry = {"language": language, "checkpoint": str(rel),
                 "condition": args.checkpoint, "shots": args.shots, "splits": {}}
        for split in args.splits:
            s, records = evaluate_split(model, language, split, prefix,
                                        args.device, args.batch_size, args.limit)
            entry["splits"][split] = s
            all_records.extend({"split": split, **r} for r in records)
            logger.info("  %-14s n=%-5d accuracy %.4f | off-answer %.4f",
                        split, s["n"], s["accuracy"], s["off_answer_rate"])
        results.append(entry)

        # The summary JSON keeps only a handful of examples per split. Error analysis
        # needs every prediction, so all of them are written here, one per line. Each
        # record carries its index into the split, so the prompt can be recovered from
        # the (regenerable) dataset rather than stored a second time.
        condition = args.checkpoint if args.checkpoint in ("pretrained", "finetuned",
                                                           "finetuned_fullseq") \
            else Path(args.checkpoint).parent.name
        rec_path = (PROJECT_ROOT / language / "eval"
                    / f"reasoning_{condition}_{args.shots}shot.jsonl")
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rec_path, "w", encoding="utf-8") as f:
            for r in all_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("  wrote %d predictions to %s", len(all_records), rec_path)

    out = Path(args.out) if args.out else (
        PROJECT_ROOT / "report" /
        f"reasoning_{args.checkpoint}_{args.shots}shot.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
