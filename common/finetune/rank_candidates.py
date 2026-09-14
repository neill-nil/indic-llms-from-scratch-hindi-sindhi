"""
Forced-choice evaluation: does the model KNOW the answer, even when it cannot SAY it?

    python -m common.finetune.rank_candidates --checkpoint finetuned
    python -m common.finetune.rank_candidates --checkpoint pretrained

Greedy generation conflates two separate abilities — deciding which answer is right, and
producing that answer's tokens. This evaluation removes the second. For each question it
scores every *possible* answer (each entity named in the question, plus "equal" for a
numeric comparison, or "more"/"less" for a relation question) by the log-probability the
model assigns to it after the prompt, and takes the highest.

Why this exists
---------------
Finetuned greedy accuracy on held-out entity names collapsed to 3.9% on questions whose
answer is a name, against 85.3% for training names — while held-out phrasings (80%) and
longer problems (48%, chance 23%) generalised well. The wrong answers were mostly the
fallback "equal" or a *training* name absent from the question, almost never a wrong name
from the question. Tokenization was ruled out: train and held-out names are both about one
token each.

The likely mechanism is the output layer. Every finetuning step raises the logit of the
correct answer token and lowers all others, so the tokens of held-out names were pushed
down on every step and never raised. A model can then attend to the right entity and still
be unable to emit it, because its output prior says that token is never an answer.

Ranking only the candidates that appear in the question cancels that prior — every
held-out candidate is suppressed alike — and exposes whether the context still favours
the right one. If forced-choice accuracy on held-out names recovers, the comparison
transferred and only answer production failed. If it does not, the model did not know.

Scoring details
---------------
A candidate's score is the summed log-probability of its tokens plus the end-of-answer
separator. Including the separator rewards a *complete* answer rather than a prefix of a
longer one. Names are almost all a single token, so length bias between candidates is
small, but it is not zero, and mean-per-token scoring is reported alongside as a check.

Candidate sequences are right-padded to a common length within a batch. That is safe for
scoring for the same reason it is safe for training: under the causal mask a padded
position can only affect positions after it, and scores are read only at real positions.
Right padding also leaves every real token at its trained position.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from common.finetune.analyse_reasoning import entities_in_order, load_lexicon, load_split
from common.finetune.dataset import DOC_SEPARATOR
from common.finetune.evaluate import TEST_SPLITS, load_model, resolve_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rank_candidates")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def candidates_for(example: dict, lexicon: dict) -> list[str]:
    """Every answer the question makes possible."""
    frag = lexicon["fragments"]
    if example["question_type"] == "relation":
        return [frag["quant_more"], frag["quant_less"]]
    names = entities_in_order(example, lexicon)
    if example["family"] == "value_compare":
        return names + [frag["equal"]]
    return names


@torch.no_grad()
def score_batch(model, sequences: list[list[int]], answer_starts: list[int],
                pad_id: int, device: str) -> tuple[list[float], list[float]]:
    """Summed and mean log-probability of each sequence's answer span."""
    width = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    for i, s in enumerate(sequences):
        ids[i, :len(s)] = torch.tensor(s)
    ids = ids.to(device)

    logits, _ = model(ids)
    logp = F.log_softmax(logits.float(), dim=-1)

    sums, means = [], []
    for i, s in enumerate(sequences):
        start = answer_starts[i]
        # logits at position t predict token t+1
        positions = torch.arange(start - 1, len(s) - 1, device=device)
        targets = ids[i, start:len(s)]
        tok_lp = logp[i, positions, targets]
        sums.append(float(tok_lp.sum()))
        means.append(float(tok_lp.mean()))
    return sums, means


def rank_split(model, language: str, split: str, device: str,
               batch_size: int, limit: int | None) -> dict:
    import sentencepiece as spm
    lexicon = load_lexicon(language)
    sp = spm.SentencePieceProcessor(
        model_file=str(PROJECT_ROOT / language / "tokenizer"
                       / f"{language}_unigram_16000.model"))
    eos_id = sp.piece_to_id(DOC_SEPARATOR)
    pool = {e["name"] for k in lexicon["entities"].values() for e in k}

    data = load_split(language, split)[:limit] if limit else load_split(language, split)

    # Flatten (example, candidate) pairs so batches can mix examples.
    jobs = []
    for idx, ex in enumerate(data):
        prompt = sp.encode(ex["prompt"])
        for cand in candidates_for(ex, lexicon):
            seq = prompt + sp.encode(" " + cand) + [eos_id]
            jobs.append((idx, cand, seq, len(prompt)))

    scored = defaultdict(list)
    for b in range(0, len(jobs), batch_size):
        chunk = jobs[b:b + batch_size]
        sums, means = score_batch(model, [j[2] for j in chunk], [j[3] for j in chunk],
                                  sp.pad_id(), device)
        for (idx, cand, _, _), s, m in zip(chunk, sums, means):
            scored[idx].append((cand, s, m))

    hits_sum, hits_mean, chance = [], [], []
    names_only_hits, names_only_chance = [], []
    by_type = defaultdict(lambda: {"sum": [], "mean": []})
    for idx, ex in enumerate(data):
        options = scored[idx]
        pick_sum = max(options, key=lambda o: o[1])[0]
        pick_mean = max(options, key=lambda o: o[2])[0]
        ok_s, ok_m = pick_sum == ex["answer"], pick_mean == ex["answer"]
        hits_sum.append(ok_s)
        hits_mean.append(ok_m)
        chance.append(1 / len(options))
        kind = "name" if ex["answer"] in pool else "equal/more/less"

        # The strictest form of the question: among the NAMES in the question only, with
        # "equal" removed, is the right one most likely? The first forced-choice run
        # showed the model scoring "equal" so highly on held-out names that it won even
        # against the correct candidate, which leaves open whether the model could tell
        # the entities apart at all. Restricting to names — on questions whose answer is
        # a name — takes that escape route away.
        if kind == "name":
            name_opts = [o for o in options if o[0] in pool]
            names_only_hits.append(max(name_opts, key=lambda o: o[1])[0] == ex["answer"])
            names_only_chance.append(1 / len(name_opts))
        by_type[kind]["sum"].append(ok_s)
        by_type[kind]["mean"].append(ok_m)

    acc = lambda v: round(sum(v) / len(v), 4) if v else None      # noqa: E731
    return {
        "split": split, "n": len(data),
        "accuracy_sum_logprob": acc(hits_sum),
        "accuracy_mean_logprob": acc(hits_mean),
        "chance": round(sum(chance) / len(chance), 4),
        "names_only": {"n": len(names_only_hits), "accuracy": acc(names_only_hits),
                       "chance": round(sum(names_only_chance) / len(names_only_chance), 4)
                       if names_only_chance else None},
        "by_answer_type": {k: {"n": len(v["sum"]), "accuracy_sum_logprob": acc(v["sum"]),
                               "accuracy_mean_logprob": acc(v["mean"])}
                           for k, v in by_type.items()},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--language", choices=["hindi", "sindhi"], help="default: both")
    p.add_argument("--checkpoint", default="finetuned")
    p.add_argument("--splits", nargs="+", default=TEST_SPLITS)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    results = []
    for language in ([args.language] if args.language else ["hindi", "sindhi"]):
        model = load_model(resolve_checkpoint(language, args.checkpoint), args.device)
        entry = {"language": language, "condition": args.checkpoint, "splits": {}}
        for split in args.splits:
            r = rank_split(model, language, split, args.device, args.batch_size, args.limit)
            entry["splits"][split] = r
            names = r["by_answer_type"].get("name", {})
            logger.info("%s %-14s forced-choice %.4f (chance %.4f) | names-only %.4f "
                        "(chance %.4f)", language, split, r["accuracy_sum_logprob"],
                        r["chance"], r["names_only"]["accuracy"] or float("nan"),
                        r["names_only"]["chance"] or float("nan"))
        results.append(entry)

    out = Path(args.out) if args.out else \
        PROJECT_ROOT / "report" / f"reasoning_forced_choice_{args.checkpoint}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
