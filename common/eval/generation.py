"""
Generation quality evaluation (Step 9).

Generates continuations from held-out prefixes at several decoding temperatures and scores
them against the true continuations with BLEU, chrF++ and ROUGE-L, plus diversity and
repetition diagnostics.

A caveat that belongs with these numbers rather than buried in a footnote: overlap metrics
like BLEU and ROUGE were designed for tasks with a single correct answer — translation,
summarisation — where the reference is *the* right output. Open-ended continuation has no
single correct answer. A fluent, sensible continuation that simply differs from what the
author happened to write next scores near zero. These metrics are reported because the
brief asks for them and because they are comparable *between* our two models, but they
measure agreement with one arbitrary reference, not text quality.

Usage:
    python -m common.eval.generation --language hindi
    python -m common.eval.generation --language hindi --split val --n_prompts 200
"""

import argparse
import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import torch
import sentencepiece as spm
from sacrebleu.metrics import BLEU, CHRF

from common.model.config import GPTConfig
from common.model.gpt import GPT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("generation")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOC_SEPARATOR = "<|endoftext|>"

# Decoding settings required by the brief. Temperature 0 is greedy: we take the argmax
# rather than dividing by zero.
TEMPERATURES = [0.0, 0.5, 1.0, 1.5]


# ─────────────────────────────────────────────────────────────────────────────
# Prompt selection
# ─────────────────────────────────────────────────────────────────────────────

def sample_prompts(tokens: np.ndarray, sep_id: int, n_prompts: int,
                   prompt_len: int, gen_len: int, seed: int = 0):
    """
    Draw (prompt, reference) pairs from the held-out token stream.

    Windows that contain a document separator are rejected. If a prompt and its reference
    straddle a document boundary, the "true continuation" is the start of an unrelated
    document, and every overlap metric would be scoring the model against text no model
    could predict. That would depress the scores for a reason that has nothing to do with
    model quality, and would do so unevenly between the two languages, since their
    document lengths differ by a factor of three.
    """
    rng = np.random.default_rng(seed)
    span = prompt_len + gen_len
    prompts, references = [], []
    attempts = 0
    max_attempts = n_prompts * 200

    while len(prompts) < n_prompts and attempts < max_attempts:
        attempts += 1
        start = int(rng.integers(0, len(tokens) - span))
        window = np.asarray(tokens[start:start + span])
        if (window == sep_id).any():
            continue                     # crosses a document boundary
        prompts.append(window[:prompt_len].tolist())
        references.append(window[prompt_len:].tolist())

    if len(prompts) < n_prompts:
        logger.warning("only found %d/%d separator-free windows after %d attempts",
                       len(prompts), n_prompts, attempts)
    return prompts, references


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def rouge_l(hypothesis: str, reference: str) -> float:
    """
    ROUGE-L F1 over whitespace tokens, via longest common subsequence.

    Implemented here rather than taken from a library because the common Python ROUGE
    packages apply English-specific preprocessing — Porter stemming, English stopword
    handling, ASCII-oriented tokenisation — which is meaningless for Devanagari and
    Perso-Arabic and would silently mangle both languages. Whitespace tokens are the
    honest unit for a script-agnostic comparison.
    """
    h, r = hypothesis.split(), reference.split()
    if not h or not r:
        return 0.0

    # Standard LCS dynamic program, kept to two rows so memory is O(min(len)).
    if len(h) < len(r):
        h, r = r, h
    prev = [0] * (len(r) + 1)
    for a in h:
        cur = [0] * (len(r) + 1)
        for j, b in enumerate(r, 1):
            cur[j] = prev[j - 1] + 1 if a == b else max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[-1]

    if lcs == 0:
        return 0.0
    precision = lcs / len(hypothesis.split())
    recall = lcs / len(reference.split())
    return 2 * precision * recall / (precision + recall)


def distinct_n(texts: List[str], n: int) -> float:
    """
    Fraction of generated n-grams that are unique, pooled across all samples.

    Low values mean the model keeps emitting the same phrases. Computed over the pooled
    corpus rather than per-sample so it also catches a model that produces the same output
    for different prompts.
    """
    grams = Counter()
    for t in texts:
        w = t.split()
        grams.update(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    total = sum(grams.values())
    return len(grams) / total if total else 0.0


def repetition_rate(texts: List[str], n: int = 4) -> float:
    """
    Mean fraction of a sample's n-grams that repeat *within that sample*.

    This is the degenerate-loop diagnostic: a model stuck in a cycle scores near 1.0.
    Measured per sample and averaged, so one pathological output does not hide among many
    healthy ones the way a pooled statistic would.
    """
    rates = []
    for t in texts:
        w = t.split()
        if len(w) < n + 1:
            continue
        grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
        counts = Counter(grams)
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        rates.append(repeated / len(grams))
    return sum(rates) / len(rates) if rates else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(language: str, split: str, n_prompts: int, prompt_len: int,
             gen_len: int, batch_size: int, device: str, seed: int) -> dict:
    """Generate at every temperature and score against the true continuations."""
    ckpt_path = PROJECT_ROOT / language / "train" / "run" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ckpt["model_config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    sp = spm.SentencePieceProcessor(
        model_file=str(PROJECT_ROOT / language / "tokenizer"
                       / f"{language}_unigram_16000.model"))
    sep_id = sp.piece_to_id(DOC_SEPARATOR)

    data = np.memmap(PROJECT_ROOT / language / "data" / "tokenized" / f"{split}.bin",
                     dtype=np.uint16, mode="r")
    prompts, references = sample_prompts(data, sep_id, n_prompts, prompt_len, gen_len, seed)
    logger.info("%s/%s: %d prompts, %d-token prefix -> %d-token continuation",
                language, split, len(prompts), prompt_len, gen_len)

    ref_texts = [sp.decode(r) for r in references]
    bleu, chrf = BLEU(), CHRF(word_order=2)     # chrF++ is chrF with word bigrams
    results, samples = {}, []

    for temp in TEMPERATURES:
        label = "greedy" if temp == 0.0 else f"temp_{temp}"
        hyps = []
        torch.manual_seed(seed)                 # reproducible sampling
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            idx = torch.tensor(chunk, dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.startswith("cuda")):
                out = model.generate(idx, max_new_tokens=gen_len, temperature=temp)
            hyps.extend(sp.decode(row[prompt_len:].tolist()) for row in out.cpu())

        scored = {
            "bleu": bleu.corpus_score(hyps, [ref_texts]).score,
            "chrf++": chrf.corpus_score(hyps, [ref_texts]).score,
            "rouge_l": sum(rouge_l(h, r) for h, r in zip(hyps, ref_texts)) / len(hyps),
            "distinct_1": distinct_n(hyps, 1),
            "distinct_2": distinct_n(hyps, 2),
            "repetition_rate_4gram": repetition_rate(hyps, 4),
            "mean_chars": sum(len(h) for h in hyps) / len(hyps),
        }
        results[label] = scored
        logger.info("  %-10s BLEU %5.2f | chrF++ %5.2f | ROUGE-L %.4f | "
                    "D-1 %.4f | D-2 %.4f | rep %.4f",
                    label, scored["bleu"], scored["chrf++"], scored["rouge_l"],
                    scored["distinct_1"], scored["distinct_2"],
                    scored["repetition_rate_4gram"])

        # Rank this setting's outputs so the report can show the actual spread rather than
        # one arbitrary draw. Ranking is by per-sample chrF++ against the true continuation:
        # an objective, reproducible criterion rather than a subjective read.
        #
        # What that ranks is AGREEMENT WITH THE REFERENCE, which is not the same as quality.
        # A fluent continuation that simply goes somewhere else than the author did will
        # rank low. The ranking is therefore a way of surfacing range, not a quality verdict,
        # and the report says so.
        per_sample = sorted(
            ({"rank_score_chrf": chrf.sentence_score(h, [r]).score,
              "prompt": sp.decode(p_ids), "generated": h, "reference": r}
             for h, r, p_ids in zip(hyps, ref_texts, prompts)),
            key=lambda d: -d["rank_score_chrf"])

        n = len(per_sample)
        samples.append({
            "setting": label,
            # Kept for backward compatibility with the previous report format.
            "prompt": per_sample[0]["prompt"],
            "generated": per_sample[0]["generated"],
            "reference": per_sample[0]["reference"],
            "ranked": {"best": per_sample[0],
                       "upper_quartile": per_sample[n // 4],
                       "median": per_sample[n // 2],
                       "lower_quartile": per_sample[3 * n // 4],
                       "worst": per_sample[-1]},
            "chrf_spread": {"best": per_sample[0]["rank_score_chrf"],
                            "median": per_sample[n // 2]["rank_score_chrf"],
                            "worst": per_sample[-1]["rank_score_chrf"]},
        })

    return {
        "language": language, "split": split, "n_prompts": len(prompts),
        "prompt_tokens": prompt_len, "generated_tokens": gen_len,
        "context": cfg.max_seq_len, "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
        "metrics": results, "samples": samples,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--language", choices=["hindi", "sindhi"], help="default: both")
    p.add_argument("--split", default="val", choices=["val", "test"],
                   help="val by default: it carries far less encoding corruption than test")
    p.add_argument("--n_prompts", type=int, default=200)
    p.add_argument("--prompt_tokens", type=int, default=64)
    p.add_argument("--generate_tokens", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    languages = [args.language] if args.language else ["hindi", "sindhi"]
    out = []
    for lang in languages:
        out.append(evaluate(lang, args.split, args.n_prompts, args.prompt_tokens,
                            args.generate_tokens, args.batch_size, args.device, args.seed))

    path = PROJECT_ROOT / "report" / "generation_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
