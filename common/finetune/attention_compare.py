"""
Attention before and after reasoning finetuning (brief section 3.2).

Computation only — figures are drawn by plot_attention_compare.py from the JSON written
here, so the numbers can be checked without re-running a model.

    python -m common.finetune.attention_compare

For each language, the pretrained and the finetuned checkpoint are run over IDENTICAL
inputs, so every difference between them is attributable to finetuning:

  corpus     the exact held-out windows Phase 2 analysed (val, 64 windows of 256 tokens,
             seed 0). Answers "did finetuning change how the model reads ordinary text?"
             The pretrained numbers must reproduce Phase 2's stored statistics; that is
             checked, and is what licenses comparing against them.

  reasoning  comparative-reasoning prompts from test_iid (names seen in finetuning) and
             test_entity (held-out names). Answers the brief's "especially on
             comparative-reasoning prompts", and tests the Chunk D finding that the
             finetuned model knows which held-out entity wins but cannot say its name.

Phase 2's toolkit is reused rather than reimplemented: `attention_statistics` for entropy,
distance, self- and previous-token mass, and `sample_clean_windows` for the corpus inputs.
Two measurements are added.

  attention sink       mass on position 0. Phase 2 estimated this from a single example
                       window; here it is averaged over every window and every query.

  answer position      the final prompt token — the one whose logits produce the answer.
                       Its attention row is split into mass on entity-name tokens, on the
                       correct entity versus each distractor, on digit tokens, and on the
                       sink. This is the most direct view available of what the model
                       consults at the moment it answers.

Why the reasoning statistics use MIN_QUERY = 8, not Phase 2's 32
----------------------------------------------------------------
Phase 2 skipped the first 32 queries because a causal mask forces early queries to be
short-range. Reasoning prompts are only 38-58 tokens, so 32 would leave almost nothing.
A threshold of 8 keeps most of each prompt. This makes reasoning-prompt distances NOT
comparable with corpus distances — but that comparison is never made. Pretrained and
finetuned are always compared on the same prompts under the same threshold.

Why prompts are batched by exact length
---------------------------------------
Padding would put tokens into the attention rows being measured. Equal-length buckets
need none, exactly as in the reasoning evaluator.
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch

from common.eval.attention import attention_statistics, sample_clean_windows
from common.finetune.analyse_reasoning import entities_in_order, load_lexicon, load_split
from common.finetune.evaluate import load_model, resolve_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("attention_compare")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOC_SEPARATOR = "<|endoftext|>"

# Phase 2's corpus settings, reproduced exactly.
CORPUS_WINDOWS, CORPUS_LEN, CORPUS_SEED, CORPUS_MIN_QUERY = 64, 256, 0, 32

REASONING_SPLITS = ["test_iid", "test_entity"]
REASONING_MIN_QUERY = 8


def tokenizer(language: str) -> spm.SentencePieceProcessor:
    return spm.SentencePieceProcessor(
        model_file=str(PROJECT_ROOT / language / "tokenizer"
                       / f"{language}_unigram_16000.model"))


# ─────────────────────────────────────────────────────────────────────────────
# Corpus text
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def corpus_statistics(model, language: str, device: str) -> dict:
    """Phase 2's corpus analysis on any checkpoint, plus a properly averaged sink."""
    sp = tokenizer(language)
    data = np.memmap(PROJECT_ROOT / language / "data" / "tokenized" / "val.bin",
                     dtype=np.uint16, mode="r")
    seq_len = min(CORPUS_LEN, model.config.max_seq_len)
    windows = sample_clean_windows(data, sp.piece_to_id(DOC_SEPARATOR),
                                   CORPUS_WINDOWS, seq_len, CORPUS_SEED)
    idx = torch.tensor(windows.astype(np.int64), device=device)

    per_layer = [[] for _ in range(model.config.n_layers)]
    sink = [[] for _ in range(model.config.n_layers)]
    chunk = max(1, 2048 // seq_len)            # Phase 2's chunking, so averages match
    for i in range(0, len(idx), chunk):
        _, _, attns = model(idx[i:i + chunk], return_attention=True)
        for layer, a in enumerate(attns):
            per_layer[layer].append(attention_statistics(a, CORPUS_MIN_QUERY))
            sink[layer].append(a.float()[..., CORPUS_MIN_QUERY:, 0].mean(dim=(0, 2)).cpu().numpy())

    out = {}
    for layer, chunks in enumerate(per_layer):
        out[str(layer)] = {k: np.mean([c[k] for c in chunks], axis=0).tolist() for k in chunks[0]}
        out[str(layer)]["sink"] = np.mean(sink[layer], axis=0).tolist()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning prompts
# ─────────────────────────────────────────────────────────────────────────────

def find_positions(ids: list[int], pattern: list[int]) -> list[int]:
    """Every token position covered by an occurrence of `pattern` in `ids`."""
    n, m, hits = len(ids), len(pattern), []
    for i in range(n - m + 1):
        if ids[i:i + m] == pattern:
            hits.extend(range(i, i + m))
    return hits


def locate(example: dict, ids: list[int], sp, lexicon: dict) -> dict:
    """
    Token positions of each entity mention and of every digit token in a prompt.

    Entities appear in text in their oblique form where one exists, so that is the form
    searched for; it is keyed by citation form, which is what the answer uses. A name is
    tokenized as a word-initial piece (`▁name`), which SentencePiece segments the same way
    wherever it occurs because pieces never span whitespace.
    """
    oblique = {e["name"]: e.get("oblique", e["name"])
               for kind in lexicon["entities"].values() for e in kind}
    spans = {}
    for name in entities_in_order(example, lexicon):
        spans[name] = find_positions(ids, sp.encode(" " + oblique[name]))
    digits = [i for i, t in enumerate(ids)
              if (p := sp.id_to_piece(t).replace("▁", "")) and p.isdigit()]
    return {"entities": spans, "digits": digits}


@torch.no_grad()
def reasoning_statistics(model, language: str, split: str, n_examples: int,
                         device: str, seed: int) -> dict:
    """Generic head statistics plus what the answer position attends to."""
    sp = tokenizer(language)
    lexicon = load_lexicon(language)
    data = load_split(language, split)
    chosen = random.Random(seed).sample(range(len(data)), min(n_examples, len(data)))

    buckets = defaultdict(list)
    for i in chosen:
        ids = sp.encode(data[i]["prompt"])
        buckets[len(ids)].append((data[i], ids))

    L, H = model.config.n_layers, model.config.n_heads
    generic = [[] for _ in range(L)]
    acc = {k: np.zeros((L, H)) for k in
           ("entity", "number", "sink", "gold", "distractor", "first", "last")}
    counts = defaultdict(int)
    located = found = 0

    for length, items in buckets.items():
        if length <= REASONING_MIN_QUERY:
            continue
        batch = torch.tensor([ids for _, ids in items], device=device)
        _, _, attns = model(batch, return_attention=True)
        maps = [locate(ex, ids, sp, lexicon) for ex, ids in items]

        for layer, a in enumerate(attns):
            generic[layer].append((len(items), attention_statistics(a, REASONING_MIN_QUERY)))
            row = a.float()[:, :, -1, :].cpu().numpy()          # (B, H, T) — answer query

            for b, ((ex, _), m) in enumerate(zip(items, maps)):
                ent = sorted({p for ps in m["entities"].values() for p in ps})
                acc["entity"][layer] += row[b][:, ent].sum(-1) if ent else 0
                acc["sink"][layer] += row[b][:, 0]
                if m["digits"]:
                    acc["number"][layer] += row[b][:, m["digits"]].sum(-1)
                if layer == 0:
                    counts["all"] += 1
                    counts["number"] += bool(m["digits"])

                names = list(m["entities"])
                if ex["answer"] in m["entities"] and m["entities"][ex["answer"]]:
                    others = [n for n in names if n != ex["answer"] and m["entities"][n]]
                    if others:
                        acc["gold"][layer] += row[b][:, m["entities"][ex["answer"]]].sum(-1)
                        acc["distractor"][layer] += np.mean(
                            [row[b][:, m["entities"][n]].sum(-1) for n in others], axis=0)
                        if layer == 0:
                            counts["gold"] += 1
                if len(names) >= 2 and m["entities"][names[0]] and m["entities"][names[-1]]:
                    acc["first"][layer] += row[b][:, m["entities"][names[0]]].sum(-1)
                    acc["last"][layer] += row[b][:, m["entities"][names[-1]]].sum(-1)
                    if layer == 0:
                        counts["first_last"] += 1

        for (ex, _), m in zip(items, maps):
            located += len(m["entities"])
            found += sum(bool(v) for v in m["entities"].values())

    out = {}
    for layer in range(L):
        total = sum(n for n, _ in generic[layer])
        stats = {k: (sum(n * np.asarray(s[k]) for n, s in generic[layer]) / total).tolist()
                 for k in generic[layer][0][1]}
        stats["answer_entity_mass"] = (acc["entity"][layer] / counts["all"]).tolist()
        stats["answer_sink_mass"] = (acc["sink"][layer] / counts["all"]).tolist()
        stats["answer_number_mass"] = (acc["number"][layer] / max(1, counts["number"])).tolist()
        stats["answer_gold_mass"] = (acc["gold"][layer] / max(1, counts["gold"])).tolist()
        stats["answer_distractor_mass"] = (acc["distractor"][layer] / max(1, counts["gold"])).tolist()
        stats["answer_first_entity_mass"] = (acc["first"][layer] / max(1, counts["first_last"])).tolist()
        stats["answer_last_entity_mass"] = (acc["last"][layer] / max(1, counts["first_last"])).tolist()
        out[str(layer)] = stats

    return {"n_examples": counts["all"], "n_gold_comparisons": counts["gold"],
            "entity_location_rate": round(found / max(1, located), 4),
            "min_query": REASONING_MIN_QUERY, "layers": out}


def heatmap_example(language: str) -> tuple[dict, list[int]]:
    """
    One fixed prompt for the side-by-side heatmaps: three entities, numeric, no tie, a
    name as the answer. Chosen by rule rather than by eye so it cannot be picked for
    looking interesting.
    """
    sp = tokenizer(language)
    for ex in load_split(language, "test_iid"):
        ids = sp.encode(ex["prompt"])
        if (ex["family"] == "value_compare" and ex["n_entities"] == 3 and not ex["is_tie"]
                and ex["template_author"] == "generated" and len(ids) <= 48):
            return ex, ids
    raise RuntimeError("no suitable heatmap example")


@torch.no_grad()
def example_attention(model, ids: list[int], device: str) -> list:
    _, _, attns = model(torch.tensor([ids], device=device), return_attention=True)
    return [np.round(a[0].float().cpu().numpy(), 5).tolist() for a in attns]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--language", choices=["hindi", "sindhi"], help="default: both")
    p.add_argument("--n_examples", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=str(PROJECT_ROOT / "report" / "attention_finetune_results.json"))
    args = p.parse_args()

    results = {}
    for language in ([args.language] if args.language else ["hindi", "sindhi"]):
        sp = tokenizer(language)
        lexicon = load_lexicon(language)
        ex, ids = heatmap_example(language)
        entry = {"example": {
            "text": ex["prompt"], "gold": ex["answer"],
            "tokens": [sp.id_to_piece(t) for t in ids],
            "entity_positions": locate(ex, ids, sp, lexicon)["entities"],
            "attention": {}}}

        for condition in ("pretrained", "finetuned"):
            model = load_model(resolve_checkpoint(language, condition), args.device)
            logger.info("%s %s: corpus windows", language, condition)
            entry[condition] = {"corpus": corpus_statistics(model, language, args.device),
                                "reasoning": {}}
            for split in REASONING_SPLITS:
                logger.info("%s %s: reasoning prompts, %s", language, condition, split)
                entry[condition]["reasoning"][split] = reasoning_statistics(
                    model, language, split, args.n_examples, args.device, args.seed)
            entry["example"]["attention"][condition] = example_attention(model, ids, args.device)
            del model
            torch.cuda.empty_cache()
        results[language] = entry

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
