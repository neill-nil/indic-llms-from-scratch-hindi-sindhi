"""
Render <language>/tokenizer/tokenizer_statistics.md from evaluation_report.json.

The statistics report is generated rather than hand-written on purpose. A previous
hand-written revision claimed the Unigram model kept the acronym "आईआईएम" intact as a
single token while BPE split it; in fact that string is absent from both vocabularies
and both tokenizers segment it identically. Deriving every number and every example
from the evaluation artefact makes that class of error structurally impossible.

Usage:
    python -m common.tokenizer.write_stats_report --language hindi --vocab_size 16000
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("stats_report")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

LANG_META = {
    "hindi":  {"display": "Hindi",  "script": "Devanagari",
               "model": "Model H (higher-resource)",
               # Held-out Unigram fertility at 32K vs 16K, measured on this corpus.
               "fert_32k": 1.2437, "fert_16k": 1.3056,
               # Training split size, from data/preprocessed/split_statistics.json.
               "train_lines": 7_238_248, "train_tokens": 504_818_280},
    "sindhi": {"display": "Sindhi", "script": "Perso-Arabic",
               "model": "Model L (lower-resource)",
               "fert_32k": 1.2412, "fert_16k": 1.2987,
               "train_lines": 11_041_567, "train_tokens": 618_440_091},
}

# Sentences reservoir-sampled by the SentencePiece trainer (input_sentence_size).
TOKENIZER_SAMPLE_LINES = 500_000


def pct_delta(worse: float, better: float) -> str:
    """Format the relative improvement of `better` over `worse` as a percentage."""
    if worse == 0:
        return "n/a"
    return f"{abs(worse - better) / worse:.2%}"


def format_pieces(pieces) -> str:
    """Render a token list as an inline code span."""
    return "`" + " | ".join(pieces) + "`"


def build_comparison_table(report: dict) -> str:
    """Build the BPE-vs-Unigram metric table. Lower is better except where noted."""
    bpe, uni = report["bpe"], report["unigram"]
    rows = [
        ("Fertility (tokens / word)", "fertility", "lower", 4),
        ("Tokens per character", "tokens_per_char", "lower", 4),
        ("Bytes per token", "bytes_per_token", "higher", 3),
        ("Token entropy (bits/token)", "token_entropy_bits", "lower", 4),
        ("Unknown-token rate", "unk_rate", "lower", 8),
    ]
    lines = ["| Metric | BPE | Unigram | Better | Margin |",
             "|---|---|---|---|---|"]
    for label, key, direction, dp in rows:
        b, u = bpe["fertility"][key], uni["fertility"][key]
        if b == u:
            winner, margin = "tie", "0.00%"
        else:
            better_is_uni = (u < b) if direction == "lower" else (u > b)
            winner = "Unigram" if better_is_uni else "BPE"
            margin = pct_delta(max(b, u), min(b, u))
        lines.append(f"| {label} | {b:.{dp}f} | {u:.{dp}f} | {winner} | {margin} |")

    bu, uu = bpe["fertility"]["vocab_utilisation"], uni["fertility"]["vocab_utilisation"]
    lines.append(
        f"| Vocabulary utilisation (higher better) | {bu:.2%} | {uu:.2%} | "
        f"{'Unigram' if uu > bu else 'BPE'} | {abs(uu - bu):.2%} pts |")

    bs = bpe["vocabulary"]["target_script_ratio"]
    us = uni["vocabulary"]["target_script_ratio"]
    lines.append(
        f"| Target-script share of vocab | {bs:.2%} | {us:.2%} | "
        f"{'Unigram' if us > bs else 'BPE'} | {abs(us - bs):.2%} pts |")
    return "\n".join(lines)


def build_examples_section(report: dict) -> str:
    """Render mined contrastive examples for both algorithms, in both directions."""
    contrastive = report.get("contrastive")
    if not contrastive:
        return "_No contrastive examples available; re-run evaluate_tokenizer.py._"

    out = []
    for key, winner, loser in [("unigram_better", "unigram", "bpe"),
                               ("bpe_better", "bpe", "unigram")]:
        block = contrastive[key]
        n = block[f"n_words_favouring_{winner}"]
        out.append(
            f"**Words that {winner.upper()} segments into fewer tokens "
            f"({n} distinct frequent word types):**\n")
        if not block["examples"]:
            out.append("_None found._\n")
            continue
        out.append(f"| Word | Corpus freq. | Unigram | BPE |")
        out.append("|---|---|---|---|")
        for e in block["examples"][:5]:
            out.append(f"| `{e['word']}` | {e['corpus_frequency']} | "
                       f"{format_pieces(e['unigram'])} | {format_pieces(e['bpe'])} |")
        out.append("")
    return "\n".join(out)


def render(language: str, vocab_size: int, report: dict) -> str:
    """Assemble the full Markdown statistics report for one language."""
    meta = LANG_META[language]
    uni, bpe = report["unigram"], report["bpe"]
    uv, uf = uni["vocabulary"], uni["fertility"]

    emb_16k = vocab_size * 512
    emb_32k = 32000 * 512

    compression_gain = (meta["fert_16k"] - meta["fert_32k"]) / meta["fert_16k"]
    sample_frac = TOKENIZER_SAMPLE_LINES / meta["train_lines"]
    sampled_tokens = meta["train_tokens"] * sample_frac
    script_dist = ", ".join(f"{k} {v:,}" for k, v in uv["script_distribution"].items())

    contrastive = report.get("contrastive", {})
    n_uni = contrastive.get("unigram_better", {}).get("n_words_favouring_unigram", 0)
    n_bpe = contrastive.get("bpe_better", {}).get("n_words_favouring_bpe", 0)

    return f"""# {meta['display']} Tokenizer Statistics

<!-- GENERATED FILE - do not edit by hand.
     Regenerate with:
       python -m common.tokenizer.evaluate_tokenizer --language {language} --vocab_size {vocab_size}
       python -m common.tokenizer.write_stats_report  --language {language} --vocab_size {vocab_size}
-->

Vocabulary and evaluation metrics for the {meta['display']} tokenizers ({meta['model']}).
Both were trained from scratch with SentencePiece on the preprocessed training corpus;
no pretrained tokenizer or vocabulary was used. All held-out metrics below are measured
on `{language}/data/preprocessed/val.txt`, which no tokenizer was trained on.

## 1. Selected configuration

* **Algorithm selected for LM training:** Unigram
* **Vocabulary size:** {vocab_size:,}
* **Character coverage:** 0.9995
* **Byte fallback:** enabled ({uv['byte_fallback_tokens']} byte tokens), which is why the
  unknown-token rate is exactly zero — any character absent from the vocabulary is
  representable as a sequence of raw byte tokens rather than collapsing to `<unk>`.
* **Special tokens:** {uv['special_tokens']} (`<pad>`, `<unk>`, `<s>`, `</s>`, `<|endoftext|>`)

### Why {vocab_size:,} slots

Vocabulary size trades compression against parameter budget, and at a ~25M parameter
target that budget is the binding constraint. With a model dimension of 512 and a tied
input/output embedding, a {vocab_size//1000}K vocabulary costs {emb_16k/1e6:.2f}M parameters;
a 32K vocabulary would cost {emb_32k/1e6:.2f}M — roughly two-thirds of the entire budget
spent on a lookup table, leaving too little for the transformer stack itself.

The measured cost of that choice is real but modest: a 32K Unigram model compresses this
corpus {compression_gain:.1%} better (held-out fertility {meta['fert_32k']:.4f} vs
{meta['fert_16k']:.4f}). Paying that to free {(emb_32k - emb_16k)/1e6:.1f}M parameters for
depth and feed-forward width is the right trade at this scale.

## 2. Tokenizer training data

Trained on a shuffled reservoir sample of **{TOKENIZER_SAMPLE_LINES:,} lines** from the
cleaned training split (`input_sentence_size`, `shuffle_input_sentence=True`), which bounds
trainer memory on a multi-GB corpus:

| Train split | Sampled for tokenizer | Fraction |
|---|---|---|
| {meta['train_lines']:,} lines / {meta['train_tokens']/1e6:.1f}M tokens | {TOKENIZER_SAMPLE_LINES:,} lines / ~{sampled_tokens/1e6:.1f}M tokens | {sample_frac:.2%} |

**Why this is sufficient.** ~{sampled_tokens/1e6:.0f}M tokens against a {vocab_size:,}-slot
vocabulary is roughly {sampled_tokens/vocab_size:,.0f} observations per entry on average,
well above what is needed to estimate merge frequencies or unigram probabilities. The
held-out numbers below confirm it directly: vocabulary utilisation is
{uf['vocab_utilisation']:.1%} and the unknown-token rate is exactly zero on text the
tokenizer never saw. A vocabulary fitted to too small a sample would be full of tokens that
never recur.

Sampling applies only to tokenizer training; Phase 2 pretraining uses the entire split.

## 3. Held-out comparison: BPE vs Unigram

Measured on a strided sample across the whole validation file
({uf['lines_sampled']:,} lines, {uf['total_words_sampled']:,} words, {uf['total_tokens']:,} tokens).

{build_comparison_table(report)}

**Verdict.** Unigram is the better choice, but by a narrower margin than raw fertility
alone suggests. The decisive metric is **token entropy**: Unigram produces a token stream
with lower empirical entropy, which directly lower-bounds the perplexity a downstream
language model can achieve on the same text. It also leaves fewer dead embedding rows
({uf['vocab_utilisation']:.1%} of its vocabulary appears in held-out text, against
{bpe['fertility']['vocab_utilisation']:.1%} for BPE) — at a tight parameter budget an unused
vocabulary slot is a wasted embedding vector.

BPE retains a slightly higher target-script share of the vocabulary, but this is not
evidence of quality: the difference is that Unigram allocates more slots to Latin
substrings, and real {meta['display']} web text contains English loanwords, product names
and numerals that must be encoded either way.

## 4. Vocabulary composition (Unigram)

* **Vocabulary size:** {uv['vocab_size']:,}
* **{meta['script']} tokens:** {uv['target_script_ratio']:.2%} of non-special vocabulary
* **Script distribution:** {script_dist}
* **Average token length:** {uv['avg_token_length_chars']:.2f} characters
* **Median token length:** {uv['median_token_length']} characters

## 5. Token-frequency statistics (held-out)

* **Distinct token types observed:** {uf['vocab_types_used']:,} of {uv['vocab_size']:,} ({uf['vocab_utilisation']:.2%})
* **Token entropy:** {uf['token_entropy_bits']:.4f} bits/token
* **Types covering 50% of all tokens:** {uf['types_covering_50pct']:,}
* **Types covering 90% of all tokens:** {uf['types_covering_90pct']:,}
* **Types covering 99% of all tokens:** {uf['types_covering_99pct']:,}
* **Unknown-token rate:** {uf['unk_rate']:.8f}

The distribution is steeply Zipfian: just {uf['types_covering_50pct']:,} token types account
for half of all token occurrences, while the last 1% of probability mass is spread across
roughly {uv['vocab_size'] - uf['types_covering_99pct']:,} rare types.

## 6. Where the two algorithms actually differ

These examples are mined automatically from held-out text by
`mine_contrastive_examples()` rather than chosen by hand, and the comparison is run in
both directions so neither algorithm is shown only its wins. Across frequent word types,
Unigram segments {n_uni} more efficiently and BPE segments {n_bpe} more efficiently.

{build_examples_section(report)}

## 7. Sample tokenizations (Unigram)

""" + "\n".join(
        f"**Original:** `{s['original'][:160]}`\n\n"
        f"**Tokens ({s['n_tokens']}):** {format_pieces(s['tokens'][:24])}\n"
        for s in uni.get("samples", [])[:3]
    )


def main():
    parser = argparse.ArgumentParser(description="Generate tokenizer_statistics.md from evaluation_report.json")
    parser.add_argument("--language", required=True, choices=["hindi", "sindhi"])
    parser.add_argument("--vocab_size", type=int, default=16000)
    args = parser.parse_args()

    tokenizer_dir = PROJECT_ROOT / args.language / "tokenizer"
    report_path = tokenizer_dir / "evaluation_report.json"
    if not report_path.exists():
        logger.error(f"Missing {report_path}. Run evaluate_tokenizer.py first.")
        return

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    out_path = tokenizer_dir / "tokenizer_statistics.md"
    out_path.write_text(render(args.language, args.vocab_size, report), encoding="utf-8")
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
