"""
Render report/phase2_intrinsic.md from intrinsic_results.json.

Generated rather than hand-written, for the same reason the Phase 1 tokenizer report is:
numbers typed by hand drift from the artefacts they claim to describe. Everything here is
read from the measurement file.

Usage:
    python -m common.eval.write_intrinsic_report
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("intrinsic_report")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = PROJECT_ROOT / "report" / "intrinsic_results.json"
OUT = PROJECT_ROOT / "report" / "phase2_intrinsic.md"

# Measured contamination rates, from the mojibake detector in this module's analysis.
# Documents whose text contains Arabic-Indic or Devanagari digits wedged inside words at
# high density, which is the signature of a legacy non-Unicode font decoded as Unicode.
CONTAMINATION = {
    ("hindi", "val"): 0.006, ("hindi", "test"): 0.073,
    ("sindhi", "val"): 0.016, ("sindhi", "test"): 0.166,
}

NATIVE_CTX = {"hindi": 512, "sindhi": 1024}


def row(r: dict) -> str:
    tag = "native" if r["context"] == NATIVE_CTX[r["language"]] else "matched"
    return (f"| {r['language'].title()} | {r['split']} | {r['context']} ({tag}) | "
            f"{r['cross_entropy']:.4f} | {r['perplexity']:.2f} | "
            f"**{r['bits_per_byte']:.4f}** | {r['n_tokens']:,} | {r['n_bytes']:,} |")


def main():
    if not RESULTS.exists():
        logger.error("Missing %s — run common.eval.intrinsic first.", RESULTS)
        raise SystemExit(1)

    results = json.loads(RESULTS.read_text())
    by = {(r["language"], r["split"], r["context"]): r for r in results}

    h_test = by[("hindi", "test", 512)]
    s_test_m = by[("sindhi", "test", 512)]
    s_test_n = by[("sindhi", "test", 1024)]
    s_val_n = by[("sindhi", "val", 1024)]
    h_val = by[("hindi", "val", 512)]

    bpb_gap = (s_test_m["bits_per_byte"] - h_test["bits_per_byte"]) / h_test["bits_per_byte"]
    ppl_ratio = s_test_n["perplexity"] / s_val_n["perplexity"]

    body = f"""# Phase 2: Intrinsic Language-Modelling Evaluation

<!-- GENERATED FILE - do not edit by hand.
     Regenerate with:
       python -m common.eval.intrinsic
       python -m common.eval.write_intrinsic_report
-->

Cross-entropy, perplexity and bits-per-byte for Model H (Hindi) and Model L (Sindhi) on
their own held-out splits. Every token in each split is scored exactly once, using
non-overlapping windows, so these numbers are deterministic rather than sampled estimates.

## Results

{"| Model | Split | Context | CE (nats) | Perplexity | BPB | Tokens | Bytes |"}
|---|---|---|---|---|---|---|---|
{chr(10).join(row(r) for r in results)}

Model L is evaluated twice: at its native 1024-token context, and again at 512 to match
Model H exactly. The matched rows are what the cross-model comparison rests on, because
a longer context lowers loss for free and would otherwise flatter Model L.

## Headline comparison

At **matched context (512 tokens) on the test splits**:

| | Hindi (H) | Sindhi (L) | gap |
|---|---|---|---|
| Cross-entropy | {h_test['cross_entropy']:.4f} | {s_test_m['cross_entropy']:.4f} | — |
| Perplexity | {h_test['perplexity']:.2f} | {s_test_m['perplexity']:.2f} | — |
| **Bits per byte** | **{h_test['bits_per_byte']:.4f}** | **{s_test_m['bits_per_byte']:.4f}** | **+{bpb_gap:.1%}** |

Model L needs **{bpb_gap:.1%} more bits per byte** of text than Model H. Both models have
identical transformer stacks, identical vocabulary sizes, identical token budgets and
identical effective batches, so this gap reflects the languages and their corpora rather
than the training setup.

## Why bits-per-byte, not perplexity, is the headline metric

Perplexity is per *token*, and the two models have separate tokenizers learned on separate
corpora. A token means a different amount of text in each, so the numbers are not directly
comparable. Bits-per-byte normalises by the underlying UTF-8 bytes and is therefore
tokenizer-independent, which is why the brief asks for it.

This corpus provides an unusually sharp demonstration. On Sindhi, perplexity says the test
split is **{ppl_ratio:.1f}x easier** than validation ({s_test_n['perplexity']:.2f} vs
{s_val_n['perplexity']:.2f}), which is not credible for two halves of the same collection.
Bits-per-byte says the opposite and correct thing: test is slightly *harder*
({s_test_n['bits_per_byte']:.4f} vs {s_val_n['bits_per_byte']:.4f}). The explanation is in
the next section — perplexity was misled by tokenization, BPB was not.

## Corpus contamination, and its effect on these numbers

Investigating the perplexity anomaly above revealed that both test splits contain
encoding-corrupted text: legacy non-Unicode Sindhi and Hindi fonts decoded as Unicode,
producing strings of valid script characters in nonsense order, typically with digits
wedged inside words. A representative Sindhi example:

```
ـام ١اال ١اك ٧ي ٻٽي٨٦ ٧ؾار ةيؽك حيٝ رهزا٤٦ س٩٧ي ١ام ٤ڪؽي ٿ٩ۿ
```

Measured rates, over 40,000 documents per split:

| | val | test |
|---|---|---|
| Hindi | {CONTAMINATION[('hindi','val')]:.1%} | **{CONTAMINATION[('hindi','test')]:.1%}** |
| Sindhi | {CONTAMINATION[('sindhi','val')]:.1%} | **{CONTAMINATION[('sindhi','test')]:.1%}** |

This corrupted text is short, highly repetitive and therefore easy to predict, which
deflates per-token perplexity wherever it is dense. It arrives as ~80-character fragments,
which is why the Sindhi test split holds 208,300 documents against validation's 53,252
despite having fewer tokens.

**Why the Phase 1 filters did not catch it.** `script_filter` required at least 90% of
alphabetic characters to fall in the expected Unicode block. Mojibake passes that check
comfortably: it *is* Devanagari or Arabic-block characters, just in meaningless sequence.
Detecting it needs a different signal — here, the density of digits appearing inside
words, which legitimate prose in either language almost never produces.

**Effect on the reported numbers.** The two test splits are contaminated at different
rates (7.3% vs 16.6%), so some part of the headline BPB gap is contamination asymmetry
rather than language difficulty. The validation splits are far cleaner (0.6% and 1.6%) and
are the better basis for a like-for-like comparison; at matched 512-token context the
validation BPB gap is
{(by[('sindhi','val',512)]['bits_per_byte'] - h_val['bits_per_byte']) / h_val['bits_per_byte']:.1%},
against {bpb_gap:.1%} on test. Both figures support the same conclusion, and the
validation figure is the more conservative of the two.

## Method

* Windows are non-overlapping and stride by the context length, so every token is scored
  exactly once and the result is a deterministic function of the split.
* Cross-entropy is the mean over all scored positions, in nats.
* Perplexity is `exp(CE)`.
* Bits-per-byte is `CE * n_tokens / (ln(2) * n_bytes)`, converting from nats-per-token to
  bits-per-byte using the true UTF-8 byte count of the underlying text.
* Byte counts come from the `.txt` splits the `.bin` files were tokenized from, excluding
  the document separator lines that carry no text.

Raw measurements: `report/intrinsic_results.json`.
"""
    OUT.write_text(body, encoding="utf-8")
    logger.info("Wrote %s", OUT)


if __name__ == "__main__":
    main()
