"""
Render report/phase2_generation.md from generation_results.json.

Generated rather than hand-written so the tables and the sample outputs cannot drift from
the measurements they claim to describe.

Usage:
    python -m common.eval.write_generation_report
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("generation_report")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = PROJECT_ROOT / "report" / "generation_results.json"
OUT = PROJECT_ROOT / "report" / "phase2_generation.md"

DISPLAY = {"hindi": "Model H (Hindi)", "sindhi": "Model L (Sindhi)"}
ORDER = ["greedy", "temp_0.5", "temp_1.0", "temp_1.5"]
LABEL = {"greedy": "Greedy (T=0)", "temp_0.5": "T = 0.5",
         "temp_1.0": "T = 1.0", "temp_1.5": "T = 1.5"}


def metric_table(result: dict) -> str:
    rows = ["| Decoding | BLEU-4 | chrF++ | ROUGE-L | Distinct-1 | Distinct-2 | Repetition (4-gram) |",
            "|---|---|---|---|---|---|---|"]
    m = result["metrics"]
    best_bleu = max(m[s]["bleu"] for s in ORDER)
    best_chrf = max(m[s]["chrf++"] for s in ORDER)
    for s in ORDER:
        v = m[s]
        bleu = f"**{v['bleu']:.2f}**" if v["bleu"] == best_bleu else f"{v['bleu']:.2f}"
        chrf = f"**{v['chrf++']:.2f}**" if v["chrf++"] == best_chrf else f"{v['chrf++']:.2f}"
        rows.append(f"| {LABEL[s]} | {bleu} | {chrf} | {v['rouge_l']:.4f} | "
                    f"{v['distinct_1']:.3f} | {v['distinct_2']:.3f} | "
                    f"{v['repetition_rate_4gram']:.3f} |")
    return "\n".join(rows)


RANK_LABEL = {
    "best": "Best of 200",
    "upper_quartile": "Upper quartile (75th percentile)",
    "median": "Median",
    "lower_quartile": "Lower quartile (25th percentile)",
    "worst": "Worst of 200",
}
RANK_ORDER = ["best", "upper_quartile", "median", "lower_quartile", "worst"]


def samples_block(result: dict) -> str:
    """One prompt/output/reference triple per decoding setting, first-drawn prompt."""
    out = []
    for s in result["samples"]:
        out.append(f"**{LABEL[s['setting']]}**\n")
        out.append(f"> **Prompt (64 tokens):** {s['prompt'][:220]}\n")
        out.append(f"> **Model output (128 tokens):** {s['generated'][:260]}\n")
        out.append(f"> **True continuation:** {s['reference'][:220]}\n")
    return "\n".join(out)


def ranked_block(result: dict, setting: str = "temp_1.0") -> str:
    """
    The spread of outputs at one decoding setting, best to worst.

    A single example says nothing about a model that produces 200 different continuations.
    These are ranked by per-sample chrF++ against the true continuation — an objective,
    reproducible criterion — and five points across the distribution are shown.
    """
    s = next(x for x in result["samples"] if x["setting"] == setting)
    if "ranked" not in s:
        return "_Re-run `common.eval.generation` to populate ranked samples._"

    spread = s["chrf_spread"]
    out = [f"Ranked by per-sample chrF++ against the true continuation. Across the 200 "
           f"prompts the score ranges from **{spread['best']:.1f}** down to "
           f"**{spread['worst']:.1f}** (median **{spread['median']:.1f}**), so a single "
           f"example is not representative of anything.\n"]
    for key in RANK_ORDER:
        r = s["ranked"][key]
        out.append(f"#### {RANK_LABEL[key]} — chrF++ {r['rank_score_chrf']:.1f}\n")
        out.append(f"> **Prompt:** {r['prompt'][:200]}\n")
        out.append(f"> **Model output:** {r['generated'][:240]}\n")
        out.append(f"> **True continuation:** {r['reference'][:200]}\n")
    return "\n".join(out)


def main():
    if not RESULTS.exists():
        logger.error("Missing %s — run common.eval.generation first.", RESULTS)
        raise SystemExit(1)
    results = json.loads(RESULTS.read_text())
    by = {r["language"]: r for r in results}
    h, s = by["hindi"], by["sindhi"]

    body = f"""# Phase 2: Generation Quality Evaluation

<!-- GENERATED FILE - do not edit by hand.
     Regenerate with:
       python -m common.eval.generation
       python -m common.eval.write_generation_report
-->

Each model was given **{h['n_prompts']} prompts of {h['prompt_tokens']} tokens** drawn from
its own validation split, asked to continue for **{h['generated_tokens']} tokens**, and
scored against the text that actually followed. Four decoding settings: greedy plus
temperatures 0.5, 1.0 and 1.5.

Prompts are screened to contain no document separator. A window straddling a boundary
would pair a prompt with the opening of an *unrelated* document, so the "correct
continuation" would be text no model could predict — depressing every score for a reason
unconnected to model quality, and doing so unevenly between two languages whose document
lengths differ by a factor of three.

Figure: `figures/generation_metrics.png`

## How to read these metrics

| Metric | What it measures | Direction |
|---|---|---|
| **BLEU-4** | overlap of word 4-grams with the reference | higher better |
| **chrF++** | overlap of *character* n-grams plus word bigrams | higher better |
| **ROUGE-L** | longest common subsequence with the reference, as F1 | higher better |
| **Distinct-1 / -2** | fraction of generated unigrams / bigrams that are unique | higher = more varied |
| **Repetition rate** | fraction of 4-grams repeating *within one sample* | lower better |

**A caveat that belongs up front, not in a footnote.** BLEU and ROUGE were designed for
tasks with a single correct answer — translation, summarisation — where the reference *is*
the right output. Open-ended continuation has no single correct answer: a fluent, sensible
continuation that simply differs from what the author happened to write next scores near
zero. **BLEU values of 1–2 are therefore normal here and do not indicate a broken model.**
These metrics are reported because the brief requires them and because they are comparable
*between our two models*, not as absolute quality scores.

## {DISPLAY['hindi']}

{metric_table(h)}

## {DISPLAY['sindhi']}

{metric_table(s)}

## Findings

### 1. Greedy decoding degenerates into loops

Repetition rate is **{h['metrics']['greedy']['repetition_rate_4gram']:.3f}** for Model H and
**{s['metrics']['greedy']['repetition_rate_4gram']:.3f}** for Model L: over 80% of 4-grams
in a greedy sample are repeats of something earlier in that same sample. Distinct-2 collapses
to {h['metrics']['greedy']['distinct_2']:.3f} and {s['metrics']['greedy']['distinct_2']:.3f}.

Always taking the most likely token drives the model into a cycle it cannot escape, because
each repetition makes the next repetition more likely still. This is the expected failure
mode for a model of this size under greedy decoding, and it is exactly what the diversity
diagnostics exist to expose. The sample outputs below show it plainly.

### 2. The metrics disagree about the best temperature, and the disagreement is informative

BLEU and ROUGE-L peak at **T = 0.5**; chrF++ peaks at **T = 1.0**.

BLEU rewards exact word-sequence matches, so it favours the conservative setting that
reproduces common phrasing. chrF++ operates on character n-grams and therefore tolerates
morphological variation — which matters enormously in Devanagari and Perso-Arabic, where an
inflected form differing by one suffix is a total miss for BLEU but largely correct for
chrF. For morphologically rich languages, **chrF++ is the more trustworthy of the two**,
and its peak at T = 1.0 is the better guide to where the model actually generates well.

### 3. Temperature 1.5 is too hot

BLEU falls to {h['metrics']['temp_1.5']['bleu']:.2f} / {s['metrics']['temp_1.5']['bleu']:.2f}
and Distinct-2 reaches {h['metrics']['temp_1.5']['distinct_2']:.3f} /
{s['metrics']['temp_1.5']['distinct_2']:.3f} — almost every bigram is unique. That is not
creativity; it is the distribution flattened until the output stops being constrained by
the language. Repetition of exactly 0.000 alongside near-perfect distinctness is the
signature of noise, not fluency.

### 4. Model H edges Model L, but the margin is small

At T = 1.0: chrF++ **{h['metrics']['temp_1.0']['chrf++']:.2f}** vs
**{s['metrics']['temp_1.0']['chrf++']:.2f}**, ROUGE-L
{h['metrics']['temp_1.0']['rouge_l']:.4f} vs {s['metrics']['temp_1.0']['rouge_l']:.4f}.

The direction agrees with the intrinsic evaluation, but the margin is far smaller than the
78% bits-per-byte gap would suggest. **This is itself a finding:** overlap metrics are noisy
enough on open-ended generation that they barely resolve a difference perplexity shows
unambiguously. It is a caution against treating BLEU-style scores as a primary measure of
language-model quality.

Model L is also markedly more repetitive at T = 0.5
({s['metrics']['temp_0.5']['repetition_rate_4gram']:.3f} vs
{h['metrics']['temp_0.5']['repetition_rate_4gram']:.3f}), suggesting it leans harder on
memorised phrasing once the distribution is sharpened.

## Generated samples — one per decoding setting

The same prompt at each of the four settings, so the effect of temperature is visible
directly.

### {DISPLAY['hindi']}

{samples_block(h)}

### {DISPLAY['sindhi']}

{samples_block(s)}

## The spread of output quality at T = 1.0

**Important caveat on this ranking.** chrF++ measures *agreement with the one continuation
the author happened to write*, which is not the same as quality. A fluent, grammatical
continuation that simply goes somewhere else ranks low. The ranking is a way of surfacing
the range of behaviour, not a verdict on which outputs are good — read the text, not just
the score.

### {DISPLAY['hindi']} — best to worst

{ranked_block(h)}

### {DISPLAY['sindhi']} — best to worst

{ranked_block(s)}

## Qualitative assessment

**Both models learned the language; neither learned to stay on topic.**

The greedy samples are grammatical — correct word order, plausible morphology, sensible
collocations — and then repeat a single clause indefinitely. The T = 1.0 samples read as
fluent native text, with agreement and inflection largely correct, but drift away from the
prompt's subject within a sentence or two. In the Hindi example the prompt concerns women
overcoming obstacles and the continuation moves to plastic pipes and flooding; the Sindhi
example drifts from a 17th-century physician to unrelated dynastic history.

That combination — local fluency, absent long-range coherence — is what a 24M-parameter
model trained on 500M tokens should be expected to produce. Syntax and morphology are
learnable from local statistics; sustained topical control needs far more capacity and data.

Raw measurements and full samples: `report/generation_results.json`.
"""
    OUT.write_text(body, encoding="utf-8")
    logger.info("Wrote %s", OUT)


if __name__ == "__main__":
    main()
