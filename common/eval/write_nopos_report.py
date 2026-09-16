"""
Bonus ablation report: the no-positional-embeddings model against the standard Model H.

    python -m common.eval.write_nopos_report

Writes report/nopos_ablation.md. Every number is read from a results file — the Phase 2
files for the standard model, and the `*_nopos.json` files for the ablated one — so the
comparison cannot drift from the measurements it describes.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
REPORT = ROOT / "report"

STANDARD = "Model H (standard)"
ABLATED = "no positional embeddings"
TEMPS = [("greedy", "greedy"), ("temp_0.5", "T = 0.5"), ("temp_1.0", "T = 1.0"), ("temp_1.5", "T = 1.5")]
METRICS = [("bleu", "BLEU", 2), ("chrf++", "chrF++", 2), ("rouge_l", "ROUGE-L", 3),
           ("distinct_1", "Distinct-1", 3), ("distinct_2", "Distinct-2", 3),
           ("repetition_rate_4gram", "Repetition (4-gram)", 3)]


def load(name):
    return json.loads((REPORT / name).read_text(encoding="utf-8"))


def hindi(rows):
    return [r for r in rows if r["language"] == "hindi"]


def table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def delta(new, old, pct=True):
    if old == 0:
        return "—"
    return f"{new / old - 1:+.1%}" if pct else f"{new - old:+.3f}"


def section_setup():
    std = json.loads((ROOT / "hindi" / "configs" / "model_config.json").read_text())
    abl = json.loads((ROOT / "hindi" / "configs" / "model_config_nopos.json").read_text())
    from common.model.config import GPTConfig
    s, a = GPTConfig(**std), GPTConfig(**abl)
    diff = [k for k in std if std.get(k) != abl.get(k)] + \
           [k for k in abl if k not in std]
    return "\n".join([
        "## 1. Setup",
        "",
        "Hindi was chosen for the ablation: the shorter 512-token context makes it the faster "
        "run, and Model H is the stronger and cleaner baseline of the two.",
        "",
        table(["", STANDARD, ABLATED],
              [["Positional embeddings", "learned absolute, 512 × 512", "**none**"],
               ["Parameters", f"{s.n_params():,}", f"{a.n_params():,}"],
               ["Non-embedding parameters", f"{s.n_params(non_embedding=True):,}",
                f"{a.n_params(non_embedding=True):,}"],
               ["Layers / width / heads", f"{s.n_layers} / {s.d_model} / {s.n_heads}",
                f"{a.n_layers} / {a.d_model} / {a.n_heads}"],
               ["Context length", s.max_seq_len, a.max_seq_len],
               ["Training tokens", "500,023,296", "500,023,296"],
               ["Tokens per optimizer step", "24,576", "24,576"],
               ["Config fields that differ", ", ".join(diff) or "none", ""]]),
        "",
        "The only architectural difference is the position table, worth "
        f"{s.n_params() - a.n_params():,} parameters ({s.max_seq_len} × {s.d_model}). Everything "
        "else — corpus, tokenizer, schedule, batch, seed — is identical.",
        "",
        "**One difference that could not be held fixed:** Model H was pretrained on a Kaggle T4 "
        "in fp16, while the ablation ran on the local RTX 3050 in bf16. Precision and hardware "
        "therefore differ alongside the ablation, which is a limitation of the comparison.",
    ])


def section_training():
    def curve(path):
        recs = [json.loads(l) for l in (ROOT / path).open() if "val_loss" in l]
        return recs

    std, abl = curve("hindi/train/run/training_log.jsonl"), curve("hindi/train/run_nopos/training_log.jsonl")
    rows = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        i = max(0, int(len(std) * frac) - 1)
        j = max(0, int(len(abl) * frac) - 1)
        rows.append([f"{frac:.0%} of training", f"{std[i]['step']:,}", f"{std[i]['val_loss']:.4f}",
                     f"{abl[j]['step']:,}", f"{abl[j]['val_loss']:.4f}",
                     delta(abl[j]["val_loss"], std[i]["val_loss"], pct=False)])
    return "\n".join([
        "## 2. Training",
        "",
        table(["progress", "standard step", "standard val loss", "ablated step",
               "ablated val loss", "difference"], rows),
        "",
        f"Best validation loss during training: standard **{min(r['val_loss'] for r in std):.4f}**, "
        f"ablated **{min(r['val_loss'] for r in abl):.4f}**.",
    ])


def section_intrinsic():
    std = {(r["split"], r["context"]): r for r in hindi(load("intrinsic_results.json"))}
    abl = {(r["split"], r["context"]): r for r in hindi(load("intrinsic_results_nopos.json"))}
    rows = []
    for key in sorted(set(std) & set(abl)):
        s, a = std[key], abl[key]
        assert s["n_tokens"] == a["n_tokens"], f"different token counts for {key}"
        rows.append([key[0], key[1], f"{s['perplexity']:.2f}", f"{a['perplexity']:.2f}",
                     f"{s['bits_per_byte']:.4f}", f"{a['bits_per_byte']:.4f}",
                     delta(a["bits_per_byte"], s["bits_per_byte"])])
    return "\n".join([
        "## 3. Perplexity and bits per byte",
        "",
        "Same held-out text, identical token counts.",
        "",
        table(["split", "context", "PPL standard", "PPL ablated", "BPB standard",
               "BPB ablated", "BPB change"], rows),
    ])


def section_generation():
    std = hindi(load("generation_results.json"))[0]["metrics"]
    abl = hindi(load("generation_results_nopos.json"))[0]["metrics"]
    parts = ["## 4. Generation quality", "",
             "200 prompts of 64 tokens from the validation split, continued for 128 tokens, "
             "scored against the true continuation.", ""]
    for key, label in TEMPS:
        rows = [[name, f"{std[key][m]:.{d}f}", f"{abl[key][m]:.{d}f}",
                 delta(abl[key][m], std[key][m])] for m, name, d in METRICS]
        parts += [f"### {label}", "", table(["metric", STANDARD, ABLATED, "change"], rows), ""]
    return "\n".join(parts)


def section_attention():
    std = hindi(load("attention_results.json"))[0]
    abl = hindi(load("attention_results_nopos.json"))[0]
    import numpy as np

    def stat(res, name):
        return np.array([res["layers"][str(l)][name] for l in range(res["n_layers"])])

    rows = []
    for l in range(std["n_layers"]):
        rows.append([l,
                     f"{stat(std, 'entropy_normalised')[l].mean():.3f}",
                     f"{stat(abl, 'entropy_normalised')[l].mean():.3f}",
                     f"{stat(std, 'mean_distance')[l].mean():.1f}",
                     f"{stat(abl, 'mean_distance')[l].mean():.1f}",
                     f"{stat(std, 'prev_token')[l].mean():.3f}",
                     f"{stat(abl, 'prev_token')[l].mean():.3f}"])
    ps, pa = stat(std, "prev_token"), stat(abl, "prev_token")
    si, hi = np.unravel_index(ps.argmax(), ps.shape)
    sj, hj = np.unravel_index(pa.argmax(), pa.shape)
    return "\n".join([
        "## 5. Attention",
        "",
        "64 held-out windows of 256 tokens, seed 0 — the same windows Phase 2 used. Means over "
        "the 8 heads of each layer.",
        "",
        table(["layer", "entropy std", "entropy ablated", "distance std", "distance ablated",
               "prev-token std", "prev-token ablated"], rows),
        "",
        f"Strongest previous-token head: standard **L{si}H{hi}** at {ps[si, hi]:.3f}; "
        f"ablated **L{sj}H{hj}** at {pa[sj, hj]:.3f}. Heads with more than half their mass on "
        f"the previous token: standard **{int((ps > 0.5).sum())}**, ablated "
        f"**{int((pa > 0.5).sum())}**.",
        "",
        "Heatmaps: `figures/attention_nopos_layer0.png`, `figures/attention_nopos_layer5.png` "
        "(compare with `figures/attention_hindi_layer0.png` and `attention_hindi_layer5.png`).",
    ])


def main():
    doc = "\n\n".join([
        "# Bonus ablation — a Hindi model trained with no positional embeddings",
        "<!-- GENERATED by common/eval/write_nopos_report.py. Do not edit by hand; regenerate. -->",
        "The optional ablation from the brief: retrain one model without positional embeddings, "
        "run the full Phase 2 evaluation suite on it, and compare against the standard model.",
        section_setup(), section_training(), section_intrinsic(), section_generation(),
        section_attention(),
        "## 6. Figures\n\n"
        "![Ablation comparison](figures/nopos_comparison.png)\n\n"
        "![Attention without positions](figures/nopos_attention.png)",
    ]) + "\n"
    out = REPORT / "nopos_ablation.md"
    out.write_text(doc, encoding="utf-8")
    print(f"wrote {out} ({len(doc.splitlines())} lines)")


if __name__ == "__main__":
    main()
