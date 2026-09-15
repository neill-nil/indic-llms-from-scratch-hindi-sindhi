"""
Write report/phase3_results.md: every Phase 3 table, generated from the committed results.

    python -m common.finetune.write_results_report

No number in the output is typed by hand. The narrative report (phase3.md) quotes from this
document, so a result can be traced to the file it came from and regenerated after any
change. The same convention as Phase 2's write_*_report.py modules.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
REPORT = ROOT / "report"
LANGS = ["hindi", "sindhi"]
NAME = {"hindi": "Model H (Hindi)", "sindhi": "Model L (Sindhi)"}
SPLITS = ["test_iid", "test_entity", "test_template", "test_hop"]
SPLIT_NOTE = {"test_iid": "in-distribution", "test_entity": "held-out names",
              "test_template": "held-out phrasings", "test_hop": "4-5 entities (trained on 2-3)"}


def load(name):
    return json.loads((REPORT / name).read_text(encoding="utf-8"))


def by_lang(rows):
    return {r["language"]: r for r in rows}


def pct(v, d=1):
    return "—" if v is None else f"{100 * v:.{d}f}%"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def val_records(lang, run):
    path = ROOT / lang / "finetune" / run / "training_log.jsonl"
    return [json.loads(l) for l in path.open() if "val_accuracy" in l]


def section_dataset():
    rows, meta = [], {}
    for lang in LANGS:
        st = json.loads((ROOT / lang / "finetune" / "data" / "dataset_statistics.json").read_text())
        meta[lang] = st
        for split, s in st["splits"].items():
            rows.append([NAME[lang], split, f"{s['n']:,}", pct(s["tie_rate"]),
                         pct(s["most_common_answer_share"])])
    tpl = {}
    for lang in LANGS:
        t = json.loads((ROOT / "common" / "reasoning" / f"templates_{lang}.json").read_text())["templates"]
        tpl[lang] = (len(t), sum(x.get("author") == "user" for x in t),
                     sum(x["split"] == "heldout" for x in t))
    audit = meta["hindi"]["audit"]
    return "\n".join([
        "## 1. Reasoning corpus",
        "",
        table(["model", "split", "examples", "ties", "most common answer"], rows),
        "",
        "The most common answer is the equality word in every split; its share equals the tie rate.",
        "",
        table(["model", "templates", "human-authored", "held out"],
              [[NAME[l], *tpl[l]] for l in LANGS]),
        "",
        "Leakage audit (asserted at build time, both languages): "
        + ", ".join(f"`{k}` = {v}" for k, v in audit.items()) + ".",
    ])


def section_probe():
    p = load("finetune_lr_probe.json")
    rows = []
    for lr, v in p["summary"].items():
        loss = {r["language"]: r["best_val_loss"] for r in p["runs"] if r["learning_rate"] == float(lr)}
        rows.append([f"{float(lr):g}" + (" **(chosen)**" if float(lr) == p["chosen_learning_rate"] else ""),
                     pct(v["per_language"]["hindi"]), pct(v["per_language"]["sindhi"]), pct(v["mean"]),
                     f"{loss['hindi']:.4f}", f"{loss['sindhi']:.4f}"])
    return "\n".join([
        "## 2. Learning-rate probe",
        "",
        f"One epoch per rate, validation split only. Selection: {p['selection']}.",
        "",
        table(["learning rate", "H best val acc", "L best val acc", "mean", "H best val loss",
               "L best val loss"], rows),
    ])


def section_runs():
    rows = []
    for lang in LANGS:
        for run, label in (("run", "loss on answer (main)"), ("run_fullseq", "loss on full sequence (ablation)")):
            recs = val_records(lang, run)
            best = min(recs, key=lambda r: r["val_loss"])
            rows.append([NAME[lang], label, best["step"], f"{best['val_loss']:.4f}",
                         pct(best["val_accuracy"]), pct(recs[-1]["val_accuracy"])])
    cfg = json.loads((ROOT / "hindi" / "finetune" / "run" / "train_config.json").read_text())
    cfg_l = json.loads((ROOT / "sindhi" / "finetune" / "run" / "train_config.json").read_text())
    keys = ["learning_rate", "min_lr", "warmup_steps", "weight_decay", "beta1", "beta2",
            "grad_clip", "micro_batch", "grad_accum", "epochs", "seed", "loss_on"]
    return "\n".join([
        "## 3. Finetuning runs",
        "",
        table(["model", "loss", "best step", "best val loss", "val acc at best", "val acc at end"], rows),
        "",
        "Validation accuracy is exact-answer accuracy under teacher forcing; validation loss is on "
        "answer tokens for every run, so the rows are comparable.",
        "",
        "Hyperparameters of the main runs (`<language>/finetune/run/train_config.json`):",
        "",
        # These configs were extracted from the checkpoints, which predate the `loss_on`
        # setting; answer-only was the only mode when they were trained.
        table(["setting", "Model H", "Model L"],
              [[k, cfg.get(k, "answer (only mode at the time)"),
                cfg_l.get(k, "answer (only mode at the time)")] for k in keys]),
        "",
        "All settings are identical between the two models.",
    ])


def section_accuracy():
    g = {c: by_lang(load(f"reasoning_{c}.json")) for c in ("pretrained_0shot", "pretrained_4shot", "finetuned_0shot")}
    fc = by_lang(load("reasoning_forced_choice_finetuned.json"))
    fcp = by_lang(load("reasoning_forced_choice_pretrained.json"))
    an = load("reasoning_analysis.json")
    parts = ["## 4. Test accuracy",
             "",
             "*Chance* is a random name from the question (what a model scores knowing only the answer "
             "format). *Forced choice* scores each possible answer by log-probability instead of "
             "generating; *names only* restricts that to the entities in the question.",
             ""]
    for lang in LANGS:
        rows = []
        for s in SPLITS:
            rows.append([f"`{s}` ({SPLIT_NOTE[s]})",
                         pct(an[f"{lang}/finetuned_0shot"]["chance_by_split"][s]["random_name"]),
                         pct(g["pretrained_0shot"][lang]["splits"][s]["accuracy"]),
                         pct(g["pretrained_4shot"][lang]["splits"][s]["accuracy"]),
                         f"**{pct(g['finetuned_0shot'][lang]['splits'][s]['accuracy'])}**",
                         pct(g["finetuned_0shot"][lang]["splits"][s]["off_answer_rate"]),
                         pct(fc[lang]["splits"][s]["accuracy_sum_logprob"]),
                         f"{pct(fc[lang]['splits'][s]['names_only']['accuracy'])} ({pct(fc[lang]['splits'][s]['names_only']['chance'])})",
                         f"{pct(fcp[lang]['splits'][s]['names_only']['accuracy'])} ({pct(fcp[lang]['splits'][s]['names_only']['chance'])})"])
        parts += [f"### {NAME[lang]}", "",
                  table(["split", "chance", "pretrained 0-shot", "pretrained 4-shot",
                         "finetuned greedy", "off-answer", "forced choice",
                         "forced, names only (chance)", "pretrained forced, names only (chance)"], rows), ""]
    def off(cond, l):
        return "/".join(pct(g[cond][l]["splits"][s]["off_answer_rate"], 2) for s in SPLITS)
    parts.append("Pretrained off-answer rate (splits in the order above) — zero-shot: "
                 + ", ".join(f"{NAME[l]} {off('pretrained_0shot', l)}" for l in LANGS)
                 + "; 4-shot: " + ", ".join(f"{NAME[l]} {off('pretrained_4shot', l)}" for l in LANGS) + ".")
    return "\n".join(parts)


def section_controlled():
    from common.finetune.plot_reasoning import task_chance
    an = load("reasoning_analysis.json")
    c = {l: an[f"{l}/finetuned_0shot"]["controlled_in_distribution"] for l in LANGS}
    bins = list(c["hindi"]["accuracy_by_margin"])
    tasks = ["numeric, no tie", "numeric, tie", "chain, argmax", "chain, argmin", "chain, relation"]
    ch = dict(zip(tasks, task_chance()))
    rel = {l: c[l]["relation_confusion"] for l in LANGS}
    return "\n".join([
        "## 5. What the finetuned models learned (test_iid only)",
        "",
        "Restricted to the in-distribution split so the unseen-name failure does not dilute it.",
        "",
        "### 5.1 Accuracy by numeric margin (gap between best and second-best, as a share of the attribute's range)",
        "",
        table(["margin", "Model H", "Model L", "n (H / L)"],
              [[b, pct(c["hindi"]["accuracy_by_margin"][b]["accuracy"]),
                pct(c["sindhi"]["accuracy_by_margin"][b]["accuracy"]),
                f"{c['hindi']['accuracy_by_margin'][b]['n']} / {c['sindhi']['accuracy_by_margin'][b]['n']}"]
               for b in bins]),
        "",
        "### 5.2 Accuracy by task type",
        "",
        table(["task", "Model H", "Model L", "chance (uniform over possible answers)"],
              [[t, pct(c["hindi"]["accuracy_by_task"][t]["accuracy"]),
                pct(c["sindhi"]["accuracy_by_task"][t]["accuracy"]), pct(ch[t])] for t in tasks]),
        "",
        "### 5.3 Relation questions — prediction against truth",
        "",
        table(["model", "counts (gold → predicted)"],
              [[NAME[l], "; ".join(f"{k.replace('gold=', '').replace(' predicted=', ' → ')}: {v}"
                                   for k, v in sorted(rel[l].items()))] for l in LANGS]),
        "",
        "### 5.4 Template author, controlled (numeric comparisons without ties)",
        "",
        table(["model", "generated", "human-authored"],
              [[NAME[l], f"{pct(c[l]['accuracy_by_template_author_numeric_no_tie']['generated']['accuracy'])} "
                         f"(n={c[l]['accuracy_by_template_author_numeric_no_tie']['generated']['n']})",
                f"{pct(c[l]['accuracy_by_template_author_numeric_no_tie']['user']['accuracy'])} "
                f"(n={c[l]['accuracy_by_template_author_numeric_no_tie']['user']['n']})"] for l in LANGS]),
        "",
        "### 5.5 Positional bias: rate at which each non-answer slot is wrongly chosen",
        "",
        table(["model", "first", "middle", "last"],
              [[NAME[l], *(pct(c[l]["wrong_pick_rate_per_slot"][k]) for k in ("first", "middle", "last"))]
               for l in LANGS]),
    ])


def section_errors():
    an = load("reasoning_analysis.json")
    cats = ["off_answer", "wrong_entity", "opposite_extreme", "missed_tie", "false_tie", "relation_flip"]
    parts = ["## 6. Error analysis — finetuned models, greedy", "",
             "Counts of wrong answers by category. *off_answer*: not a possible answer; *wrong_entity*: "
             "a name from the question but not the right one; *opposite_extreme*: the smallest when "
             "asked for the largest, or vice versa; *missed_tie* / *false_tie*: failed to answer, or "
             "wrongly answered, \"equal\"; *relation_flip*: \"more\" for \"less\" or vice versa.", ""]
    for lang in LANGS:
        e = an[f"{lang}/finetuned_0shot"]["errors"]["by_split"]
        parts += [f"### {NAME[lang]}", "",
                  table(["split", *cats], [[s, *(e.get(s, {}).get(k, 0) for k in cats)] for s in SPLITS]), ""]
    return "\n".join(parts)


def section_qualitative():
    an = load("reasoning_analysis.json")
    parts = ["## 7. Qualitative examples", "",
             "Sampled with a fixed seed from the finetuned models' predictions; not selected by hand.", ""]
    for lang in LANGS:
        q = an[f"{lang}/finetuned_0shot"]["qualitative"]
        parts += [f"### {NAME[lang]} — successes", ""]
        for x in q["successes"][:4]:
            parts.append(f"- `{x['split']}` · {x['text'].replace(chr(10), ' ')} **{x['gold']}**")
        parts += ["", f"### {NAME[lang]} — failures", ""]
        for x in q["failures"][:6]:
            parts.append(f"- `{x['split']}` · {x['text'].replace(chr(10), ' ')} — expected **{x['gold']}**, "
                         f"got **{x['predicted'] or '(empty)'}** — *{x['error']}*")
        parts.append("")
    return "\n".join(parts)


def section_lm():
    def rows_of(name):
        return {(r["language"], r["split"], r["context"]): r for r in load(name)}
    pre, ft = rows_of("intrinsic_results.json"), rows_of("intrinsic_results_finetuned.json")
    fs = rows_of("intrinsic_results_finetuned_fullseq.json")
    rows = []
    for k in pre:
        rows.append([NAME[k[0]], k[1], k[2], f"{pre[k]['perplexity']:.2f}", f"{ft[k]['perplexity']:.2f}",
                     f"{pre[k]['bits_per_byte']:.4f}", f"{ft[k]['bits_per_byte']:.4f}",
                     f"{ft[k]['bits_per_byte'] / pre[k]['bits_per_byte'] - 1:+.1%}",
                     f"{fs[k]['bits_per_byte']:.4f}" if k in fs else "—"])
    h, s = ("hindi", "test", 512), ("sindhi", "test", 512)
    return "\n".join([
        "## 8. Language modelling before and after finetuning",
        "",
        "The same held-out corpus text as Phase 2, token counts identical.",
        "",
        table(["model", "split", "context", "PPL pretrained", "PPL finetuned", "BPB pretrained",
               "BPB finetuned", "BPB change", "BPB full-seq ablation (val only)"], rows),
        "",
        f"Matched-context test gap, Model L over Model H in bits per byte: pretrained "
        f"{pre[s]['bits_per_byte'] / pre[h]['bits_per_byte'] - 1:+.1%}, finetuned "
        f"{ft[s]['bits_per_byte'] / ft[h]['bits_per_byte'] - 1:+.1%}.",
    ])


def section_ablation():
    g = {c: by_lang(load(f"reasoning_{c}.json")) for c in ("finetuned_0shot", "finetuned_fullseq_0shot")}
    f = {c: by_lang(load(f"reasoning_forced_choice_{c}.json")) for c in ("finetuned", "finetuned_fullseq")}
    rows = []
    for s in SPLITS:
        row = [s]
        for lang in LANGS:
            row += [pct(g["finetuned_0shot"][lang]["splits"][s]["accuracy"]),
                    pct(g["finetuned_fullseq_0shot"][lang]["splits"][s]["accuracy"]),
                    pct(f["finetuned"][lang]["splits"][s]["names_only"]["accuracy"]),
                    pct(f["finetuned_fullseq"][lang]["splits"][s]["names_only"]["accuracy"])]
        rows.append(row)
    off = [[s, *(pct(g[c][l]["splits"][s]["off_answer_rate"]) for l in LANGS
                 for c in ("finetuned_0shot", "finetuned_fullseq_0shot"))] for s in SPLITS]
    return "\n".join([
        "## 9. Ablation: loss on the full sequence",
        "",
        "Identical to the main runs except that every token is supervised, not only the answer.",
        "",
        table(["split", "H greedy (answer)", "H greedy (full)", "H names-only (answer)", "H names-only (full)",
               "L greedy (answer)", "L greedy (full)", "L names-only (answer)", "L names-only (full)"], rows),
        "",
        table(["split", "H off-answer (answer)", "H off-answer (full)", "L off-answer (answer)",
               "L off-answer (full)"], off),
        "",
        "Language-modelling effect: see the last column of section 8.",
    ])


def section_attention():
    R = load("attention_finetune_results.json")

    def grid(lang, cond, src, stat, split="test_iid"):
        b = R[lang][cond]["corpus"] if src == "corpus" else R[lang][cond]["reasoning"][split]["layers"]
        return np.array([b[str(l)][stat] for l in range(len(b))])

    sel = []
    for lang in LANGS:
        for split in ("test_iid", "test_entity"):
            row = [NAME[lang], SPLIT_NOTE[split]]
            for cond in ("pretrained", "finetuned"):
                gold = grid(lang, cond, "r", "answer_gold_mass", split)
                dis = grid(lang, cond, "r", "answer_distractor_mass", split)
                l, h = np.unravel_index((gold - dis).argmax(), gold.shape)
                row.append(f"L{l}H{h}: {gold[l, h]:.3f} vs {dis[l, h]:.3f}")
            sel.append(row)

    depth = []
    for lang in LANGS:
        for layer in range(6):
            depth.append([NAME[lang], layer] + [
                f"{grid(lang, 'pretrained', 'r', st)[layer].mean():.3f} → {grid(lang, 'finetuned', 'r', st)[layer].mean():.3f}"
                for st in ("answer_number_mass", "answer_entity_mass", "answer_sink_mass")])

    corpus = []
    for lang in LANGS:
        for layer in range(6):
            corpus.append([NAME[lang], layer] + [
                f"{grid(lang, 'pretrained', 'corpus', st)[layer].mean():.3f} → {grid(lang, 'finetuned', 'corpus', st)[layer].mean():.3f}"
                for st in ("entropy_normalised", "mean_distance", "prev_token", "sink")])

    prev = []
    for lang in LANGS:
        pt, ft = grid(lang, "pretrained", "corpus", "prev_token"), grid(lang, "finetuned", "corpus", "prev_token")
        l, h = np.unravel_index(pt.argmax(), pt.shape)
        d_pt, d_ft = grid(lang, "pretrained", "corpus", "mean_distance"), grid(lang, "finetuned", "corpus", "mean_distance")
        prev.append([NAME[lang], f"L{l}H{h}", f"{pt[l, h]:.3f} → {ft[l, h]:.3f}",
                     f"{d_pt[l, h]:.2f} → {d_ft[l, h]:.2f}", f"{int((pt > 0.5).sum())} → {int((ft > 0.5).sum())}"])

    return "\n".join([
        "## 10. Attention before and after finetuning",
        "",
        "Pretrained and finetuned checkpoints on identical inputs: Phase 2's corpus windows (val, "
        "64 × 256 tokens, seed 0) and 500 reasoning prompts per split.",
        "",
        "### 10.1 Selector head — answer-position attention to the correct entity vs the average distractor (strongest head)",
        "",
        table(["model", "prompts", "pretrained", "finetuned"], sel),
        "",
        "### 10.2 Answer-position attention by layer (mean over heads, test_iid), pretrained → finetuned",
        "",
        table(["model", "layer", "→ digits", "→ entity names", "→ position 0 (sink)"], depth),
        "",
        "### 10.3 Corpus text by layer (mean over heads), pretrained → finetuned",
        "",
        table(["model", "layer", "normalised entropy", "mean distance (tokens)", "previous-token mass", "sink"], corpus),
        "",
        "### 10.4 Phase 2's strongest previous-token head",
        "",
        table(["model", "head", "previous-token mass", "mean distance (tokens)",
               "heads with >50% previous-token mass (model-wide)"], prev),
    ])


def main():
    doc = "\n\n".join([
        "# Phase 3 — results tables",
        "<!-- GENERATED by common/finetune/write_results_report.py from the files in report/. "
        "Do not edit by hand; regenerate. -->",
        "Every number below is read from a committed results file. The narrative and "
        "interpretation are in [`phase3.md`](phase3.md).",
        section_dataset(), section_probe(), section_runs(), section_accuracy(),
        section_controlled(), section_errors(), section_qualitative(), section_lm(),
        section_ablation(), section_attention(),
    ]) + "\n"
    out = REPORT / "phase3_results.md"
    out.write_text(doc, encoding="utf-8")
    print(f"wrote {out} ({len(doc.splitlines())} lines)")


if __name__ == "__main__":
    main()
