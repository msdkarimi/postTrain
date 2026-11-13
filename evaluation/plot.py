"""Plots and a table-view report comparing evaluated models.

    python -m evaluation.plot                                   # every model found under outputs/eval and outputs/bench
    python -m evaluation.plot --models base,sft --reference base
    python -m evaluation.plot --wandb post-train                # also log the images and tables to Weights & Biases

Reads  outputs/eval/<model>/summary.json   (held-out agent tasks, from evaluation.run_eval)
       outputs/bench/<model>/bench.json    (IFEval / GSM8K / MMLU, from evaluation.bench)
Writes outputs/plots/*.png and outputs/plots/report.md (the table twin of every chart: all numbers, nothing hidden in a tooltip).

Design rules (dataviz skill): the MODEL is the category, so each model gets one validated categorical color and keeps it in every chart
(color follows the entity, assigned in a fixed order: base, instruct, then alphabetical; at most 4 models). Gains / drops use a
blue-red diverging pair around a neutral zero line. Thin marks with a surface gap, solid hairline grid, legend always shown,
values labelled selectively. Light surface only (static images).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # categorical slots 1-4 (validated light-surface set)
POS, NEG = "#2a78d6", "#e34948"                                 # diverging poles (blue = better than reference, red = worse)
BENCH_LABELS = [("bfcl_ast", "BFCL\n(function calling)"), ("ifeval_prompt_strict", "IFEval\n(prompt, strict)"), ("ifeval_inst_strict", "IFEval\n(instruction, strict)"),
                ("gsm8k_flexible", "GSM8K\n(5-shot)"), ("mmlu", "MMLU\n(5-shot)")]


def _font() -> None:
    have = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"):
        if cand in have:
            plt.rcParams["font.family"] = cand
            return


def order_models(names) -> list:
    return sorted(names, key=lambda n: (0 if n.startswith("base") else 1 if "instruct" in n else 2, n))


def load(eval_root: Path, bench_root: Path) -> dict:
    data = {}
    for root, fname, key in ((eval_root, "summary.json", "eval"), (bench_root, "bench.json", "bench")):
        if root.exists():
            for d in sorted(root.iterdir()):
                if (d / fname).exists():
                    data.setdefault(d.name, {})[key] = json.loads((d / fname).read_text())
    return data


def colors_for(all_names: list, shown: list) -> dict:
    if len(all_names) <= len(SERIES):
        return {n: SERIES[all_names.index(n)] for n in shown}
    if len(shown) > len(SERIES):
        raise SystemExit(f"{len(shown)} models is too many to tell apart by color (max {len(SERIES)}): pick some with --models")
    return {n: SERIES[i] for i, n in enumerate(shown)}              # >4 models exist: colors follow the subset you asked for


# ---- chart chrome -------------------------------------------------------------------------------------------------------------------
def new_fig(w, h, title, subtitle=None):
    fig = plt.figure(figsize=(w, h), facecolor=SURFACE)
    fig.text(0.04, 0.97, title, fontsize=13, fontweight="bold", color=INK, va="top", ha="left")
    if subtitle:
        fig.text(0.04, 0.925, subtitle, fontsize=9, color=INK2, va="top", ha="left")
    return fig


def style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def legend(fig, names, colors, x=0.04, y=0.885):
    fig.legend([Patch(facecolor=colors[n], edgecolor="none", label=n) for n in names], names, loc="upper left", bbox_to_anchor=(x, y), ncol=len(names),
               frameon=False, fontsize=9, labelcolor=INK2, handlelength=1.1, columnspacing=1.6)


def save(fig, path: Path):
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


# ---- charts -------------------------------------------------------------------------------------------------------------------------
def chart_by_category(models, cols, ev, path):
    cats = sorted({c for m in models for c in ev[m]["by_category"]})
    groups = ["overall"] + cats
    val = {m: {"overall": ev[m]["overall"]["pass_rate"], **{c: v["pass_rate"] for c, v in ev[m]["by_category"].items()}} for m in models}
    n = len(models)
    fig = new_fig(max(8.5, 0.95 * len(groups) + 2.5), 5.4, "Held-out agent tasks: pass rate by category",
                  "Share of tasks whose final state and reply pass the checker. Same tasks, same checkers for every model.")
    legend(fig, models, cols)
    ax = fig.add_axes([0.07, 0.2, 0.91, 0.58])
    style(ax, "y")
    w = 0.74 / n
    for i, m in enumerate(models):
        for g, grp in enumerate(groups):
            v = val[m].get(grp)
            if v is None:
                continue
            x = g + (i - (n - 1) / 2) * w
            ax.bar(x, v * 100, width=w * 0.92, color=cols[m], edgecolor=SURFACE, linewidth=1.2)
            if grp == "overall":
                ax.text(x, v * 100 + 1.5, f"{v * 100:.0f}%", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=28, ha="right")
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{t}%" for t in (0, 25, 50, 75, 100)])
    return save(fig, path)


def chart_skills(models, cols, ev, path, top=18):
    skills = {}
    for m in models:
        for k, v in ev[m]["by_skill"].items():
            skills[k] = max(skills.get(k, 0), v["n"])
    keep = sorted(skills, key=lambda k: -skills[k])[:top]
    if not keep:
        return None
    ref = models[0]
    keep = sorted(keep, key=lambda k: ev[ref]["by_skill"].get(k, {}).get("pass_rate", 0))
    n = len(models)
    fig = new_fig(9.5, 0.42 * len(keep) * max(1, n * 0.62) + 2.2, "Held-out agent tasks: pass rate by skill",
                  f"Skills = task tags (the {len(keep)} most common). Sorted by {ref}.")
    legend(fig, models, cols)
    ax = fig.add_axes([0.24, 0.07, 0.72, 0.74 - 0.0 if len(keep) > 8 else 0.6])
    style(ax, "x")
    h = 0.76 / n
    for i, m in enumerate(models):
        for j, k in enumerate(keep):
            v = ev[m]["by_skill"].get(k, {}).get("pass_rate")
            if v is not None:
                ax.barh(j + (i - (n - 1) / 2) * h, v * 100, height=h * 0.9, color=cols[m], edgecolor=SURFACE, linewidth=1.0)
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels([f"{k.replace('_', ' ')}  (n={skills[k]})" for k in keep], color=INK2)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels([f"{t}%" for t in (0, 25, 50, 75, 100)])
    return save(fig, path)


def _diverging_panel(ax, labels, deltas, annotate_all=False, note=None):
    style(ax, "x")
    y = list(range(len(labels)))
    ax.barh(y, deltas, height=0.68, color=[POS if d >= 0 else NEG for d in deltas], edgecolor=SURFACE, linewidth=1.0)
    ax.axvline(0, color=AXIS, linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK2, fontsize=8.5)
    span = max(max(abs(d) for d in deltas), 3.0 if note else 1.0)
    ax.set_xlim(-span * 1.35, span * 1.35)
    idxs = range(len(deltas)) if annotate_all else sorted({deltas.index(max(deltas)), deltas.index(min(deltas))})
    for i in idxs:
        d = deltas[i]
        txt = (note[i] if note else f"{d:+.1f}")
        if note:       # long annotations go on the empty (positive) side of the zero line so they never hit the tick labels
            ax.text(span * 0.04 if d < 0 else d + span * 0.04, i, txt, va="center", ha="left", fontsize=8.5, color=INK)
        else:
            ax.text(d + (span * 0.04) * (1 if d >= 0 else -1), i, txt, va="center", ha="left" if d >= 0 else "right", fontsize=8.5, color=INK)
    return ax


def chart_family_delta(models, cols, ev, ref, path, k=10):
    others = [m for m in models if m != ref]
    if not others:
        return None
    fig = new_fig(5.6 * len(others) + 1.2, 7.4, f"Held-out tasks: change in pass rate vs {ref}, by family",
                  "Percentage points; the 10 biggest gains and drops per model. Families have ~8 tasks, so one task = 12.5 points: read single bars with care.")
    gs = fig.add_gridspec(1, len(others), left=0.2 / len(others) + 0.04, right=0.98, top=0.83, bottom=0.08, wspace=0.95)
    for c, m in enumerate(others):
        fam = {f: (ev[m]["by_family"][f]["pass_rate"] - ev[ref]["by_family"][f]["pass_rate"]) * 100 for f in ev[m]["by_family"] if f in ev[ref]["by_family"]}
        ordered = sorted(fam.items(), key=lambda kv: kv[1])
        pick = ordered[:k] + ordered[-k:] if len(ordered) > 2 * k else ordered
        pick = sorted({f: d for f, d in pick}.items(), key=lambda kv: kv[1])
        ax = fig.add_subplot(gs[0, c])
        _diverging_panel(ax, [f.replace("_", " ") for f, _ in pick], [d for _, d in pick])
        ax.set_title(m, fontsize=10, color=INK, loc="left", fontweight="bold")
        ax.set_xlabel("percentage points  (blue: better than " + ref + ", red: worse)", fontsize=8, color=MUTED)
    return save(fig, path)


def chart_benchmarks(models, cols, bench, path):
    have = [(k, lab) for k, lab in BENCH_LABELS if any(k in bench[m]["scores"] for m in models)]
    n = len(models)
    fig = new_fig(max(7.5, 2.4 * len(have) + 2), 5.2, "General-ability benchmarks (forgetting check)",
                  "Higher is better. Whiskers = one standard error. Same settings and questions for every model (see report.md).")
    legend(fig, models, cols)
    ax = fig.add_axes([0.07, 0.17, 0.91, 0.6])
    style(ax, "y")
    w = 0.74 / n
    for i, m in enumerate(models):
        for g, (k, _) in enumerate(have):
            s = bench[m]["scores"].get(k)
            if not s:
                continue
            x = g + (i - (n - 1) / 2) * w
            ax.bar(x, s["value"] * 100, width=w * 0.92, color=cols[m], edgecolor=SURFACE, linewidth=1.2)
            if s.get("stderr"):
                ax.errorbar(x, s["value"] * 100, yerr=s["stderr"] * 100, color=INK2, linewidth=1.0, capsize=2.5, capthick=1.0)
            ax.text(x, s["value"] * 100 + (s.get("stderr") or 0) * 100 + 1.2, f"{s['value'] * 100:.0f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(have)))
    ax.set_xticklabels([lab for _, lab in have], color=INK2, fontsize=9)
    top = max([bench[m]["scores"][k]["value"] * 100 for m in models for k, _ in have if k in bench[m]["scores"]] + [10])
    ax.set_ylim(0, min(105, top * 1.25 + 4))
    ticks = [t for t in range(0, 101, 10 if top > 40 else 5) if t <= ax.get_ylim()[1]]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t}%" for t in ticks])
    return save(fig, path)


def forgetting_deltas(models, bench, ref):
    out = {}
    for m in models:
        if m == ref:
            continue
        rows = []
        for k, lab in BENCH_LABELS:
            a, b = bench[ref]["scores"].get(k), bench[m]["scores"].get(k)
            if a and b:
                se = math.sqrt((a.get("stderr") or 0) ** 2 + (b.get("stderr") or 0) ** 2)
                rows.append((lab.replace("\n", " "), (b["value"] - a["value"]) * 100, se * 100))
        out[m] = rows
    return out


def chart_forgetting(models, bench, ref, path):
    deltas = forgetting_deltas(models, bench, ref)
    deltas = {m: r for m, r in deltas.items() if r}
    if not deltas:
        return None
    fig = new_fig(5.8 * len(deltas) + 1.4, 4.3, f"Forgetting check: change vs {ref} (percentage points)",
                  "Negative (red) = the model got worse than the reference. 'within noise' = smaller than two standard errors.")
    gs = fig.add_gridspec(1, len(deltas), left=0.04 + 0.2 / len(deltas), right=0.98, top=0.78, bottom=0.2, wspace=1.0)
    for c, (m, rows) in enumerate(deltas.items()):
        ax = fig.add_subplot(gs[0, c])
        labels, ds, ses = zip(*rows)
        note = [f"{d:+.1f}" + ("  (within noise)" if abs(d) < 2 * s else "") for d, s in zip(ds, ses)]
        _diverging_panel(ax, list(labels), list(ds), annotate_all=True, note=note)
        ax.set_title(m, fontsize=10, color=INK, loc="left", fontweight="bold")
        ax.set_xlabel("percentage points  (blue: better than " + ref + ", red: worse)", fontsize=8, color=MUTED)
    return save(fig, path)


# ---- table view ---------------------------------------------------------------------------------------------------------------------
def write_report(models, ev, bench, ref, figs, path: Path):
    L = ["# Model comparison", "", f"Models: {', '.join(models)}. Reference for deltas: **{ref}**.", ""]
    L += [f"![{p.stem}]({p.name})" for p in figs if p] + [""]
    if any(m in ev for m in models):
        L += ["## Held-out agent tasks (table view)", "", "| | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
        for lab, get in (("pass rate", lambda s: f"{100 * s['overall']['pass_rate']:.1f}% ({s['overall']['pass']}/{s['overall']['n']})"),
                         ("format errors / assistant msg", lambda s: f"{s['overall']['format_error_rate']:.3f}"), ("tool/API errors / call", lambda s: f"{s['overall']['tool_error_rate']:.3f}")):
            L.append(f"| {lab} | " + " | ".join(get(ev[m]) if m in ev else "-" for m in models) + " |")
        for title, key in (("By category", "by_category"), ("By skill", "by_skill"), ("By family", "by_family")):
            keys = sorted({k for m in models if m in ev for k in ev[m][key]})
            L += ["", f"### {title}", "", "| | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
            for k in keys:
                L.append(f"| {k} | " + " | ".join((f"{100 * ev[m][key][k]['pass_rate']:.1f}% (n={ev[m][key][k]['n']})" if m in ev and k in ev[m][key] else "-") for m in models) + " |")
    if any(m in bench for m in models):
        L += ["", "## Benchmarks (table view)", "", "| | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
        for k, lab in BENCH_LABELS:
            L.append(f"| {lab.replace(chr(10), ' ')} | " + " | ".join((f"{100 * bench[m]['scores'][k]['value']:.1f}% (+-{100 * (bench[m]['scores'][k].get('stderr') or 0):.1f}, n={bench[m]['scores'][k].get('n')})"
                                                                     if m in bench and k in bench[m]["scores"] else "-") for m in models) + " |")
        L += ["", "Settings: " + json.dumps(next(iter(bench.values()))["settings"]) + f"; preset {next(iter(bench.values()))['preset']}.", ""]
        for m, rows in forgetting_deltas([x for x in models if x in bench], bench, ref).items() if ref in bench else []:
            L += [f"### {m} vs {ref}", "", "| benchmark | change (points) | noise (2 x se) |", "|---|---|---|"] + [f"| {a} | {d:+.1f} | {2 * s:.1f} |" for a, d, s in rows] + [""]
    path.write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="outputs/eval")
    ap.add_argument("--bench-root", default="outputs/bench")
    ap.add_argument("--out", default="outputs/plots")
    ap.add_argument("--models", default="", help="comma-separated subset (default: all found)")
    ap.add_argument("--reference", default="", help="model the deltas are measured against (default: base, else the first model)")
    ap.add_argument("--wandb", default="", help="W&B project to log the images and tables to")
    args = ap.parse_args()
    _font()
    data = load(Path(args.eval_root), Path(args.bench_root))
    if not data:
        raise SystemExit(f"nothing found under {args.eval_root} / {args.bench_root}: run evaluation.run_eval and evaluation.bench first")
    allm = order_models(data)
    models = [m for m in allm if not args.models or m in args.models.split(",")]
    cols = colors_for(allm, models)
    ref = args.reference or models[0]
    ev = {m: data[m]["eval"] for m in models if "eval" in data[m]}
    bn = {m: data[m]["bench"] for m in models if "bench" in data[m]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figs = []
    em, bm = [m for m in models if m in ev], [m for m in models if m in bn]
    if em:
        figs += [chart_by_category(em, cols, ev, out / "heldout_by_category.png"), chart_skills(em, cols, ev, out / "heldout_skills.png")]
        if ref in ev:
            figs.append(chart_family_delta(em, cols, ev, ref, out / "heldout_family_delta.png"))
    if bm:
        figs.append(chart_benchmarks(bm, cols, bn, out / "benchmarks.png"))
        if ref in bn:
            figs.append(chart_forgetting(bm, bn, ref, out / "forgetting_delta.png"))
    figs = [f for f in figs if f]
    write_report(models, ev, bn, ref, figs, out / "report.md")
    print("wrote:", *[str(f) for f in figs], str(out / "report.md"), sep="\n  ")
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb, job_type="eval-report", name="model-comparison")
        for m in models:                                             # scalars, so W&B's own compare / parallel-coordinates views work
            for k, v in (ev[m]["overall"].items() if m in ev else []):
                if isinstance(v, (int, float)):
                    run.summary[f"heldout/{m}/{k}"] = v
            for k, v in (bn[m]["scores"].items() if m in bn else []):
                run.summary[f"bench/{m}/{k}"] = v["value"]
        keys = [k for k, _ in BENCH_LABELS if any(k in bn[m]["scores"] for m in bn)]
        run.log({"bench_table": wandb.Table(columns=["model"] + keys, data=[[m] + [bn[m]["scores"].get(k, {}).get("value") for k in keys] for m in bn]),
                 "heldout_table": wandb.Table(columns=["model", "pass_rate"], data=[[m, ev[m]["overall"]["pass_rate"]] for m in ev])})
        run.log({f"plots/{f.stem}": wandb.Image(str(f)) for f in figs})
        run.log({"report": wandb.Html((out / "report.md").read_text().replace("\n", "<br>"))})
        run.finish()


if __name__ == "__main__":
    main()
