import json
from pathlib import Path
from math import isnan
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Config
# -------------------------
EVAL_DIR = Path("processed/ground_truth_evaluation")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_PER_VIDEO_TEX = Path("appendix_asr_per_video_longtable.tex")
OUT_AGG_TEX = Path("appendix_asr_aggregate_longtable.tex")
OUT_RANK_TEX = Path("asr_ranking_aggregate.tex")
OUT_STATS_TEX = Path("asr_stats_significance.tex")
OUT_STATS_JSON = Path("asr_stats_significance.json")
OUT_BOXPLOT = FIG_DIR / "asr_wer_boxplot.pdf"

BASELINE_MODEL = "Whisper-Large-v3"
MIN_COMMON_VIDEOS_FOR_TEST = 8  # avoid making claims on tiny overlap

# -------------------------
# Helpers
# -------------------------
def tex_escape(s: str) -> str:
    return (s.replace("\\", "\\textbackslash{}")
             .replace("&", "\\&")
             .replace("%", "\\%")
             .replace("_", "\\_")
             .replace("#", "\\#")
             .replace("{", "\\{")
             .replace("}", "\\}"))

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def bh_fdr(pvals):
    """
    Benjamini–Hochberg FDR correction.
    Returns q-values in original order.
    """
    pvals = np.array(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = q
    return out.tolist()

def wilcoxon_signed_rank(x, y):
    """
    Wilcoxon signed-rank test (two-sided), normal approximation.
    No SciPy dependency.

    Returns: (W_stat, p_value)
    Notes: Uses tie correction approximately; fine for thesis-level reporting.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    d = x - y
    d = d[d != 0]
    n = d.size
    if n == 0:
        return (0.0, 1.0)

    absd = np.abs(d)
    ranks = absd.argsort().argsort().astype(float) + 1.0

    # Average ranks for ties
    unique, inv, counts = np.unique(absd, return_inverse=True, return_counts=True)
    for i, c in enumerate(counts):
        if c > 1:
            idx = np.where(inv == i)[0]
            ranks[idx] = ranks[idx].mean()

    Wpos = ranks[d > 0].sum()
    Wneg = ranks[d < 0].sum()
    W = min(Wpos, Wneg)

    # Normal approximation
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0

    # Tie correction (approx)
    tie_term = 0.0
    for c in counts:
        if c > 1:
            tie_term += c * (c * c - 1)
    if tie_term > 0:
        var -= tie_term / 48.0

    if var <= 0:
        return (float(W), 1.0)

    z = (W - mean) / np.sqrt(var)
    # two-sided p from z using erf
    from math import erf, sqrt
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))
    return (float(W), float(p))

def paired_bootstrap_ci(x, y, iters=5000, alpha=0.05, seed=0):
    """
    Bootstrap CI for mean difference (x - y), paired resampling over videos.
    """
    rng = np.random.default_rng(seed)
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    d = x - y
    n = d.size
    if n == 0:
        return (0.0, 0.0, 0.0)
    samples = rng.choice(d, size=(iters, n), replace=True)
    means = samples.mean(axis=1)
    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)
    return (float(d.mean()), float(lo), float(hi))

# -------------------------
# Load all evaluations
# -------------------------
if not EVAL_DIR.exists():
    raise SystemExit(f"ERROR: {EVAL_DIR} not found. Run from your BasePipeline root.")

per_video_rows = []  # (video, model, wer, cer, acc, gt_words)
wer_by_model = {}    # model -> list of WER across videos
acc_by_model = {}
count_by_model = {}
gt_by_video = {}

# Map: model -> {video -> wer}
wer_matrix = {}

json_files = sorted(EVAL_DIR.glob("*_evaluation.json"))
for fp in json_files:
    data = json.loads(fp.read_text(encoding="utf-8"))
    video = data.get("video") or fp.stem.replace("_evaluation", "")
    gt_words = data.get("ground_truth_words")
    gt_by_video[video] = gt_words

    evals = data.get("evaluations", [])
    for e in evals:
        model = e.get("model")
        wer = safe_float(e.get("wer"))
        cer = safe_float(e.get("cer"))
        acc = safe_float(e.get("accuracy"))

        # Keep row even if wer exists but transcript is junk; evaluation already encoded that.
        if model is None or wer is None:
            continue

        per_video_rows.append((video, model, wer, cer, acc, gt_words))

        wer_by_model.setdefault(model, []).append(wer)
        if acc is not None:
            acc_by_model.setdefault(model, []).append(acc)
        count_by_model[model] = count_by_model.get(model, 0) + 1

        wer_matrix.setdefault(model, {})[video] = wer

# -------------------------
# 1) Per-video longtable (appendix)
# -------------------------
per_video_rows.sort(key=lambda r: (r[0].lower(), r[2]))  # by video, then best WER first

with OUT_PER_VIDEO_TEX.open("w", encoding="utf-8") as f:
    f.write("\\begin{center}\n")
    f.write("\\captionsetup{type=table}\n")
    f.write("\\captionof{table}{Per-video Ground Truth Evaluation (WER/CER/Accuracy)}\n")
    f.write("\\label{tab:asr_per_video_longtable}\n")
    f.write("\\begin{longtable}{llrrrr}\n")
    f.write("\\toprule\n")
    f.write("\\textbf{Video} & \\textbf{Model} & \\textbf{GT Words} & \\textbf{WER (\\%)} & \\textbf{CER (\\%)} & \\textbf{Acc (\\%)}\\\\\n")
    f.write("\\midrule\n\\endfirsthead\n")
    f.write("\\toprule\n")
    f.write("\\textbf{Video} & \\textbf{Model} & \\textbf{GT Words} & \\textbf{WER (\\%)} & \\textbf{CER (\\%)} & \\textbf{Acc (\\%)}\\\\\n")
    f.write("\\midrule\n\\endhead\n")
    f.write("\\midrule\\multicolumn{6}{r}{\\emph{Continued on next page}}\\\\\n\\midrule\n\\endfoot\n")
    f.write("\\bottomrule\n\\endlastfoot\n")

    for video, model, wer, cer, acc, gt_words in per_video_rows:
        gt_words_str = "" if gt_words is None else str(gt_words)
        cer_str = "" if cer is None else f"{cer:.2f}"
        acc_str = "" if acc is None else f"{acc:.2f}"
        f.write(
            f"{tex_escape(video)} & {tex_escape(model)} & {gt_words_str} & "
            f"{wer:.2f} & {cer_str} & {acc_str}\\\\\n"
        )

    f.write("\\end{longtable}\n")
    f.write("\\end{center}\n")

print("Saved:", OUT_PER_VIDEO_TEX)

# -------------------------
# 2) Aggregate longtable
# -------------------------
agg_rows = []
for model, wers in wer_by_model.items():
    avg_wer = float(np.mean(wers))
    avg_acc = float(np.mean(acc_by_model.get(model, [np.nan])))
    n = len(wers)
    agg_rows.append((model, avg_wer, avg_acc, n))

# Sort by avg WER ascending
agg_rows.sort(key=lambda r: r[1])

with OUT_AGG_TEX.open("w", encoding="utf-8") as f:
    f.write("\\begin{center}\n")
    f.write("\\captionsetup{type=table}\n")
    f.write("\\captionof{table}{Aggregate Ground Truth Evaluation Across Videos}\n")
    f.write("\\label{tab:asr_aggregate_longtable}\n")
    f.write("\\begin{longtable}{lccc}\n")
    f.write("\\toprule\n")
    f.write("\\textbf{Model} & \\textbf{Avg WER (\\%)} & \\textbf{Avg Acc (\\%)} & \\textbf{Videos Evaluated}\\\\\n")
    f.write("\\midrule\n\\endfirsthead\n")
    f.write("\\toprule\n")
    f.write("\\textbf{Model} & \\textbf{Avg WER (\\%)} & \\textbf{Avg Acc (\\%)} & \\textbf{Videos Evaluated}\\\\\n")
    f.write("\\midrule\n\\endhead\n")
    f.write("\\midrule\\multicolumn{4}{r}{\\emph{Continued on next page}}\\\\\n\\midrule\n\\endfoot\n")
    f.write("\\bottomrule\n\\endlastfoot\n")

    for model, avg_wer, avg_acc, n in agg_rows:
        acc_str = "" if isnan(avg_acc) else f"{avg_acc:.2f}"
        f.write(f"{tex_escape(model)} & {avg_wer:.2f} & {acc_str} & {n}\\\\\n")

    f.write("\\end{longtable}\n\\end{center}\n")

print("Saved:", OUT_AGG_TEX)

# -------------------------
# 3) Ranking table with bold best (WER + Acc)
# -------------------------
best_wer = min(r[1] for r in agg_rows) if agg_rows else None
best_acc = max((r[2] for r in agg_rows if not isnan(r[2])), default=None)

with OUT_RANK_TEX.open("w", encoding="utf-8") as f:
    f.write("\\begin{table}[H]\n\\centering\n")
    f.write("\\caption{Aggregate Ranking of ASR Models (Lower WER is Better)}\n")
    f.write("\\label{tab:asr_ranking_aggregate}\n")
    f.write("\\begin{tabular}{clccc}\n")
    f.write("\\toprule\n")
    f.write("\\textbf{Rank} & \\textbf{Model} & \\textbf{Avg WER (\\%)} & \\textbf{Avg Acc (\\%)} & \\textbf{Videos}\\\\\n")
    f.write("\\midrule\n")

    for i, (model, avg_wer, avg_acc, n) in enumerate(agg_rows, start=1):
        wer_str = f"{avg_wer:.2f}"
        acc_str = "" if isnan(avg_acc) else f"{avg_acc:.2f}"

        if best_wer is not None and abs(avg_wer - best_wer) < 1e-9:
            wer_str = f"\\textbf{{{wer_str}}}"
        if best_acc is not None and (not isnan(avg_acc)) and abs(avg_acc - best_acc) < 1e-9:
            acc_str = f"\\textbf{{{acc_str}}}"

        note = "" if n >= 5 else " (low $n$)"
        f.write(f"{i} & {tex_escape(model)}{note} & {wer_str} & {acc_str} & {n}\\\\\n")

    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

print("Saved:", OUT_RANK_TEX)

# -------------------------
# 4) Boxplot of WER distributions
# -------------------------
models_sorted = [r[0] for r in agg_rows]  # in ranking order
data_sorted = [wer_by_model[m] for m in models_sorted]

plt.figure(figsize=(10, 4))
plt.boxplot(data_sorted, labels=models_sorted, vert=True, showfliers=True)
plt.ylabel("WER (%)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(OUT_BOXPLOT)
print("Saved:", OUT_BOXPLOT)

# -------------------------
# 5) Significance testing vs baseline (paired)
# -------------------------
stats = []
baseline_map = wer_matrix.get(BASELINE_MODEL, {})
baseline_videos = set(baseline_map.keys())

for model in models_sorted:
    if model == BASELINE_MODEL:
        continue
    model_map = wer_matrix.get(model, {})
    common = sorted(baseline_videos.intersection(model_map.keys()))
    if len(common) < MIN_COMMON_VIDEOS_FOR_TEST:
        stats.append({
            "model": model,
            "baseline": BASELINE_MODEL,
            "n_common": len(common),
            "wilcoxon_W": None,
            "p_value": None,
            "q_value": None,
            "mean_diff": None,
            "ci_low": None,
            "ci_high": None,
            "note": "Insufficient overlap for paired test"
        })
        continue

    x = [model_map[v] for v in common]        # WER(model)
    y = [baseline_map[v] for v in common]     # WER(baseline)
    W, p = wilcoxon_signed_rank(x, y)
    mean_diff, lo, hi = paired_bootstrap_ci(x, y, iters=5000, seed=42)

    stats.append({
        "model": model,
        "baseline": BASELINE_MODEL,
        "n_common": len(common),
        "wilcoxon_W": W,
        "p_value": p,
        "q_value": None,  # fill after FDR
        "mean_diff": mean_diff,  # positive means model worse (higher WER) than baseline
        "ci_low": lo,
        "ci_high": hi,
        "note": ""
    })

# FDR correction across valid p-values
valid = [s for s in stats if s["p_value"] is not None]
if valid:
    qvals = bh_fdr([s["p_value"] for s in valid])
    for s, q in zip(valid, qvals):
        s["q_value"] = q

OUT_STATS_JSON.write_text(json.dumps(stats, indent=2), encoding="utf-8")
print("Saved:", OUT_STATS_JSON)

with OUT_STATS_TEX.open("w", encoding="utf-8") as f:
    f.write("\\begin{table}[H]\n\\centering\n")
    f.write(f"\\caption{{Paired Significance Tests on WER vs. {tex_escape(BASELINE_MODEL)} (Wilcoxon + Bootstrap CI)}}\n")
    f.write("\\label{tab:asr_significance}\n")
    f.write("\\begin{tabular}{lrrrrr}\n")
    f.write("\\toprule\n")
    f.write("\\textbf{Model} & $n$ & $W$ & $p$ & $q$ & $\\Delta\\overline{\\mathrm{WER}}$ [95\\% CI]\\\\\n")
    f.write("\\midrule\n")

    for s in stats:
        model = tex_escape(s["model"])
        n = s["n_common"]
        if s["p_value"] is None:
            f.write(f"{model} & {n} & -- & -- & -- & -- \\\\\n")
            continue
        W = s["wilcoxon_W"]
        p = s["p_value"]
        q = s["q_value"]
        md = s["mean_diff"]
        lo = s["ci_low"]
        hi = s["ci_high"]
        f.write(f"{model} & {n} & {W:.1f} & {p:.3g} & {q:.3g} & {md:.2f} [{lo:.2f}, {hi:.2f}] \\\\\n")

    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

print("Saved:", OUT_STATS_TEX)
print("DONE.")