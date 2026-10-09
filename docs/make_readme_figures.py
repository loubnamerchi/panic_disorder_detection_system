"""Regenerate README figures from saved artifacts (read-only: models and metrics are never modified).

Usage (from the repository root):
    python docs/make_readme_figures.py .
"""
import json
import os
import sys

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
os.chdir(ROOT)
sys.path.insert(0, os.getcwd())
OUT = os.path.join("docs", "figures")
os.makedirs(OUT, exist_ok=True)

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODELS = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
LABEL = {"logistic_regression": "Logistic Reg.", "random_forest": "Random Forest", "xgboost": "XGBoost", "lightgbm": "LightGBM"}
COL = {"logistic_regression": "#2a78d6", "random_forest": "#eb6834", "xgboost": "#1baf7a", "lightgbm": "#4a3aa7"}
EXP = {"experiment_1": ("Exp. 1 — Baseline", "#2a78d6"), "experiment_2": ("Exp. 2 — SMOTENC", "#eb6834")}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.spines.top": False, "axes.spines.right": False})

RES = {e: json.load(open(f"artifacts/{e}/metrics/model_comparison.json")) for e in EXP}
METRICS = [("pr_auc", "PR-AUC (↑)"), ("recall", "Recall @0.5 (↑)"), ("precision", "Precision @0.5 (↑)"),
           ("f1", "F1 @0.5 (↑)"), ("brier_score", "Brier score (↓)")]


def style(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# 1. Experiment 1 vs Experiment 2, test metrics with bootstrap 95% CI
fig, axes = plt.subplots(1, len(METRICS), figsize=(17, 4.4))
for ax, (k, title) in zip(axes, METRICS):
    y = np.arange(len(MODELS))[::-1]
    for j, e in enumerate(EXP):
        t = [RES[e][m]["test"] for m in MODELS]
        v = np.array([r[k] for r in t])
        lo = np.array([r["confidence_intervals"][k]["lower"] for r in t])
        hi = np.array([r["confidence_intervals"][k]["upper"] for r in t])
        yy = y + (0.19 if j == 0 else -0.19)
        ax.barh(yy, v, height=0.36, color=EXP[e][1], edgecolor=SURF, linewidth=2, label=EXP[e][0])
        ax.errorbar(v, yy, xerr=[v - lo, hi - v], fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
        for yi, vi, h in zip(yy, v, hi):
            ax.text(h + (0.0012 if k == "brier_score" else 0.015), yi, f"{vi:.4f}" if k == "brier_score" else f"{vi:.3f}",
                    va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[m] for m in MODELS] if ax is axes[0] else [])
    ax.set_title(title, fontsize=10.5, loc="left")
    ax.set_xlim(0, 0.075 if k == "brier_score" else 1.18)
    style(ax)
axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.12), ncol=2, frameon=False, fontsize=9.5)
fig.suptitle("Held-out test set (n = 18,000; 769 positives). Error bars: bootstrap 95% CI (2,000 resamples)",
             fontsize=11, x=0.01, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(f"{OUT}/exp1_vs_exp2_test_metrics.png", dpi=160)
plt.close(fig)

# 2. Reliability diagrams (10 equal-width bins), recomputed from saved models on the test set
Xte_raw = pd.read_parquet("data/processed/X_test.parquet")
yte = pd.read_parquet("data/processed/y_test.parquet").squeeze().values
bins = np.linspace(0, 1, 11)
fig, axes = plt.subplots(2, 4, figsize=(16, 8.4), sharex=True, sharey=True)
for r, e in enumerate(EXP):
    fe = joblib.load(f"artifacts/{e}/feature_engineering/feature_engineer.pkl")
    Xte = fe.encode_onehot(Xte_raw, fit=False)
    for ax, m in zip(axes[r], MODELS):
        p = joblib.load(f"artifacts/{e}/models/{m}/{m}_model.joblib").predict_proba(Xte)[:, 1]
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        mp, fp, n = [], [], []
        for b in range(10):
            s = idx == b
            if s.any():
                mp.append(p[s].mean()); fp.append(yte[s].mean()); n.append(int(s.sum()))
        ece = sum(c / len(p) * abs(f - q) for q, f, c in zip(mp, fp, n))
        ax.plot([0, 1], [0, 1], ls="--", color=INK2, lw=1)
        ax.plot(mp, fp, color=COL[m], lw=2, marker="o", ms=8, mec=SURF, mew=2)
        for q, f, c in list(zip(mp, fp, n))[1:]:
            ax.annotate(f"{c:,}", (q, f), textcoords="offset points", xytext=(6, 6 if f < 0.15 else -12), fontsize=7.5, color=INK2)
        ax.set_title(f"{EXP[e][0]} · {LABEL[m]}\nBrier {RES[e][m]['test']['brier_score']:.4f} · ECE {ece:.4f} · mean p̂ {p.mean():.3f}",
                     fontsize=9.5, loc="left")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
        if r == 1:
            ax.set_xlabel("Mean predicted probability")
    axes[r][0].set_ylabel("Observed fraction positive")
fig.suptitle("Reliability diagrams — test set, 10 equal-width bins (labels = samples per bin; dashed = perfect calibration; "
             "prevalence = 0.043)", fontsize=11, x=0.01, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{OUT}/reliability_exp1_vs_exp2.png", dpi=160)
plt.close(fig)

# 3. Synthetic-rule audit: positive rate by number of risk indicators within the Sleep-quality gate
T = "Panic Disorder Diagnosis"
df = pd.read_csv("data/raw/panic_disorder_dataset.csv")
sq = df[df["Lifestyle Factors"] == "Sleep quality"]
flags = ((sq["Current Stressors"] == "High").astype(int) + (sq["Severity"] == "Severe") + (sq["Impact on Life"] == "Significant")
         + (sq["Symptoms"] == "Panic attacks") + (sq["Personal History"] == "Yes") + (sq["Family History"] == "Yes")
         + (sq["Coping Mechanisms"] != "Exercise"))
g = sq.groupby(flags)[T].agg(["count", "mean"])
fig, ax = plt.subplots(figsize=(8.5, 3.8))
ax.bar(g.index, g["mean"] * 100, width=0.6, color="#2a78d6", edgecolor=SURF, linewidth=2)
for x, (c, rate) in g.iterrows():
    ax.text(x, rate * 100 + 2, f"{rate*100:.1f}%\n(n={int(c):,})", ha="center", fontsize=8, color=INK)
ax.set_xlabel("Number of risk indicators present (0–7)")
ax.set_ylabel("Positive rate (%)")
ax.set_ylim(0, 100)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
ax.set_title("Panic-disorder rate within Lifestyle Factors = Sleep quality (n = 39,842)\n"
             "Outside this category the positive rate is exactly 0%", fontsize=10.5, loc="left")
fig.tight_layout()
fig.savefig(f"{OUT}/audit_risk_indicator_gradient.png", dpi=160)
plt.close(fig)

print("Saved:", sorted(os.listdir(OUT)))
