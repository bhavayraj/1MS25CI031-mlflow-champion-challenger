"""Run the Champion (A) and Challenger (B) on the same evaluation set and log everything to MLflow.

Runs created in the experiment hf-sentiment-champion-challenger:
    compare_sentiment                 (parent / summary run with the decision)
      A_sentiment_distilbert-sst2     (nested run, Model A = current Champion)
      B_sentiment_roberta-large       (nested run, Model B = Challenger)
"""
import json
import math
import os
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import torch
import transformers
from mlflow.models import infer_signature
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import pipeline

from sentiment_wrapper import HFSentimentModel

# ------------------------------------------------------------------ settings
STUDENT_USN = "1MS25CI031"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = "hf-sentiment-champion-challenger"
TASK = "sentiment"
DATASET_NAME = "stanfordnlp/sst2 (validation, seeded subset)"
SEED = 42
BATCH_SIZE = 16
MAX_LENGTH = 512
LABELS = ["NEGATIVE", "POSITIVE"]

MODELS = {
    "A": {
        "hf_model_id": "distilbert-base-uncased-finetuned-sst-2-english",
        "short": "distilbert-sst2",
        "role": "champion",
    },
    "B": {
        "hf_model_id": "siebert/sentiment-roberta-large-english",
        "short": "roberta-large",
        "role": "challenger",
    },
}

RULE = json.load(open("champion_rule.json"))
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
DEVICE = 0 if torch.cuda.is_available() else -1
DEVICE_NAME = torch.cuda.get_device_name(0) if DEVICE == 0 else f"cpu ({platform.processor() or platform.machine()})"

transformers.set_seed(SEED)
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)

eval_path = Path("data/eval_inputs.csv")
if not eval_path.exists():
    raise SystemExit("data/eval_inputs.csv not found. Run  python prepare_data.py  first.")
eval_df = pd.read_csv(eval_path)
texts = eval_df["text"].astype(str).tolist()
refs = eval_df["reference"].tolist()

LIB_VERSIONS = f"transformers={transformers.__version__}, torch={torch.__version__}, mlflow={mlflow.__version__}"


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0),
    }


def confusion_plot(y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def evaluate_model(key, cfg):
    """Run one model on the shared inputs inside its own nested MLflow run."""
    run_name = f"{key}_{TASK}_{cfg['short']}"
    print(f"\n=== {run_name}: loading {cfg['hf_model_id']}")
    clf = pipeline("sentiment-analysis", model=cfg["hf_model_id"], device=DEVICE)
    n_params = clf.model.num_parameters()

    clf(texts[:4], batch_size=4, truncation=True, max_length=MAX_LENGTH)  # warm-up, not timed
    start = time.perf_counter()
    raw = clf(texts, batch_size=BATCH_SIZE, truncation=True, max_length=MAX_LENGTH)
    total_s = time.perf_counter() - start

    preds = [r["label"].upper() for r in raw]
    scores = [round(float(r["score"]), 4) for r in raw]
    metrics = compute_metrics(refs, preds)
    metrics.update(
        {
            "avg_latency_ms": 1000 * total_s / len(texts),
            "total_inference_s": total_s,
            "model_params_millions": n_params / 1e6,
            "num_errors": int(sum(p != r for p, r in zip(preds, refs))),
        }
    )

    pred_df = eval_df[["sst2_idx", "text", "reference"]].copy()
    pred_df["prediction"] = preds
    pred_df["score"] = scores
    pred_df["correct"] = pred_df["prediction"] == pred_df["reference"]
    pred_file = OUT / f"predictions_{cfg['short']}.csv"
    pred_df.to_csv(pred_file, index=False)

    metrics_file = OUT / f"metrics_{cfg['short']}.json"
    json.dump({k: round(v, 6) for k, v in metrics.items()}, open(metrics_file, "w"), indent=2)

    with mlflow.start_run(run_name=run_name, nested=True) as run:
        mlflow.log_params(
            {
                "hf_model_id": cfg["hf_model_id"],
                "task": "sentiment-analysis",
                "role": f"{cfg['role']} candidate",
                "batch_size": BATCH_SIZE,
                "max_length": MAX_LENGTH,
                "truncation": True,
                "eval_samples": len(texts),
                "seed": SEED,
                "device": DEVICE_NAME,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "student_usn": STUDENT_USN,
                "library_versions": LIB_VERSIONS,
                "dataset_name": DATASET_NAME,
                "hf_model_id": cfg["hf_model_id"],
                "model_letter": key,
            }
        )
        mlflow.log_artifact(str(pred_file), "predictions")
        mlflow.log_artifact(str(metrics_file), "metrics")
        mlflow.log_artifact(str(eval_path), "data")

        fig = confusion_plot(refs, preds, f"{run_name}\naccuracy {metrics['accuracy']:.3f}, macro-F1 {metrics['macro_f1']:.3f}")
        fig.savefig(OUT / f"confusion_matrix_{cfg['short']}.png", dpi=150)
        mlflow.log_figure(fig, f"plots/confusion_matrix_{cfg['short']}.png")
        plt.close(fig)

        example = pd.DataFrame({"text": texts[:2]})
        signature = infer_signature(
            example, pd.DataFrame({"text": texts[:2], "prediction": preds[:2], "score": scores[:2]})
        )
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=HFSentimentModel(),
            code_paths=["sentiment_wrapper.py"],
            model_config={"hf_model_id": cfg["hf_model_id"], "batch_size": BATCH_SIZE, "max_length": MAX_LENGTH},
            signature=signature,
            input_example=example,
            pip_requirements=[
                f"mlflow=={mlflow.__version__}",
                f"transformers=={transformers.__version__}",
                f"torch=={torch.__version__.split('+')[0]}",
                "pandas",
            ],
            tags={"hf_model_id": cfg["hf_model_id"], "role": cfg["role"]},
        )
        mlflow.set_tag("model_uri", info.model_uri)
        print(f"    accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
              f"avg_latency_ms={metrics['avg_latency_ms']:.1f}  run_id={run.info.run_id}")

    del clf
    return {"run_name": run_name, "metrics": metrics, "pred_df": pred_df, **cfg}


def mcnemar_p(a_df, b_df):
    """Exact two-sided McNemar test on the examples where exactly one model is right."""
    a_only = int(((a_df["correct"]) & (~b_df["correct"])).sum())
    b_only = int(((~a_df["correct"]) & (b_df["correct"])).sum())
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    k = min(a_only, b_only)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return a_only, b_only, min(1.0, 2 * p)


def decide(a, b):
    """Apply the rule from champion_rule.json (written before the final run)."""
    metric = RULE["primary_metric"]
    margin = RULE["margin_pct_points"]
    delta_pts = 100 * (b["metrics"][metric] - a["metrics"][metric])
    b_latency = b["metrics"]["avg_latency_ms"]
    if delta_pts >= margin and b_latency <= RULE["latency_guardrail_ms"]:
        return "B", delta_pts, f"Challenger beats Champion by {delta_pts:.2f} pts (>= {margin}) and passes the latency guardrail"
    if delta_pts >= margin:
        return "A", delta_pts, (f"Challenger wins on {metric} by {delta_pts:.2f} pts but fails the latency guardrail "
                                f"({b_latency:.1f} ms > {RULE['latency_guardrail_ms']} ms)")
    return "A", delta_pts, f"Challenger improvement of {delta_pts:.2f} pts is below the {margin}-pt margin; Champion stays"


def comparison_plot(a, b):
    names = ["accuracy", "macro_f1", "precision_macro", "recall_macro"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [3, 1.3]})
    x = range(len(names))
    w = 0.38
    for i, (m, color) in enumerate(((a, "#4C72B0"), (b, "#DD8452"))):
        vals = [m["metrics"][n] for n in names]
        bars = axes[0].bar([xi + (i - 0.5) * w for xi in x], vals, w, label=f"{m['run_name']}", color=color)
        axes[0].bar_label(bars, fmt="%.3f", fontsize=8)
    axes[0].set_xticks(list(x), names)
    lo = min(min(a["metrics"][n], b["metrics"][n]) for n in names)
    axes[0].set_ylim(max(0, lo - 0.1), 1.0)
    axes[0].set_title("Quality metrics (higher is better)")
    axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    lat = [a["metrics"]["avg_latency_ms"], b["metrics"]["avg_latency_ms"]]
    bars = axes[1].bar(["A", "B"], lat, color=["#4C72B0", "#DD8452"])
    axes[1].bar_label(bars, fmt="%.1f ms", fontsize=8)
    axes[1].axhline(RULE["latency_guardrail_ms"], color="red", ls="--", lw=1)
    axes[1].set_title("Avg latency per sentence\n(lower is better, red = guardrail)", fontsize=10)
    fig.suptitle("Champion (A) vs Challenger (B) on the same 200 SST-2 sentences", fontsize=11)
    fig.tight_layout()
    return fig


def confidence_plot(a, b):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for m, color in ((a, "#4C72B0"), (b, "#DD8452")):
        ax.hist(m["pred_df"]["score"], bins=20, range=(0.5, 1.0), alpha=0.6, label=m["run_name"], color=color)
    ax.set_xlabel("confidence score of predicted label")
    ax.set_ylabel("number of sentences")
    ax.set_title("Prediction confidence distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


with mlflow.start_run(run_name=f"compare_{TASK}") as parent:
    a = evaluate_model("A", MODELS["A"])
    b = evaluate_model("B", MODELS["B"])

    winner, delta_pts, reason = decide(a, b)
    a_only, b_only, p_value = mcnemar_p(a["pred_df"], b["pred_df"])

    mlflow.log_params(
        {
            "primary_metric": RULE["primary_metric"],
            "primary_metric_direction": RULE["primary_metric_direction"],
            "margin": RULE["margin_pct_points"],
            "tie_breaker": "champion stays if within margin",
            "latency_guardrail_ms": RULE["latency_guardrail_ms"],
            "model_A": MODELS["A"]["hf_model_id"],
            "model_B": MODELS["B"]["hf_model_id"],
            "eval_samples": len(texts),
            "seed": SEED,
            "device": DEVICE_NAME,
        }
    )
    summary_metrics = {f"A_{k}": v for k, v in a["metrics"].items()}
    summary_metrics.update({f"B_{k}": v for k, v in b["metrics"].items()})
    summary_metrics.update(
        {
            "delta_primary_metric": b["metrics"][RULE["primary_metric"]] - a["metrics"][RULE["primary_metric"]],
            "delta_primary_metric_pts": delta_pts,
            "delta_accuracy": b["metrics"]["accuracy"] - a["metrics"]["accuracy"],
            "latency_ratio_B_over_A": b["metrics"]["avg_latency_ms"] / a["metrics"]["avg_latency_ms"],
            "only_A_correct": a_only,
            "only_B_correct": b_only,
            "mcnemar_p_value": p_value,
        }
    )
    mlflow.log_metrics(summary_metrics)
    winner_cfg = MODELS[winner]
    mlflow.set_tags(
        {
            "winner": f"{winner} ({winner_cfg['hf_model_id']})",
            "winner_letter": winner,
            "decision_reason": reason,
            "student_usn": STUDENT_USN,
            "library_versions": LIB_VERSIONS,
            "dataset_name": DATASET_NAME,
        }
    )

    # disagreements: examples where the two models predict different labels
    merged = a["pred_df"][["sst2_idx", "text", "reference"]].copy()
    merged["pred_A"] = a["pred_df"]["prediction"]
    merged["score_A"] = a["pred_df"]["score"]
    merged["pred_B"] = b["pred_df"]["prediction"]
    merged["score_B"] = b["pred_df"]["score"]
    dis = merged[merged["pred_A"] != merged["pred_B"]].copy()

    def comment(row):
        who = "A" if row["pred_A"] == row["reference"] else "B"
        loser = "B" if who == "A" else "A"
        return (f"Only {who} is right; {loser} gave {row['pred_' + loser]} with "
                f"confidence {row['score_' + loser]:.2f}")

    dis["comment"] = dis.apply(comment, axis=1) if len(dis) else []
    dis.to_csv(OUT / "disagreements_all.csv", index=False)
    dis.head(5).to_csv(OUT / "disagreements_top5.csv", index=False)

    summary = {
        "task": "sentiment-analysis",
        "dataset": DATASET_NAME,
        "eval_samples": len(texts),
        "device": DEVICE_NAME,
        "library_versions": LIB_VERSIONS,
        "rule": RULE,
        "A": {"hf_model_id": a["hf_model_id"], "run_name": a["run_name"], **{k: round(v, 6) for k, v in a["metrics"].items()}},
        "B": {"hf_model_id": b["hf_model_id"], "run_name": b["run_name"], **{k: round(v, 6) for k, v in b["metrics"].items()}},
        "delta_primary_metric_pts": round(delta_pts, 4),
        "latency_ratio_B_over_A": round(summary_metrics["latency_ratio_B_over_A"], 3),
        "only_A_correct": a_only,
        "only_B_correct": b_only,
        "mcnemar_p_value": round(p_value, 4),
        "num_disagreements": int(len(dis)),
        "winner": winner,
        "winner_hf_model_id": winner_cfg["hf_model_id"],
        "decision_reason": reason,
    }
    json.dump(summary, open(OUT / "metrics_summary.json", "w"), indent=2)

    fig = comparison_plot(a, b)
    fig.savefig(OUT / "comparison_bar_chart.png", dpi=150)
    mlflow.log_figure(fig, "plots/comparison_bar_chart.png")
    plt.close(fig)
    fig = confidence_plot(a, b)
    fig.savefig(OUT / "confidence_histogram.png", dpi=150)
    mlflow.log_figure(fig, "plots/confidence_histogram.png")
    plt.close(fig)

    mlflow.log_artifact(str(OUT / "metrics_summary.json"))
    mlflow.log_artifact(str(OUT / "disagreements_top5.csv"), "qualitative")
    mlflow.log_artifact(str(OUT / "disagreements_all.csv"), "qualitative")
    mlflow.log_artifact(str(eval_path), "data")
    mlflow.log_artifact("champion_rule.json")
    mlflow.log_artifact("README.md")
    mlflow.log_artifact("requirements.txt")

    print("\n=== Decision")
    print(f"  delta {RULE['primary_metric']} (B - A): {delta_pts:+.2f} pts | only A right: {a_only} | "
          f"only B right: {b_only} | McNemar p = {p_value:.3f}")
    print(f"  winner: {winner} ({winner_cfg['hf_model_id']})")
    print(f"  reason: {reason}")
    print(f"  summary run_id: {parent.info.run_id}")
