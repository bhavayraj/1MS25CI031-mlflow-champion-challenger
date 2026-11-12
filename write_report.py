"""Write the Results and Decision sections of README.md from outputs/metrics_summary.json,
then log the finished README to the compare_sentiment run in MLflow.

The decision paragraph applies champion_rule.json to the logged numbers; read it before
submitting and make sure you can explain it in the viva.
"""
import json
import os
import re
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = "hf-sentiment-champion-challenger"

s = json.load(open("outputs/metrics_summary.json"))
rule = s["rule"]
A, B = s["A"], s["B"]
margin = f'{rule["margin_pct_points"]:g}'
guard = rule["latency_guardrail_ms"]
n = s["eval_samples"]
delta = s["delta_primary_metric_pts"]
ratio = s["latency_ratio_B_over_A"]
p = s["mcnemar_p_value"]


def pct(x):
    return f"{100 * x:.2f}"


rows = [
    ("Accuracy (%) ↑", pct(A["accuracy"]), pct(B["accuracy"]), f"{100 * (B['accuracy'] - A['accuracy']):+.2f}"),
    ("**Macro-F1 (%) ↑ – primary**", f"**{pct(A['macro_f1'])}**", f"**{pct(B['macro_f1'])}**", f"**{delta:+.2f}**"),
    ("Macro precision (%) ↑", pct(A["precision_macro"]), pct(B["precision_macro"]), f"{100 * (B['precision_macro'] - A['precision_macro']):+.2f}"),
    ("Macro recall (%) ↑", pct(A["recall_macro"]), pct(B["recall_macro"]), f"{100 * (B['recall_macro'] - A['recall_macro']):+.2f}"),
    ("Wrong predictions (of 200) ↓", str(int(A["num_errors"])), str(int(B["num_errors"])), f"{int(B['num_errors'] - A['num_errors']):+d}"),
    ("Avg latency (ms / sentence) ↓", f"{A['avg_latency_ms']:.1f}", f"{B['avg_latency_ms']:.1f}", f"× {ratio:.2f}"),
    ("Total inference time (s) ↓", f"{A['total_inference_s']:.2f}", f"{B['total_inference_s']:.2f}", f"{B['total_inference_s'] - A['total_inference_s']:+.2f}"),
    ("Parameters (millions) ↓", f"{A['model_params_millions']:.0f}", f"{B['model_params_millions']:.0f}", f"× {B['model_params_millions'] / A['model_params_millions']:.1f}"),
]
table = "| Metric | A – Champion (distilbert-sst2) | B – Challenger (roberta-large) | B − A |\n|---|---|---|---|\n"
table += "\n".join(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in rows)

dis = pd.read_csv("outputs/disagreements_top5.csv")
dis_md = "| SST-2 idx | Sentence | Reference | A | B | Comment |\n|---|---|---|---|---|---|\n"
for _, r in dis.iterrows():
    sent = str(r["text"]).replace("|", "/")
    sent = sent if len(sent) <= 90 else sent[:87] + "..."
    dis_md += (f"| {r['sst2_idx']} | {sent} | {r['reference']} | {r['pred_A']} ({r['score_A']:.2f}) | "
               f"{r['pred_B']} ({r['score_B']:.2f}) | {r['comment']} |\n")

results = f"""Run on {s['device']} with {s['library_versions']}. All numbers are on the same {n} sentences in `data/eval_inputs.csv`.

{table}

The two models disagree on {s['num_disagreements']} of the {n} sentences. On those, only A is right {s['only_A_correct']} times and only B is right {s['only_B_correct']} times (exact McNemar test, p = {p:.3f}).

![Comparison of the two models](outputs/comparison_bar_chart.png)

| Confusion matrix – Model A | Confusion matrix – Model B |
|---|---|
| ![A](outputs/confusion_matrix_distilbert-sst2.png) | ![B](outputs/confusion_matrix_roberta-large.png) |

![Confidence distribution](outputs/confidence_histogram.png)

**Five sentences where the models disagree** (full list in `outputs/disagreements_all.csv`):

{dis_md}
**Prediction on new data with the Champion** (`outputs/new_data_predictions.csv`):

![predict_from_mlflow.py output](screenshots/3_predict_from_mlflow_output.png)

**MLflow screenshots**

![Compare view](screenshots/1_mlflow_compare_view.png)

![Registry aliases](screenshots/2_registry_aliases.png)
"""

# --------------------------------------------------------------- decision paragraph
winner = s["winner"]
b_wins_metric = delta >= rule["margin_pct_points"]
b_fast_enough = B["avg_latency_ms"] <= guard
significant = p < 0.05

if winner == "B":
    verdict = (f"**Champion: Model B (roberta-large).** The Challenger improves macro-F1 from {pct(A['macro_f1'])}% to "
               f"{pct(B['macro_f1'])}%, a gain of {delta:.2f} points, which is above my {margin}-point margin, and its "
               f"average latency of {B['avg_latency_ms']:.1f} ms per sentence is within the {guard} ms guardrail. "
               f"Under the rule fixed before the run, it replaces the Champion.")
elif b_wins_metric and not b_fast_enough:
    verdict = (f"**Champion: Model A stays (distilbert-sst2).** The Challenger is more accurate, improving macro-F1 by "
               f"{delta:.2f} points (from {pct(A['macro_f1'])}% to {pct(B['macro_f1'])}%), which clears the {margin}-point "
               f"margin. However, it needs {B['avg_latency_ms']:.1f} ms per sentence, above the {guard} ms guardrail, so "
               f"the rule rejects it.")
elif delta > 0:
    verdict = (f"**Champion: Model A stays (distilbert-sst2).** The Challenger is slightly better, {pct(B['macro_f1'])}% "
               f"against {pct(A['macro_f1'])}% macro-F1, but the gain of {delta:.2f} points is below my {margin}-point "
               f"margin, so by the tie-breaker the smaller and faster Champion is kept.")
else:
    verdict = (f"**Champion: Model A stays (distilbert-sst2).** The Challenger does not improve on the Champion: its "
               f"macro-F1 is {pct(B['macro_f1'])}% against {pct(A['macro_f1'])}% ({delta:+.2f} points), so the rule "
               f"keeps Model A.")

sample = (f"With only {n} sentences each sentence is worth 0.5 points, and the models disagree on "
          f"{s['num_disagreements']} of them ({s['only_A_correct']} won by A, {s['only_B_correct']} by B). "
          + (f"The McNemar test gives p = {p:.3f}, so the difference is unlikely to be chance alone. "
             if significant else
             f"The McNemar test gives p = {p:.3f}, so the difference could easily be chance and a larger test set "
             f"would be needed to be sure. "))
tradeoff = (f"The trade-off is size and speed: B has about {B['model_params_millions'] / A['model_params_millions']:.0f}× "
            f"more parameters and was {ratio:.1f}× slower ({B['avg_latency_ms']:.1f} ms against "
            f"{A['avg_latency_ms']:.1f} ms per sentence), and it needs a much larger download. "
            f"Model A was fine-tuned on SST-2 itself, which helps it on this data, while B was trained on a wider mix "
            f"of English reviews and may generalise better to other kinds of text. The disagreement table above shows "
            f"the sentences where the two models differ.")
decision = f"{verdict}\n\n{sample}{tradeoff}"
words = len(re.findall(r"\b\w[\w.%-]*\b", re.sub(r"\*\*", "", decision)))

readme = Path("README.md").read_text(encoding="utf-8")
readme = re.sub(r"<!-- RESULTS_START -->.*<!-- RESULTS_END -->",
                f"<!-- RESULTS_START -->\n{results}\n<!-- RESULTS_END -->", readme, flags=re.S)
readme = re.sub(r"<!-- DECISION_START -->.*<!-- DECISION_END -->",
                f"<!-- DECISION_START -->\n{decision}\n<!-- DECISION_END -->", readme, flags=re.S)
Path("README.md").write_text(readme, encoding="utf-8")
print(f"README.md updated (decision paragraph: {words} words, winner {winner})")

# log the final README to the summary run
mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()
exp = client.get_experiment_by_name(EXPERIMENT)
summary_run = client.search_runs(
    [exp.experiment_id],
    filter_string="attributes.run_name = 'compare_sentiment' AND attributes.status = 'FINISHED'",
    order_by=["attributes.start_time DESC"],
    max_results=1,
)[0]
client.log_artifact(summary_run.info.run_id, "README.md")
print(f"README.md logged to run {summary_run.info.run_id}")
