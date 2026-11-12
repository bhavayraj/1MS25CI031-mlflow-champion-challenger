"""Save the three screenshots required for submission into screenshots/.

1. MLflow compare view of the A and B runs
2. Model Registry page of sentiment-model showing the @champion and @challenger aliases
3. Terminal output of predict_from_mlflow.py (rendered from outputs/new_data_predictions_terminal.txt)

Needs the MLflow server running and: pip install playwright && python -m playwright install chromium
"""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mlflow import MlflowClient
from playwright.sync_api import sync_playwright

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = "hf-sentiment-champion-challenger"
OUT = Path("screenshots")
OUT.mkdir(exist_ok=True)

client = MlflowClient(TRACKING_URI)
exp = client.get_experiment_by_name(EXPERIMENT)
summary = client.search_runs(
    [exp.experiment_id],
    filter_string="attributes.run_name = 'compare_sentiment' AND attributes.status = 'FINISHED'",
    order_by=["attributes.start_time DESC"],
    max_results=1,
)[0]
children = client.search_runs([exp.experiment_id], filter_string=f"tags.mlflow.parentRunId = '{summary.info.run_id}'")
run_ids = [r.info.run_id for r in sorted(children, key=lambda r: r.data.tags["model_letter"])]

ui = TRACKING_URI.rstrip("/") + "/#"
compare_url = f"{ui}/compare-runs?runs={json.dumps(run_ids)}&experiments={json.dumps([exp.experiment_id])}".replace(" ", "")
registry_url = f"{ui}/models/sentiment-model"


def dismiss_popups(page):
    """MLflow 3 shows a GenAI / Model training chooser and an assistant panel on first visit."""
    for action in (lambda: page.get_by_text("Model training", exact=True).click(timeout=3000),
                   lambda: page.get_by_role("button", name="Close").click(timeout=2000)):
        try:
            action()
            page.wait_for_timeout(800)
        except Exception:
            pass


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1400, "height": 1000}, device_scale_factor=1.5)
    page.goto(f"{ui}/experiments")
    page.wait_for_timeout(4000)
    dismiss_popups(page)

    page.goto(compare_url)
    page.wait_for_timeout(5000)
    height = page.evaluate(
        "() => { let m = 0; for (const e of document.querySelectorAll('*')) {"
        " if (e.scrollHeight > e.clientHeight + 5 && getComputedStyle(e).overflowY !== 'visible')"
        " m = Math.max(m, e.scrollHeight - e.clientHeight); } return m; }"
    )
    page.set_viewport_size({"width": 1400, "height": min(1000 + int(height) + 20, 3200)})
    page.wait_for_timeout(1500)
    page.screenshot(path=str(OUT / "1_mlflow_compare_view.png"))
    print("saved screenshots/1_mlflow_compare_view.png")

    page.set_viewport_size({"width": 1400, "height": 800})
    page.goto(registry_url)
    page.wait_for_timeout(5000)
    page.screenshot(path=str(OUT / "2_registry_aliases.png"))
    print("saved screenshots/2_registry_aliases.png")
    browser.close()

# terminal output of predict_from_mlflow.py drawn as a terminal window
text = Path("outputs/new_data_predictions_terminal.txt").read_text(encoding="utf-8").rstrip("\n")
lines = text.split("\n")
width = max(len(l) for l in lines)
fig = plt.figure(figsize=(0.083 * width + 0.6, 0.22 * len(lines) + 0.5), dpi=150)
fig.patch.set_facecolor("#1e1e1e")
for i, line in enumerate(lines):
    color = "#6ad06a" if line.startswith("$") else "#d4d4d4"
    fig.text(0.015, 1 - (i + 1) / (len(lines) + 1), line, family="monospace", fontsize=9.5, color=color, va="center")
fig.savefig(OUT / "3_predict_from_mlflow_output.png", facecolor=fig.get_facecolor())
print("saved screenshots/3_predict_from_mlflow_output.png")
