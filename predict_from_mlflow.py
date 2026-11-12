"""Load the current Champion from the MLflow Model Registry and predict on new sentences.

No Hugging Face code here: the model is loaded only through its alias URI.
Usage:
    python predict_from_mlflow.py                    # uses @champion
    python predict_from_mlflow.py --alias challenger # stretch goal: same script, other alias
"""
import argparse
import os
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
REGISTERED_NAME = "sentiment-model"

parser = argparse.ArgumentParser()
parser.add_argument("--alias", default="champion")
args = parser.parse_args()

mlflow.set_tracking_uri(TRACKING_URI)
model_uri = f"models:/{REGISTERED_NAME}@{args.alias}"

mv = MlflowClient().get_model_version_by_alias(REGISTERED_NAME, args.alias)
model = mlflow.pyfunc.load_model(model_uri)

new_data = pd.read_csv("data/new_data.csv")
pred = model.predict(new_data[["text"]])
out = pd.DataFrame(
    {
        "text": new_data["text"],
        "prediction": pred["prediction"],
        "score": pred["score"],
        "expected": new_data["expected"],
    }
)
out["match"] = out["prediction"] == out["expected"]

Path("outputs").mkdir(exist_ok=True)
out_file = "outputs/new_data_predictions.csv" if args.alias == "champion" else f"outputs/new_data_predictions_{args.alias}.csv"
out.to_csv(out_file, index=False)

show = out.copy()
show["text"] = show["text"].str.slice(0, 58)
lines = [
    f"Model URI          : {model_uri}",
    f"Registered version : {mv.version}",
    f"Hugging Face model : {mv.tags.get('hf_model_id')}",
    f"Source run         : {mv.tags.get('source_run_name')} (run_id {mv.run_id})",
    "",
    show[["text", "prediction", "score", "expected"]].to_string(index=False),
    "",
    f"{int(out['match'].sum())}/{len(out)} predictions match the label I expected",
    f"Saved {out_file}",
]
text = "\n".join(lines)
print(text)
command = "python predict_from_mlflow.py" + ("" if args.alias == "champion" else f" --alias {args.alias}")
Path(out_file.replace(".csv", "_terminal.txt")).write_text(f"$ {command}\n{text}\n", encoding="utf-8")
