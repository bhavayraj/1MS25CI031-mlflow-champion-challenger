# Champion–Challenger Model Comparison with Hugging Face and MLflow

**Student:** Bhavayraj Singhal · **USN:** 1MS25CI031 · **Course:** Skill Lab – MLOps with MLflow (Mini Project)

## 1. Task and models

| Item | Value |
|---|---|
| Task | Binary sentiment classification (POSITIVE / NEGATIVE) |
| Dataset | SST-2 validation set (`stanfordnlp/sst2`), 200 sentences sampled with seed 42 → `data/eval_inputs.csv` |
| Model A – current Champion | [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) (~67M parameters) |
| Model B – Challenger | [`siebert/sentiment-roberta-large-english`](https://huggingface.co/siebert/sentiment-roberta-large-english) (~355M parameters) |
| New data for Step 5.6 | `data/new_data.csv` – 8 sentences I wrote myself, not taken from SST-2 |
| MLflow experiment | `hf-sentiment-champion-challenger` |
| Registered model | `sentiment-model` with aliases `@champion` and `@challenger` |

Both models output the labels POSITIVE / NEGATIVE, the same labels as SST-2, so no label mapping is needed.

## 2. How to run

Windows (PowerShell), from this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m playwright install chromium

# terminal 1 – tracking server (keep it open)
python -m mlflow server --host 127.0.0.1 --port 5000

# terminal 2
python prepare_data.py          # saves data/eval_inputs.csv (200 SST-2 sentences, seed 42)
python evaluate.py              # runs A and B on the same inputs and logs 3 runs to MLflow
python register_champion.py     # registers both versions and sets @champion / @challenger
python predict_from_mlflow.py   # loads models:/sentiment-model@champion and predicts on data/new_data.csv
python predict_from_mlflow.py --alias challenger   # stretch goal: same script, other alias
python capture_screenshots.py   # saves the 3 screenshots into screenshots/
python write_report.py          # fills the results tables below from outputs/metrics_summary.json
```

`run_all.ps1` (Windows) and `run_all.sh` (macOS / Linux) run all of these steps in order and start the server for you. Open http://127.0.0.1:5000 to see the runs.

## 3. Champion rule (written before the final run)

This rule was fixed before any model was evaluated and is stored in `champion_rule.json`, which `evaluate.py` reads. It was committed to the repository before the first evaluation run.

1. **Primary metric:** macro-F1 on the 200 evaluation sentences. **Higher is better.** Macro-F1 gives equal weight to the positive and negative class, so a model cannot score well by favouring one label.
2. **Minimum improvement margin:** the Challenger must beat the Champion by **at least 2 percentage points** of macro-F1. With 200 sentences, one sentence is 0.5 points, so a 2-point gain means about four more sentences classified correctly. Anything smaller is too close to random variation in the sample to justify switching to a model that is five times larger.
3. **Tie-breaker:** if the difference is smaller than 2 points (or the Challenger is worse), the **Champion stays**, because it is the smaller and faster model.
4. **Guardrail:** the Challenger is rejected if its **average latency is above 250 ms per sentence** on the same machine, even if it wins on macro-F1. Above that, a sentiment check would feel slow in an interactive application.
5. **Direction of every metric:** accuracy, macro-F1, macro precision and macro recall – higher is better; `avg_latency_ms` and `total_inference_s` – lower is better.

The decision is logged in the `compare_sentiment` run as the parameters `primary_metric` and `margin`, the metric `delta_primary_metric` (Challenger minus Champion) and the tag `winner`.

## 4. Results

<!-- RESULTS_START -->
_Filled in by `write_report.py` after `evaluate.py` has run._
<!-- RESULTS_END -->

## 5. Decision

<!-- DECISION_START -->
_Filled in after the evaluation run._
<!-- DECISION_END -->

## 6. Limitations and declaration

**Limitations**
- Only 200 sentences are used, so each sentence is worth 0.5 percentage points of accuracy. Small differences between the models may not hold on a larger test set.
- SST-2 sentences are short movie-review phrases. Results may differ on other kinds of text, such as tweets, product reviews or Hinglish.
- Latency was measured once, on my laptop, with batches of 16 sentences. It depends on the hardware, the batch size and on other programs running at the same time.
- The models were evaluated as published on the Hugging Face Hub. Model A was itself fine-tuned on the SST-2 training set, which can favour it on this data set.

**Declaration**
- An AI assistant (Claude by Anthropic) was used to help write the code in this repository and to draft this README. The results come from running the code on my own laptop.
- Libraries used: Hugging Face `transformers`, `datasets`, PyTorch, MLflow, scikit-learn, pandas, matplotlib and Playwright (screenshots only). Exact versions are logged in the `library_versions` tag of every run.

## Repository structure

```
README.md                 this report
requirements.txt          libraries
champion_rule.json        the decision rule (fixed before the run)
prepare_data.py           builds data/eval_inputs.csv from SST-2
sentiment_wrapper.py      custom mlflow.pyfunc.PythonModel around the Hugging Face pipeline
evaluate.py               runs both models and logs everything to MLflow
register_champion.py      registers both models and sets the aliases
predict_from_mlflow.py    loads @champion and predicts on new data
capture_screenshots.py    saves the three required screenshots
write_report.py           writes the results and decision sections of this README
run_all.ps1 / run_all.sh  run everything in order
data/                     eval_inputs.csv, new_data.csv
outputs/                  predictions, metrics_summary.json, plots, disagreements, new_data_predictions.csv
screenshots/              compare view, registry aliases, predict_from_mlflow.py output
```
