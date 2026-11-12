"""Custom MLflow pyfunc wrapper around a Hugging Face sentiment pipeline (as in Lab 9).

The Hugging Face code lives only inside this class. Anyone who loads the model from
MLflow (for example predict_from_mlflow.py) just calls model.predict(DataFrame).
"""
import mlflow
import pandas as pd


class HFSentimentModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from transformers import pipeline

        cfg = context.model_config
        self.hf_model_id = cfg["hf_model_id"]
        self.pipe = pipeline(
            "sentiment-analysis",
            model=self.hf_model_id,
            device=-1,  # CPU when loaded from MLflow, so it works on any machine
        )
        self.batch_size = int(cfg.get("batch_size", 16))
        self.max_length = int(cfg.get("max_length", 512))

    def predict(self, context, model_input, params=None):
        texts = model_input["text"].astype(str).tolist()
        results = self.pipe(
            texts, batch_size=self.batch_size, truncation=True, max_length=self.max_length
        )
        return pd.DataFrame(
            {
                "text": texts,
                "prediction": [r["label"].upper() for r in results],
                "score": [round(float(r["score"]), 4) for r in results],
            }
        )
