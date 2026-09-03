from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.preprocess import Preprocessor
from src.train import Trainer


class Evaluator:

    def __init__(self, preprocessor: Preprocessor, trainer: Trainer,
                 test_size: float = 0.2, random_state: int = 42):
        self.preprocessor = preprocessor
        self.trainer = trainer
        self.test_size = test_size
        self.random_state = random_state
        self.metrics: dict = {}

    def evaluate(self, X: pd.DataFrame, y: pd.Series, best_model_name: str,
                 output_dir: str | None = None) -> dict:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        model = self.trainer.get_candidate_models()[best_model_name]
        pipeline = self.preprocessor.build_pipeline(model)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        self.metrics = {
            "model": best_model_name,
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "mae": float(mean_absolute_error(y_test, preds)),
            "r2": float(r2_score(y_test, preds)),
        }

        if output_dir:
            output_path=Path(output_dir) / f"{best_model_name}.joblib"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, output_path)
            self.metrics["saved_to"] = output_path

        return self.metrics
