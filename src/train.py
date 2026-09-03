import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor

from src.preprocess import Preprocessor


class Trainer:

    def __init__(self, preprocessor: Preprocessor, cv_folds: int = 5,
                 scoring: str = "neg_root_mean_squared_error", random_state: int = 42):
        self.preprocessor = preprocessor
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        self.results: pd.DataFrame | None = None

    def get_candidate_models(self) -> dict:
        return {
            "DecisionTreeRegressor": DecisionTreeRegressor(random_state=self.random_state),
            "Ridge": Ridge(),
            "RandomForestRegressor": RandomForestRegressor(random_state=self.random_state),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=self.random_state),
            "LightGBMRegressor": LGBMRegressor(random_state=self.random_state, verbose=-1),
        }

    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        models = self.get_candidate_models()
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        results = []
        for name, model in models.items():
            pipeline = self.preprocessor.build_pipeline(model)
            scores = cross_val_score(pipeline, X, y, cv=kfold, scoring=self.scoring, n_jobs=-1)
            results.append(
                {
                    "model": name,
                    "mean_score": np.mean(scores),
                    "std_score": np.std(scores),
                    "rmse_mean": -np.mean(scores) if self.scoring.startswith("neg_") else np.mean(scores),
                }
            )

        self.results = pd.DataFrame(results).sort_values("rmse_mean").reset_index(drop=True)
        return self.results
