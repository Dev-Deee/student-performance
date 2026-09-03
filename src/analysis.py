import pandas as pd

class Analyzer:
    def __init__(self, target_column: str):
        self.target_column = target_column
        self.summary: dict = {}

    def summarize(self, df: pd.DataFrame) -> dict:
        numeric_df = df.select_dtypes(include="number")
        correlations = (
            numeric_df.corr()[self.target_column]
            .drop(self.target_column)
            .sort_values(ascending=False)
            if self.target_column in numeric_df.columns
            else pd.Series(dtype=float)
        )

        self.summary = {
            "describe": df.describe(include="all").to_dict(),
            "correlations_with_target": correlations.to_dict(),
            "categorical_columns": df.select_dtypes(include="object").columns.tolist(),
            "numeric_columns": numeric_df.columns.tolist(),
        }
        return self.summary

    def print_top_correlations(self, top_n: int = 10) -> None:
        corr = self.summary["correlations_with_target"]
        print(f"=== Top {top_n} numeric features correlated with target ===")
        for feature, value in list(corr.items())[:top_n]:
            print(f"{feature:15s} {value:+.3f}")
