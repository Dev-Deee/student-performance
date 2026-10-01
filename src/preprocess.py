import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Preprocessor:
    COLUMN_RENAME_MAP = {
        "famsize": "family size",
        "Pstatus": "parent's cohabitation status",
        "Medu": "mother's education level",
        "Fedu": "father's education level",
        "Mjob": "mother's job",
        "Fjob": "father's job",
        "famsup": "family support",
        "schoolsup": "school support",
        "activities": "extra-curricular activities",
        "famrel": "family relationships",
        "paid": "extra paid classes",
        "goout": "going out with friends",
        "Dalc": "workday alcohol consumption",
        "Walc": "weekend alcohol consumption",
        "G1": "first period grade",
        "G2": "second period grade",
        "G3": "final grade",
    }

    def __init__(self):
        self.preprocessor: ColumnTransformer | None = None

    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        renamed = df.rename(columns=self.COLUMN_RENAME_MAP)
        print(f"Renamed {len(self.COLUMN_RENAME_MAP)} columns to readable names.")
        return renamed

    def build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        categorical_cols = X.select_dtypes(include="object").columns.tolist()
        numeric_cols = X.select_dtypes(include="number").columns.tolist()

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ]
        )
        return self.preprocessor

    def build_pipeline(self, model) -> Pipeline:
        if self.preprocessor is None:
            raise RuntimeError("Call build_preprocessor(X) before build_pipeline(model).")
        return Pipeline(steps=[("preprocessor", self.preprocessor), ("model", model)])
