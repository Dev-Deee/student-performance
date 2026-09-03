import pandas as pd


class ValidationError(Exception):
    pass


class Validator:
    def __init__(self, target_column: str):
        self.target_column = target_column
        self.report: dict = {}

    def validate(self, df: pd.DataFrame) -> dict:
        if df.empty:
            raise ValidationError("Loaded dataframe is empty.")

        if self.target_column not in df.columns:
            raise ValidationError(
                f"Target variable column '{self.target_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        self.report = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "target_column": self.target_column,
            "target_nulls": int(df[self.target_column].isnull().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "null_counts": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict(),
        }

        if self.report["target_nulls"] > 0:
            raise ValidationError(
                f"Target column '{self.target_column}' has {self.report['target_nulls']} null values."
            )

        return self.report

    def print_report(self) -> None:
        report = self.report
        print("=== Data Quality Report ===")
        print(f"Rows: {report['n_rows']}  Columns: {report['n_columns']}")
        print(f"Target column: {report['target_column']} (nulls: {report['target_nulls']})")
        print(f"Duplicate rows: {report['duplicate_rows']}")
        if report["null_counts"]:
            print(f"Columns with nulls: {report['null_counts']}")
        else:
            print("No missing values in feature columns.")
