from pathlib import Path
import pandas as pd

class Ingestor:
    def __init__(self, raw_path: str):
        self.raw_path = Path(raw_path)

    # def load_raw_data(self) -> pd.DataFrame:
    #     if not self.raw_path.exists():
    #         raise FileNotFoundError(f"Raw file not found at: {self.raw_path}")
    #     df = pd.read_excel(self.raw_path)
    #     return df

    def load_raw_data(self) -> pd.DataFrame:
        if not self.raw_path.exists():
            raise FileNotFoundError(f"Raw file not found at: {self.raw_path}")

        suffix = self.raw_path.suffix.lower()
        if suffix in (".xlsx", ".xls"):
            df = pd.read_excel(self.raw_path)
        elif suffix == ".csv":
            df = pd.read_csv(self.raw_path, sep=None, engine="python")
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        return df

    @staticmethod
    def split_features_target(df: pd.DataFrame, target_column: str):
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y
