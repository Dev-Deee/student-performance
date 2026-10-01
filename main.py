import yaml

from src.data import Ingestor
from src.validate import Validator
from src.analysis import Analyzer
from src.preprocess import Preprocessor
from src.train import Trainer
from src.evaluate import Evaluator


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    data_cfg = config["data"]
    model_cfg = config["model"]

    # 1. Ingest
    ingestor = Ingestor(data_cfg["raw_path"])
    df = ingestor.load_raw_data()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {data_cfg['raw_path']}\n")

    # 2. Rename columns to readable names (before validation, so the target
    #    column check below matches the renamed name in config.yaml)
    preprocessor = Preprocessor()
    df = preprocessor.rename_columns(df)

    # 3. Validate
    validator = Validator(target_column=data_cfg["target_column"])
    validator.validate(df)
    validator.print_report()

    # 4. Analyze
    analyzer = Analyzer(target_column=data_cfg["target_column"])
    analyzer.summarize(df)
    analyzer.print_top_correlations()

    # 5. Preprocess: split features/target, build the encoding pipeline
    X, y = ingestor.split_features_target(df, data_cfg["target_column"])
    preprocessor.build_preprocessor(X)

    # 6. Train: cross-validate all candidate models
    trainer = Trainer(
        preprocessor=preprocessor,
        cv_folds=data_cfg["cv_folds"],
        scoring=data_cfg["scoring"],
        random_state=data_cfg["random_state"],
    )
    results = trainer.cross_validate(X, y)
    print("\n=== Cross-validation results ===")
    print(results)

    best_model_name = results.iloc[0]["model"]
    print(f"\nBest model: {best_model_name}")

    # 7. Evaluate: fit best model on a held-out split, save it
    evaluator = Evaluator(
        preprocessor=preprocessor,
        trainer=trainer,
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
    )
    metrics = evaluator.evaluate(X, y, best_model_name, output_dir=model_cfg["output_path"])
    print("\n=== Final evaluation on held-out test set ===")
    print(metrics)

if __name__ == "__main__":
    main()
