import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

from src.utils import load_config, ensure_directories


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def main():
    cfg = load_config()
    ensure_directories()

    data_path = cfg["dataset"]["out_path"]
    test_start = pd.to_datetime(cfg["split"]["test_start_date"])

    df = pd.read_parquet(data_path)
    df["game_date"] = pd.to_datetime(df["game_date"])

    # time split
    train_df = df[df["game_date"] < test_start].copy()
    test_df = df[df["game_date"] >= test_start].copy()

    print("Train rows:", len(train_df), "Test rows:", len(test_df))

    y_train = train_df["strikeouts"].values
    y_test = test_df["strikeouts"].values

    # feature columns: drop identifiers + target
    drop_cols = ["game_pk", "game_date", "game_year", "pitcher", "strikeouts"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X_train = train_df[feature_cols].fillna(0.0).values
    X_test = test_df[feature_cols].fillna(0.0).values

    # Baseline 1: mean predictor
    mean_pred = np.full_like(y_test, y_train.mean(), dtype=float)

    # Baseline 2: linear regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    results = [
        ("mean_baseline", mean_absolute_error(y_test, mean_pred), rmse(y_test, mean_pred)),
        ("linear_regression", mean_absolute_error(y_test, lr_pred), rmse(y_test, lr_pred)),
    ]

    print("\n[BASELINE RESULTS]")
    for name, mae, r in results:
        print(f"{name:18s}  MAE={mae:.3f}  RMSE={r:.3f}")

    # Optional: save
    out_path = "reports/metrics_baselines.csv"
    pd.DataFrame(results, columns=["model", "mae", "rmse"]).to_csv(out_path, index=False)
    print("\nSaved metrics to:", out_path)


if __name__ == "__main__":
    main()