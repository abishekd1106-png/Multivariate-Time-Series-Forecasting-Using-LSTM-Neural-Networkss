import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.statespace.sarimax import SARIMAX

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import tensorflow as tf

warnings.filterwarnings("ignore")

# If SHAP is not installed, you may need: pip install shap
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not installed. Install with `pip install shap` to run explainability.")


# ---------------------------------------------------------
# 1. Synthetic multivariate time-series dataset
# ---------------------------------------------------------
def generate_synthetic_dataset(n_steps=10000, random_state=42):
    """
    Generate a multivariate time series with 5 interacting features and 1 target.
    Simulates realistic seasonal patterns, noise, and cross-feature dependencies.
    """
    np.random.seed(random_state)
    t = np.arange(n_steps)

    feature1 = 0.5 * np.sin(0.01 * t) + np.random.normal(0, 0.1, n_steps)
    feature2 = 0.3 * np.cos(0.015 * t + 1) + np.random.normal(0, 0.1, n_steps)
    feature3 = 0.2 * np.sin(0.02 * t + 2) + 0.1 * feature1
    feature4 = 0.4 * np.cos(0.005 * t) + 0.2 * feature2
    feature5 = 0.1 * np.sin(0.03 * t) + 0.3 * feature3

    target = (
        0.3 * feature1
        + 0.25 * feature2
        + 0.15 * feature3
        + 0.2 * feature4
        + 0.1 * feature5
        + np.random.normal(0, 0.05, n_steps)
    )

    df = pd.DataFrame(
        {
            "feature1": feature1,
            "feature2": feature2,
            "feature3": feature3,
            "feature4": feature4,
            "feature5": feature5,
            "target": target,
        }
    )
    return df


# ---------------------------------------------------------
# 2. Supervised windowing: lookback -> horizon
# ---------------------------------------------------------
def create_supervised_data(df, lookback=30, horizon=5):
    """
    Convert a multivariate time series into supervised 3D tensors:
    X: [samples, lookback, n_features]
    y: [samples, horizon] (multi-step target)
    """
    data = df.values
    n_features = data.shape[1] - 1  # exclude target
    X, y = [], []

    for i in range(len(data) - lookback - horizon):
        X.append(data[i : i + lookback, :n_features])
        y.append(data[i + lookback : i + lookback + horizon, -1])

    return np.array(X), np.array(y)


# ---------------------------------------------------------
# 3. LSTM model definition
# ---------------------------------------------------------
def build_lstm_model(input_shape, horizon=5):
    """
    Simple but strong LSTM architecture for multi-step forecasting.
    """
    model = Sequential(
        [
            LSTM(64, input_shape=input_shape),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(horizon),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


# ---------------------------------------------------------
# 4. MASE metric
# ---------------------------------------------------------
def mase(y_true, y_pred):
    """
    Mean Absolute Scaled Error: scale MAE by a naive one-step forecast.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    naive = np.mean(np.abs(np.diff(y_true)))
    return np.mean(np.abs(y_true - y_pred)) / naive


# ---------------------------------------------------------
# 5. Walk-forward validation on training data
# ---------------------------------------------------------
def walk_forward_validation(X_train, y_train, n_splits=5, batch_size=64, epochs=30):
    """
    Expanding-window walk-forward validation.
    Returns metrics per split and the best-performing model.
    """
    split_size = len(X_train) // n_splits
    metrics = []
    best_model = None
    best_mae = np.inf

    for i in range(n_splits):
        print(f"\n🔹 Walk-forward split {i+1}/{n_splits}")
        end_idx = split_size * (i + 1)

        X_sub = X_train[:end_idx]
        y_sub = y_train[:end_idx]

        # Use last 10% of this subset as validation
        val_size = max(1, int(0.1 * len(X_sub)))
        X_tr, X_val = X_sub[:-val_size], X_sub[-val_size:]
        y_tr, y_val = y_sub[:-val_size], y_sub[-val_size:]

        model = build_lstm_model(input_shape=X_train.shape[1:], horizon=y_train.shape[1])

        es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)

        history = model.fit(
            X_tr,
            y_tr,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es, lr],
            verbose=0,
        )

        # Evaluate on validation slice
        val_pred = model.predict(X_val, verbose=0)
        val_mae = mean_absolute_error(y_val.flatten(), val_pred.flatten())
        val_rmse = mean_squared_error(y_val.flatten(), val_pred.flatten(), squared=False)
        val_mase = mase(y_val, val_pred)

        print(f"   Split {i+1} - MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, MASE: {val_mase:.4f}")
        metrics.append((val_mae, val_rmse, val_mase))

        # Track best model for final explainability & test evaluation
        if val_mae < best_mae:
            best_mae = val_mae
            best_model = model

    return metrics, best_model


# ---------------------------------------------------------
# 6. SARIMAX multi-step baseline (5-step horizon)
# ---------------------------------------------------------
def sarimax_multi_step_baseline(train_series, test_series, horizon=5, max_windows=50):
    """
    Rolling SARIMAX baseline that generates 5-step forecasts for a subset
    of test windows to compare with the LSTM's 5-step outputs.

    NOTE: Fitting SARIMAX for every window is expensive, so we limit
    to `max_windows` most recent windows for demonstration.
    """
    y_true_list = []
    y_pred_list = []

    # Use only the last `max_windows` offset positions in the test set
    n_windows = max(1, min(len(test_series) - horizon, max_windows))
    start_offsets = range(len(test_series) - n_windows - horizon, len(test_series) - horizon)

    for offset in start_offsets:
        history = np.concatenate([train_series, test_series[:offset]])
        model = SARIMAX(history, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
        model_fit = model.fit(disp=False)

        forecast = model_fit.forecast(steps=horizon)
        y_pred_list.append(forecast)
        y_true_list.append(test_series[offset : offset + horizon])

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    return y_true, y_pred


# ---------------------------------------------------------
# 7. SHAP-based sequence explainability
# ---------------------------------------------------------
def shap_explain_lstm(model, X_background, X_explain, feature_names=None):
    """
    Use SHAP DeepExplainer to obtain time-series-aware feature attributions.
    Returns average |SHAP| per feature (aggregated over time and samples).
    """
    if not SHAP_AVAILABLE:
        print("⚠️ SHAP not available, skipping explainability.")
        return None, None

    print("\n🧠 Running SHAP explainability on LSTM model...")

    # DeepExplainer expects a list for Keras models
    explainer = shap.DeepExplainer(model, X_background)
    shap_values = explainer.shap_values(X_explain)  # list with one array (for output)

    # shap_values shape: [1, n_samples, lookback, n_features] for regression
    shap_values = np.array(shap_values)[0]  # remove output-dim list wrapper

    # Aggregate |SHAP| over time and samples
    mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 1))  # -> [n_features]

    if feature_names is None:
        feature_names = [f"feature{i+1}" for i in range(mean_abs_shap.shape[0])]

    feature_importance = dict(zip(feature_names, mean_abs_shap))

    # Sort by importance
    sorted_importance = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )

    print("\n🔍 Top 3 most influential input features (by mean |SHAP|):")
    for i, (fname, score) in enumerate(list(sorted_importance.items())[:3], start=1):
        print(f"   {i}. {fname}: {score:.4f}")

    return shap_values, sorted_importance


# ---------------------------------------------------------
# 8. MAIN EXECUTION PIPELINE
# ---------------------------------------------------------
def main():
    lookback = 30
    horizon = 5
    test_ratio = 0.2

    print("\n🚀 Generating synthetic multivariate dataset...")
    df = generate_synthetic_dataset(n_steps=10000)

    # Train / Test split (time-based)
    split_idx = int(len(df) * (1 - test_ratio))
    df_train_raw = df.iloc[:split_idx].copy()
    df_test_raw = df.iloc[split_idx:].copy()

    # Scale features using only train data (no leakage)
    scaler = StandardScaler()
    scaler.fit(df_train_raw)
    df_train = pd.DataFrame(scaler.transform(df_train_raw), columns=df.columns)
    df_test = pd.DataFrame(scaler.transform(df_test_raw), columns=df.columns)

    print("📌 Creating supervised train and test datasets...")
    X_train, y_train = create_supervised_data(df_train, lookback=lookback, horizon=horizon)
    X_test, y_test = create_supervised_data(df_test, lookback=lookback, horizon=horizon)

    print(f"   Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"   Input shape for LSTM: {X_train.shape[1:]}  |  Horizon: {horizon}")

    # ------------------- LSTM: Walk-forward on train -------------------
    print("\n📊 Running walk-forward validation on training data...")
    wf_metrics, best_model = walk_forward_validation(
        X_train, y_train, n_splits=5, batch_size=64, epochs=30
    )

    wf_mae = np.mean([m[0] for m in wf_metrics])
    wf_rmse = np.mean([m[1] for m in wf_metrics])
    wf_mase = np.mean([m[2] for m in wf_metrics])

    print("\n📊 Average Walk-Forward Validation Performance (LSTM)")
    print(f"   MAE : {wf_mae:.4f}")
    print(f"   RMSE: {wf_rmse:.4f}")
    print(f"   MASE: {wf_mase:.4f}")

    # ------------------- Final LSTM evaluation on test -------------------
    print("\n✅ Evaluating best LSTM model on held-out test set...")
    lstm_test_pred = best_model.predict(X_test, verbose=0)

    lstm_mae = mean_absolute_error(y_test.flatten(), lstm_test_pred.flatten())
    lstm_rmse = mean_squared_error(y_test.flatten(), lstm_test_pred.flatten(), squared=False)
    lstm_mase = mase(y_test, lstm_test_pred)

    print("\n📊 LSTM Test Performance (5-step horizon)")
    print(f"   MAE : {lstm_mae:.4f}")
    print(f"   RMSE: {lstm_rmse:.4f}")
    print(f"   MASE: {lstm_mase:.4f}")

    # ------------------- SARIMAX baseline (multi-step) -------------------
    print("\n📌 Building SARIMAX multi-step baseline...")
    target_train = df_train["target"].values
    target_test = df_test["target"].values

    sarimax_y_true, sarimax_y_pred = sarimax_multi_step_baseline(
        target_train, target_test, horizon=horizon, max_windows=50
    )

    sarimax_mae = mean_absolute_error(sarimax_y_true.flatten(), sarimax_y_pred.flatten())
    sarimax_rmse = mean_squared_error(
        sarimax_y_true.flatten(), sarimax_y_pred.flatten(), squared=False
    )
    sarimax_mase = mase(sarimax_y_true, sarimax_y_pred)

    print("\n📊 SARIMAX Baseline Performance (5-step horizon, last windows)")
    print(f"   MAE : {sarimax_mae:.4f}")
    print(f"   RMSE: {sarimax_rmse:.4f}")
    print(f"   MASE: {sarimax_mase:.4f}")

    # ------------------- SHAP Explainability -------------------
    # Use a small background set and a few test samples to keep SHAP runtime manageable
    if SHAP_AVAILABLE:
        background_size = min(200, X_train.shape[0])
        explain_size = min(50, X_test.shape[0])

        X_background = X_train[np.random.choice(X_train.shape[0], background_size, replace=False)]
        X_explain = X_test[:explain_size]

        feature_names = ["feature1", "feature2", "feature3", "feature4", "feature5"]
        shap_values, feature_importance = shap_explain_lstm(
            best_model, X_background, X_explain, feature_names=feature_names
        )
    else:
        feature_importance = None

    print("\n🎯 Pipeline complete – data, model, validation, baseline, and explainability all implemented.")


if __name__ == "__main__":
    main()
