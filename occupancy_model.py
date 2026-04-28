"""
occupancy_model.py
------------------
Trains and wraps a small MLP (Multi-Layer Perceptron) neural network
that predicts parking occupancy given contextual features.

Architecture
------------
  Input  : [hour, weekday, lot_id, class_density, event]  — 5 features
  Hidden : 64 → 32 neurons, ReLU activation
  Output : 1 neuron (occupancy fraction 0-1)

The same interface works whether scikit-learn or Keras is available;
scikit-learn MLPRegressor is used by default (no GPU required).
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle, os

from campus_graph import LOT_ID_MAP
from generate_data import generate_occupancy_data

FEATURE_COLS = ["hour", "weekday", "lot_id", "class_density", "event"]
TARGET_COL   = "occupancy_pct"
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "occupancy_model.pkl")


class OccupancyPredictor:
    """
    Thin wrapper around a trained MLPRegressor + StandardScaler.

    Usage
    -----
        predictor = OccupancyPredictor()
        predictor.train()                        # train on synthetic data
        occ = predictor.predict("P_CENTRAL", hour=9, weekday=1, event=0)
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.model  = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )
        self._trained = False

    # ------------------------------------------------------------------
    def train(self, n_days: int = 180, verbose: bool = True) -> dict:
        """Generate synthetic data, train, and return evaluation metrics."""
        df = generate_occupancy_data(n_days=n_days)

        X = df[FEATURE_COLS].values
        y = df[TARGET_COL].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        self.model.fit(X_train_s, y_train)
        self._trained = True

        y_pred = np.clip(self.model.predict(X_test_s), 0, 1)
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)

        metrics = {
            "MAE":  round(mae,  4),
            "R²":   round(r2,   4),
            "samples_train": len(X_train),
            "samples_test":  len(X_test),
        }

        if verbose:
            print("=== Occupancy Model Training Complete ===")
            for k, v in metrics.items():
                print(f"  {k}: {v}")

        return metrics

    # ------------------------------------------------------------------
    def _class_density_estimate(self, hour: int, weekday: int) -> int:
        """Quick estimate when caller doesn't supply class density."""
        if weekday >= 5:
            return 0
        return 7 if 8 <= hour <= 17 else (2 if 18 <= hour <= 20 else 0)

    def predict(
        self,
        lot_name: str,
        hour: int,
        weekday: int,
        event: int = 0,
        class_density: int | None = None,
    ) -> float:
        """
        Return predicted occupancy fraction (0-1) for a specific lot.

        Parameters
        ----------
        lot_name      : e.g. "P_CENTRAL"
        hour          : 0-23
        weekday       : 0 (Mon) – 6 (Sun)
        event         : 1 if campus event today, else 0
        class_density : number of concurrent classes (auto-estimated if None)
        """
        if not self._trained:
            raise RuntimeError("Call .train() before .predict()")

        if lot_name not in LOT_ID_MAP:
            raise ValueError(f"Unknown lot: {lot_name}")

        if class_density is None:
            class_density = self._class_density_estimate(hour, weekday)

        features = np.array([[
            hour, weekday, LOT_ID_MAP[lot_name], class_density, event
        ]])
        features_s = self.scaler.transform(features)
        occ = float(np.clip(self.model.predict(features_s)[0], 0.0, 1.0))
        return round(occ, 4)

    def predict_all_lots(
        self, hour: int, weekday: int, event: int = 0
    ) -> dict[str, float]:
        """Return predicted occupancy for every parking lot."""
        return {
            lot: self.predict(lot, hour, weekday, event)
            for lot in LOT_ID_MAP
        }

    # ------------------------------------------------------------------
    def save(self, path: str = MODEL_PATH):
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "model": self.model}, f)
        print(f"Model saved → {path}")

    def load(self, path: str = MODEL_PATH):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.scaler = obj["scaler"]
        self.model  = obj["model"]
        self._trained = True
        print(f"Model loaded ← {path}")


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = OccupancyPredictor()
    p.train()
    print("\nSample predictions (Monday 9 AM, no event):")
    for lot, occ in sorted(p.predict_all_lots(hour=9, weekday=0).items()):
        print(f"  {lot:12s}: {occ:.1%} occupied")
