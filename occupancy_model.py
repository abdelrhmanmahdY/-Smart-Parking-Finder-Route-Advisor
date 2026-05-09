"""
occupancy_model.py
------------------
MLP neural network that predicts parking lot occupancy.

Behaviour
---------
- If model file EXISTS  --> delete it and retrain fresh every time
- If model file MISSING --> train and save it
- Always shows: Train R2, Test R2, MAE, MSE, Recall, Precision, F1, Overfit check
- Always saves CSV dataset to campus_parking/parking_dataset.csv
"""

import os, pickle
import numpy as np
from sklearn.neural_network  import MLPRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, r2_score, mean_squared_error,

)
from generate_data import compute_hour_weight
from campus_graph  import LOT_ID_MAP
from generate_data import generate_occupancy_data, save_dataset_csv

FEATURE_COLS = ["hour", "weekday", "lot_id", "class_density", "event"]
TARGET_COL   = "occupancy_pct"

# Model is saved right next to this file
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "occupancy_model.pkl"
)


class OccupancyPredictor:

    def __init__(self):
        self.scaler   = StandardScaler()
        self.model    = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,

        )
        self._trained = False

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #
    def initialise(self, verbose: bool = True):
        """
        Always retrain:
          - If a saved model exists  -> delete it first, then retrain
          - If no saved model exists -> just train
        After training, save the new model.
        """
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
            if verbose:
                print(f"  Existing model deleted → retraining from scratch")

        self.train(verbose=verbose)

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #
    def train(self, n_days: int = 12, verbose: bool = True):
        """Generate data, train model, save CSV + model, print metrics."""

        # 1. Generate dataset
        df = generate_occupancy_data(n_days=n_days)

        # 2. Save CSV  (always, so user can open it)
        csv_path = save_dataset_csv(df)
        if verbose:
            print(f"  Dataset  : {len(df)} rows saved → {csv_path}")

        # 3. Prepare features / target
        X = df[FEATURE_COLS].values
        y = df[TARGET_COL].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_tr = self.scaler.fit_transform(X_train)
        X_te = self.scaler.transform(X_test)

        # 4. Train
        self.model.fit(X_tr, y_train)
        self._trained = True

        # 5. Predictions
        yp_tr = np.clip(self.model.predict(X_tr), 0, 1)
        yp_te = np.clip(self.model.predict(X_te), 0, 1)

        # 6. Regression metrics
        train_mae = mean_absolute_error(y_train, yp_tr)
        test_mae  = mean_absolute_error(y_test,  yp_te)
        train_mse = mean_squared_error(y_train,  yp_tr)
        test_mse  = mean_squared_error(y_test,   yp_te)
        train_r2  = r2_score(y_train, yp_tr)
        test_r2   = r2_score(y_test,  yp_te)
        overfit   = train_r2 - test_r2

        metrics = {
            "rows_total":      len(df),
            "rows_train":      len(X_train),
            "rows_test":       len(X_test),
            "train_R2":        round(train_r2,  4),
            "test_R2":         round(test_r2,   4),
            "train_MAE":       round(train_mae, 4),
            "test_MAE":        round(test_mae,  4),
            "train_MSE":       round(train_mse, 4),
            "test_MSE":        round(test_mse,  4),
            "overfit_gap":     round(overfit,   4),
        }

        if verbose:
            self._print_report(metrics)

        # 8. Save model
        self._save()
        return metrics

    # ------------------------------------------------------------------ #
    #  Save / Load                                                         #
    # ------------------------------------------------------------------ #
    def _save(self):
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"scaler": self.scaler, "model": self.model}, f)
        print(f"  Model saved  → {MODEL_PATH}")

    # ------------------------------------------------------------------ #
    #  Report                                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _print_report(m: dict):
        sep = "=" * 54
        print(sep)
        print("           MODEL EVALUATION REPORT")
        print(sep)
        print(f"  Dataset   : {m['rows_total']} rows  "
              f"({m['rows_train']} train / {m['rows_test']} test)")
        print()
        print("  ── Regression Metrics ──────────────────────────")
        print(f"  Train R²  (train accuracy) : {m['train_R2']:.4f}")
        print(f"  Test  R²  (test  accuracy) : {m['test_R2']:.4f}")
        print(f"  Train MAE                  : {m['train_MAE']:.4f}")
        print(f"  Test  MAE                  : {m['test_MAE']:.4f}")
        print(f"  Train MSE                  : {m['train_MSE']:.4f}")
        print(f"  Test  MSE                  : {m['test_MSE']:.4f}")
        print()
        print("  ── Classification Metrics  (threshold > 80%) ───")
        print(f"  Train Recall               : {m['train_recall']:.4f}")
        print(f"  Test  Recall               : {m['test_recall']:.4f}")
        print(f"  Train Precision            : {m['train_precision']:.4f}")
        print(f"  Test  Precision            : {m['test_precision']:.4f}")
        print(f"  Train F1                   : {m['train_F1']:.4f}")
        print(f"  Test  F1                   : {m['test_F1']:.4f}")
        print()
        g = m["overfit_gap"]
        print("  ── Overfit Check ───────────────────────────────")
        print(f"  Train R² − Test R²         : {g:.4f}")
        if   g < 0.05:  status = "✔  No overfitting"
        elif g < 0.10:  status = "⚠  Slight overfit — acceptable"
        else:           status = "✘  Overfitting — add more data"
        print(f"  Result                     : {status}")
        print(sep)

    # ------------------------------------------------------------------ #
    #  Prediction                                                          #
    # ------------------------------------------------------------------ #
    def _density_estimate(self, hour, weekday):
        
        hw = compute_hour_weight(hour, weekday)
        if weekday >= 5: return 0
        if hw > 0.70:    return 8
        if hw > 0.30:    return 4
        if hw > 0.05:    return 1
        return 0

    def predict(self, lot_name, hour, weekday, event=0, class_density=None):
        if not self._trained:
            raise RuntimeError("Call initialise() before predict()")
        if lot_name not in LOT_ID_MAP:
            raise ValueError(f"Unknown lot: {lot_name}")
        if class_density is None:
            class_density = self._density_estimate(hour, weekday)
        X   = np.array([[hour, weekday, LOT_ID_MAP[lot_name], class_density, event]])
        occ = float(np.clip(self.model.predict(self.scaler.transform(X))[0], 0, 1))
        return round(occ, 4)

    def predict_all_lots(self, hour, weekday, event=0):
        return {lot: self.predict(lot, hour, weekday, event) for lot in LOT_ID_MAP}
if __name__ == "__main__":
    model=OccupancyPredictor()

    model.initialise()
