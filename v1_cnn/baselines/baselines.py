import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def _as_class_indices(labels):
    """Convert integer labels or one-hot labels into a one-dimensional array."""
    labels = np.asarray(labels)
    if labels.ndim == 2:
        return np.argmax(labels, axis=1)
    if labels.ndim == 1:
        return labels
    raise ValueError("Labels must be a 1D class array or a 2D one-hot array.")


def run_baseline_models(X_train_3d, X_val_3d, y_train_encoded, y_val_encoded):
    print("\n--- Running Baseline Models ---")

    if X_train_3d.ndim != 3 or X_val_3d.ndim != 3:
        raise ValueError("Baseline models expect feature tensors shaped (samples, time, channels).")

    y_train_encoded = _as_class_indices(y_train_encoded)
    y_val_encoded = _as_class_indices(y_val_encoded)

    # BASELINE 1: "Last Value"
    # We slice the tensor to only look at the final time step (t249) for all 23 sensors
    # Shape goes from (Samples, 250, 23) -> (Samples, 23)
    X_train_last = X_train_3d[:, -1, :] 
    X_val_last = X_val_3d[:, -1, :]
    
    rf_last = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_last.fit(X_train_last, y_train_encoded)
    pred_last = rf_last.predict(X_val_last)
    f1_last = f1_score(y_val_encoded, pred_last, average='macro')
    print(f"Baseline 1 (Last Value Only) Macro F1-Score: {f1_last:.4f}")
    
    # BASELINE 2: "Temporal Mean"
    # We collapse the 250 time steps into a single average reading per sensor
    # Shape goes from (Samples, 250, 23) -> (Samples, 23)
    X_train_mean = X_train_3d.mean(axis=1)
    X_val_mean = X_val_3d.mean(axis=1)
    
    rf_mean = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_mean.fit(X_train_mean, y_train_encoded)
    pred_mean = rf_mean.predict(X_val_mean)
    f1_mean = f1_score(y_val_encoded, pred_mean, average='macro')
    print(f"Baseline 2 (Time-Averaged Mean) Macro F1-Score: {f1_mean:.4f}")
    
    return f1_last, f1_mean
