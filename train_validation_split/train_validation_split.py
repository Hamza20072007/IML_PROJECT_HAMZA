import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / 'output' / 'preprocessed_trainData.csv'

def create_train_validation_split(csv_path=DATA_PATH, random_state=42):
    """Create a reproducible, stratified train/validation split.

    Sequence IDs are returned for traceability but excluded from model features.
    """
    # 1. Load the fully sanitized 2D data.
    df = pd.read_csv(csv_path)

    required_columns = {'id', 'target'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    # 2. Separate model features from labels and sequence IDs.
    X = df.drop(columns=['id', 'target'])
    y = df['target']
    sequence_ids = df['id']

    # 3. Make an approximately 80/20 stratified split without putting rows with
    # the same ID in both partitions. This avoids possible sequence leakage.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    train_indices, val_indices = next(splitter.split(X, y, groups=sequence_ids))

    return (
        X.iloc[train_indices],
        X.iloc[val_indices],
        y.iloc[train_indices],
        y.iloc[val_indices],
        sequence_ids.iloc[train_indices],
        sequence_ids.iloc[val_indices],
    )


if __name__ == '__main__':
    X_train, X_val, y_train, y_val, ids_train, ids_val = create_train_validation_split()

    print(f"Training features shape: {X_train.shape}")
    print(f"Validation features shape: {X_val.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Validation labels shape: {y_val.shape}")
    print(f"Training IDs shape: {ids_train.shape}")
    print(f"Validation IDs shape: {ids_val.shape}")
