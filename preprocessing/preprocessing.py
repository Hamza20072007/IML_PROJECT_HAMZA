import pandas as pd
import numpy as np
from pathlib import Path

def preprocess_har_data(csv_path, output_path=None):
    print("1. Loading dataset...")
    # Read all rows before inferring column types so mixed numeric/string values
    # do not produce inconsistent inference warnings.
    df = pd.read_csv(csv_path, low_memory=False)
    
    print("2. Separating metadata and features...")
    # Isolate the target and id so they aren't mangled by numeric conversions
    metadata = df[['id', 'target']].copy()
    features = df.drop(columns=['id', 'target'])
    
    print("3. Cleaning string anomalies and commas...")
    # Values in this list represent missing sensor readings, not real values.
    anomalies = ['?', '9999', '9999.0', 9999, 9999.0, '-', 'unknown', 'N/A', 'null']
    
    # Target only the object (string) columns to fix commas and hidden spaces
    object_cols = features.select_dtypes(include=['object']).columns
    for col in object_cols:
        # Strip whitespaces and replace European decimal commas with periods
        features[col] = (
            features[col]
            .astype('string')
            .str.strip()
            .str.replace(',', '.', regex=False)
        )

    # Run this after whitespace cleanup so values such as " ? " are also found.
    features = features.replace(anomalies, np.nan)
        
    print("4. Casting entire feature matrix to float64...")
    features = features.astype(np.float64)
    
    print("5. Reshaping into 3D Tensor (Samples, 250 time steps, 23 sensors)...")
    N = features.shape[0]
    time_steps = 250
    channels = 23
    
    # Verify the math before reshaping to prevent silent data mangling
    assert features.shape[1] == time_steps * channels, f"Expected {time_steps * channels} columns, got {features.shape[1]}"
    
    # Fold the flat rows into the 3D structure
    tensor_3d = features.values.reshape(N, time_steps, channels)
    
    print("6. Applying Time-Aware Imputation...")
    imputed_tensor = np.empty_like(tensor_3d)
    
    # Iterate through each sample to impute along its specific timeline
    for i in range(N):
        # Isolate the (250, 23) matrix for the current sample
        sample_df = pd.DataFrame(tensor_3d[i])
        
        # Interpolate strictly down the time axis (axis=0 of this specific slice)
        try:
            # Attempt cubic spline for smooth physical waveforms
            sample_df = sample_df.interpolate(method='cubic', axis=0)
        except ValueError:
            # Fallback to linear if a sensor drops too many points for cubic math to work
            sample_df = sample_df.interpolate(method='linear', axis=0)
            
        # Patch the terminal edges (t000 and t249) where interpolation cannot anchor
        sample_df = sample_df.ffill(axis=0).bfill(axis=0)
        
        # If a sensor was completely dead for the entire 2.5 seconds, fill with 0
        sample_df = sample_df.fillna(0)
        
        # Save the cleaned sample back into the tensor
        imputed_tensor[i] = sample_df.values
        
    print("Preprocessing complete!")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # A CSV is two-dimensional, so flatten each (250, 23) sample back into
        # the original feature-column layout while retaining id and target.
        preprocessed_df = pd.DataFrame(
            imputed_tensor.reshape(N, -1), columns=features.columns
        )
        preprocessed_df.insert(0, 'id', metadata['id'].to_numpy())
        preprocessed_df['target'] = metadata['target'].to_numpy()
        preprocessed_df.to_csv(output_path, index=False)
        print(f"Saved preprocessed CSV to: {output_path}")

    return imputed_tensor, metadata

# ==========================================
# EXECUTION IN VS CODE
# ==========================================
if __name__ == "__main__":
    # Run the pipeline
    X, y_df = preprocess_har_data(
        'data/trainData.csv', 'output/preprocessed_trainData.csv'
    )
    
    print("\n--- Sanity Checks ---")
    print("Final Tensor Shape:", X.shape)
    print("Contains NaNs:", np.isnan(X).any())
    print("Target classes:", y_df['target'].unique())
