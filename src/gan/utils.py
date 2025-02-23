import numpy as np
import pandas as pd
import torch

def load_and_normalize_data(file_path, features):
    """Loads CSV data and normalizes numerical features to [0,1]."""
    data = pd.read_csv(file_path)
    selected_data = data[features].values
    
    min_vals = np.min(selected_data, axis=0)
    max_vals = np.max(selected_data, axis=0)
    normalized_data = (selected_data - min_vals) / (max_vals - min_vals)
    
    return torch.tensor(normalized_data, dtype=torch.float32), min_vals, max_vals

def denormalize_data(normalized_data, min_vals, max_vals):
    """Converts normalized data back to the original scale."""
    return normalized_data * (max_vals - min_vals) + min_vals

def save_synthetic_data(data, features, file_name="synthetic_supply_chain_data.csv"):
    """Saves generated synthetic data to a CSV file."""
    df = pd.DataFrame(data, columns=features)
    df["Date"] = pd.date_range(start="2025-01-01", periods=len(data), freq="D")
    df.to_csv(file_name, index=False)
    print(f"Synthetic data saved to {file_name}")
