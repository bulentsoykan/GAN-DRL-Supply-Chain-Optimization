import numpy as np
import pandas as pd

def load_historical_data(file_path, features):
    """Loads and normalizes historical supply chain data."""
    df = pd.read_csv(file_path)
    selected_data = df[features].values

    min_vals = np.min(selected_data, axis=0)
    max_vals = np.max(selected_data, axis=0)
    normalized_data = (selected_data - min_vals) / (max_vals - min_vals)

    return normalized_data, min_vals, max_vals

def save_results(results, file_name="training_results.csv"):
    """Saves episode results to a CSV file."""
    df = pd.DataFrame(results, columns=["Episode", "Total_Reward"])
    df.to_csv(file_name, index=False)
    print(f"Results saved to {file_name}")
