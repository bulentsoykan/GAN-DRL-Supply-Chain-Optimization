import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load historical data
file_path = "historical_supply_chain_data.csv"
df_hist = pd.read_csv(file_path)

# Load synthetic data
synthetic_file_path = "synthetic_supply_chain_data.csv"
df_synth = pd.read_csv(synthetic_file_path)

# Combine both datasets
df = pd.concat([df_hist, df_synth], ignore_index=True)

# Convert date column if exists
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])

# Drop duplicates and check missing values
df = df.drop_duplicates()
missing_values = df.isnull().sum()
print("Missing Values:")
print(missing_values[missing_values > 0])

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Visualizing distributions
numeric_cols = df.select_dtypes(include=np.number).columns
plt.figure(figsize=(15, 6))
for i, col in enumerate(numeric_cols[:4]):  # Adjust number of plots as needed
    plt.subplot(2, 2, i+1)
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Time-series visualization
if 'date' in df.columns:
    plt.figure(figsize=(12, 5))
    sns.lineplot(x='date', y='inventory', data=df)
    plt.title('Inventory Levels Over Time')
    plt.xticks(rotation=45)
    plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlations")
plt.show()

# Feature Engineering for DRL
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

df_scaled.to_csv("processed_supply_chain_data.csv", index=False)
print("Processed data saved successfully!")
