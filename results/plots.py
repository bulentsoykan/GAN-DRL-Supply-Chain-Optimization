import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
historical_data = pd.read_csv("../historical_supply_chain_data.csv")
synthetic_data = pd.read_csv("../synthetic_supply_chain_data.csv")

# Combine datasets
historical_data["Type"] = "Historical"
synthetic_data["Type"] = "Synthetic"
full_data = pd.concat([historical_data, synthetic_data], ignore_index=True)

# Demand Over Time
plt.figure(figsize=(10, 5))
sns.lineplot(data=full_data, x="Date", y="Demand", hue="Type")
plt.title("Demand Over Time")
plt.xticks(rotation=45)
plt.savefig("../figures/demand_over_time.png")

# Inventory Levels
plt.figure(figsize=(10, 5))
sns.histplot(full_data, x="Inventory_Level", hue="Type", kde=True, bins=30)
plt.title("Inventory Level Distribution")
plt.savefig("../figures/inventory_levels.png")

# Lead Time Distribution
plt.figure(figsize=(10, 5))
sns.boxplot(data=full_data, x="Type", y="Lead_Time")
plt.title("Lead Time Distribution")
plt.savefig("../figures/lead_time_distribution.png")

# Historical vs. Synthetic Data Comparison
plt.figure(figsize=(10, 5))
sns.kdeplot(historical_data["Demand"], label="Historical", fill=True)
sns.kdeplot(synthetic_data["Demand"], label="Synthetic", fill=True)
plt.title("Comparison of Demand Distributions")
plt.legend()
plt.savefig("../figures/historical_vs_synthetic_demand.png")

print("Plots saved in the figures directory.")
