import os
import pandas as pd
from model import GAN
from train import train_gan
from environment import SupplyChainEnv
from age import DRLAgent

def load_data(filepath):
    """Loads CSV data into a Pandas DataFrame."""
    return pd.read_csv(filepath)

def save_data(data, filepath):
    """Saves a Pandas DataFrame to a CSV file."""
    data.to_csv(filepath, index=False)

def main():
    # Load historical and synthetic data
    historical_data = load_data("../historical_supply_chain_data.csv")
    synthetic_data = load_data("../synthetic_supply_chain_data.csv")
    
    # Combine datasets
    full_data = pd.concat([historical_data, synthetic_data], ignore_index=True)
    save_data(full_data, "../processed_supply_chain_data.csv")
    
    # Train GAN for synthetic data generation
    gan = GAN()
    train_gan(gan, full_data)
    
    # Initialize Supply Chain Environment
    env = SupplyChainEnv(full_data)
    
    # Train DRL Agent
    agent = DRLAgent(env)
    agent.train()
    
    print("Pipeline execution complete!")

if __name__ == "__main__":
    main()
