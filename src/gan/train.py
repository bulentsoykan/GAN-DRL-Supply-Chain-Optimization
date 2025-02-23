import torch
import torch.optim as optim
import torch.nn as nn
from model import Generator, Discriminator
from utils import load_and_normalize_data, denormalize_data, save_synthetic_data

# Parameters
historical_data_file = "historical_supply_chain_data.csv"
features = ["Demand", "Lead_Time", "Inventory_Level", "Order_Quantity", "Supplier_Reliability"]
latent_dim = 10
batch_size = 64
epochs = 5000
lr = 0.001

# Load and normalize historical data
data_tensor, min_vals, max_vals = load_and_normalize_data(historical_data_file, features)

# Initialize models
num_features = len(features)
generator = Generator(latent_dim, num_features)
discriminator = Discriminator(num_features)

# Optimizers and loss function
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
criterion = nn.BCELoss()

# Training loop
for epoch in range(epochs):
    # Train Discriminator
    real_data = data_tensor[torch.randint(0, len(data_tensor), (batch_size,))]
    fake_noise = torch.randn(batch_size, latent_dim)
    fake_data = generator(fake_noise)

    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)

    optimizer_D.zero_grad()
    loss_real = criterion(discriminator(real_data), real_labels)
    loss_fake = criterion(discriminator(fake_data.detach()), fake_labels)
    loss_D = (loss_real + loss_fake) / 2
    loss_D.backward()
    optimizer_D.step()

    # Train Generator
    optimizer_G.zero_grad()
    loss_G = criterion(discriminator(fake_data), real_labels)
    loss_G.backward()
    optimizer_G.step()

    # Print progress every 500 epochs
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss_D={loss_D.item():.4f}, Loss_G={loss_G.item():.4f}")

# Generate synthetic data
num_synthetic_samples = 1000
noise = torch.randn(num_synthetic_samples, latent_dim)
synthetic_data = generator(noise).detach().numpy()

# Convert synthetic data back to original scale
synthetic_data = denormalize_data(synthetic_data, min_vals, max_vals)

# Save synthetic data
save_synthetic_data(synthetic_data, features)
