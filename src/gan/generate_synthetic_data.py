import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# --- Define Generator and Discriminator ---
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# --- Generate Synthetic Data ---
np.random.seed(42)
real_data = torch.tensor([
    [100, 5, 50], [120, 7, 40], [110, 6, 45], [140, 8, 55], [90, 5, 35],
    [130, 7, 60], [150, 9, 70], [105, 6, 48], [95, 5, 38], [125, 8, 52]
], dtype=torch.float32)

input_dim = 2  # Latent space (random noise)
output_dim = 3  # Generated demand, lead time, inventory level
generator = Generator(input_dim, output_dim)
discriminator = Discriminator(output_dim)

# --- Training GAN ---
gen_optimizer = optim.Adam(generator.parameters(), lr=0.01)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

for epoch in range(500):
    z = torch.randn(real_data.shape[0], input_dim)  # Random noise
    fake_data = generator(z)  # Generate synthetic data
    
    # Train Discriminator
    real_labels = torch.ones(real_data.shape[0], 1)
    fake_labels = torch.zeros(real_data.shape[0], 1)
    disc_optimizer.zero_grad()
    real_loss = loss_fn(discriminator(real_data), real_labels)
    fake_loss = loss_fn(discriminator(fake_data.detach()), fake_labels)
    disc_loss = real_loss + fake_loss
    disc_loss.backward()
    disc_optimizer.step()
    
    # Train Generator
    gen_optimizer.zero_grad()
    gen_loss = loss_fn(discriminator(fake_data), real_labels)
    gen_loss.backward()
    gen_optimizer.step()

# --- Generate Synthetic Data for Analysis ---
z = torch.randn(100, input_dim)
generated_data = generator(z).detach().numpy()

# --- Visualize Generated vs Real Data ---
plt.figure(figsize=(10, 5))
plt.scatter(real_data[:, 0], real_data[:, 1], label='Real Data', marker='o', color='blue')
plt.scatter(generated_data[:, 0], generated_data[:, 1], label='Generated Data', marker='x', color='red')
plt.xlabel('Demand')
plt.ylabel('Lead Time')
plt.legend()
plt.title('Real vs Generated Supply Chain Data')
plt.show()
