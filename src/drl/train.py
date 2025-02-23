import torch
import numpy as np
from agent import DRLAgent
from environment import SupplyChainEnv

# Hyperparameters
episodes = 500
target_update_freq = 10

env = SupplyChainEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DRLAgent(state_dim, action_dim)

for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    for t in range(100):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            break

    agent.replay()

    if episode % target_update_freq == 0:
        agent.update_target_network()

    print(f"Episode {episode}, Total Reward: {total_reward}")

torch.save(agent.policy_net.state_dict(), "drl_supply_chain_model.pth")
print("Training completed and model saved.")
