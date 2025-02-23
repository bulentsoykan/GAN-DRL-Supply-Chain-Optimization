import numpy as np
import gym
from gym import spaces

class SupplyChainEnv(gym.Env):
    def __init__(self):
        super(SupplyChainEnv, self).__init__()

        self.max_inventory = 500
        self.max_order = 100
        self.max_demand = 50

        self.action_space = spaces.Discrete(3)  # Order 0, 50, or 100 units
        self.observation_space = spaces.Box(low=0, high=self.max_inventory, shape=(2,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.inventory = np.random.randint(100, 300)
        self.demand = np.random.randint(10, 50)
        return np.array([self.inventory, self.demand], dtype=np.float32)

    def step(self, action):
        order_amount = [0, 50, 100][action]

        self.inventory += order_amount - self.demand
        self.demand = np.random.randint(10, self.max_demand)

        reward = -abs(self.inventory - self.demand)  # Reward for balancing inventory
        done = False

        return np.array([self.inventory, self.demand], dtype=np.float32), reward, done, {}

    def render(self):
        print(f"Inventory: {self.inventory}, Demand: {self.demand}")
