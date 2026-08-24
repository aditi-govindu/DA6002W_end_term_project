# @title Inference by reading from PPO saved model
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from pathlib import Path

# --- Constants (must match values used during training) ---
# These values are derived from your notebook's global constants and environment setup.
# Ensure these match the definitions in your training notebook.
STATE_DIM = 38  # inv(3) + pipeline(12) + demand_hist(21) + day(1) + capacity(1)
ACTION_DIM = 1331 # 11^3 (11 choices per product, 3 products)
QUANTITIES = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
NUM_PRODUCTS = 3
NUM_CHOICES_PER_PRODUCT = len(QUANTITIES)

# --- Helper Functions (copied from notebook for self-containment) ---
def continuous_state_normalization(obs: dict) -> np.ndarray:
    """Normalizes and flattens the raw observation dictionary into a continuous vector."""
    inventory_norm = obs["inventory"].astype(float) / 500.0
    pipeline_norm = obs["arrival_pipeline"].flatten().astype(float) / 200.0
    demand_history_norm = obs["demand_history"].flatten().astype(float) / 100.0
    day_norm = obs["day"].astype(float) / 50.0
    capacity_utilisation_norm = obs["capacity_utilisation"].astype(float)
    return np.concatenate([
        inventory_norm,
        pipeline_norm,
        demand_history_norm,
        day_norm,
        capacity_utilisation_norm
    ], axis=0)

def map_action_index_to_multi_discrete(action_idx: int) -> list[int]:
    """Converts a single integer action index to a MultiDiscrete action tuple."""
    multi_discrete_action = []
    temp_idx = action_idx
    for _ in range(NUM_PRODUCTS):
        multi_discrete_action.insert(0, temp_idx % NUM_CHOICES_PER_PRODUCT)
        temp_idx //= NUM_CHOICES_PER_PRODUCT
    return multi_discrete_action

def action_index_to_quantity(action_indices) -> list[int]:
    """Convert env discrete indices [a1, a2, a3] (0-10) to actual quantities [q1, q2, q3]"""
    return [int(QUANTITIES[idx]) for idx in action_indices]

# --- PPO Model Definition (copied from notebook's ActorCritic class) ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.actor_net = nn.Sequential(
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic_net = nn.Sequential(
            nn.Linear(256, 1)
        )

    def forward(self, x):
        shared_features = self.shared_net(x)
        action_probs = self.actor_net(shared_features)
        state_values = self.critic_net(shared_features)
        return action_probs, state_values
        
ppo_policy_model = ActorCritic(STATE_DIM, ACTION_DIM)
model_path = Path('./ppo_model.pth') # Assuming model is saved in the same directory

if model_path.exists():
    try:
        # Load the model's state_dict. map_location='cpu' is good practice for deployment.
        ppo_policy_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"PPO model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print("PPO model initialized with random weights instead.")
else:
    print(f"Warning: Model file not found at {model_path}. PPO model initialized with random weights.")
    print("Please ensure your trained 'ppo_model.pth' is saved in the correct location.")

ppo_policy_model.eval() # Set the model to evaluation mode; no gradients needed during inference.

# --- Main Policy Function for Submission ---
def run_policy(observation: dict) -> list[int]:
    """Return order quantities for Products 1, 2 and 3 using the trained PPO policy."""

    # Preprocess the observation
    state = continuous_state_normalization(observation)
    state_tensor = torch.FloatTensor(state).unsqueeze(0) # Add batch dimension

    # Perform inference
    with torch.no_grad(): # Disable gradient calculations for efficiency
        action_probs, _ = ppo_policy_model(state_tensor)
        dist = Categorical(action_probs)
        # For a deterministic policy during inference.
        action_idx = action_probs.argmax().item()

    # Convert the action index back to quantities
    multi_discrete_action = map_action_index_to_multi_discrete(action_idx)
    quantities = action_index_to_quantity(multi_discrete_action)

    return quantities