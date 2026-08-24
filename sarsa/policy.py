# @title Inference using saved Tabular SARSA model
import numpy as np
from pathlib import Path

# --- Constants (must match values used during training) ---
# These values are derived from your notebook's global constants and environment setup.
# Ensure these match the definitions in your training notebook.
N = 3 # Number of Products
QUANTITIES = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
NUM_PRODUCTS = N
NUM_CHOICES_PER_PRODUCT = len(QUANTITIES) # 11 (0 to 100 in steps of 10)
NUM_TOTAL_ACTIONS = NUM_CHOICES_PER_PRODUCT ** NUM_PRODUCTS # 11^3 = 1331
NUM_DISCRETE_STATES = 27 * 8 * 3  # From discrete_state_aggregation (Inventory * Pipeline * DayPhase)

# --- Helper Functions (copied from notebook for self-containment) ---
def discrete_state_aggregation(obs: dict) -> int:
    """Aggregates continuous state variables into a single discrete state index."""
    # Inventory Bins (per product): Low ([0, 30)), Medium ([30, 80)), High (>= 80).
    inventory_bins = []
    for i in range(N):
        if obs["inventory"][i] < 30:
            inventory_bins.append(0) # Low
        elif obs["inventory"][i] < 80:
            inventory_bins.append(1) # Medium
        else:
            inventory_bins.append(2) # High
    inventory_state = inventory_bins[0] * 9 + inventory_bins[1] * 3 + inventory_bins[2] * 1

    # Total Pipeline Sum Bins (per product): Low ([0, 30)), High (>= 30).
    pipeline_bins = []
    for i in range(N):
        total_pipeline_sum = obs["arrival_pipeline"][i].sum()
        if total_pipeline_sum < 30:
            pipeline_bins.append(0) # Low
        else:
            pipeline_bins.append(1) # High
    pipeline_state = pipeline_bins[0] * 4 + pipeline_bins[1] * 2 + pipeline_bins[2] * 1

    # Episode Phase: Early (t <= 15), Mid (15 < t <= 35), Late (t > 35).
    day = obs["day"][0]
    if day <= 15:
        day_state = 0 # Early
    elif day <= 35:
        day_state = 1 # Mid
    else:
        day_state = 2 # Late

    discrete_state = inventory_state * (8 * 3) + pipeline_state * 3 + day_state
    return discrete_state

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

# --- Global Q-table Instance and Loading for policy.py ---
# This Q-table will be loaded once when the policy.py module is imported.
initial_sarsa_q_table = np.zeros((NUM_DISCRETE_STATES, NUM_TOTAL_ACTIONS))

# Define the path to your saved model file.
# In a submission, 'sarsa_table_trained.npy' should be in the same directory as 'policy.py'
model_path = Path('./sarsa_table_trained.npy') # Assuming model is saved in the same directory

if model_path.exists():
    try:
        sarsa_q_table = np.load(model_path)
        print(f"SARSA Q-table loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading SARSA Q-table from {model_path}: {e}")
        print("SARSA Q-table initialized with zeros instead.")
        sarsa_q_table = initial_sarsa_q_table
else:
    print(f"Warning: Model file not found at {model_path}. SARSA Q-table initialized with zeros.")
    print("Please ensure your trained 'sarsa_table_trained.npy' is saved in the correct location.")
    sarsa_q_table = initial_sarsa_q_table


# --- Main Policy Function for Submission ---
def run_policy(observation: dict) -> list[int]:
    """Return order quantities for Products 1, 2 and 3 using the trained Tabular SARSA policy."""

    # Get discrete state from observation
    state = discrete_state_aggregation(observation)

    # Choose action with the highest Q-value (exploitation only)
    action_idx = np.argmax(sarsa_q_table[state, :])

    # Convert single action index to MultiDiscrete action tuple
    multi_discrete_action = map_action_index_to_multi_discrete(action_idx)

    # Convert MultiDiscrete indices to actual quantities
    quantities = action_index_to_quantity(multi_discrete_action)

    return quantities

# 
