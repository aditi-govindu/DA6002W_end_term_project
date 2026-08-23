# DA6002W_end_term_project
Industrial Inventory Control using reinforcement learning algorithms.

# Design for a Reinforcement Learning Industrial Inventory Control System

This document serves as the system design document for the development, training, evaluation, and deployment of reinforcement learning (RL) policies for the multi-product industrial inventory control system.

> NOTE: This project is was commissioned as part of the Course DA6002W under the IITM Web M.Tech Program. All rights to the code and use are reserved by the author and the DS AI community at IITM.

---

## 1. System Architecture & Module Overview

The system architecture consists of 5 modules designed to manage environment interactions, feature engineering, policy training pipelines, local validation and automated testing.

```
                  +-----------------------------------+
                  |  industrial_inventory_env (Gym)   |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |    Module 1: Preprocessing &      |
                  |        Feature Engineering        |
                  +-----------------------------------+
                                    |
            +-----------------------+-----------------------+
            |                                               |
            v                                               v
+-----------------------+                       +-----------------------+
|  Module 2: Tabular    |                       |  Module 3: Neural     |
|   RL Pipelines        |                       |   RL Pipelines        |
| (Q-Learn, SARSA, TD)  |                       | (DQN, DDQN, PPO, etc.)|
+-----------------------+                       +-----------------------+
            |                                               |
            +-----------------------+-----------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |   Module 4: Reward Shaping &      |
                  |        Cost Accounting            |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |  Module 5: Local Validation &     |
                  |       Scenario Testing            |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |  Module 6: Policy Export &        |
                  |     Inference Interface           |
                  +-----------------------------------+
```

### 1.1 Project modules
1. **Module 1: Preprocessing & Feature Engineering**
   - Receives raw Gymnasium observation dictionaries.
   - For Neural Methods: Normalizes vector components to $[0, 1]$ bounds.
   - For Tabular Methods: Discretizes continuous spaces (inventory, arrival pipeline) into discrete state bins to prevent state-space explosion.
2. **Module 2: 2 Tabular RL Pipelines**
   - Implements model-free tabular algorithms: Tabular Q-Learning and Tabular SARSA
   - Manages state-action lookup tables ($Q$-tables and $E$-tables) with memory-efficient indexing.
3. **Module 3: 4 Deep RL Pipelines**
   - Implements deep reinforcement learning algorithms: Deep Q-Network (DQN), Double Deep Q-Network (DDQN), Asynchronous Advantage Actor-Critic (A3C), and `Proximal Policy Optimisation (PPO)`.
   - Contains PyTorch neural network architectures, experience replay buffers, policy/value heads, and optimization algorithms.
4. **Module 4: Reward Shaping & Cost Accounting**
   - Computes official daily costs and base rewards.
   - Provides rewards for training acceleration while retaining strict unscaled cost tracking.
5. **Module 5: Local Validation Engine**
   - Executes deterministic policy evaluations across 20+ local validation episodes across 4 declared demand scenarios - `Stationary, Seasonal, Trend, Temporary Shock`.
6. **Module 6: Policy Export & Submission Interface**
   - Packages frozen weights and encapsulates the deterministic `run_policy(observation)` interface.

---

## 2. Core Environment Mechanics & Data Specifications

### 2.1 State Space Specification
The environment returns a dictionary observation `obs` at each step $t$:

| Variable used | Purpose | Data type | No. of items |
| :--- | :--- | :--- | :--- |
| `inventory` | Current on-hand inventory per product | `np.ndarray` shape `(3,)` | $[0, 1000]$ items |
| `arrival_pipeline` | Outstanding orders queued across arrival days | `np.ndarray` shape `(3, 4)` | $[0, 1000]$ items |
| `demand_history` | Historical demand observed during past 7 days | `np.ndarray` shape `(7, 3)` | $[0, \infty)$ items |
| `day` | Current simulation day index | `int` or `np.ndarray` shape `(1,)` | $[1, 50]$ |
| `capacity_utilisation` | Ratio of active inventory volume to maximum capacity | `float` or `np.ndarray` shape `(1,)` | $[0.0, 1.0]$ |

### 2.2 Product Parameters & Cost Functions

#### Constants & Product Attributes
- **Number of Products ($N$):** $3$
- **Warehouse Capacity ($V_{\max}$):** $1000.0$ volume units
- **Product Volumes ($\mathbf{v}$):** $v_1 = 2.0$, $v_2 = 3.0$, $v_3 = 1.5$ per item
- **Daily Holding Cost ($c_h$):** $	ext{Rs. } 5.00$ per unit volume per day
- **Stockout Costs ($\mathbf{c}_s$):** $c_{s,1} = 400.0$, $c_{s,2} = 500.0$, $c_{s,3} = 300.0$ per unfulfilled item
- **Fixed Ordering Costs ($\mathbf{c}_o$):** $c_{o,1} = 80.0$, $c_{o,2} = 200.0$, $c_{o,3} = 120.0$ per non-zero order
- **Discarding Costs ($\mathbf{c}_d$):** $c_{d,1} = 200.0$, $c_{d,2} = 250.0$, $c_{d,3} = 150.0$ per discarded item
- **Base Lead Times ($\mathbf{L}$):** $L_1 = 3$, $L_2 = 2$, $L_3 = 1$ days
- **Reference Demand Means (${\mu}_d$):** $\mu_{d,1} = 30$, $\mu_{d,2} = 25$, $\mu_{d,3} = 35$ units/day
- **Episode Duration ($T$):** $50$ days

#### Cost per variable to be optimised
For each day $t \in \{1, \dots, 50\}$ and product $i \in \{1, 2, 3\}$:

1. **Holding Cost ($C_{h,t}$):**
   $$V_t = \sum_{i=1}^3 v_i \cdot I_{i,t}$$
   $$C_{h,t} = V_t \cdot c_h$$
   where $I_{i,t}$ is the on-hand inventory of product $i$ after serving day $t$'s arrivals and demand.

2. **Stockout Cost ($C_{s,t}$):**
   $$C_{s,t} = \sum_{i=1}^3 c_{s,i} \cdot \max(0, D_{i,t} - I_{i,t}^{	ext{available}})$$
   where $D_{i,t}$ is realized demand and $I_{i,t}^{	ext{available}}$ is on-hand stock prior to fulfillment.

3. **Ordering Cost ($C_{o,t}$):**
   $$C_{o,t} = \sum_{i=1}^3 c_{o,i} \cdot \mathbb{I}(q_{i,t} > 0)$$
   where $q_{i,t} \in \{0, 10, 20, \dots, 100\}$ and $\mathbb{I}(\cdot)$ is the indicator function.

4. **Discarding Cost ($C_{d,t}$):**
When arriving stock $A_{i,t}$ pushes total volume over $V_{\max}$:

$$V_{\text{excess}, t} = \max\left(0, \sum_{i=1}^3 v_i \cdot (I_{i,t-1} + A_{i,t}) - V_{\max}\right)$$

Allocated discard quantities $K_{i,t}$ incur:

$$C_{d,t} = \sum_{i=1}^3 c_{d,i} \cdot K_{i,t}$$

5. **Total Daily Cost & Official Base Reward:**
   $$	ext{Daily Cost}_t = C_{h,t} + C_{s,t} + C_{o,t} + C_{d,t}$$
   $$R_{	ext{base}, t} = -rac{	ext{Daily Cost}_t}{100.0}$$
   $$	ext{Episode Return} = \sum_{t=1}^{50} R_{	ext{base}, t}$$

---

## 3. Preprocessing, State Vector Design & Action Space Mapping

### 3.1 Continuous State Normalization (Deep RL Input Layer)
The raw dictionary observation is flattened into a continuous vector $\mathbf{x}_t \in \mathbb{R}^{38}$ defined as:

$$\mathbf{x}_t = \left[ \frac{\mathbf{I}_t}{500.0}, \frac{\text{vec}(\mathbf{P}_t)}{200.0}, \frac{\text{vec}(\mathbf{H}_t)}{100.0}, \frac{t}{50.0}, U_t \right]$$

* $\mathbf{I}_t \in \mathbb{R}^3$: Current inventory vector.
* $\mathbf{P}_t \in \mathbb{R}^{3 \times 4}$: Arrival pipeline matrix flattened to 12 elements.
* $\mathbf{H}_t \in \mathbb{R}^{7 \times 3}$: Demand history matrix flattened to 21 elements.
* $t \in \mathbb{R}^1$: Current day index.
* $U_t \in \mathbb{R}^1$: Capacity utilization ratio ($0.0$ to $1.0$).

### 3.2 Discrete State Aggregation (Tabular RL Input Layer)
To make tabular RL tractable, state variables are binned into a single discrete state index $S_t$:
- **Inventory Bins (per product):** Low ($[0, 30)$), Medium ($[30, 80)$), High ($\ge 80$). ($3^3 = 27$ states).
- **Total Pipeline Sum Bins (per product):** Low ($[0, 30)$), High ($\ge 30$). ($2^3 = 8$ states).
- **Episode Phase:** Early ($t \le 15$), Mid ($15 < t \le 35$), Late ($t > 35$). ($3$ states).
- **Total Tabular State Space Size:** $|\mathcal{S}| = 27 	imes 8 	imes 3 = 648$ state indices.

### 3.3 Action Space Mapping & Translation

#### Internal Gymnasium Environment Action Representation
The environment expects action indices $a_i \in \{0, 1, 2, \dots, 10\}$ for each product, representing quantities $10 \cdot a_i$.
- **Action Space Type:** `MultiDiscrete([11, 11, 11])`
- **Total Joint Discrete Actions:** $11^3 = 1331$ combinations.

#### Policy Interface Return Specification
The `run_policy(observation)` function must return explicit quantities:
$$\mathbf{q}_t = [q_1, q_2, q_3]^T \quad 	ext{where } q_i \in \{0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100\}$$

**Mapping Logic:**


$$\mathbf{a}_t = \left[ \left\lfloor \frac{q_1}{10} \right\rfloor, \left\lfloor \frac{q_2}{10} \right\rfloor, \left\lfloor \frac{q_3}{10} \right\rfloor \right]^T, \quad q_i = 10 \cdot a_i$$

---

## 5. Algorithms evaluated

The errors stem from broken closing brackets (`\right` detached from `]`), missing backslashes on commands (`\theta` turned into `heta`, `\argmax` turned into `rg\max`), and mismatched parenthesis/bracket pairing in the DQN loss equation.

Here is the corrected Markdown:

### 5.1 Tabular Q-Learning
* **Update Rule:**
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_t + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t) \right]$$
* **Inference Policy:** $A^* = \arg\max_{a} Q(S, a)$

### 5.2 Tabular SARSA
* **Update Rule:**
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_t + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$
* **Inference Policy:** $A^* = \arg\max_{a} Q(S, a)$

### 5.3 Deep Q-Network (DQN)
* **Loss Function:**
$$L(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$
where $\theta^-$ denotes parameters of the target network updated every $C$ steps.

### 5.4 Double Deep Q-Network (DDQN)
- **Target Value Formulation:**
  $$Y_t^{	ext{DoubleQ}} = R_{t+1} + \gamma Q\left(S_{t+1}, \arg\max_{a} Q(S_{t+1}, a; 	heta_t); 	heta_t^- 
\right)$$
- **Loss Function:**
  $$L(	heta) = \mathbb{E} \left[ \left( Y_t^{	ext{DoubleQ}} - Q(S_t, A_t; 	heta_t) 
\right)^2 
\right]$$

### 5.5 Proximal Policy Optimization (PPO)
- **Probability Ratio:**
  $$r_t(	heta) = \frac{\pi_{	heta}(a_t | s_t)}{\pi_{	heta_{	ext{old}}}(a_t | s_t)}$$
- **Clipped Surrogate Objective:**
  $$L^{	ext{CLIP}}(	heta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(	heta)\hat{A}_t, \, 	ext{clip}(r_t(	heta), 1-\epsilon, 1+\epsilon)\hat{A}_t 
\right) 
\right]$$
  where $\hat{A}_t$ is calculated using Generalised Advantage Estimation (GAE-$\lambda$).

### 5.6 Asynchronous Actor Critic (A3C)
* **$n$-Step Advantage Formulation:**

$$A(s_t, a_t) = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n}; \phi') - V(s_t; \phi')$$


* **Actor Loss with Entropy Regularization:**

$$L_{\text{actor}}(\theta') = -\frac{1}{n} \sum_{i=0}^{n-1} \left[ \log \pi(a_{t+i} \vert{} s_{t+i}; \theta') A(s_{t+i}, a_{t+i}) + \beta H(\pi(\cdot \vert{} s_{t+i}; \theta')) \right]$$

where gradients calculated across 8 parallel workers update the shared global parameters asynchronously.

---

## 6. Training Dynamics & Reward Shaping Strategy

### 6.1 Reward Shaping Formulation
To accelerate learning in sparse-cost or high-stockout regimes, optional training reward shaping may be applied:
$$R_{	ext{shaped}, t} = R_{	ext{base}, t} - \psi_{	ext{stockout}} \sum_{i=1}^3 \max(0, D_{i,t} - I_{i,t}) - \psi_{	ext{cap}} \cdot \max(0, U_t - 0.90)$$
- **Default Hyperparameters:** $\psi_{	ext{stockout}} = 2.0$, $\psi_{	ext{cap}} = 5.0$.
- **Validation Rule:** All local validation and leaderboard logging MUST bypass $R_{	ext{shaped}}$ and evaluate strictly using unscaled actual daily cost $	ext{Daily Cost}_t$.

### 6.2 Domain Randomization Bounds
Each training episode must apply domain randomization according to assigned student variant boundaries:
- **Demand Level Multiplier:** $\eta_i \sim U(0.85, 1.15)$ applied to reference demand mean $\mu_{d,i}$.
- **Initial Inventory:** $I_{i,0} \sim 	ext{UniformDiscrete}(\{80, 90, 100, 110, 120\})$.
- **Lead-Time Delay Probability:** $p_{	ext{delay}} \sim U(0.00, 0.10)$. Probability of $+1$ day delay.

---

## 7. Performance Evaluation Metrics

### 7.1 Quantitative Metrics

1. **Episode Total Unscaled Cost ($C_{\text{ep}}$):**

$$C_{\text{ep}} = \sum_{t=1}^{50} \text{Daily Cost}_t$$


2. **Mean Episode Cost ($\bar{C}$):**

$$\bar{C} = \frac{1}{M} \sum_{k=1}^M C_{\text{ep}}^{(k)} \quad \text{for } M \ge 20 \text{ validation episodes}$$


3. **Cost Standard Deviation ($\sigma_C$):**

$$\sigma_C = \sqrt{\frac{1}{M-1} \sum_{k=1}^M \left( C_{\text{ep}}^{(k)} - \bar{C} \right)^2}$$


4. **Stockout Frequency ($F_{\text{stockout}}$):**

$$F_{\text{stockout}} = \frac{1}{50 \cdot M} \sum_{k=1}^M \sum_{t=1}^{50} \mathbb{I}\left( \sum_{i=1}^3 \max(0, D_{i,t} - I_{i,t}) > 0 \right)$$


5. **Capacity Discard Event Rate ($R_{\text{discard}}$):**

$$R_{\text{discard}} = \frac{1}{50 \cdot M} \sum_{k=1}^M \sum_{t=1}^{50} \mathbb{I}(C_{d,t} > 0)$$

---

## 8. Verification & Test Suite

Automated validation checks are mandated prior to leaderboard submission.

```
+--------------------------------------------------------------------------+
|                       Automated Verification Suite                       |
+--------------------------------------------------------------------------+
| 1. Interface Signature Test   --> run_policy(obs) returns list of length 3 |
| 2. Action Range & Type Test   --> q_i in {0,10,20,...,100}, dtype == int  |
| 3. Determinism Test           --> run_policy(obs) returns identical output|
| 4. Runtime Latency Test       --> Execution time < 10ms per step         |
| 5. Zero-Side-Effect Test      --> No env.step(), file writes, or network |
+--------------------------------------------------------------------------+
```

### 8.1 Test Implementation Details

#### Test 1: Policy Interface Conformance Test
- **Requirement:** Inspect policy file to ensure `run_policy(observation)` exists.
- **Assertion:**
  ```python
  output = run_policy(sample_obs)
  assert isinstance(output, list), "Output must be a Python list"
  assert len(output) == 3, "Output list must contain exactly 3 elements"
  ```

#### Test 2: Discrete Action Mapping Validity Test
- **Requirement:** Validate that every element returned by `run_policy` belongs to the allowed discrete set $\{0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100\}$.
- **Assertion:**
  ```python
  allowed = set(range(0, 101, 10))
  for val in output:
      assert val in allowed, f"Invalid order quantity returned: {val}"
      assert isinstance(val, (int, np.integer)), f"Quantity {val} must be integer type"
  ```

#### Test 3: Deterministic Policy Inference Test
- **Requirement:** Given the exact same observation input, `run_policy(observation)` must return identical actions across consecutive invocations.
- **Assertion:**
  ```python
  action1 = run_policy(sample_obs)
  action2 = run_policy(sample_obs)
  assert action1 == action2, "Policy inference is non-deterministic!"
  ```

#### Test 4: Runtime Environment Isolation & Latency Test
- **Requirement:** Inference execution per step must complete within $10	ext{ ms}$. Internet access or side-effect calls (`env.step()`, `env.reset()`, file write operations) are strictly prohibited.
- **Assertion:**
  ```python
  import time
  start = time.perf_counter()
  _ = run_policy(sample_obs)
  elapsed = time.perf_counter() - start
  assert elapsed < 0.010, f"Inference execution time too high: {elapsed:.4f}s"
  ```
