import numpy as np
import matplotlib.pyplot as plt

# ---------- Environment ----------
class BanditEnv_2:
    def __init__(self, std=1.0):
        self.means = [-10, 6, 8, 0, -2]
        self.std = std
        
    def reset(self):
        # Means are constant in this problem
        pass
    
    def step(self, action):
        return np.random.normal(self.means[action], self.std)

# ---------- Helpers ----------
def incremental_mean(prev_mean, count, new_value):
    return prev_mean + (new_value - prev_mean) / count

def run_epsilon_greedy(T=200, n_arms=5, epsilon=0.1, decay_rate=None, std=1.0, seed=42):
    """
    Runs epsilon greedy experiment.
    
    Args:
        T: total trials
        n_arms: number of arms
        epsilon: initial epsilon
        decay_rate: if not None, eps(t+1) = eps(t) * decay_rate. 
                    if None, use fixed epsilon.
        std: standard deviation of bandit rewards
        seed: random seed
    """
    rng = np.random.default_rng(seed)
    env = BanditEnv_2(std=std)
    
    counts = np.zeros(n_arms, dtype=int)
    means = np.zeros(n_arms, dtype=float)
    rewards = []
    
    current_eps = epsilon
    eps_history = [] # To track epsilon over time for decay plots

    for t in range(T):
        # Epsilon-Greedy Logic
        if rng.random() < current_eps:
            a = rng.integers(0, n_arms) # Explore
        else:
            # Exploit: Random tie-breaking for equal means
            # (Standard np.argmax always picks first occurrence of max)
            max_val = np.max(means)
            candidates = np.where(means == max_val)[0]
            a = rng.choice(candidates)

        r = env.step(a)
        
        # Update
        counts[a] += 1
        means[a] = incremental_mean(means[a], counts[a], r)
        rewards.append(r)
        
        eps_history.append(current_eps)

        # Decay Epsilon if applicable
        if decay_rate is not None:
            current_eps = current_eps * decay_rate

    return {
        "total_reward": np.sum(rewards),
        "rewards": np.array(rewards),
        "counts": counts,
        "means": means,
        "eps_history": eps_history
    }

def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# ==========================================
# 1. Influence of Epsilon
# ==========================================
def experiment_epsilon_influence():
    print("\n--- 1. Influence of Epsilon ---")
    eps_values = [0.01, 0.1, 0.3, 0.5]
    T = 500
    
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Rewards over time
    plt.subplot(1, 2, 1)
    
    for eps in eps_values:
        # Run multiple seeds to get smoother curves
        avg_rewards = np.zeros(T)
        n_seeds = 50
        total_r = 0
        
        for s in range(n_seeds):
            res = run_epsilon_greedy(T=T, epsilon=eps, seed=s)
            avg_rewards += res["rewards"]
            total_r += res["total_reward"]
            
        avg_rewards /= n_seeds
        total_r /= n_seeds
        
        # Plot smoothed reward
        plt.plot(moving_average(avg_rewards, 20), label=f"ε={eps}")
        print(f"ε={eps}: Avg Total Reward = {total_r:.1f}")

    plt.axhline(8, color='k', linestyle='--', label="Optimal Mean")
    plt.xlabel("Trial")
    plt.ylabel("Average Reward (Smoothed)")
    plt.title("Reward History by Epsilon")
    plt.legend()
    
    # Subplot 2: Arm Selection Counts (from single representative run)
    plt.subplot(1, 2, 2)
    width = 0.2
    x = np.arange(5)
    
    for i, eps in enumerate(eps_values):
        # Just one seed for bar chart to show typical behavior distribution
        res = run_epsilon_greedy(T=T, epsilon=eps, seed=42)
        plt.bar(x + i*width, res["counts"], width, label=f"ε={eps}")
        
    plt.xticks(x + width*1.5, [f"Arm {i}" for i in range(5)])
    plt.xlabel("Bandit Arm")
    plt.ylabel("Selection Count")
    plt.title("Arm Selection Frequency")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("epsilon_influence.png")
    print("Saved plot: epsilon_influence.png")

# ==========================================
# 2. Optimal Epsilon
# ==========================================
def experiment_optimal_epsilon():
    print("\n--- 2. Finding Optimal Epsilon ---")
    eps_range = np.linspace(0, 0.5, 21) # 0.0 to 0.5
    final_rewards = []
    
    T = 500
    n_seeds = 100 # High number of seeds for statistical significance
    
    for eps in eps_range:
        current_total = 0
        for s in range(n_seeds):
            res = run_epsilon_greedy(T=T, epsilon=eps, seed=s)
            current_total += res["total_reward"]
        final_rewards.append(current_total / n_seeds)
        
    best_idx = np.argmax(final_rewards)
    best_eps = eps_range[best_idx]
    
    plt.figure(figsize=(8, 5))
    plt.plot(eps_range, final_rewards, marker='o')
    plt.axvline(best_eps, color='r', linestyle='--', label=f"Optimal ε={best_eps:.2f}")
    plt.xlabel("Epsilon")
    plt.ylabel("Average Total Reward")
    plt.title(f"Parameter Sweep: Optimal Epsilon (T={T})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("optimal_epsilon.png")
    print(f"Optimal Epsilon found: {best_eps:.2f} with reward {final_rewards[best_idx]:.1f}")
    print("Saved plot: optimal_epsilon.png")

# ==========================================
# 3. Epsilon Decay
# ==========================================
def experiment_epsilon_decay():
    print("\n--- 3. Epsilon Decay ---")
    # Rule: eps(t+1) = decay_rate * eps(t)
    decay_rates = [0.9, 0.99, 0.999]
    fixed_eps = 0.1 # Compare against a good fixed value
    T = 1000
    n_seeds = 50
    
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Reward Performance
    plt.subplot(1, 2, 1)
    
    # Run fixed first
    avg_rewards_fixed = np.zeros(T)
    for s in range(n_seeds):
        avg_rewards_fixed += run_epsilon_greedy(T=T, epsilon=fixed_eps, seed=s)["rewards"]
    plt.plot(moving_average(avg_rewards_fixed/n_seeds, 50), 'k--', label=f"Fixed ε={fixed_eps}")
    
    # Run decays
    for decay in decay_rates:
        avg_rewards = np.zeros(T)
        avg_totals = 0
        for s in range(n_seeds):
            res = run_epsilon_greedy(T=T, epsilon=1.0, decay_rate=decay, seed=s)
            avg_rewards += res["rewards"]
            avg_totals += res["total_reward"]
            
        plt.plot(moving_average(avg_rewards/n_seeds, 50), label=f"Decay η={decay}")
        print(f"Decay η={decay}: Avg Total = {avg_totals/n_seeds:.1f}")
        
    plt.axhline(8, color='gray', linestyle=':', label="Optimal")
    plt.xlabel("Trial")
    plt.ylabel("Avg Reward (Smoothed)")
    plt.title("Decay Strategy Performance")
    plt.legend()
    
    # Subplot 2: Epsilon over time
    plt.subplot(1, 2, 2)
    t_steps = np.arange(T)
    for decay in decay_rates:
        # eps(t) = eps0 * decay^t
        eps_curve = 1.0 * (np.array([decay] * T) ** t_steps)
        plt.plot(t_steps, eps_curve, label=f"η={decay}")
    
    plt.axhline(fixed_eps, color='k', linestyle='--', label=f"Fixed ε={fixed_eps}")
    plt.xlabel("Trial")
    plt.ylabel("Epsilon Value")
    plt.title("Epsilon Schedule")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("epsilon_decay.png")
    print("Saved plot: epsilon_decay.png")

# ==========================================
# 4. Standard Deviation Influence
# ==========================================
def experiment_std_influence():
    print("\n--- 4. Influence of Standard Deviation ---")
    stds = [0.5, 1.0, 2.0, 5.0]
    eps = 0.1
    T = 500
    n_seeds = 50
    
    plt.figure(figsize=(10, 6))
    
    for std in stds:
        avg_rewards = np.zeros(T)
        for s in range(n_seeds):
            res = run_epsilon_greedy(T=T, epsilon=eps, std=std, seed=s)
            avg_rewards += res["rewards"]
            
        # Normalize reward curve relative to optimal mean (8) so we can compare learning speed
        # If we plotted raw rewards, high noise graphs would just look "noisier"
        # We want to see how quickly it approaches the mean of 8.
        
        smoothed = moving_average(avg_rewards/n_seeds, 20)
        plt.plot(smoothed, label=f"σ={std}")
        
    plt.axhline(8, color='k', linestyle='--', label="Optimal Mean (8.0)")
    plt.xlabel("Trial")
    plt.ylabel("Average Reward")
    plt.title("Learning Performance vs Noise Level (σ)")
    plt.legend()
    plt.savefig("std_influence.png")
    print("Saved plot: std_influence.png")

if __name__ == "__main__":
    experiment_epsilon_influence()
    experiment_optimal_epsilon()
    experiment_epsilon_decay()
    experiment_std_influence()
