"""
CartPole-v1 — Search-Based Optimization Playground
==================================================

Implements the assignment’s Parts 1–5 (+ adaptive noise + conclusions hooks):

1) Random actions (1000 ep)  -> histogram + avg
2) Angle-based actions       -> histogram + avg
3) Random search (1000 iters; each 20 eps) for linear policy weights
   - Return best weights; test over 1000 eps
   - 3D scatter for weight vectors with avg >= 100 (red) vs <100 (black)
4) Hill climbing (1000 steps; each score = avg over 20 eps)
   - Return best; test 1000 eps; histogram
5) Simulated annealing (1000 steps; 20 eps/score)
   - With/without adaptive noise scaling
   - Return best; test 1000 eps; histogram

CLI examples (after installing gymnasium + matplotlib + seaborn):

    python cartpole_search.py --task random --episodes 1000 --plot random_hist.png
    python cartpole_search.py --task angle  --episodes 1000 --plot angle_hist.png

    # Random search (stores history + 3D scatter)
    python cartpole_search.py --task random_search --iters 1000 --per_iter 20 --scatter search_scatter.png

    # Hill climbing
    python cartpole_search.py --task hill --iters 1000 --per_iter 20 --sigma 0.1 --plot hill_hist.png

    # Simulated annealing (with adaptive noise scaling)
    python cartpole_search.py --task anneal --iters 1000 --per_iter 20 --sigma 0.1 \
        --t0 1.0 --cool 0.995 --adaptive --plot sa_hist.png

Notes:
- Uses Gymnasium API (returns (obs, info) on reset; step returns (obs, reward, terminated, truncated, info))
- Linear policy: action = 1 if dot(w, obs) >= 0 else 0  (obs = [x, x_dot, theta, theta_dot])
"""

import argparse
import math
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Any

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D)

# ------------------------- Utilities & Env helpers -------------------------

def make_env(seed: int = 0):
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return env

def run_episode(env: gym.Env, policy_fn: Callable[[np.ndarray], int], max_steps: int = 1000) -> int:
    obs, _ = env.reset()
    total = 0
    for _ in range(max_steps):
        a = policy_fn(obs)
        obs, r, terminated, truncated, _ = env.step(a)
        total += r
        if terminated or truncated:
            break
    return total

def eval_policy(env_maker: Callable[[], gym.Env],
                policy_fn: Callable[[np.ndarray], int],
                episodes: int,
                max_steps: int = 1000) -> List[int]:
    env = env_maker()
    scores = [run_episode(env, policy_fn, max_steps) for _ in range(episodes)]
    env.close()
    return scores

def plot_hist(scores: List[float], title: str, out: str = None):
    sns.set(style="whitegrid")
    plt.figure(figsize=(8,4.5))
    # distplot is deprecated; histplot with kde=True is fine for the assignment
    sns.histplot(scores, kde=True, bins=30)
    plt.title(title)
    plt.xlabel("Episode reward (timesteps alive)")
    plt.ylabel("Count")
    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=140)
        print(f"[saved] {out}")
    else:
        plt.show()

# ------------------------- Policies ---------------------------------------

def random_policy(_: np.ndarray) -> int:
    return np.random.randint(0, 2)

def angle_policy(obs: np.ndarray) -> int:
    # obs: [x, x_dot, theta, theta_dot]
    theta = obs[2]
    # push toward falling side: if theta > 0 (tilt right), push right (action 1), else left (0)
    return 1 if theta > 0 else 0

def linear_policy(weights: np.ndarray) -> Callable[[np.ndarray], int]:
    def policy(obs: np.ndarray) -> int:
        return 1 if float(np.dot(weights, obs)) >= 0.0 else 0
    return policy

# ------------------------- Scoring wrappers --------------------------------

def avg_return_for_weights(w: np.ndarray,
                           env_maker: Callable[[], gym.Env],
                           episodes: int,
                           max_steps: int = 1000) -> float:
    policy = linear_policy(w)
    scores = eval_policy(env_maker, policy, episodes, max_steps)
    return float(np.mean(scores))

# ------------------------- Part 1 & 2 Runners ------------------------------

def run_random(episodes: int, plot: str = None):
    scores = eval_policy(lambda: make_env(0), random_policy, episodes)
    print(f"[Random] avg over {episodes} episodes: {np.mean(scores):.2f}")
    plot_hist(scores, f"Random policy ({episodes} episodes)", plot)

def run_angle(episodes: int, plot: str = None):
    scores = eval_policy(lambda: make_env(1), angle_policy, episodes)
    print(f"[Angle] avg over {episodes} episodes: {np.mean(scores):.2f}")
    plot_hist(scores, f"Angle policy ({episodes} episodes)", plot)

# ------------------------- Part 3: Random Search ---------------------------

@dataclass
class SearchRecord:
    w: np.ndarray
    avg: float

def random_search(iters: int, per_iter: int, w_scale: float = 1.0,
                  seed: int = 42, max_steps: int = 1000) -> Tuple[np.ndarray, List[SearchRecord]]:
    rng = np.random.default_rng(seed)
    env_maker = lambda: make_env(seed)
    history: List[SearchRecord] = []
    best_w = None
    best_avg = -np.inf
    for t in range(1, iters+1):
        w = rng.normal(0.0, w_scale, size=4)
        avg = avg_return_for_weights(w, env_maker, per_iter, max_steps)
        history.append(SearchRecord(w=w, avg=avg))
        if avg > best_avg:
            best_avg = avg; best_w = w.copy()
        if t % 50 == 0:
            print(f"[RandomSearch] iter {t}/{iters}  best={best_avg:.1f}")
    return best_w, history

def scatter_3d(history: List[SearchRecord], out: str = None):
    # Determine importance by absolute weight magnitude
    W = np.stack([rec.w for rec in history], axis=0)
    A = np.abs(W).mean(0)
    idx = np.argsort(-A)[:3]  # top-3
    w123 = W[:, idx]
    c = ['red' if rec.avg >= 100 else 'black' for rec in history]

    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(w123[:,0], w123[:,1], w123[:,2], c=c, s=18, alpha=0.8)
    ax.set_xlabel(f"w[{idx[0]}]"); ax.set_ylabel(f"w[{idx[1]}]"); ax.set_zlabel(f"w[{idx[2]}]")
    ax.set_title("Random Search — weight vectors (red: avg ≥ 100)")
    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=140)
        print(f"[saved] {out}")
    else:
        plt.show()
    print(f"[Top-3 features by |w|]: indices {idx.tolist()} (descending importance)")

# ------------------------- Part 4: Hill Climbing ---------------------------

def hill_climb(iters: int, per_iter: int, sigma: float = 0.1,
               seed: int = 123, max_steps: int = 1000) -> Tuple[np.ndarray, List[SearchRecord]]:
    rng = np.random.default_rng(seed)
    env_maker = lambda: make_env(seed)
    w = rng.normal(0.0, 1.0, size=4)
    base = avg_return_for_weights(w, env_maker, per_iter, max_steps)
    history = [SearchRecord(w=w.copy(), avg=base)]
    best_w, best_avg = w.copy(), base
    for t in range(1, iters+1):
        candidate = w + rng.normal(0.0, sigma, size=4)
        cand_avg = avg_return_for_weights(candidate, env_maker, per_iter, max_steps)
        if cand_avg > base:
            w = candidate; base = cand_avg
            if cand_avg > best_avg: best_avg, best_w = cand_avg, candidate.copy()
        history.append(SearchRecord(w=w.copy(), avg=base))
        if t % 50 == 0:
            print(f"[Hill] iter {t}/{iters}  cur={base:.1f}  best={best_avg:.1f}")
    return best_w, history

# ------------------------- Part 5: Simulated Annealing --------------------

def accept_prob(delta: float, T: float) -> float:
    # Accept downhill with Boltzmann probability
    return math.exp(delta / max(T, 1e-8))

def simulated_annealing(iters: int, per_iter: int,
                        sigma: float = 0.1,
                        t0: float = 1.0,
                        cool: float = 0.995,
                        adaptive: bool = False,
                        seed: int = 7,
                        max_steps: int = 1000) -> Tuple[np.ndarray, List[SearchRecord]]:
    rng = np.random.default_rng(seed)
    env_maker = lambda: make_env(seed)
    w = rng.normal(0.0, 1.0, size=4)
    cur = avg_return_for_weights(w, env_maker, per_iter, max_steps)
    history = [SearchRecord(w=w.copy(), avg=cur)]
    best_w, best_avg = w.copy(), cur
    T = t0
    min_sigma, max_sigma = 0.01, 2.0
    for t in range(1, iters+1):
        noise = rng.normal(0.0, sigma, size=4)
        cand = w + noise
        val  = avg_return_for_weights(cand, env_maker, per_iter, max_steps)
        delta = val - cur
        take = False
        if val >= cur:
            take = True
            if adaptive:
                sigma = max(min_sigma, sigma * 0.5)  # reward up → shrink noise
        else:
            if rng.random() < accept_prob(delta, T):
                take = True
                if adaptive:
                    sigma = min(max_sigma, sigma * 2.0)  # reward down → enlarge noise
        if take:
            w, cur = cand, val
            if val > best_avg: best_avg, best_w = val, cand.copy()
        history.append(SearchRecord(w=w.copy(), avg=cur))
        T *= cool
        if t % 50 == 0:
            print(f"[SA{'(adaptive)' if adaptive else ''}] iter {t}/{iters}  cur={cur:.1f}  best={best_avg:.1f}  T={T:.4f}  sigma={sigma:.4f}")
    return best_w, history

# ------------------------- Shared evaluation / reporting -------------------

def test_weights(w: np.ndarray, episodes: int, seed: int = 101) -> Tuple[float, List[float]]:
    env_maker = lambda: make_env(seed)
    scores = eval_policy(env_maker, linear_policy(w), episodes)
    return float(np.mean(scores)), scores

def report_best(tag: str, w: np.ndarray, test_avg: float):
    print(f"[{tag}] Best weights: {np.round(w, 4)}")
    print(f"[{tag}] Test average reward (1000 episodes): {test_avg:.2f}")

# ------------------------- CLI --------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True,
                   choices=["random", "angle", "random_search", "hill", "anneal"])
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--plot", type=str, default=None)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--per_iter", type=int, default=20)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--t0", type=float, default=1.0)
    p.add_argument("--cool", type=float, default=0.995)
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--scatter", type=str, default=None)
    args = p.parse_args()

    if args.task == "random":
        run_random(args.episodes, plot=args.plot)
        return

    if args.task == "angle":
        run_angle(args.episodes, plot=args.plot)
        return

    if args.task == "random_search":
        best_w, history = random_search(args.iters, args.per_iter)
        test_avg, test_scores = test_weights(best_w, 1000)
        report_best("RandomSearch", best_w, test_avg)
        # 3D scatter
        if args.scatter: scatter_3d(history, out=args.scatter)
        return

    if args.task == "hill":
        best_w, history = hill_climb(args.iters, args.per_iter, sigma=args.sigma)
        test_avg, test_scores = test_weights(best_w, 1000)
        report_best("HillClimb", best_w, test_avg)
        if args.plot:
            plot_hist(test_scores, f"Hill Climb — best weights over 1000 episodes (avg={test_avg:.1f})", out=args.plot)
        return

    if args.task == "anneal":
        best_w, history = simulated_annealing(args.iters, args.per_iter,
                                              sigma=args.sigma, t0=args.t0,
                                              cool=args.cool, adaptive=args.adaptive)
        test_avg, test_scores = test_weights(best_w, 1000)
        tag = f"SimAnneal{' (adaptive)' if args.adaptive else ''}"
        report_best(tag, best_w, test_avg)
        if args.plot:
            plot_hist(test_scores, f"{tag} — best weights over 1000 episodes (avg={test_avg:.1f})", out=args.plot)
        return

if __name__ == "__main__":
    main()
