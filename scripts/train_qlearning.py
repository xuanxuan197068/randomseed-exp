"""对比实验：只有终点奖励（稀疏） vs 沿途金币（中间奖励），同一个世界、同样的训练预算。

    python scripts/train_qlearning.py --seed 7 --episodes 600

两组都关闭最短路塑形（reward_shaping="none"），只比较金币本身的作用。
输出学习曲线和贪心策略的路线图到 out/。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from agents import TabularQAgent, run_episode  # noqa: E402
from envs import MinecraftLikeGridEnv  # noqa: E402
from worldgen.render import terrain_rgb  # noqa: E402


def make_env(args, coins: bool) -> MinecraftLikeGridEnv:
    return MinecraftLikeGridEnv(
        world_seed=args.seed, goal_distance=args.goal_distance, reward_shaping="none",
        coins=coins, coin_spacing=args.coin_spacing, coin_reward=args.coin_reward,
        max_steps=args.max_steps,
    )


def train(args, coins: bool) -> tuple[TabularQAgent, list[dict], MinecraftLikeGridEnv]:
    env = make_env(args, coins)
    agent = TabularQAgent(env.action_space.n, alpha=args.alpha, gamma=args.gamma, seed=args.agent_seed)
    history = []
    t = time.time()
    for ep in range(args.episodes):
        frac = min(1.0, ep / max(1, args.episodes * args.explore_frac))
        eps = args.eps_start + frac * (args.eps_end - args.eps_start)
        stats = run_episode(env, agent, eps, seed=0)
        stats["epsilon"] = eps
        history.append(stats)
        if (ep + 1) % max(1, args.episodes // 10) == 0:
            recent = history[-max(1, args.episodes // 10):]
            print(f"  [{'coins ' if coins else 'sparse'}] ep {ep + 1:5d}  eps {eps:.2f}  "
                  f"success {np.mean([h['success'] for h in recent]):.2f}  "
                  f"coins {np.mean([h['coins'] for h in recent]):5.1f}/{stats['coins_total']}  "
                  f"steps {np.mean([h['steps'] for h in recent]):6.0f}  ({time.time() - t:.0f}s)")
    return agent, history, env


def moving(x, k):
    x = np.asarray(x, dtype=float)
    return np.convolve(x, np.ones(k) / k, mode="valid") if len(x) >= k else x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="7", help="世界种子")
    ap.add_argument("--goal-distance", type=float, default=150)
    ap.add_argument("--coin-spacing", type=int, default=12)
    ap.add_argument("--coin-reward", type=float, default=0.1)
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--explore-frac", type=float, default=0.6, help="ε 在前多少比例的回合内衰减到 eps-end")
    ap.add_argument("--agent-seed", type=int, default=0)
    args = ap.parse_args()

    results = {}
    for coins in (False, True):
        name = "coins" if coins else "sparse"
        print(f"== {name} ==")
        agent, history, env = train(args, coins)
        greedy = run_episode(env, agent, epsilon=0.0, learn=False, seed=0)
        q = env.quest
        print(f"  greedy: success={greedy['success']} steps={greedy['steps']} (optimal {q.optimal_length}) "
              f"coins={greedy['coins']}/{greedy['coins_total']} states={len(agent.q)}")
        results[name] = (history, greedy, env)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = max(1, args.episodes // 30)
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))
    for name, (history, _, _) in results.items():
        axes[0].plot(moving([h["success"] for h in history], k), label=name)
        axes[1].plot(moving([h["steps"] for h in history], k), label=name)
    axes[0].set_title(f"success rate (moving avg {k})")
    axes[0].set_xlabel("episode")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_title("episode length")
    axes[1].set_xlabel("episode")
    axes[1].axhline(results["coins"][2].quest.optimal_length, color="gray", ls="--", label="optimal")
    for ax in axes[:2]:
        ax.legend()

    _, greedy, env = results["coins"]
    q = env.quest
    x0, z0, x1, z1 = q.box
    path = np.array(greedy["path"])
    sparse_path = np.array(results["sparse"][1]["path"])
    pad = 24
    vx0, vz0 = min(path[:, 0].min(), q.goal.x) - pad, min(path[:, 1].min(), q.goal.z) - pad
    vx1, vz1 = max(path[:, 0].max(), q.goal.x) + pad, max(path[:, 1].max(), q.goal.z) + pad
    area = env.world.generate_area(vx0, vz0, vx1 - vx0, vz1 - vz0)
    ax = axes[2]
    ax.imshow(terrain_rgb(area), extent=(vx0, vx1, vz1, vz0), interpolation="nearest")
    cx = [c[0] + 0.5 for c in env.coin_layout]
    cz = [c[1] + 0.5 for c in env.coin_layout]
    ax.scatter(cx, cz, s=40, c="gold", edgecolors="black", zorder=3, label="coins")
    ax.plot(sparse_path[:, 0] + 0.5, sparse_path[:, 1] + 0.5, color="white", lw=1, alpha=0.7, label="greedy (sparse)")
    ax.plot(path[:, 0] + 0.5, path[:, 1] + 0.5, color="red", lw=2, label="greedy (coins)")
    ax.plot(*q.start, marker="P", color="white", markersize=14, markeredgecolor="black", ls="")
    ax.plot(*q.goal_pos, marker="*", color="magenta", markersize=20, markeredgecolor="black", ls="")
    ax.set_xlim(vx0, vx1)
    ax.set_ylim(vz1, vz0)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(f"greedy policy: {greedy['steps']} steps (optimal {q.optimal_length}), "
                 f"coins {greedy['coins']}/{greedy['coins_total']}")
    fig.suptitle(f"Tabular Q-learning, world seed {env.world.seed}, goal ~{args.goal_distance:.0f} blocks, "
                 f"{args.episodes} episodes each")
    out = os.path.join("out", f"qlearning_{env.world.seed}.png")
    os.makedirs("out", exist_ok=True)
    plt.tight_layout()
    fig.savefig(out, dpi=80)
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
