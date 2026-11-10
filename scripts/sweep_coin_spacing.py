"""金币间距扫描：金币越稀疏，表格 Q-learning 还能不能学会走到传送门？

    python scripts/sweep_coin_spacing.py --spacings 12 24 36 48 72 0 --episodes 600

spacing=0 表示不放金币（只有终点奖励）。每个 (世界, 智能体种子, 间距) 独立训练，并行运行。
记录：贪心策略是否成功、步数、沿路线最远推进到百分之几、第一次成功出现在第几回合。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402


def run_job(job: tuple) -> dict:
    sys.path.insert(0, ROOT)
    from agents import TabularQAgent, run_episode
    from envs import MinecraftLikeGridEnv

    world, agent_seed, spacing, episodes, goal_distance, max_steps = job
    env = MinecraftLikeGridEnv(world_seed=world, goal_distance=goal_distance, reward_shaping="none",
                               coins=spacing > 0, coin_spacing=max(spacing, 1), max_steps=max_steps)
    agent = TabularQAgent(env.action_space.n, seed=agent_seed)
    first_success = None
    for ep in range(episodes):
        eps = 1.0 + min(1.0, ep / (0.6 * episodes)) * (0.05 - 1.0)
        stats = run_episode(env, agent, eps, seed=0)
        if stats["success"] and first_success is None:
            first_success = ep + 1
    greedy = run_episode(env, agent, epsilon=0.0, learn=False, seed=0)
    q = env.quest
    best = min(q.distance(*p) for p in greedy["path"])
    return {
        "world": world, "agent": agent_seed, "spacing": spacing,
        "coins_total": env.coins_total, "optimal": q.optimal_length,
        "success": greedy["success"], "steps": greedy["steps"], "coins": greedy["coins"],
        "progress": 1.0 - best / q.optimal_length, "first_success": first_success,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", nargs="+", default=["7", "12345", "2024"])
    ap.add_argument("--agents", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--spacings", nargs="+", type=int, default=[12, 24, 36, 48, 72, 0])
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--goal-distance", type=float, default=150)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    jobs = [(w, a, s, args.episodes, args.goal_distance, args.max_steps)
            for s in args.spacings for w in args.worlds for a in args.agents]
    t = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(run_job, jobs, chunksize=1)
    print(f"{len(jobs)} runs, {time.time() - t:.0f}s\n")

    label = lambda s: "无金币" if s == 0 else str(s)  # noqa: E731
    print("| 间距 | 世界 | 金币数 | 成功 | 贪心步数 / 最短 | 最远推进 | 首次成功回合 |")
    print("|---|---|---|---|---|---|---|")
    for s in args.spacings:
        for w in args.worlds:
            rs = [r for r in results if r["spacing"] == s and r["world"] == w]
            ok = [r for r in rs if r["success"]]
            steps = f"{min(r['steps'] for r in ok)}–{max(r['steps'] for r in ok)} / {rs[0]['optimal']}" if ok else f"— / {rs[0]['optimal']}"
            prog = " ".join(f"{r['progress']:.0%}" for r in rs)
            first = " ".join(str(r["first_success"]) if r["first_success"] else "—" for r in rs)
            print(f"| {label(s)} | {w} | {rs[0]['coins_total']} | {len(ok)}/{len(rs)} | {steps} | {prog} | {first} |")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [s for s in args.spacings if s > 0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for w in args.worlds:
        succ = [np.mean([r["success"] for r in results if r["spacing"] == s and r["world"] == w]) for s in xs]
        prog = [np.mean([r["progress"] for r in results if r["spacing"] == s and r["world"] == w]) for s in xs]
        axes[0].plot(xs, succ, marker="o", label=f"world {w}")
        axes[1].plot(xs, prog, marker="o", label=f"world {w}")
    if 0 in args.spacings:
        for ax, key in zip(axes, ("success", "progress")):
            ax.axhline(np.mean([r[key] for r in results if r["spacing"] == 0]), color="gray", ls="--",
                       label="no coins")
    axes[0].set_title("greedy policy reaches portal (fraction of agent seeds)")
    axes[1].set_title("furthest progress along route (mean)")
    for ax in axes:
        ax.set_xlabel("coin spacing (steps)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(xs)
        ax.legend()
    fig.suptitle(f"Tabular Q-learning, {args.episodes} episodes, goal ~{args.goal_distance:.0f} blocks")
    plt.tight_layout()
    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    out = os.path.join(ROOT, "out", f"coin_spacing_sweep{args.tag}.png")
    fig.savefig(out, dpi=80)
    print(f"\nplot -> {out}")


if __name__ == "__main__":
    main()
