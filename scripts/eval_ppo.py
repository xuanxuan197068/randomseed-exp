"""评估应变能力：训练好的 PPO 在**没见过的岩浆布局**和**没见过的世界**上表现如何。

    python scripts/eval_ppo.py --model out/ppo_lava.pt --episodes 200

测试集：
  A 训练世界 + 新岩浆布局      B 新世界 + 岩浆      C 新世界 + 2 倍岩浆（比训练更难）      D 新世界 + 无岩浆
对照智能体：
  compass   只朝末影之眼方向走（看不见岩浆、不管摔伤）——衡量陷阱本身有多致命
  rule      手写规则：朝目标方向走，但绝不踩岩浆、绝不跳会摔伤的落差——强的反应式基线
  ppo       PPO + CNN（按策略分布采样）
另外在单个世界上训练一个表格 Q-learning（只记坐标），看"背地图"能否应对每回合变化的岩浆。
"""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

HELDOUT_BASE = 5000
EVAL_SEED_BASE = 100_000  # 回合种子与训练时的完全不同 → 岩浆布局都是新的


def heuristic_action(env, rng, avoid: bool) -> int:
    """朝目标方向走；avoid=True 时跳过岩浆格和会摔伤的落差。"""
    from worldgen.navigation import fall_damage

    gx, gz = env.quest.goal_pos
    x, z = env.pos
    scores = []
    for a, (dx, dz) in enumerate(env.moves):
        dst = (x + dx, z + dz)
        ok = env.can_move(env.pos, dst)
        if ok and avoid:
            into_water = int(env._cell(*dst).height[0, 0]) < env.world_config.sea_level
            drop = env.stand_height(x, z) - env.stand_height(*dst)
            ok = dst not in env.lava and fall_damage(drop, into_water, env.safe_fall) == 0
        progress = -(abs(gx - dst[0]) + abs(gz - dst[1]))
        scores.append((ok, progress + rng.random() * 0.5))
    if rng.random() < 0.15:  # 少量随机，避免卡在凹角
        ok_actions = [a for a, (ok, _) in enumerate(scores) if ok]
        if ok_actions:
            return int(rng.choice(ok_actions))
    return max(range(len(scores)), key=lambda a: (scores[a][0], scores[a][1]))


def run_episodes(job: tuple) -> list[dict]:
    sys.path.insert(0, ROOT)
    import torch

    torch.set_num_threads(1)
    from agents.ppo import load, stack_obs
    from envs import MinecraftLikeGridEnv

    agent_name, model_path, env_kwargs, world, episode_seeds = job
    env = MinecraftLikeGridEnv(world_seed=world, **env_kwargs)
    net = load(model_path)[0] if agent_name == "ppo" else None
    rng = np.random.default_rng(world)
    out = []
    for s in episode_seeds:
        obs, info = env.reset(seed=s)
        torch.manual_seed(s)
        path = [info["pos"]]
        while True:
            if agent_name == "ppo":
                a = int(net.act(stack_obs([obs]))[0])
            else:
                a = heuristic_action(env, rng, avoid=(agent_name == "rule"))
            obs, r, te, tr, info = env.step(a)
            path.append(info["pos"])
            if te or tr:
                break
        out.append({"agent": agent_name, "world": world, "seed": s, "success": info["success"],
                    "death": info["death"], "steps": info["steps"], "optimal": info["optimal_length"],
                    "coins": info["coins_collected"], "path": path, "lava": sorted(env.lava)})
    return out


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    spl = [r["success"] * r["optimal"] / max(r["steps"], r["optimal"]) for r in rows]
    return {
        "n": n,
        "success": np.mean([r["success"] for r in rows]),
        "lava": np.mean([r["death"] == "lava" for r in rows]),
        "fall": np.mean([r["death"] == "fall" for r in rows]),
        "timeout": np.mean([not r["success"] and r["death"] is None for r in rows]),
        "spl": np.mean(spl),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "out", "ppo_lava.pt"))
    ap.add_argument("--episodes", type=int, default=200, help="每个测试集每个智能体的回合数")
    ap.add_argument("--worlds", type=int, default=20, help="每个测试集用多少个世界")
    ap.add_argument("--tabular-episodes", type=int, default=600)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--plot-only", action="store_true", help="只跑 B 测试集上的 PPO 并画轨迹图")
    args = ap.parse_args()

    import torch

    ckpt = torch.load(args.model, weights_only=False)
    base_kwargs = ckpt["env_kwargs"]
    train_worlds = list(range(1000, 1000 + args.worlds))
    heldout = list(range(HELDOUT_BASE, HELDOUT_BASE + args.worlds))
    per_world = max(1, args.episodes // args.worlds)
    seeds = [EVAL_SEED_BASE + k for k in range(per_world)]
    suites = {
        "A 训练世界·新岩浆": (train_worlds, base_kwargs),
        "B 新世界·岩浆": (heldout, base_kwargs),
        "C 新世界·2倍岩浆": (heldout, {**base_kwargs, "lava_density": 2 * base_kwargs["lava_density"]}),
        "D 新世界·无岩浆": (heldout, {**base_kwargs, "lava_density": 0.0}),
    }
    agents = ["compass", "rule", "ppo"]
    if args.plot_only:
        suites = {"B 新世界·岩浆": suites["B 新世界·岩浆"]}
        agents = ["ppo"]
    jobs, keys = [], []
    for suite, (worlds, kw) in suites.items():
        for agent in agents:
            for w in worlds:
                jobs.append((agent, args.model, kw, w, seeds))
                keys.append((suite, agent))
    with Pool(args.workers) as pool:
        chunks = pool.map(run_episodes, jobs, chunksize=1)
    results: dict = {}
    for key, rows in zip(keys, chunks):
        results.setdefault(key, []).extend(rows)

    if args.plot_only:
        plot_trajectories(results, base_kwargs)
        return
    print(f"每格 {per_world * args.worlds} 回合（{args.worlds} 个世界 × {per_world} 个岩浆布局）\n")
    print("| 测试集 | 智能体 | 成功率 | 岩浆死亡 | 摔死 | 超时 | SPL |")
    print("|---|---|---|---|---|---|---|")
    for suite in suites:
        for agent in agents:
            m = summarize(results[(suite, agent)])
            print(f"| {suite} | {agent} | {m['success']:.0%} | {m['lava']:.0%} | {m['fall']:.0%} | "
                  f"{m['timeout']:.0%} | {m['spl']:.2f} |")

    # 表格 Q-learning：在训练世界 1000 上学 600 回合（每回合岩浆不同），再在新岩浆布局上测
    from agents import TabularQAgent, run_episode
    from envs import MinecraftLikeGridEnv

    env = MinecraftLikeGridEnv(world_seed=1000, **base_kwargs)
    agent = TabularQAgent(env.action_space.n, seed=0)
    for ep in range(args.tabular_episodes):
        eps = 1.0 + min(1.0, ep / (0.6 * args.tabular_episodes)) * (0.05 - 1.0)
        run_episode(env, agent, eps)
    tab = []
    for s in range(EVAL_SEED_BASE, EVAL_SEED_BASE + 100):
        obs, info = env.reset(seed=s)
        st = (info["pos"][0], info["pos"][1], info["coins_mask"])
        while True:
            obs, r, te, tr, info = env.step(agent.act(st, 0.0))
            st = (info["pos"][0], info["pos"][1], info["coins_mask"])
            if te or tr:
                break
        tab.append({"success": info["success"], "death": info["death"], "steps": info["steps"],
                    "optimal": info["optimal_length"]})
    m = summarize(tab)
    ppo_same = summarize([r for r in results[("A 训练世界·新岩浆", "ppo")] if r["world"] == 1000])
    print(f"\n世界 1000，新岩浆布局（100 回合）：表格 Q 成功 {m['success']:.0%}、岩浆死亡 {m['lava']:.0%}、"
          f"超时 {m['timeout']:.0%}；PPO 在同一世界成功 {ppo_same['success']:.0%}、岩浆死亡 {ppo_same['lava']:.0%}")

    plot_trajectories(results, base_kwargs)


def plot_trajectories(results: dict, env_kwargs: dict) -> None:
    """PPO 在没见过的世界上：岩浆恰好挡住"无岩浆最短路"时它怎么绕；外加一个超时的失败案例。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from envs import MinecraftLikeGridEnv
    from worldgen.render import terrain_rgb

    rows = results[("B 新世界·岩浆", "ppo")] + results.get(("C 新世界·2倍岩浆", "ppo"), [])
    near_miss, timeouts = [], []
    envs: dict = {}
    for r in rows:
        env = envs.setdefault(r["world"], MinecraftLikeGridEnv(world_seed=r["world"], **env_kwargs))
        lava = set(map(tuple, r["lava"]))
        touching = sum(any((x + dx, z + dz) in lava for dx, dz in env.moves) for x, z in r["path"])
        if r["success"]:
            near_miss.append((touching, r))
        elif r["death"] is None:
            timeouts.append(r)
    picks, seen = [], set()
    for _, r in sorted(near_miss, key=lambda t: (-t[0], t[1]["world"], t[1]["seed"])):  # 贴着岩浆走过最多的成功回合
        if r["world"] not in seen and len(picks) < 2:
            picks.append(r)
            seen.add(r["world"])
    by_world: dict = {}
    for r in timeouts:
        by_world.setdefault(r["world"], []).append(r)
    for w in sorted(by_world, key=lambda w: -len(by_world[w]))[:2]:  # 最常超时的两个世界
        picks.append(by_world[w][0])
    fig, axes = plt.subplots(1, len(picks), figsize=(6.5 * len(picks), 7))
    for ax, r in zip(np.atleast_1d(axes), picks):
        env = envs[r["world"]]
        env.reset(seed=r["seed"])  # 地形、起点、终点只取决于世界；岩浆直接用记录下来的
        q = env.quest
        pts = np.array(list(r["path"]) + [q.goal_pos, q.start])
        pad = 10
        x0, z0 = int(pts[:, 0].min() - pad), int(pts[:, 1].min() - pad)
        x1, z1 = int(pts[:, 0].max() + pad), int(pts[:, 1].max() + pad)
        area = env.world.generate_area(x0, z0, x1 - x0, z1 - z0)
        img = terrain_rgb(area).copy()
        for x, z in r["lava"]:
            if x0 <= x < x1 and z0 <= z < z1:
                img[z - z0, x - x0] = (235, 95, 15)
        ax.imshow(img, extent=(x0, x1, z1, z0), interpolation="nearest")
        p = np.array(r["path"])
        ax.plot(p[:, 0] + 0.5, p[:, 1] + 0.5, color="red", lw=1.8, label="PPO")
        ax.plot(*q.start, marker="P", color="white", markersize=13, markeredgecolor="black", ls="")
        ax.plot(*q.goal_pos, marker="*", color="magenta", markersize=18, markeredgecolor="black", ls="")
        status = "success" if r["success"] else (f"died: {r['death']}" if r["death"] else "TIMEOUT")
        ax.set_title(f"world {r['world']}, lava seed {r['seed']}: {status}\n{r['steps']} steps (optimal {r['optimal']})")
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("PPO on unseen worlds (lava re-randomized each episode). Left: passing right beside lava. "
                 "Right: the two worlds where it times out most")
    plt.tight_layout()
    out = os.path.join(ROOT, "out", "ppo_lava_trajectories.png")
    fig.savefig(out, dpi=80)
    print(f"trajectories -> {out}  (successes: {len(near_miss)}, timeouts: {len(timeouts)})")


if __name__ == "__main__":
    main()
