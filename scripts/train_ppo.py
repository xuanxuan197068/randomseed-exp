"""训练 PPO + CNN 智能体：多个世界、每回合随机岩浆、摔落伤害。

    python scripts/train_ppo.py --steps 5000000 --out out/ppo_lava.pt

训练世界是 --train-worlds 个固定种子；岩浆每回合重新随机布置，智能体背不了答案。
评估见 scripts/eval_ppo.py（新的岩浆布局 + 从没见过的世界）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from agents.ppo import PPOConfig, save, train_ppo  # noqa: E402

TRAIN_WORLD_BASE = 1000

# 训练奖励：成功 +1、金币 +0.1、死亡 −1，外加最短路塑形（整条最短路总和 +1）。
# 塑形用的最短路**无视岩浆**——它一直把智能体往直线方向拉，绕开岩浆只能从死亡惩罚和观测里学到。
# 评估只看成功率、死亡率、路径长度，与奖励设计无关。
ENV_KWARGS = dict(
    goal_distance=150, coin_spacing=24, reward_shaping="geodesic",
    fall_damage=True, lava_density=4.0, hazard_layout="episode", death_penalty=1.0,
)


def make_env_fn(env_kwargs: dict, world_seeds: list[int], num_envs: int):
    """第 i 个子环境只用 world_seeds[i::num_envs]：所有世界都会被训练到，
    但每个进程只需生成少数几个世界（首次生成一个世界要几秒，否则会拖慢整个训练）。"""
    def make(i: int):
        shard = world_seeds[i % len(world_seeds)::num_envs] or world_seeds

        def _thunk():
            sys.path.insert(0, ROOT)
            from envs import MinecraftLikeGridEnv
            return MinecraftLikeGridEnv(world_seeds=shard, **env_kwargs)
        return _thunk
    return make


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5_000_000)
    ap.add_argument("--train-worlds", type=int, default=48)
    ap.add_argument("--num-envs", type=int, default=24)
    ap.add_argument("--torch-threads", type=int, default=6)
    ap.add_argument("--lava-density", type=float, default=ENV_KWARGS["lava_density"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "ppo_lava.pt"))
    args = ap.parse_args()

    env_kwargs = {**ENV_KWARGS, "lava_density": args.lava_density}
    worlds = list(range(TRAIN_WORLD_BASE, TRAIN_WORLD_BASE + args.train_worlds))
    cfg = PPOConfig(total_steps=args.steps, num_envs=args.num_envs, seed=args.seed, torch_threads=args.torch_threads)
    print(f"env: {env_kwargs}\ntrain worlds: {worlds[0]}..{worlds[-1]}\nppo: {cfg}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    log_path = os.path.splitext(args.out)[0] + "_episodes.jsonl"

    def checkpoint(update, net, episodes):
        if update % 50 == 49:
            save(net, cfg, env_kwargs, args.out)

    t = time.time()
    net, episodes = train_ppo(make_env_fn(env_kwargs, worlds, cfg.num_envs), cfg, on_update=checkpoint)
    save(net, cfg, env_kwargs, args.out)
    with open(log_path, "w") as f:
        for e in episodes:
            f.write(json.dumps({**e, "death": e["death"] if e["death"] else None}) + "\n")
    last = episodes[-500:]
    print(f"done in {time.time() - t:.0f}s, {len(episodes)} episodes; last 500: "
          f"success {np.mean([e['success'] for e in last]):.2f}, "
          f"lava {np.mean([e['death'] == 'lava' for e in last]):.2f}, "
          f"fall {np.mean([e['death'] == 'fall' for e in last]):.2f}")
    print(f"model -> {args.out}\nepisode log -> {log_path}")


if __name__ == "__main__":
    main()
