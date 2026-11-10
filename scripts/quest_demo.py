"""主线演示：画出起点、目标要塞（末地传送门）、最短路径，并让"神谕"智能体沿最短路径走完一回合。

    python scripts/quest_demo.py --seed 12345
    python scripts/quest_demo.py --seed 7 --goal-distance 800
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from envs import MinecraftLikeGridEnv  # noqa: E402
from worldgen.quest import nominal_goal_distance  # noqa: E402
from worldgen.render import terrain_rgb  # noqa: E402


def run_oracle(env: MinecraftLikeGridEnv) -> tuple[bool, int, float]:
    path = env.quest.optimal_path()
    action_of = {m: a for a, m in enumerate(env.moves)}
    total, terminated, info = 0.0, False, {}
    for (x0, z0), (x1, z1) in zip(path, path[1:]):
        _, r, terminated, truncated, info = env.step(action_of[(x1 - x0, z1 - z0)])
        total += r
        if terminated or truncated:
            break
    return terminated, info.get("steps", 0), total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="12345")
    ap.add_argument("--goal-distance", type=float, default=None, help="终点名义距离（方块）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    env = MinecraftLikeGridEnv(world_seed=args.seed, goal_distance=args.goal_distance)
    _, info = env.reset(seed=0)
    q, world = env.quest, env.world
    print(f"world_seed={world.seed} start={q.start} goal={q.goal_pos} "
          f"直线距离={q.straight_distance:.0f} 最短步数={q.optimal_length} max_steps={env.max_steps}")

    x0, z0, x1, z1 = q.box
    area = world.generate_area(x0, z0, x1 - x0, z1 - z0)
    rgb = terrain_rgb(area)
    unreachable = ~np.isfinite(q.dist)
    rgb[unreachable] = (rgb[unreachable] * 0.45).astype(np.uint8)  # 暗色：从这里走不到终点

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(rgb, extent=(x0, x1, z1, z0), interpolation="nearest")
    path = np.array(q.optimal_path())
    ax.plot(path[:, 0] + 0.5, path[:, 1] + 0.5, color="red", lw=2, label=f"shortest path ({q.optimal_length} steps)")
    if env.coin_layout:
        ax.scatter([c[0] + 0.5 for c in env.coin_layout], [c[1] + 0.5 for c in env.coin_layout], s=30,
                   c="gold", edgecolors="black", zorder=3, label=f"coins ({len(env.coin_layout)})")
    ax.plot(*q.start, marker="P", color="white", markersize=16, markeredgecolor="black", ls="", label="start")
    ax.plot(*q.goal_pos, marker="*", color="magenta", markersize=22, markeredgecolor="black", ls="", label="end portal (goal)")
    others = [s for s in world.structures.strongholds() if s is not q.goal and x0 <= s.x < x1 and z0 <= s.z < z1]
    if others:
        ax.plot([s.x for s in others], [s.z for s in others], marker="*", color="gray", markersize=14,
                markeredgecolor="black", ls="", label="other strongholds")
    r = nominal_goal_distance(world.config)
    t = np.linspace(0, 2 * math.pi, 400)
    ax.plot(r * np.cos(t), r * np.sin(t), "--", color="magenta", lw=1, label=f"ring 1 nominal radius {r:.0f}")
    ax.set_xlim(x0, x1)
    ax.set_ylim(z1, z0)
    ax.legend(loc="lower left")
    ax.set_title(f"seed={world.seed}  straight={q.straight_distance:.0f}  shortest={q.optimal_length}  "
                 "(dimmed = cannot reach portal)")
    out = args.out or os.path.join("out", f"quest_{world.seed}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=80, bbox_inches="tight")
    print(f"map -> {out}")

    success, steps, total = run_oracle(env)
    print(f"oracle: success={success} steps={steps} return={total:.3f}")


if __name__ == "__main__":
    main()
