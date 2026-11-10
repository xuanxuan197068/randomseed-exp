"""最简单的强化学习智能体：表格 Q-learning。

- 状态：(x, z, 已捡金币的位掩码)。只记位置意味着它学的是"这张地图"，不会泛化到新世界——
  这正适合用来检验奖励设计本身（稀疏 vs 金币）是否让任务可学。
  注意必须用"捡了哪些"而不是"捡了几枚"：只记数量时，某回合跳过一枚金币会让同一个
  (位置, 数量) 状态学到指向那枚金币的价值，之后它已被捡走，智能体就会追一个不存在的奖励而原地打转。
- 探索：ε-greedy，ε 线性衰减；Q 值相同时随机打破平局。
- 更新：回合结束后把整条轨迹**倒序**做一遍 Q-learning 更新（episodic backward update）。
  在确定性环境里，一次发现的金币/终点奖励能沿整条轨迹一次性传回起点，比逐步更新快得多。
- 超时截断（truncated）不是终止，仍然自举。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


def state_of(info: dict) -> tuple[int, int, int]:
    x, z = info["pos"]
    return x, z, info.get("coins_mask", 0)


@dataclass
class TabularQAgent:
    n_actions: int
    alpha: float = 0.5
    gamma: float = 0.99
    seed: int = 0
    q: dict = field(default_factory=dict)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.q = defaultdict(lambda: np.zeros(self.n_actions))

    def act(self, state, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        values = self.q[state]
        best = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best))

    def update_episode(self, transitions: list[tuple]) -> None:
        for s, a, r, s2, terminated in reversed(transitions):
            target = r if terminated else r + self.gamma * self.q[s2].max()
            self.q[s][a] += self.alpha * (target - self.q[s][a])


def run_episode(env, agent: TabularQAgent, epsilon: float, learn: bool = True, seed: int | None = None) -> dict:
    obs, info = env.reset(seed=seed)
    s = state_of(info)
    transitions, total, path = [], 0.0, [info["pos"]]
    while True:
        a = agent.act(s, epsilon)
        obs, r, terminated, truncated, info = env.step(a)
        s2 = state_of(info)
        transitions.append((s, a, r, s2, terminated))
        total += r
        path.append(info["pos"])
        s = s2
        if terminated or truncated:
            break
    if learn:
        agent.update_episode(transitions)
    return {
        "success": bool(info["success"]),
        "steps": info["steps"],
        "return": total,
        "coins": info.get("coins_collected", 0),
        "coins_total": info.get("coins_total", 0),
        "path": path,
    }
