"""主线任务：从出生点出发，找到末地传送门（要塞的传送门房间）。

对应 MC 的主线：要塞按同心环分布在出生点周围固定距离上，玩家用末影之眼指向**最近的**要塞。
这里加一条 MC 里不需要（因为能挖方块）、但 2.5D 网格必须有的规则：
**终点必须按环境的移动规则可达**。规划时依次尝试离起点最近的要塞，
第一个可达的作为目标；同时得到整片区域到传送门的最短步数场，用于奖励塑形与评估（最优路径长度）。

规划只依赖 (世界种子, WorldConfig, 起点, 移动规则)，因此同样可复现。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from scipy.sparse.csgraph import dijkstra

from .navigation import MOVES_4, distance_from_source, distance_to_targets, free_mask, stand_height, step_graph
from .rng import Random
from .structures import StructureInstance
from .tiles import Obj

COIN_SALT = 0xC01


@dataclass
class EndPortalQuest:
    start: tuple[int, int]
    goal: StructureInstance  # 目标要塞（传送门房间中心 = goal.x, goal.z）
    box: tuple[int, int, int, int]  # 距离场覆盖的方块范围 [x0,x1)×[z0,z1)
    dist: np.ndarray  # box 内每格到任一传送门的最少步数（inf = 走不到）
    stand: np.ndarray
    free: np.ndarray
    objects: np.ndarray  # box 内的物体层（金币不放在物体上）
    water: np.ndarray  # box 内的水面（落水不摔伤；岩浆不挨着水）
    max_climb: int
    moves: tuple

    def portal_mask(self) -> np.ndarray:
        return self.objects == Obj.PORTAL

    @property
    def goal_pos(self) -> tuple[int, int]:
        return self.goal.x, self.goal.z

    @property
    def optimal_length(self) -> int:
        return int(self.distance(*self.start))

    @property
    def straight_distance(self) -> float:
        return math.hypot(self.goal.x - self.start[0], self.goal.z - self.start[1])

    def _index(self, x: int, z: int) -> tuple[int, int] | None:
        x0, z0, x1, z1 = self.box
        if x0 <= x < x1 and z0 <= z < z1:
            return z - z0, x - x0
        return None

    def distance(self, x: int, z: int) -> float:
        """到传送门的最短步数；在 box 外或走不到返回 inf。"""
        ij = self._index(x, z)
        return float(self.dist[ij]) if ij is not None else math.inf

    def optimal_path(self) -> list[tuple[int, int]]:
        """沿距离场下降得到一条最短路径（起点 → 传送门）。"""
        x, z = self.start
        path = [(x, z)]
        d = self.distance(x, z)
        if not math.isfinite(d):
            return path
        while d > 0:
            i, j = self._index(x, z)
            for dx, dz in self.moves:
                nij = self._index(x + dx, z + dz)
                if nij is None or not self.free[nij] or self.stand[nij] - self.stand[i, j] > self.max_climb:
                    continue
                if self.dist[nij] == d - 1:
                    x, z, d = x + dx, z + dz, d - 1
                    path.append((x, z))
                    break
            else:  # 理论上不会发生
                raise RuntimeError("distance field is inconsistent")
        return path


def nominal_goal_distance(config) -> float:
    """第一环要塞到原点的名义距离（方块）：4 × distance 区块，实际在 ±31% 内随机。"""
    return 4 * config.stronghold.distance * config.chunk_size


def stronghold_distance_for(goal_distance: float, chunk_size: int = 16) -> int:
    """反过来：想让终点大约在 goal_distance 方块外，StrongholdConfig.distance 应取多少区块。"""
    return max(1, round(goal_distance / (4 * chunk_size)))


def plan_end_portal_quest(world, start: tuple[int, int], max_climb: int = 1, moves=MOVES_4,
                          margin: int = 96, bounds: tuple[int, int, int, int] | None = None,
                          max_candidates: int = 3, min_distance: float | None = None) -> EndPortalQuest | None:
    """选离起点最近、且可达的要塞作为终点。最近的 max_candidates 个都走不到时返回 None。

    min_distance：终点到起点的最小直线距离。要塞环以世界原点为圆心，而出生点可能因为原点是海洋
    被挪开，此时最近的要塞可能贴着起点；默认取第一环名义半径的一半，过近的要塞会被跳过。
    """
    sx, sz = start
    if min_distance is None:
        min_distance = 0.5 * nominal_goal_distance(world.config)
    candidates = [s for s in world.structures.strongholds()
                  if math.hypot(s.x - sx, s.z - sz) >= min_distance]
    if bounds is not None:
        bx0, bz0, bx1, bz1 = bounds
        candidates = [s for s in candidates if bx0 <= s.x < bx1 and bz0 <= s.z < bz1]
    candidates = sorted(candidates, key=lambda s: ((s.x - sx) ** 2 + (s.z - sz) ** 2, s.x, s.z))
    sea = world.config.sea_level

    for goal in candidates[:max_candidates]:
        x0 = min(sx, goal.x) - margin
        z0 = min(sz, goal.z) - margin
        x1 = max(sx, goal.x) + margin + 1
        z1 = max(sz, goal.z) + margin + 1
        if bounds is not None:
            x0, z0 = max(x0, bounds[0]), max(z0, bounds[1])
            x1, z1 = min(x1, bounds[2]), min(z1, bounds[3])
        area = world.generate_area(x0, z0, x1 - x0, z1 - z0)
        stand = stand_height(area.height, sea)
        free = free_mask(area.objects)
        gx0, gz0, gx1, gz1 = goal.bbox
        targets = np.zeros(area.shape, dtype=bool)
        targets[max(gz0 - z0, 0):max(gz1 - z0, 0), max(gx0 - x0, 0):max(gx1 - x0, 0)] = True
        targets &= area.objects == Obj.PORTAL
        dist = distance_to_targets(stand, free, targets, max_climb, moves)
        if math.isfinite(dist[sz - z0, sx - x0]):
            return EndPortalQuest(start, goal, (x0, z0, x1, z1), dist, stand, free, area.objects,
                                  area.height < sea, max_climb, tuple(moves))
    return None


def place_coins(quest: EndPortalQuest, seed: int, spacing: int = 24, max_detour: int = 4,
                uphill_spacing: int | None = None, uphill_threshold: float = 0.5) -> list[tuple[int, int]]:
    """沿途放置金币（中间奖励）：位置随机，但受三条规则约束。

    1. 走廊：d_start(c) + d_goal(c) ≤ L + max_detour（L = 最短步数）
       → 去捡这枚金币最多多走 max_detour 步；它从起点可达，捡完也一定还能走到传送门
    2. 进度：金币按路程分层，第 k 层满足 |d_start(c) − p_k| ≤ 1，相邻层相隔 spacing 步；
       若接下来 spacing 步里上坡步占比 ≥ uphill_threshold，则改用 uphill_spacing（默认 spacing/2）
       → 上坡时随机探索会被"往下走容易、往上走受限"的规则推回坡底，金币需要更密
    3. 连贯：第 k+1 枚离第 k 枚（第 1 枚离起点）不超过 该层间距 + max_detour 步
       → 按顺序捡完所有金币再进传送门，总路程 ≤ L + (n+1)·max_detour + n，不会来回折返
    另外不放在起点、传送门、障碍物或结构物体上。
    随机：每层在所有合格格子里按种子随机选一个，不偏向某一条特定的最短路。
    """
    L = quest.optimal_length
    x0, z0 = quest.box[0], quest.box[1]
    w = quest.stand.shape[1]
    nav = (quest.stand, quest.free)
    start_ij = (quest.start[1] - z0, quest.start[0] - x0)
    d_start = distance_from_source(*nav, start_ij, quest.max_climb, quest.moves)
    valid = quest.free & (quest.objects == Obj.NONE) & np.isfinite(quest.dist) & (quest.dist > 0)
    valid &= d_start + quest.dist <= L + max_detour

    # 用一条最短路估计每一段的上坡程度，决定该段的金币间距
    path = quest.optimal_path()
    rises = np.array([quest.stand[b[1] - z0, b[0] - x0] > quest.stand[a[1] - z0, a[0] - x0]
                      for a, b in zip(path, path[1:])], dtype=bool)
    uphill = uphill_spacing or max(1, spacing // 2)
    levels: list[tuple[int, int]] = []  # (进度, 该层间距)
    progress = 0
    while True:
        ahead = rises[progress:progress + spacing]
        gap = uphill if len(ahead) and ahead.mean() >= uphill_threshold else spacing
        progress += gap
        if progress > L - gap // 2:
            break
        levels.append((progress, gap))

    rng = Random.at(seed, quest.start[0], quest.start[1], COIN_SALT)
    coins: list[tuple[int, int]] = []
    d_prev = d_start
    for progress, gap in levels:
        layer = valid & (np.abs(d_start - progress) <= 1) & (d_prev <= gap + max_detour)
        cand = np.flatnonzero(layer.ravel())
        if len(cand) == 0:
            continue
        i, j = divmod(int(cand[rng.next_int(len(cand))]), w)
        coins.append((x0 + j, z0 + i))
        d_prev = distance_from_source(*nav, (i, j), quest.max_climb, quest.moves)
    return coins


def route_through(quest: EndPortalQuest, waypoints: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """起点 → 依次经过 waypoints → 传送门 的路线（每段都是最短路）。用于演示与测试。"""
    x0, z0 = quest.box[0], quest.box[1]
    w = quest.stand.shape[1]
    route = [quest.start]
    for wx, wz in waypoints:
        src = (route[-1][1] - z0, route[-1][0] - x0)
        _, pred = dijkstra(step_graph(quest.stand, quest.free, quest.max_climb, quest.moves), directed=True,
                           indices=src[0] * w + src[1], unweighted=True, return_predecessors=True)
        node, seg = (wz - z0) * w + (wx - x0), []
        while node != src[0] * w + src[1]:
            if node < 0:
                raise ValueError(f"waypoint {(wx, wz)} unreachable")
            i, j = divmod(int(node), w)
            seg.append((x0 + j, z0 + i))
            node = pred[node]
        route.extend(reversed(seg))
    tail = EndPortalQuest(route[-1], quest.goal, quest.box, quest.dist, quest.stand, quest.free,
                          quest.objects, quest.water, quest.max_climb, quest.moves)
    return route + tail.optimal_path()[1:]
