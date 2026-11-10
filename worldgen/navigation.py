"""可达性与最短路：把 RL 环境的移动规则写成图，用于保证"终点一定走得到"。

移动规则（与 envs/grid_env.py 共用这一份定义）：
- 站立高度 = max(地表高度, 海平面)：水面上游泳
- 从 u 走到相邻格 v：v 上没有不可穿过的物体，且 stand(v) - stand(u) ≤ max_climb（只能跳 1 格，往下任意）

由于上坡受限、下坡不限，图是**有向**的：能下去不代表能上来。
所以"到终点的距离"要在反向图上从终点做 BFS。
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .tiles import IMPASSABLE_OBJS

MOVES_4 = ((0, -1), (1, 0), (0, 1), (-1, 0))  # 北 东 南 西（z 轴向南为正）
MOVES_8 = MOVES_4 + ((1, -1), (1, 1), (-1, 1), (-1, -1))


def stand_height(height: np.ndarray, sea_level: int) -> np.ndarray:
    return np.maximum(height.astype(np.int32), sea_level)


def free_mask(objects: np.ndarray) -> np.ndarray:
    return ~np.isin(objects, list(IMPASSABLE_OBJS))


def can_step(src_stand: int, dst_stand: int, dst_obj: int, max_climb: int) -> bool:
    return int(dst_obj) not in IMPASSABLE_OBJS and dst_stand - src_stand <= max_climb


def step_graph(stand: np.ndarray, free: np.ndarray, max_climb: int, moves=MOVES_4) -> csr_matrix:
    """有向图：边 u→v 表示可以一步从 u 走到 v。"""
    h, w = stand.shape
    idx = np.arange(h * w).reshape(h, w)
    rows, cols = [], []
    for dx, dz in moves:
        zs = slice(max(0, -dz), h - max(0, dz))
        xs = slice(max(0, -dx), w - max(0, dx))
        zd = slice(max(0, dz), h - max(0, -dz))
        xd = slice(max(0, dx), w - max(0, -dx))
        ok = free[zd, xd] & (stand[zd, xd] - stand[zs, xs] <= max_climb)
        rows.append(idx[zs, xs][ok])
        cols.append(idx[zd, xd][ok])
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    return csr_matrix((np.ones(len(r), dtype=np.float64), (r, c)), shape=(h * w, h * w))


def distance_to_targets(stand: np.ndarray, free: np.ndarray, targets: np.ndarray,
                        max_climb: int, moves=MOVES_4) -> np.ndarray:
    """每一格走到任一目标格的最少步数（走不到为 inf）。在反向图上做多源 BFS。"""
    if not targets.any():
        return np.full(stand.shape, np.inf)
    g = step_graph(stand, free, max_climb, moves)
    d = dijkstra(g.T.tocsr(), directed=True, indices=np.flatnonzero(targets), unweighted=True, min_only=True)
    return d.reshape(stand.shape)


def distance_from_source(stand: np.ndarray, free: np.ndarray, source: tuple[int, int],
                         max_climb: int, moves=MOVES_4) -> np.ndarray:
    """从 source（数组下标 (i, j)）出发走到每一格的最少步数（正向图上的 BFS）。"""
    g = step_graph(stand, free, max_climb, moves)
    d = dijkstra(g, directed=True, indices=source[0] * stand.shape[1] + source[1], unweighted=True)
    return d.reshape(stand.shape)


SAFE_FALL = 3  # MC：下落超过 3 格开始扣血，伤害 = 下落高度 − 3；落进水里不扣血


def fall_damage(drop: int, into_water: bool, safe_fall: int = SAFE_FALL) -> int:
    return 0 if into_water else max(0, int(drop) - safe_fall)


def min_damage_to_targets(stand: np.ndarray, free: np.ndarray, water: np.ndarray, blocked: np.ndarray,
                          targets: np.ndarray, max_climb: int, moves=MOVES_4,
                          safe_fall: int = SAFE_FALL) -> np.ndarray:
    """每一格走到目标、途中不踩 blocked（岩浆）时，最少要承受多少摔落伤害（走不到为 inf）。

    边权 = 摔落伤害 + 1e-6（同等伤害下偏好更短的路），在反向图上从目标做 Dijkstra。
    """
    if not targets.any():
        return np.full(stand.shape, np.inf)
    h, w = stand.shape
    idx = np.arange(h * w).reshape(h, w)
    ok_cell = free & ~blocked
    rows, cols, wts = [], [], []
    for dx, dz in moves:
        zs = slice(max(0, -dz), h - max(0, dz))
        xs = slice(max(0, -dx), w - max(0, dx))
        zd = slice(max(0, dz), h - max(0, -dz))
        xd = slice(max(0, dx), w - max(0, -dx))
        diff = stand[zd, xd] - stand[zs, xs]
        ok = ok_cell[zd, xd] & (diff <= max_climb)
        dmg = np.where(water[zd, xd], 0, np.maximum(0, -diff - safe_fall))
        rows.append(idx[zs, xs][ok])
        cols.append(idx[zd, xd][ok])
        wts.append(dmg[ok] + 1e-6)
    g = csr_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(h * w, h * w))
    d = dijkstra(g.T.tocsr(), directed=True, indices=np.flatnonzero(targets & ~blocked), min_only=True)
    return np.floor(d.reshape(stand.shape) + 1e-3)
