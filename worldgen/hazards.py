"""陷阱：每回合随机布置的岩浆池，配合摔落伤害，考验智能体看到危险后的应变能力。

岩浆不属于世界本身（地形由世界种子决定，永远不变），而是回合级的一层：
默认每回合按回合种子重新布置，智能体无法背答案，必须看着观测临场绕开。

布置规则（随机，但受约束）：
1. 相关：池心只选在路线附近的带状区域 d_start + d_goal ≤ L + band 内，且离起点、传送门有安全距离
2. 形状：从池心随机向四邻扩张出大小在 [min_size, max_size] 的不规则块
3. 合法：只铺在干燥陆地上（无物体或只有花），不铺在金币、起点上；同 MC，岩浆不挨着水（会变黑曜石）
4. 可解：放完后在"不踩岩浆"的图上算到传送门的最小摔落伤害，必须 < 最大生命值；
   否则按放置的逆序撤掉岩浆池，直到存在一条活得下来的路线
"""

from __future__ import annotations

import numpy as np

from .navigation import SAFE_FALL, distance_from_source, min_damage_to_targets
from .quest import EndPortalQuest
from .rng import Random
from .tiles import Obj

LAVA_SALT = 0x1A7A


def place_lava(quest: EndPortalQuest, seed: int, n_pools: int, coins: list[tuple[int, int]] = (),
               min_size: int = 3, max_size: int = 10, band: int = 16, keep_clear: int = 6,
               max_health: int | None = 20, safe_fall: int = SAFE_FALL) -> set[tuple[int, int]]:
    """返回岩浆格的世界坐标集合。max_health=None 表示不计摔落伤害，只要求不踩岩浆能走到。"""
    if n_pools <= 0:
        return set()
    x0, z0 = quest.box[0], quest.box[1]
    h, w = quest.stand.shape
    L = quest.optimal_length
    si, sj = quest.start[1] - z0, quest.start[0] - x0
    d_start = distance_from_source(quest.stand, quest.free, (si, sj), quest.max_climb, quest.moves)
    d_goal = quest.dist

    near_water = quest.water.copy()
    near_water[1:, :] |= quest.water[:-1, :]
    near_water[:-1, :] |= quest.water[1:, :]
    near_water[:, 1:] |= quest.water[:, :-1]
    near_water[:, :-1] |= quest.water[:, 1:]
    paintable = (np.isin(quest.objects, (Obj.NONE, Obj.FLOWER)) & ~near_water
                 & (d_start >= keep_clear) & (d_goal >= keep_clear))
    for x, z in coins:
        paintable[z - z0, x - x0] = False
    centers = np.flatnonzero((paintable & (d_start + d_goal <= L + band)).ravel())
    if len(centers) == 0:
        return set()

    rng = Random(seed ^ LAVA_SALT)
    lava = np.zeros((h, w), dtype=bool)
    pools: list[list[tuple[int, int]]] = []
    for _ in range(n_pools):
        ci, cj = divmod(int(centers[rng.next_int(len(centers))]), w)
        if lava[ci, cj]:
            continue
        size = min_size + rng.next_int(max_size - min_size + 1)
        cells, frontier = [(ci, cj)], [(ci, cj)]
        lava[ci, cj] = True
        for _ in range(20 * size):  # 有限次尝试：被水/树/金币包围的小块长不满也就算了
            if len(cells) >= size:
                break
            i, j = frontier[rng.next_int(len(frontier))]
            di, dj = ((-1, 0), (1, 0), (0, -1), (0, 1))[rng.next_int(4)]
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w and paintable[ni, nj] and not lava[ni, nj]:
                lava[ni, nj] = True
                cells.append((ni, nj))
                frontier.append((ni, nj))
        pools.append(cells)

    targets = quest.portal_mask()
    health_cap = np.inf if max_health is None else max_health
    big = 10**6 if max_health is None else safe_fall  # 不计伤害时让任何下落都不扣血
    while pools:
        dmg = min_damage_to_targets(quest.stand, quest.free, quest.water, lava, targets,
                                    quest.max_climb, quest.moves, safe_fall=big)
        if dmg[si, sj] < health_cap:
            break
        for i, j in pools.pop():
            lava[i, j] = False
    return {(x0 + int(j), z0 + int(i)) for i, j in zip(*np.nonzero(lava))}
