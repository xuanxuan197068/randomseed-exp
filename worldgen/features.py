"""装饰物：每个区块一个独立随机源 Random.at(seed, cx, cz, salt)。

每个 FeatureConfig 相当于 MC 的一条 PlacementModifier 链：
    count(tries) → in_square(区块内随机坐标) → biome_filter(按群系概率) →
    surface(地表方块) → slope → spacing / isolated
每次尝试固定消耗 3 个随机数，保证随机流与地形结果解耦（地形改了不会"串"到别的尝试）。
"""

from __future__ import annotations

import numpy as np

from .config import FeatureConfig, WorldConfig
from .rng import Random
from .tiles import Biome, Obj, Tile


class FeaturePlacer:
    def __init__(self, seed: int, config: WorldConfig):
        self.seed = seed
        self.config = config
        self.compiled = [self._compile(f) for f in config.features]

    @staticmethod
    def _compile(f: FeatureConfig):
        default = dict(f.density).get("*", 0.0)
        density = np.full(len(Biome), default, dtype=np.float64)
        for name, p in f.density:
            if name != "*":
                density[Biome[name.upper()]] = p
        surfaces = np.zeros(len(Tile), dtype=bool)
        for t in f.surfaces:
            surfaces[Tile[t]] = True
        return f, Obj[f.obj], density, surfaces

    def place(self, cx: int, cz: int, terrain: np.ndarray, biome: np.ndarray,
              slope: np.ndarray, objects: np.ndarray) -> None:
        """在一个区块（16×16 视图）内原地写入 objects。"""
        cs = self.config.chunk_size
        for f, obj, density, surfaces in self.compiled:
            rng = Random.at(self.seed, cx, cz, f.salt)
            placed: list[tuple[int, int]] = []
            lo, hi = (1, cs - 1) if f.isolated else (0, cs)
            for _ in range(f.tries):
                lx = lo + rng.next_int(hi - lo)
                lz = lo + rng.next_int(hi - lo)
                roll = rng.next_float()
                if roll >= density[biome[lz, lx]]:
                    continue
                if not surfaces[terrain[lz, lx]] or slope[lz, lx] > f.max_slope:
                    continue
                if objects[lz, lx] != Obj.NONE:
                    continue
                if any(max(abs(px - lx), abs(pz - lz)) < f.min_spacing for px, pz in placed):
                    continue
                if f.isolated and any(
                    objects[lz + dz, lx + dx] != Obj.NONE for dz, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                ):
                    continue
                objects[lz, lx] = obj
                placed.append((lx, lz))
