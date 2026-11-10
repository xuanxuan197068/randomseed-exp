"""World：把各层拼起来，对外提供按区块 / 按区域的查询。

生成流水线（全部是世界坐标的纯函数）：
    TerrainGenerator  气候 → 高度 → 河流 → 群系 → 表面规则
    StructureManager  与本区域相交的结构 → 写入物体层
    FeaturePlacer     每个区块按自己的种子放置装饰物
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from .config import WorldConfig
from .features import FeaturePlacer
from .rng import normalize_seed
from .structures import StructureInstance, StructureManager
from .terrain import TerrainGenerator
from .tiles import IMPASSABLE_OBJS, WATER_TILES, Biome

SPAWN_BIOMES = (Biome.PLAINS, Biome.FOREST, Biome.TAIGA, Biome.SAVANNA, Biome.JUNGLE,
                Biome.SNOWY_PLAINS, Biome.DESERT)


@dataclass
class Area:
    x0: int
    z0: int
    height: np.ndarray
    terrain: np.ndarray
    biome: np.ndarray
    slope: np.ndarray
    objects: np.ndarray
    structures: list[StructureInstance] = field(default_factory=list)

    LAYERS = ("height", "terrain", "biome", "slope", "objects")

    @property
    def shape(self) -> tuple[int, int]:
        return self.height.shape

    def crop(self, x0: int, z0: int, w: int, h: int) -> "Area":
        ox, oz = x0 - self.x0, z0 - self.z0
        sl = (slice(oz, oz + h), slice(ox, ox + w))
        x1, z1 = x0 + w, z0 + h
        structs = [s for s in self.structures
                   if s.bbox[0] < x1 and x0 < s.bbox[2] and s.bbox[1] < z1 and z0 < s.bbox[3]]
        return Area(x0, z0, *(getattr(self, k)[sl].copy() for k in self.LAYERS), structures=structs)


class World:
    def __init__(self, seed: int | str, config: WorldConfig | None = None, chunk_cache_size: int = 4096):
        self.seed = normalize_seed(seed)
        self.config = config or WorldConfig()
        self.terrain = TerrainGenerator(self.seed, self.config)
        self.structures = StructureManager(self.seed, self.config, self.terrain)
        self.features = FeaturePlacer(self.seed, self.config)
        self._chunks: OrderedDict[tuple[int, int], Area] = OrderedDict()
        self._chunk_cache_size = chunk_cache_size
        self._spawn: tuple[int, int] | None = None

    # ---- 生成 -------------------------------------------------------------
    def generate_area(self, x0: int, z0: int, w: int, h: int) -> Area:
        """一次性向量化生成任意矩形区域（内部按区块对齐计算后裁剪）。"""
        cs = self.config.chunk_size
        cx0, cz0 = x0 // cs, z0 // cs
        cx1, cz1 = (x0 + w - 1) // cs, (z0 + h - 1) // cs
        bx0, bz0 = cx0 * cs, cz0 * cs
        bw, bh = (cx1 - cx0 + 1) * cs, (cz1 - cz0 + 1) * cs

        t = self.terrain.generate(bx0, bz0, bw, bh)
        objects = np.zeros((bh, bw), dtype=np.uint8)
        structs = self.structures.in_box(bx0, bz0, bx0 + bw, bz0 + bh)
        for s in structs:
            for x, z, obj in s.cells:
                i, j = z - bz0, x - bx0
                if 0 <= i < bh and 0 <= j < bw and t.terrain[i, j] not in WATER_TILES:
                    objects[i, j] = obj
        for cz in range(cz0, cz1 + 1):
            for cx in range(cx0, cx1 + 1):
                sl = (slice((cz - cz0) * cs, (cz - cz0 + 1) * cs), slice((cx - cx0) * cs, (cx - cx0 + 1) * cs))
                self.features.place(cx, cz, t.terrain[sl], t.biome[sl], t.slope[sl], objects[sl])

        full = Area(bx0, bz0, t.height, t.terrain, t.biome, t.slope, objects, structs)
        return full.crop(x0, z0, w, h)

    def get_chunk(self, cx: int, cz: int) -> Area:
        key = (cx, cz)
        chunk = self._chunks.get(key)
        if chunk is not None:
            self._chunks.move_to_end(key)
            return chunk
        cs = self.config.chunk_size
        chunk = self.generate_area(cx * cs, cz * cs, cs, cs)
        self._chunks[key] = chunk
        if len(self._chunks) > self._chunk_cache_size:
            self._chunks.popitem(last=False)
        return chunk

    def get_area(self, x0: int, z0: int, w: int, h: int) -> Area:
        """从缓存的区块拼出区域（适合 RL 每步的小窗口查询）。结果与 generate_area 按位相同。"""
        cs = self.config.chunk_size
        cx0, cz0 = x0 // cs, z0 // cs
        cx1, cz1 = (x0 + w - 1) // cs, (z0 + h - 1) // cs
        bw, bh = (cx1 - cx0 + 1) * cs, (cz1 - cz0 + 1) * cs
        layers = {k: None for k in Area.LAYERS}
        structs: dict[tuple, StructureInstance] = {}
        for cz in range(cz0, cz1 + 1):
            for cx in range(cx0, cx1 + 1):
                ch = self.get_chunk(cx, cz)
                sl = (slice((cz - cz0) * cs, (cz - cz0 + 1) * cs), slice((cx - cx0) * cs, (cx - cx0 + 1) * cs))
                for k in Area.LAYERS:
                    arr = getattr(ch, k)
                    if layers[k] is None:
                        layers[k] = np.empty((bh, bw), dtype=arr.dtype)
                    layers[k][sl] = arr
                for s in ch.structures:
                    structs[(s.kind, s.x, s.z)] = s
        full = Area(cx0 * cs, cz0 * cs, *(layers[k] for k in Area.LAYERS), structures=list(structs.values()))
        return full.crop(x0, z0, w, h)

    # ---- 查询 -------------------------------------------------------------
    def structures_in(self, x0: int, z0: int, x1: int, z1: int) -> list[StructureInstance]:
        return self.structures.in_box(x0, z0, x1, z1)

    def find_spawn(self) -> tuple[int, int]:
        """离原点最近的、适合出生的陆地格（同 MC：保证出生在陆地上）。"""
        if self._spawn is not None:
            return self._spawn
        for radius in (128, 512, 1024):
            step = 1 if radius <= 128 else 4
            area = self.generate_area(-radius, -radius, 2 * radius, 2 * radius)
            ok = (
                np.isin(area.biome, SPAWN_BIOMES)
                & ~np.isin(area.terrain, list(WATER_TILES))
                & ~np.isin(area.objects, list(IMPASSABLE_OBJS))
                & (area.slope <= 1)
            )
            ok = ok[::step, ::step]
            if ok.any():
                jz, ix = np.nonzero(ok)
                xs, zs = ix * step - radius, jz * step - radius
                k = np.lexsort((xs, zs, xs * xs + zs * zs))[0]
                self._spawn = (int(xs[k]), int(zs[k]))
                return self._spawn
        self._spawn = (0, 0)
        return self._spawn
