"""结构放置：随机性被约束在"网格 + 校验"的规则里。

RandomSpreadPlacement（村庄、沙漠神殿）：
    区域 (rx, rz) 覆盖 spacing×spacing 个区块，候选区块 = 区域原点 + [0, spacing-separation) 的随机偏移，
    随机源 = Random.at(world_seed, rx, rz, salt)。
    → 同类结构切比雪夫距离 ≥ separation+1 区块（规则），而具体位置随种子变化（随机）。
    候选点还需满足：中心群系合法、占地全是陆地、高差不超过 max_relief、远离被排斥的结构。
    任一条件不满足，这个区域就没有该结构（与 MC 相同）。

ConcentricRings（要塞 / 末地传送门，主线终点）：
    全局只按世界种子算一次：第 k 环的数量与半径固定，角度随机，再在附近搜索平坦陆地。
    → 数量固定、到原点（出生点附近）距离有界（规则），方位随机（随机）。

结构的内部布局只依赖它自己的种子，与周围地形/加载顺序无关。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .config import StructureConfig, WorldConfig
from .rng import Random, derive_seed
from .terrain import TerrainGenerator
from .tiles import Biome, Obj

LAND_BIOMES = frozenset(b for b in Biome if b not in (Biome.DEEP_OCEAN, Biome.OCEAN, Biome.RIVER))


@dataclass(frozen=True)
class StructureInstance:
    kind: str
    x: int  # 中心方块坐标
    z: int
    footprint: int
    cells: tuple[tuple[int, int, int], ...] = field(repr=False)  # (世界 x, 世界 z, Obj)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        half = self.footprint // 2
        return self.x - half, self.z - half, self.x - half + self.footprint, self.z - half + self.footprint


def _layout_village(rng: Random, cx: int, cz: int, size: int) -> list[tuple[int, int, int]]:
    """简化的"拼图"布局：十字主路 + 沿路随机分布的 3×3 房屋。"""
    half = size // 2
    occupied: dict[tuple[int, int], int] = {}
    for d in range(-half + 1, half - 1):
        occupied[(cx + d, cz)] = Obj.PATH
        occupied[(cx, cz + d)] = Obj.PATH
    houses = 0
    for _ in range(20):
        along = rng.next_int(size - 6) - (half - 3)
        side = 1 if rng.next_int(2) else -1
        horizontal = rng.next_int(2) == 0
        gap = 2 + rng.next_int(2)
        hx, hz = (cx + along, cz + side * gap) if horizontal else (cx + side * gap, cz + along)
        cells = [(hx + i, hz + j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
        ring = [(hx + i, hz + j) for i in range(-2, 3) for j in range(-2, 3)]
        if any(abs(x - cx) >= half or abs(z - cz) >= half for x, z in ring):
            continue
        if any(occupied.get(p) in (Obj.HOUSE,) for p in ring) or any(p in occupied for p in cells):
            continue
        for p in cells:
            occupied[p] = Obj.HOUSE
        houses += 1
        if houses >= 8:
            break
    return [(x, z, o) for (x, z), o in sorted(occupied.items())]


def _layout_square(cx: int, cz: int, size: int, obj: int) -> list[tuple[int, int, int]]:
    half = size // 2
    return [(cx + i, cz + j, obj) for i in range(-half, size - half) for j in range(-half, size - half)]


def _layout_portal_room(cx: int, cz: int, size: int) -> list[tuple[int, int, int]]:
    """要塞的传送门房间：石砖地面中央是 3×3 末地传送门，四周 12 个框架（5×5 外圈去掉四角，同 MC）。"""
    half = size // 2
    cells = []
    for i in range(-half, size - half):
        for j in range(-half, size - half):
            a, b = abs(i), abs(j)
            if a <= 1 and b <= 1:
                obj = Obj.PORTAL
            elif a <= 2 and b <= 2 and not (a == 2 and b == 2):
                obj = Obj.PORTAL_FRAME
            else:
                obj = Obj.STRONGHOLD
            cells.append((cx + i, cz + j, obj))
    return cells


class StructureManager:
    def __init__(self, seed: int, config: WorldConfig, terrain: TerrainGenerator):
        self.seed = seed
        self.config = config
        self.terrain = terrain
        self.by_name = {s.name: s for s in config.structures}
        self._strongholds: list[StructureInstance] | None = None
        self.candidate = lru_cache(maxsize=4096)(self._candidate)
        self.instance = lru_cache(maxsize=4096)(self._instance)

    # ---- RandomSpread ---------------------------------------------------
    def region_of_chunk(self, s: StructureConfig, cx: int, cz: int) -> tuple[int, int]:
        return cx // s.spacing, cz // s.spacing

    def _candidate(self, name: str, rx: int, rz: int) -> tuple[int, int]:
        """区域 (rx, rz) 内的候选区块（只依赖种子与区域坐标）。"""
        s = self.by_name[name]
        rng = Random.at(self.seed, rx, rz, s.salt)
        span = s.spacing - s.separation
        return rx * s.spacing + rng.next_int(span), rz * s.spacing + rng.next_int(span)

    def _instance(self, name: str, rx: int, rz: int) -> StructureInstance | None:
        s = self.by_name[name]
        cx, cz = self.candidate(name, rx, rz)
        cs = self.config.chunk_size
        x, z = cx * cs + cs // 2, cz * cs + cs // 2
        if not self._site_ok(s, x, z):
            return None
        for other, min_chunks in s.exclusion:
            if self._near(other, cx, cz, min_chunks):
                return None
        rng = Random.at(self.seed, cx, cz, s.salt, 1)
        if name == "village":
            cells = _layout_village(rng, x, z, s.footprint)
        else:
            cells = _layout_square(x, z, s.footprint, Obj.TEMPLE)
        return StructureInstance(name, x, z, s.footprint, tuple(cells))

    def _site_ok(self, s: StructureConfig, x: int, z: int) -> bool:
        half = s.footprint // 2
        area = self.terrain.generate(x - half, z - half, s.footprint, s.footprint)
        center_biome = Biome(int(area.biome[half, half])).name.lower()
        if center_biome not in s.allowed_biomes:
            return False
        if (area.height < self.config.sea_level).any():
            return False
        return int(area.height.max()) - int(area.height.min()) <= s.max_relief

    def _near(self, name: str, cx: int, cz: int, min_chunks: int) -> bool:
        s = self.by_name[name]
        rx0, rz0 = (cx - min_chunks) // s.spacing, (cz - min_chunks) // s.spacing
        rx1, rz1 = (cx + min_chunks) // s.spacing, (cz + min_chunks) // s.spacing
        for rx in range(rx0, rx1 + 1):
            for rz in range(rz0, rz1 + 1):
                inst = self.instance(name, rx, rz)
                if inst is None:
                    continue
                ocx, ocz = self.candidate(name, rx, rz)
                if max(abs(ocx - cx), abs(ocz - cz)) < min_chunks:
                    return True
        return False

    # ---- ConcentricRings -------------------------------------------------
    def strongholds(self) -> list[StructureInstance]:
        if self._strongholds is not None:
            return self._strongholds
        cfg = self.config.stronghold
        cs = self.config.chunk_size
        rng = Random(derive_seed(self.seed, "stronghold") ^ cfg.salt)
        angle = rng.next_float() * math.pi * 2.0
        ring, ring_count, spread = 0, 0, cfg.spread
        result: list[StructureInstance] = []
        for i in range(cfg.count):
            dist = 4 * cfg.distance + cfg.distance * ring * 6 + (rng.next_float() - 0.5) * cfg.distance * 2.5
            cx = int(math.floor(math.cos(angle) * dist + 0.5))
            cz = int(math.floor(math.sin(angle) * dist + 0.5))
            x, z = self._find_site(cx * cs + cs // 2, cz * cs + cs // 2, 112, cfg.footprint, cfg.max_relief)
            cells = _layout_portal_room(x, z, cfg.footprint)
            result.append(StructureInstance("stronghold", x, z, cfg.footprint, tuple(cells)))
            angle += 2.0 * math.pi / spread
            ring_count += 1
            if ring_count == spread:
                ring += 1
                ring_count = 0
                spread += 2 * spread // (ring + 1)
                spread = min(spread, cfg.count - i - 1) or 1
                angle += rng.next_float() * math.pi * 2.0
        self._strongholds = result
        return result

    def _find_site(self, x: int, z: int, radius: int, footprint: int, max_relief: int) -> tuple[int, int]:
        """在 (x,z) 周围 radius 内找最近的合格场地：中心是陆地群系、占地全在海平面以上、高差 ≤ max_relief。
        作用同 MC 的 findBiomeHorizontal（要塞会挪到附近合适的位置）。找不到就保持原位。"""
        step = 8
        half = footprint // 2
        area = self.terrain.generate(x - radius - half, z - radius - half,
                                     2 * radius + footprint, 2 * radius + footprint)
        sea = self.config.sea_level
        offsets = [(ox, oz) for oz in range(-radius, radius + 1, step) for ox in range(-radius, radius + 1, step)]
        offsets.sort(key=lambda o: (o[0] * o[0] + o[1] * o[1], o[1], o[0]))
        for ox, oz in offsets:
            j, i = ox + radius, oz + radius  # 占地左上角在 area 中的下标
            h = area.height[i : i + footprint, j : j + footprint]
            if Biome(int(area.biome[i + half, j + half])) not in LAND_BIOMES:
                continue
            if (h < sea).any() or int(h.max()) - int(h.min()) > max_relief:
                continue
            return x + ox, z + oz
        return x, z

    # ---- 查询 -------------------------------------------------------------
    def in_box(self, x0: int, z0: int, x1: int, z1: int) -> list[StructureInstance]:
        """与方块范围 [x0,x1)×[z0,z1) 相交的全部结构。"""
        cs = self.config.chunk_size
        found: list[StructureInstance] = []
        for s in self.config.structures:
            pad = s.footprint
            rx0 = ((x0 - pad) // cs) // s.spacing
            rx1 = ((x1 + pad) // cs) // s.spacing
            rz0 = ((z0 - pad) // cs) // s.spacing
            rz1 = ((z1 + pad) // cs) // s.spacing
            for rx in range(rx0, rx1 + 1):
                for rz in range(rz0, rz1 + 1):
                    inst = self.instance(s.name, rx, rz)
                    if inst is not None and _overlaps(inst.bbox, (x0, z0, x1, z1)):
                        found.append(inst)
        found.extend(i for i in self.strongholds() if _overlaps(i.bbox, (x0, z0, x1, z1)))
        return found


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
