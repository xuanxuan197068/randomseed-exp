"""地形：气候 → 样条高度 → 河流开凿 → 坡度 → 群系 → 表面规则。

``TerrainGenerator.generate(x0, z0, w, h)`` 是世界坐标的纯函数：
同一块区域无论整片算还是拆成区块算，结果按位相同。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .biomes import BIOME_SURFACE, select_biomes, temperature_level
from .climate import Climate, ClimateSampler
from .config import WorldConfig
from .noise import NormalNoise
from .rng import derive_seed
from .spline import build_detail_spline, build_offset_spline
from .tiles import Biome, Tile


@dataclass
class TerrainArea:
    x0: int
    z0: int
    height: np.ndarray  # int16，地表高度
    terrain: np.ndarray  # uint8，Tile
    biome: np.ndarray  # uint8，Biome
    slope: np.ndarray  # int16，与四邻最大高差
    temp_level: np.ndarray  # uint8，温度档 0..4


# 表面规则：有序 (名称, 条件, 结果) 列表，first-match（同 MC SurfaceRules.sequence）
SurfaceRule = tuple[str, Callable[[dict], np.ndarray], "int | Callable[[dict], np.ndarray]"]


def default_surface_rules(cfg: WorldConfig) -> list[SurfaceRule]:
    sea = cfg.sea_level
    return [
        ("frozen_water", lambda c: (c["height"] < sea) & (c["temp_level"] == 0), Tile.ICE),
        ("deep_water", lambda c: c["height"] < sea - 10, Tile.DEEP_WATER),
        ("water", lambda c: c["height"] < sea, Tile.WATER),
        ("shore", lambda c: (c["height"] <= sea + 2) & np.isin(c["biome"], (Biome.BEACH, Biome.RIVER)), Tile.SAND),
        ("snow_line", lambda c: c["height"] >= cfg.snow_line, Tile.SNOW),
        ("steep", lambda c: c["slope"] >= cfg.steep_slope, Tile.STONE),
        ("biome_top", lambda c: np.ones(c["height"].shape, bool), lambda c: BIOME_SURFACE[c["biome"]]),
    ]


def apply_surface_rules(rules: list[SurfaceRule], ctx: dict) -> np.ndarray:
    shape = ctx["height"].shape
    out = np.zeros(shape, dtype=np.uint8)
    unset = np.ones(shape, dtype=bool)
    for _, cond, result in rules:
        m = unset & cond(ctx)
        if not m.any():
            continue
        out[m] = result(ctx)[m] if callable(result) else result
        unset &= ~m
    return out


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class TerrainGenerator:
    def __init__(self, seed: int, config: WorldConfig):
        self.config = config
        self.climate = ClimateSampler(seed, config)
        self.detail = NormalNoise(derive_seed(seed, "detail"), config.detail)
        self.offset_spline = build_offset_spline()
        self.detail_spline = build_detail_spline()
        self.surface_rules = default_surface_rules(config)

    def river_strength(self, c: Climate) -> np.ndarray:
        """谷底 |weirdness| < river_width 处开凿河流；只在海岸线以内，靠近海岸逐渐淡出。"""
        r = 1.0 - np.abs(c.weirdness) / self.config.river_width
        coast_fade = _smoothstep((c.continentalness + 0.15) / 0.1)
        return np.clip(r, 0.0, 1.0) * coast_fade

    def heights(self, x: np.ndarray, z: np.ndarray, c: Climate) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        ctx = c.as_ctx()
        offset = self.offset_spline.evaluate(ctx)
        amp = self.detail_spline.evaluate(ctx)
        h = cfg.sea_level + offset + amp * self.detail.sample(x, z)

        river = self.river_strength(c)
        s = _smoothstep(river * 1.6)
        river_bed = cfg.sea_level - 3.0
        h = np.minimum(h, h + s * (river_bed - h))
        h = np.clip(np.floor(h + 0.5), cfg.min_height, cfg.max_height).astype(np.int16)
        return h, river

    def generate(self, x0: int, z0: int, w: int, h: int) -> TerrainArea:
        # 多算一圈边，用于计算坡度
        zs, xs = np.meshgrid(
            np.arange(z0 - 1, z0 + h + 1, dtype=np.float64),
            np.arange(x0 - 1, x0 + w + 1, dtype=np.float64),
            indexing="ij",
        )
        c = self.climate.sample(xs, zs)
        height_p, river_p = self.heights(xs, zs, c)

        hp = height_p.astype(np.int32)
        center = hp[1:-1, 1:-1]
        slope = np.zeros_like(center)
        for dz, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = hp[1 + dz : hp.shape[0] - 1 + dz, 1 + dx : hp.shape[1] - 1 + dx]
            slope = np.maximum(slope, np.abs(center - nb))

        inner = (slice(1, -1), slice(1, -1))
        climate = Climate(**{k: v[inner] for k, v in c.__dict__.items()})
        biome = select_biomes(climate, river_p[inner])
        height = height_p[inner]
        temp_level = temperature_level(climate.temperature)
        ctx = {"height": height, "slope": slope, "biome": biome, "temp_level": temp_level}
        terrain = apply_surface_rules(self.surface_rules, ctx)
        return TerrainArea(x0, z0, height, terrain, biome, slope.astype(np.int16), temp_level)
