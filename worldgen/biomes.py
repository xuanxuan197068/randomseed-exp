"""群系选择：在气候参数空间里查表（仿 MC OverworldBiomeBuilder）。

规则层次（先匹配者优先）：
1. 大陆性 continentalness 决定 深海 / 海洋 / 海岸 / 内陆
2. 内陆的河谷（河流强度 > 0.5）→ 河流
3. 低侵蚀 + 高峰谷值 → 山峰（按温度分雪峰 / 石峰）
4. 高侵蚀 + 湿润 + 温和 → 沼泽
5. 其余按 温度档 × 湿度档 的 5×5 表查"中间群系"

温度/湿度都是连续噪声，量化成 5 档后相邻格最多相差 1 档。
所有"冰冻"群系都要求温度档 ≤ 1，所有"炎热"群系都要求温度档 ≥ 3，两者至少隔 2 档，
所以"沙漠/热带草原紧挨雪原/雪峰"这种不合理的拼接在结构上不可能出现。
（温和地区的高山仍会被表面规则的雪线盖上雪，只是群系记为石峰。）
"""

from __future__ import annotations

import numpy as np

from .climate import Climate
from .tiles import Biome, Tile

# MC 原版的温度/湿度分档阈值
TEMPERATURE_THRESHOLDS = (-0.45, -0.15, 0.2, 0.55)
HUMIDITY_THRESHOLDS = (-0.35, -0.1, 0.1, 0.3)

B = Biome
MIDDLE_BIOMES = np.array(
    [
        # 湿度:  极干            干              中             湿            极湿
        [B.SNOWY_PLAINS, B.SNOWY_PLAINS, B.SNOWY_PLAINS, B.TAIGA, B.TAIGA],  # 冰冻
        [B.PLAINS, B.PLAINS, B.FOREST, B.TAIGA, B.TAIGA],  # 寒冷
        [B.PLAINS, B.PLAINS, B.FOREST, B.FOREST, B.FOREST],  # 温和
        [B.SAVANNA, B.SAVANNA, B.FOREST, B.JUNGLE, B.JUNGLE],  # 温暖
        [B.DESERT, B.DESERT, B.DESERT, B.DESERT, B.DESERT],  # 炎热
    ],
    dtype=np.uint8,
)

# 每个群系的默认地表方块（表面规则的最后一条）
BIOME_SURFACE = np.zeros(len(Biome), dtype=np.uint8)
for _b, _t in {
    B.DEEP_OCEAN: Tile.GRAVEL, B.OCEAN: Tile.SAND, B.BEACH: Tile.SAND, B.RIVER: Tile.SAND,
    B.PLAINS: Tile.GRASS, B.FOREST: Tile.GRASS, B.DESERT: Tile.SAND, B.SAVANNA: Tile.GRASS,
    B.TAIGA: Tile.GRASS, B.SNOWY_PLAINS: Tile.SNOW, B.JUNGLE: Tile.GRASS, B.SWAMP: Tile.MUD,
    B.STONY_PEAKS: Tile.STONE, B.SNOWY_PEAKS: Tile.SNOW,
}.items():
    BIOME_SURFACE[_b] = _t

HOT_BIOMES = frozenset({B.DESERT, B.SAVANNA, B.JUNGLE})
FROZEN_BIOMES = frozenset({B.SNOWY_PLAINS, B.SNOWY_PEAKS})


def temperature_level(t: np.ndarray) -> np.ndarray:
    return np.digitize(t, TEMPERATURE_THRESHOLDS).astype(np.uint8)


def humidity_level(h: np.ndarray) -> np.ndarray:
    return np.digitize(h, HUMIDITY_THRESHOLDS).astype(np.uint8)


def select_biomes(c: Climate, river: np.ndarray) -> np.ndarray:
    t = temperature_level(c.temperature)
    h = humidity_level(c.humidity)
    cont, ero, pv = c.continentalness, c.erosion, c.pv
    inland = cont >= -0.05

    conditions = [
        cont < -0.455,
        cont < -0.11,
        cont < -0.05,
        river > 0.5,
        inland & (ero < -0.3) & (pv > 0.5) & (t <= 1),
        inland & (ero < -0.3) & (pv > 0.5),
        (ero > 0.55) & (h >= 2) & (t >= 1) & (t <= 3),
    ]
    choices = [B.DEEP_OCEAN, B.OCEAN, B.BEACH, B.RIVER, B.SNOWY_PEAKS, B.STONY_PEAKS, B.SWAMP]
    return np.select(conditions, choices, default=MIDDLE_BIOMES[t, h]).astype(np.uint8)
