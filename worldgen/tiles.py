"""网格中每一格的三种离散类型：地表方块 Tile、地表物体 Obj、生物群系 Biome。"""

from __future__ import annotations

from enum import IntEnum


class Tile(IntEnum):
    VOID = 0  # 世界边界之外（仅出现在 RL 观测中）
    DEEP_WATER = 1
    WATER = 2
    ICE = 3
    SAND = 4
    GRAVEL = 5
    GRASS = 6
    SNOW = 7
    STONE = 8
    MUD = 9
    LAVA = 10  # 岩浆（每回合随机布置的陷阱，不属于世界本身）


class Obj(IntEnum):
    NONE = 0
    TREE = 1
    CACTUS = 2
    FLOWER = 3
    ORE = 4
    PATH = 5
    HOUSE = 6
    TEMPLE = 7
    PORTAL = 8  # 末地传送门（主线终点）
    STRONGHOLD = 9  # 要塞传送门房间的石砖地面
    PORTAL_FRAME = 10  # 末地传送门框架（12 个，围住 3×3 传送门）


class Biome(IntEnum):
    DEEP_OCEAN = 0
    OCEAN = 1
    BEACH = 2
    RIVER = 3
    PLAINS = 4
    FOREST = 5
    DESERT = 6
    SAVANNA = 7
    TAIGA = 8
    SNOWY_PLAINS = 9
    JUNGLE = 10
    SWAMP = 11
    STONY_PEAKS = 12
    SNOWY_PEAKS = 13


WATER_TILES = frozenset({Tile.DEEP_WATER, Tile.WATER, Tile.ICE})
IMPASSABLE_OBJS = frozenset({Obj.TREE, Obj.CACTUS, Obj.HOUSE})

# 渲染颜色（RGB 0-255）
TILE_COLORS = {
    Tile.VOID: (0, 0, 0),
    Tile.DEEP_WATER: (28, 52, 120),
    Tile.WATER: (52, 96, 190),
    Tile.ICE: (170, 210, 240),
    Tile.SAND: (222, 205, 140),
    Tile.GRAVEL: (140, 135, 130),
    Tile.GRASS: (98, 160, 70),
    Tile.SNOW: (245, 248, 252),
    Tile.STONE: (125, 125, 125),
    Tile.MUD: (90, 85, 60),
    Tile.LAVA: (235, 95, 15),
}

OBJ_COLORS = {
    Obj.TREE: (30, 90, 35),
    Obj.CACTUS: (60, 140, 40),
    Obj.FLOWER: (235, 90, 160),
    Obj.ORE: (60, 210, 220),
    Obj.PATH: (180, 150, 100),
    Obj.HOUSE: (130, 70, 30),
    Obj.TEMPLE: (230, 180, 60),
    Obj.PORTAL: (20, 10, 40),
    Obj.STRONGHOLD: (110, 115, 105),
    Obj.PORTAL_FRAME: (190, 210, 150),
}

BIOME_COLORS = {
    Biome.DEEP_OCEAN: (20, 30, 100),
    Biome.OCEAN: (40, 70, 170),
    Biome.BEACH: (240, 225, 150),
    Biome.RIVER: (60, 130, 230),
    Biome.PLAINS: (140, 190, 90),
    Biome.FOREST: (40, 120, 50),
    Biome.DESERT: (245, 190, 80),
    Biome.SAVANNA: (190, 180, 90),
    Biome.TAIGA: (60, 100, 90),
    Biome.SNOWY_PLAINS: (235, 240, 250),
    Biome.JUNGLE: (30, 160, 40),
    Biome.SWAMP: (80, 100, 70),
    Biome.STONY_PEAKS: (150, 140, 130),
    Biome.SNOWY_PEAKS: (200, 210, 230),
}
