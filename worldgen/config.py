"""世界生成的全部可调参数。同一个 (seed, WorldConfig) 永远生成同一个世界。"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from .noise import NoiseParams


@dataclass(frozen=True)
class StructureConfig:
    """RandomSpread 结构放置（单位：区块）。

    世界被划分成 spacing×spacing 区块的大格；每格内在 [0, spacing-separation) 范围随机偏移
    放一个候选点 → 同类结构之间至少相隔 separation+1 个区块。候选点再经过群系/地形校验。
    """

    name: str
    spacing: int
    separation: int
    salt: int
    allowed_biomes: tuple[str, ...]
    footprint: int  # 占地边长（方块）
    max_relief: int  # 占地内最大高差
    exclusion: tuple[tuple[str, int], ...] = ()  # (其他结构名, 最小区块距离)


@dataclass(frozen=True)
class StrongholdConfig:
    """同心环放置（MC ConcentricRingsStructurePlacement）：数量固定、按环分布、离原点距离受控。"""

    count: int = 9
    distance: int = 6  # 区块；第一环半径约 4*distance 区块（默认 ≈ 384 方块，±31%）
    spread: int = 3  # 第一环个数
    footprint: int = 9  # 传送门房间边长（方块）
    max_relief: int = 3  # 房间占地内最大高差
    salt: int = 0x5E7A11


@dataclass(frozen=True)
class FeatureConfig:
    """装饰物放置（MC PlacementModifier 链：count → in_square → biome → surface → spacing）。"""

    name: str
    obj: str
    salt: int
    tries: int  # 每区块尝试次数
    density: tuple[tuple[str, float], ...]  # 群系名 → 每次尝试的接受概率；"*" 为默认
    surfaces: tuple[str, ...]  # 允许的地表 Tile 名
    max_slope: int = 2
    min_spacing: int = 1  # 同一区块内同类物体的最小切比雪夫距离
    isolated: bool = False  # 四邻不能有任何物体（仙人掌规则）


def _default_structures() -> tuple[StructureConfig, ...]:
    return (
        # salt 取自 Minecraft 原版；MC 原版 spacing/separation 为 村庄 34/8、沙漠神殿 32/8。
        # 这里的气候尺度比 MC 小约 3 倍，因此间距也按比例缩小，保证 RL 尺寸的地图里能看到结构。
        StructureConfig(
            "village", 16, 5, 10387312,
            ("plains", "desert", "savanna", "taiga", "snowy_plains"),
            footprint=32, max_relief=7,
        ),
        StructureConfig(
            "desert_temple", 12, 4, 14357617, ("desert",),
            footprint=12, max_relief=5, exclusion=(("village", 4),),
        ),
    )


def _default_features() -> tuple[FeatureConfig, ...]:
    return (
        FeatureConfig(
            "tree", "TREE", 0x7EE, tries=14,
            density=(("forest", 0.55), ("taiga", 0.5), ("jungle", 0.85), ("swamp", 0.25),
                     ("plains", 0.04), ("savanna", 0.07), ("snowy_plains", 0.03)),
            surfaces=("GRASS", "SNOW", "MUD"), max_slope=2, min_spacing=2,
        ),
        FeatureConfig(
            "cactus", "CACTUS", 0xCAC, tries=5, density=(("desert", 0.45),),
            surfaces=("SAND",), max_slope=1, isolated=True,
        ),
        FeatureConfig(
            "flower", "FLOWER", 0xF10, tries=6,
            density=(("plains", 0.35), ("forest", 0.12), ("savanna", 0.08)),
            surfaces=("GRASS",), max_slope=2,
        ),
        FeatureConfig(
            "ore", "ORE", 0x0E5, tries=6, density=(("*", 0.25),),
            surfaces=("STONE",), max_slope=255,
        ),
    )


@dataclass(frozen=True)
class WorldConfig:
    chunk_size: int = 16
    sea_level: int = 64
    min_height: int = 1
    max_height: int = 255
    # 气候噪声：波长比 MC 小（MC 大陆尺度上千区块），方便在 RL 可用的地图尺寸内看到多种群系
    temperature: NoiseParams = NoiseParams(1000.0, (1.5, 0.0, 1.0, 0.0, 0.0, 0.0), std=0.42)
    humidity: NoiseParams = NoiseParams(700.0, (1.0, 1.0, 0.0, 0.0, 0.0, 0.0), std=0.42)
    continentalness: NoiseParams = NoiseParams(1200.0, (1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0), std=0.40)
    erosion: NoiseParams = NoiseParams(600.0, (1.0, 1.0, 0.0, 1.0, 1.0), std=0.40)
    weirdness: NoiseParams = NoiseParams(500.0, (1.0, 2.0, 1.0, 0.0, 0.0, 0.0), std=0.45)
    continentalness_bias: float = 0.12  # >0 偏向陆地（RL 环境需要更多可行走区域）
    shift: NoiseParams = NoiseParams(128.0, (1.0, 1.0, 1.0, 0.0), std=0.4)
    shift_amplitude: float = 12.0  # 气候采样前的坐标扰动（MC 的 shift_x/shift_z），让群系边界不那么圆润
    detail: NoiseParams = NoiseParams(80.0, (1.0, 1.0, 1.0, 1.0), std=0.4)
    river_width: float = 0.05  # |weirdness| 小于此值的谷底开凿河流
    snow_line: int = 150
    steep_slope: int = 4  # 坡度 ≥ 此值的地表露出石头
    structures: tuple[StructureConfig, ...] = field(default_factory=_default_structures)
    stronghold: StrongholdConfig = field(default_factory=StrongholdConfig)
    features: tuple[FeatureConfig, ...] = field(default_factory=_default_features)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "WorldConfig":
        kwargs = dict(data)
        for name in ("temperature", "humidity", "continentalness", "erosion", "weirdness", "shift", "detail"):
            if name in kwargs and isinstance(kwargs[name], dict):
                p = kwargs[name]
                kwargs[name] = NoiseParams(p["scale"], tuple(p["amplitudes"]), p.get("std", 0.4))
        if "structures" in kwargs:
            kwargs["structures"] = tuple(
                StructureConfig(**{**s, "allowed_biomes": tuple(s["allowed_biomes"]),
                                   "exclusion": tuple(tuple(e) for e in s.get("exclusion", ()))})
                for s in kwargs["structures"]
            )
        if "stronghold" in kwargs and isinstance(kwargs["stronghold"], dict):
            kwargs["stronghold"] = StrongholdConfig(**kwargs["stronghold"])
        if "features" in kwargs:
            kwargs["features"] = tuple(
                FeatureConfig(**{**f, "density": tuple(tuple(d) for d in f["density"]),
                                 "surfaces": tuple(f["surfaces"])})
                for f in kwargs["features"]
            )
        return cls(**kwargs)
