"""气候参数场（MC 1.18+ 的 multi-noise）：5 个独立噪声 + 派生的峰谷值。

每个参数用 derive_seed(seed, 名字) 得到独立种子，所以改动其中一个不会影响其他。
采样前先用 shift 噪声对坐标做扰动（domain warping），这一步对所有参数相同，
因此参数之间仍保持空间连续，群系边界变得自然但不会出现跳变。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import WorldConfig
from .noise import NormalNoise
from .rng import derive_seed

PARAMS = ("temperature", "humidity", "continentalness", "erosion", "weirdness")


@dataclass
class Climate:
    temperature: np.ndarray
    humidity: np.ndarray
    continentalness: np.ndarray
    erosion: np.ndarray
    weirdness: np.ndarray
    pv: np.ndarray  # peaks & valleys

    def as_ctx(self) -> dict[str, np.ndarray]:
        return {
            "temperature": self.temperature,
            "humidity": self.humidity,
            "continentalness": self.continentalness,
            "erosion": self.erosion,
            "weirdness": self.weirdness,
            "pv": self.pv,
        }


def peaks_and_valleys(weirdness: np.ndarray) -> np.ndarray:
    """MC 的 PV 折叠：|w|≈0 → -1（谷），|w|≈2/3 → 1（峰）。"""
    return 1.0 - np.abs(3.0 * np.abs(weirdness) - 2.0)


class ClimateSampler:
    def __init__(self, seed: int, config: WorldConfig):
        self.config = config
        self.noises = {name: NormalNoise(derive_seed(seed, name), getattr(config, name)) for name in PARAMS}
        self.shift_x = NormalNoise(derive_seed(seed, "shift_x"), config.shift)
        self.shift_z = NormalNoise(derive_seed(seed, "shift_z"), config.shift)

    def sample(self, x: np.ndarray, z: np.ndarray) -> Climate:
        amp = self.config.shift_amplitude
        sx = x + self.shift_x.sample(x, z) * amp
        sz = z + self.shift_z.sample(z, x) * amp
        v = {name: noise.sample(sx, sz) for name, noise in self.noises.items()}
        v["continentalness"] = v["continentalness"] + self.config.continentalness_bias
        return Climate(pv=peaks_and_valleys(v["weirdness"]), **v)
