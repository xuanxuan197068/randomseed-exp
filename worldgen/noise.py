"""相干噪声：随机性的主要来源，同时提供"连续性"这一最基本的规则。

- ``ImprovedNoise``：2D Perlin 梯度噪声，置换表由种子洗牌得到（同 MC ImprovedNoise）
- ``OctaveNoise``：多倍频叠加（fBm），第 i 层频率 ×2^i、振幅 ×0.5^i，再乘以配置的权重
- ``NormalNoise``：两个独立 OctaveNoise 以略微不同的频率相加（同 MC DoublePerlinNoiseSampler），
  减少网格感并让分布更接近正态；输出被缩放到配置的目标标准差

所有运算只用 +、-、*、floor 和查表，不使用 sin/exp 等超越函数：这些逐元素运算在任何数组
形状下都给出完全相同的 IEEE 结果，因此"整片生成"和"逐区块生成"的结果按位一致。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .rng import Random, hash_ints

# 8 个单位梯度方向
_GRAD_X = np.array([math.cos(k * math.pi / 4) for k in range(8)])
_GRAD_Z = np.array([math.sin(k * math.pi / 4) for k in range(8)])

# 单层 ImprovedNoise 输出的经验标准差（与种子无关，用于把 NormalNoise 缩放到目标标准差）
PERLIN_STD = 0.2157


@dataclass(frozen=True)
class NoiseParams:
    """scale：最低倍频的波长（方块数）；amplitudes：各倍频权重（0 表示跳过该倍频）；
    std：NormalNoise 输出的目标标准差。"""

    scale: float
    amplitudes: tuple[float, ...]
    std: float = 0.4


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


class ImprovedNoise:
    def __init__(self, rng: Random):
        self.xo = rng.next_float() * 256.0
        self.zo = rng.next_float() * 256.0
        perm = list(range(256))
        for i in range(256):  # MC 的洗牌方式：j ∈ [i, 256)
            j = i + rng.next_int(256 - i)
            perm[i], perm[j] = perm[j], perm[i]
        self.perm = np.array(perm + perm, dtype=np.int64)

    def sample(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        x = x + self.xo
        z = z + self.zo
        xf = np.floor(x)
        zf = np.floor(z)
        xi = xf.astype(np.int64) & 255
        zi = zf.astype(np.int64) & 255
        dx = x - xf
        dz = z - zf
        u = _fade(dx)
        v = _fade(dz)
        p = self.perm
        a = p[xi]
        b = p[xi + 1]
        h00 = p[a + zi] & 7
        h01 = p[a + zi + 1] & 7
        h10 = p[b + zi] & 7
        h11 = p[b + zi + 1] & 7
        g00 = _GRAD_X[h00] * dx + _GRAD_Z[h00] * dz
        g10 = _GRAD_X[h10] * (dx - 1.0) + _GRAD_Z[h10] * dz
        g01 = _GRAD_X[h01] * dx + _GRAD_Z[h01] * (dz - 1.0)
        g11 = _GRAD_X[h11] * (dx - 1.0) + _GRAD_Z[h11] * (dz - 1.0)
        top = g00 + u * (g10 - g00)
        bottom = g01 + u * (g11 - g01)
        return top + v * (bottom - top)


class OctaveNoise:
    def __init__(self, seed: int, params: NoiseParams):
        self.octaves: list[tuple[ImprovedNoise, float, float]] = []
        for i, weight in enumerate(params.amplitudes):
            if weight == 0:
                continue
            noise = ImprovedNoise(Random.at(seed, i))
            freq = (2.0 ** i) / params.scale
            amp = weight * (0.5 ** i)
            self.octaves.append((noise, freq, amp))
        if not self.octaves:
            raise ValueError("NoiseParams needs at least one non-zero amplitude")
        total = sum(abs(a) for _, _, a in self.octaves)
        self.octaves = [(n, f, a / total) for n, f, a in self.octaves]
        # 理论标准差：独立倍频之和
        self.std = PERLIN_STD * math.sqrt(sum(a * a for _, _, a in self.octaves))

    def sample(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        total = np.zeros(np.shape(x), dtype=np.float64)
        for noise, freq, amp in self.octaves:
            total += amp * noise.sample(x * freq, z * freq)
        return total


class NormalNoise:
    INPUT_FACTOR = 1.0181268882175227  # 同 MC：第二个采样器的坐标略微缩放，打破对齐

    def __init__(self, seed: int, params: NoiseParams):
        self.first = OctaveNoise(hash_ints(seed, 1), params)
        self.second = OctaveNoise(hash_ints(seed, 2), params)
        self.factor = params.std / (self.first.std * math.sqrt(2.0))

    def sample(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        f = self.INPUT_FACTOR
        return (self.first.sample(x, z) + self.second.sample(x * f, z * f)) * self.factor
