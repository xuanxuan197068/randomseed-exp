"""确定性随机数工具。

Minecraft 的可复现性来自一条铁律：**所有随机数都由 (世界种子, 坐标, salt) 派生**，
不存在任何"全局随机状态"。于是任一区块的内容只取决于它的坐标，与加载顺序无关。

这里刻意不使用 Python 的 ``hash()``（字符串哈希每个进程随机化）和 ``numpy.random``
（NumPy 不保证 Generator 的数值流跨版本稳定），而是自己实现：

- ``splitmix64``：64 位整数混合函数，用于把多个整数折叠成一个种子（位置哈希）
- ``Random``：Xoroshiro128++，Minecraft 1.18+ 世界生成使用的同款 PRNG
"""

from __future__ import annotations

MASK64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15


def splitmix64(x: int) -> int:
    x = (x + _GOLDEN) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def hash_ints(seed: int, *values: int) -> int:
    """把种子与任意多个整数（坐标、salt）折叠成一个 64 位种子。"""
    h = splitmix64(seed & MASK64)
    for v in values:
        h = splitmix64(h ^ (int(v) & MASK64))
    return h


def _fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & MASK64
    return h


def derive_seed(seed: int, name: str) -> int:
    """为某个子系统（"temperature"、"village"…）派生独立种子，类似 MC 的 PositionalRandomFactory.fromHashOf。"""
    return hash_ints(seed, _fnv1a64(name.encode("utf-8")))


def java_string_hash(s: str) -> int:
    """Java ``String.hashCode``：MC 中非数字的种子字符串就是这样转成种子的。"""
    h = 0
    for ch in s:
        h = (31 * h + ord(ch)) & 0xFFFFFFFF
    return h - (1 << 32) if h >= (1 << 31) else h


def normalize_seed(seed: int | str) -> int:
    """接受整数或字符串种子（与 MC 一致：能解析成整数就用整数，否则取 hashCode），返回 [0, 2^64)。"""
    if isinstance(seed, str):
        s = seed.strip()
        try:
            value = int(s)
        except ValueError:
            value = java_string_hash(s)
    else:
        value = int(seed)
    return value & MASK64


def _rotl(x: int, k: int) -> int:
    return ((x << k) | (x >> (64 - k))) & MASK64


def _mix_stafford13(z: int) -> int:
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


class Random:
    """Xoroshiro128++（与 MC 的 XoroshiroRandomSource 同算法与同种子扩展方式）。"""

    __slots__ = ("_lo", "_hi")

    def __init__(self, seed: int):
        lo = (seed ^ 0x6A09E667F3BCC909) & MASK64
        hi = (lo + 0x9E3779B97F4A7C15) & MASK64
        self._lo = _mix_stafford13(lo)
        self._hi = _mix_stafford13(hi)
        if self._lo == 0 and self._hi == 0:
            self._lo, self._hi = 0x9E3779B97F4A7C15, 0x6A09E667F3BCC909

    @classmethod
    def at(cls, seed: int, *values: int) -> "Random":
        """位置随机源：Random.at(world_seed, cx, cz, salt)。"""
        return cls(hash_ints(seed, *values))

    def next_u64(self) -> int:
        s0, s1 = self._lo, self._hi
        result = (_rotl((s0 + s1) & MASK64, 17) + s0) & MASK64
        s1 ^= s0
        self._lo = _rotl(s0, 49) ^ s1 ^ ((s1 << 21) & MASK64)
        self._hi = _rotl(s1, 28)
        return result

    def next_float(self) -> float:
        """[0, 1) 的 double（53 位精度）。"""
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))

    def next_int(self, bound: int) -> int:
        """[0, bound) 的无偏整数（Lemire 方法，同 MC）。"""
        if bound <= 0:
            raise ValueError("bound must be positive")
        r = self.next_u64() & 0xFFFFFFFF
        m = r * bound
        low = m & 0xFFFFFFFF
        if low < bound:
            threshold = ((1 << 32) - bound) % bound
            while low < threshold:
                r = self.next_u64() & 0xFFFFFFFF
                m = r * bound
                low = m & 0xFFFFFFFF
        return m >> 32

    def shuffle(self, items: list) -> None:
        for i in range(len(items) - 1, 0, -1):
            j = self.next_int(i + 1)
            items[i], items[j] = items[j], items[i]
