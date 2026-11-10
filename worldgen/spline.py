"""嵌套三次样条：把气候参数映射为地形高度，这是"设计者写死的规则"。

与 MC 的 ``CubicSpline`` 相同：每个节点 (location, value, derivative)，value 可以是常数，
也可以是另一条以别的坐标为输入的样条。例如：

    continentalness → (每个节点) erosion 样条 → (每个节点) peaks&valleys 样条 → 高度

于是"越往内陆越高""侵蚀度低的地方多山""峰谷参数高处是山峰"这些规则都体现在样条节点上，
而噪声只负责决定每个位置的参数取值。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Union

import numpy as np

Value = Union[float, "Spline"]


@dataclass(frozen=True)
class Point:
    location: float
    value: Value
    derivative: float = 0.0


class Spline:
    def __init__(self, coordinate: str, points: list[Point | tuple]):
        pts = [p if isinstance(p, Point) else Point(*p) for p in points]
        locs = [p.location for p in pts]
        if locs != sorted(locs) or len(set(locs)) != len(locs):
            raise ValueError("spline locations must be strictly increasing")
        self.coordinate = coordinate
        self.points = pts
        self.locations = np.array(locs, dtype=np.float64)

    def evaluate(self, ctx: Mapping[str, np.ndarray]) -> np.ndarray:
        x = np.asarray(ctx[self.coordinate], dtype=np.float64)
        values = [_eval_value(p.value, ctx, x.shape) for p in self.points]
        derivs = [p.derivative for p in self.points]
        locs = self.locations
        n = len(self.points)

        out = np.empty(x.shape, dtype=np.float64)
        # 左右两端外：按端点导数线性外推
        left = x < locs[0]
        right = x >= locs[-1]
        out[left] = values[0][left] + derivs[0] * (x[left] - locs[0])
        out[right] = values[-1][right] + derivs[-1] * (x[right] - locs[-1])

        for i in range(n - 1):
            m = (x >= locs[i]) & (x < locs[i + 1])
            if not m.any():
                continue
            x0, x1 = locs[i], locs[i + 1]
            span = x1 - x0
            t = (x[m] - x0) / span
            y0 = values[i][m]
            y1 = values[i + 1][m]
            p = derivs[i] * span - (y1 - y0)
            q = -derivs[i + 1] * span + (y1 - y0)
            lin = y0 + t * (y1 - y0)
            out[m] = lin + t * (1.0 - t) * (p + t * (q - p))
        return out


def _eval_value(value: Value, ctx: Mapping[str, np.ndarray], shape) -> np.ndarray:
    if isinstance(value, Spline):
        return value.evaluate(ctx)
    return np.full(shape, float(value), dtype=np.float64)


# ---------------------------------------------------------------------------
# 本项目使用的地形样条（偏移量单位：方块，相对海平面）
# ---------------------------------------------------------------------------


def _pv(valley: float, low: float, mid: float, high: float, peak: float) -> Spline:
    """峰谷样条：pv=-1 是谷底（河流所在），pv=1 是山峰。"""
    return Spline(
        "pv",
        [
            (-1.0, valley, 0.0),
            (-0.4, low, (mid - valley) * 0.6),
            (0.0, mid, (high - low) * 0.8),
            (0.4, high, (peak - mid) * 0.9),
            (1.0, peak, 0.0),
        ],
    )


def _erosion(s: float) -> Spline:
    """侵蚀样条：侵蚀度低 → 山地；侵蚀度高 → 平原/沼泽。s 随内陆程度增大。"""
    return Spline(
        "erosion",
        [
            (-0.9, _pv(3, 12 * s, 36 * s, 70 * s, 110 * s)),
            (-0.5, _pv(3, 9 * s, 24 * s, 44 * s, 66 * s)),
            (-0.2, _pv(2, 6 * s, 13 * s, 21 * s, 30 * s)),
            (0.1, _pv(2, 4, 4 + 5 * s, 6 + 8 * s, 8 + 10 * s)),
            (0.45, _pv(1, 3, 4, 6, 8)),
            (0.7, _pv(0, 1, 2, 2.5, 3)),
        ],
    )


def build_offset_spline() -> Spline:
    return Spline(
        "continentalness",
        [
            (-1.05, -48.0, 0.0),
            (-0.455, -30.0, 30.0),
            (-0.19, -8.0, 60.0),
            (-0.11, 0.5, 40.0),
            (-0.05, _erosion(0.25)),
            (0.1, _erosion(0.6)),
            (0.3, _erosion(1.0)),
            (0.6, _erosion(1.2)),
        ],
    )


def build_detail_spline() -> Spline:
    """细节噪声振幅：山地崎岖、平原平坦（相当于 MC 的 jaggedness/factor）。"""
    return Spline(
        "erosion",
        [(-0.8, 9.0, 0.0), (-0.3, 6.0, -6.0), (0.3, 2.5, -3.0), (0.7, 1.2, 0.0)],
    )
