"""把 Area 渲染成图片：群系图、带光照的地表图、高度图。"""

from __future__ import annotations

import numpy as np

from .tiles import BIOME_COLORS, OBJ_COLORS, TILE_COLORS, Biome, Obj, Tile

_TILE_LUT = np.array([TILE_COLORS[Tile(i)] for i in range(len(Tile))], dtype=np.float64)
_BIOME_LUT = np.array([BIOME_COLORS[Biome(i)] for i in range(len(Biome))], dtype=np.float64)
_OBJ_LUT = np.array([OBJ_COLORS.get(Obj(i), (0, 0, 0)) for i in range(len(Obj))], dtype=np.float64)


def hillshade(height: np.ndarray, strength: float = 0.06) -> np.ndarray:
    h = height.astype(np.float64)
    gx = np.zeros_like(h)
    gz = np.zeros_like(h)
    gx[:, 1:-1] = h[:, :-2] - h[:, 2:]
    gz[1:-1, :] = h[:-2, :] - h[2:, :]
    return np.clip(1.0 + strength * (gx + gz), 0.55, 1.35)


def terrain_rgb(area, shade: bool = True, objects: bool = True) -> np.ndarray:
    rgb = _TILE_LUT[area.terrain]
    if objects:
        mask = area.objects != Obj.NONE
        rgb[mask] = _OBJ_LUT[area.objects[mask]]
    if shade:
        rgb = rgb * hillshade(area.height)[..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def biome_rgb(area) -> np.ndarray:
    return _BIOME_LUT[area.biome].astype(np.uint8)


def save_world_png(world, x0: int, z0: int, w: int, h: int, path: str, mark_spawn: bool = True) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    area = world.generate_area(x0, z0, w, h)
    extent = (x0, x0 + w, z0 + h, z0)
    fig, axes = plt.subplots(1, 3, figsize=(27, 10))
    axes[0].imshow(biome_rgb(area), extent=extent, interpolation="nearest")
    axes[0].set_title("Biomes")
    axes[1].imshow(terrain_rgb(area), extent=extent, interpolation="nearest")
    axes[1].set_title("Surface + objects + structures")
    im = axes[2].imshow(area.height, extent=extent, cmap="terrain", interpolation="nearest")
    axes[2].set_title("Height")
    fig.colorbar(im, ax=axes[2], fraction=0.046)

    markers = {"village": ("o", "red"), "desert_temple": ("^", "orange"), "stronghold": ("*", "magenta")}
    for s in area.structures:
        m, c = markers.get(s.kind, ("x", "black"))
        for ax in axes[:2]:
            ax.plot(s.x, s.z, marker=m, color=c, markersize=11, markeredgecolor="black")
    if mark_spawn:
        sx, sz = world.find_spawn()
        if x0 <= sx < x0 + w and z0 <= sz < z0 + h:
            for ax in axes[:2]:
                ax.plot(sx, sz, marker="P", color="white", markersize=13, markeredgecolor="black")

    present = sorted(set(np.unique(area.biome).tolist()))
    handles = [Patch(color=np.array(BIOME_COLORS[Biome(b)]) / 255, label=Biome(b).name.lower()) for b in present]
    handles += [
        plt.Line2D([], [], marker=m, color=c, linestyle="", markeredgecolor="black", label=k)
        for k, (m, c) in markers.items()
    ]
    handles.append(plt.Line2D([], [], marker="P", color="white", linestyle="", markeredgecolor="black", label="spawn"))
    axes[0].legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.85)
    fig.suptitle(f"seed={world.seed}  area=({x0},{z0}) {w}x{h}", fontsize=14)
    plt.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
