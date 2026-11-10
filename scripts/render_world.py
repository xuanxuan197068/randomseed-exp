"""渲染一块世界区域到 PNG。

    python scripts/render_world.py --seed 12345 --size 1024
    python scripts/render_world.py --seed "hello world" --x 2000 --z -500 --size 512
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worldgen import World  # noqa: E402
from worldgen.render import save_world_png  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="12345")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--x", type=int, default=None, help="区域中心 x（默认原点）")
    ap.add_argument("--z", type=int, default=None, help="区域中心 z（默认原点）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    world = World(args.seed)
    cx = args.x if args.x is not None else 0
    cz = args.z if args.z is not None else 0
    x0, z0 = cx - args.size // 2, cz - args.size // 2
    out = args.out or os.path.join("out", f"seed_{world.seed}_{args.size}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    t = time.time()
    save_world_png(world, x0, z0, args.size, args.size, out)
    print(f"seed={world.seed} spawn={world.find_spawn()} -> {out} ({time.time() - t:.1f}s)")


if __name__ == "__main__":
    main()
