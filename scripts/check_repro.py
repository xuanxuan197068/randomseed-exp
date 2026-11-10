"""打印可复现性检查结果：同种子 / 不同种子 / 乱序区块 / 跨进程。

    python scripts/check_repro.py --seed 12345
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests.test_determinism import area_digest  # noqa: E402
from worldgen import World  # noqa: E402
from worldgen.rng import Random  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="12345")
    args = ap.parse_args()
    x0, z0, w, h = -100, -80, 200, 160

    base = area_digest(World(args.seed).generate_area(x0, z0, w, h))
    print(f"[1] seed={args.seed} 一次性生成           {base[:16]}")
    again = area_digest(World(args.seed).generate_area(x0, z0, w, h))
    print(f"[2] 同种子重新生成                 {again[:16]}  {'一致' if again == base else '不一致!'}")

    world = World(args.seed)
    cs = world.config.chunk_size
    chunks = [(cx, cz) for cx in range(x0 // cs, (x0 + w - 1) // cs + 1)
              for cz in range(z0 // cs, (z0 + h - 1) // cs + 1)]
    Random(1).shuffle(chunks)
    for c in chunks:
        world.get_chunk(*c)
    shuffled = area_digest(world.get_area(x0, z0, w, h))
    print(f"[3] 乱序逐区块生成后拼接           {shuffled[:16]}  {'一致' if shuffled == base else '不一致!'}")

    code = (f"import sys; sys.path.insert(0, {ROOT!r})\n"
            "from tests.test_determinism import area_digest\nfrom worldgen import World\n"
            f"print(area_digest(World({args.seed!r}).generate_area({x0}, {z0}, {w}, {h})))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONHASHSEED": "random"}).stdout.strip()
    print(f"[4] 新进程(随机 PYTHONHASHSEED)    {out[:16]}  {'一致' if out == base else '不一致!'}")

    other = area_digest(World(str(World(args.seed).seed + 1)).generate_area(x0, z0, w, h))
    print(f"[5] 种子+1                         {other[:16]}  {'不同(符合预期)' if other != base else '相同?!'}")


if __name__ == "__main__":
    main()
