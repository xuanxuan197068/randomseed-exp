"""可复现性：同种子同世界；与生成顺序、进程、Python 哈希随机化无关。"""

import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest

from worldgen import World
from worldgen.rng import Random, java_string_hash, normalize_seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYERS = ("height", "terrain", "biome", "slope", "objects")


def area_digest(area) -> str:
    h = hashlib.sha256()
    for k in LAYERS:
        h.update(np.ascontiguousarray(getattr(area, k)).tobytes())
    for s in sorted(area.structures, key=lambda s: (s.kind, s.x, s.z)):
        h.update(f"{s.kind},{s.x},{s.z}".encode())
    return h.hexdigest()


def assert_same(a, b):
    for k in LAYERS:
        np.testing.assert_array_equal(getattr(a, k), getattr(b, k), err_msg=k)
    key = lambda s: (s.kind, s.x, s.z)  # noqa: E731
    assert sorted(map(key, a.structures)) == sorted(map(key, b.structures))


def test_random_is_deterministic():
    a, b = Random(42), Random(42)
    assert [a.next_u64() for _ in range(100)] == [b.next_u64() for _ in range(100)]
    r = Random(1)
    vals = [r.next_int(7) for _ in range(2000)]
    assert min(vals) == 0 and max(vals) == 6
    assert all(0.0 <= Random(i).next_float() < 1.0 for i in range(100))


def test_seed_normalization():
    assert normalize_seed("123") == 123
    assert normalize_seed("hello") == normalize_seed(java_string_hash("hello"))
    assert World("hello").seed == World(java_string_hash("hello")).seed


def test_same_seed_same_world():
    a = World(12345).generate_area(-200, 300, 384, 256)
    b = World(12345).generate_area(-200, 300, 384, 256)
    assert_same(a, b)


def test_different_seed_different_world():
    a = World(1).generate_area(0, 0, 256, 256)
    b = World(2).generate_area(0, 0, 256, 256)
    assert not np.array_equal(a.height, b.height)


@pytest.mark.parametrize("seed", [7, 12345])
def test_chunk_order_independence(seed):
    """乱序逐区块生成再拼接 == 一次性整片生成（不对齐区块边界）。"""
    x0, z0, w, h = -37, 51, 150, 117
    whole = World(seed).generate_area(x0, z0, w, h)
    world = World(seed)
    cs = world.config.chunk_size
    chunks = [(cx, cz) for cx in range(x0 // cs, (x0 + w - 1) // cs + 1)
              for cz in range(z0 // cs, (z0 + h - 1) // cs + 1)]
    Random(99).shuffle(chunks)
    for cx, cz in chunks:
        world.get_chunk(cx, cz)
    assert_same(whole, world.get_area(x0, z0, w, h))


def test_cross_process_with_different_hash_seed():
    """新进程 + 不同 PYTHONHASHSEED 得到完全相同的世界。"""
    expected = area_digest(World(2024).generate_area(-64, -64, 128, 128))
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from tests.test_determinism import area_digest\n"
        "from worldgen import World\n"
        "print(area_digest(World(2024).generate_area(-64, -64, 128, 128)))\n" % ROOT
    )
    for hash_seed in ("0", "1", "random"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT, check=True)
        assert out.stdout.strip() == expected
