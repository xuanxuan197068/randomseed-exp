"""规则性：随机世界必须满足的硬约束。"""

import itertools

import numpy as np
import pytest

from worldgen import Biome, Obj, Tile, World
from worldgen.biomes import FROZEN_BIOMES, HOT_BIOMES
from worldgen.tiles import WATER_TILES

SEEDS = [7, 12345, 2024]


@pytest.fixture(scope="module", params=SEEDS)
def world(request):
    return World(request.param)


@pytest.fixture(scope="module")
def area(world):
    return world.generate_area(-512, -512, 1024, 1024)


def _pairs_4(a):
    """所有 4 邻接对 (左, 右) 和 (上, 下)。"""
    return [(a[:, :-1], a[:, 1:]), (a[:-1, :], a[1:, :])]


def test_no_hot_next_to_frozen(area):
    hot = np.isin(area.biome, list(HOT_BIOMES))
    cold = np.isin(area.biome, list(FROZEN_BIOMES))
    for (h1, h2), (c1, c2) in zip(_pairs_4(hot), _pairs_4(cold)):
        assert not (h1 & c2).any() and not (c1 & h2).any()


def test_temperature_changes_gradually(world):
    t = world.terrain.generate(-512, -512, 1024, 1024).temp_level.astype(int)
    for a, b in _pairs_4(t):
        assert np.abs(a - b).max() <= 1


def test_surface_rules_water_iff_below_sea(area, world):
    sea = world.config.sea_level
    is_water = np.isin(area.terrain, list(WATER_TILES))
    np.testing.assert_array_equal(is_water, area.height < sea)
    land = ~is_water
    assert (area.terrain[land & (area.height >= world.config.snow_line)] == Tile.SNOW).all()


def test_objects_never_on_water(area):
    on_water = np.isin(area.terrain, list(WATER_TILES)) & (area.objects != Obj.NONE)
    assert not on_water.any()


def test_feature_surfaces_and_cactus_isolation(area):
    trees = area.objects == Obj.TREE
    assert np.isin(area.terrain[trees], [Tile.GRASS, Tile.SNOW, Tile.MUD]).all()
    cactus = area.objects == Obj.CACTUS
    assert (area.biome[cactus] == Biome.DESERT).all()
    occupied = np.pad(area.objects != Obj.NONE, 1)
    neighbours = occupied[:-2, 1:-1] | occupied[2:, 1:-1] | occupied[1:-1, :-2] | occupied[1:-1, 2:]
    assert not (cactus & neighbours).any()


def _instances(world, name, r=6):
    return [(rx, rz, world.structures.instance(name, rx, rz))
            for rx in range(-r, r) for rz in range(-r, r)]


@pytest.mark.parametrize("name", ["village", "desert_temple"])
def test_structure_spacing(world, name):
    s = world.structures.by_name[name]
    chunks = [world.structures.candidate(name, rx, rz) for rx, rz, _ in _instances(world, name)]
    for (ax, az), (bx, bz) in itertools.combinations(chunks, 2):
        assert max(abs(ax - bx), abs(az - bz)) >= s.separation + 1


@pytest.mark.parametrize("name", ["village", "desert_temple"])
def test_structure_sites_valid(world, name):
    s = world.structures.by_name[name]
    found = [i for _, _, i in _instances(world, name) if i is not None]
    if name == "village":  # 神殿只在沙漠，某些种子原点附近可能根本没有沙漠
        assert found, "expected at least one village"
    for inst in found:
        half = s.footprint // 2
        a = world.terrain.generate(inst.x - half, inst.z - half, s.footprint, s.footprint)
        assert Biome(int(a.biome[half, half])).name.lower() in s.allowed_biomes
        assert (a.height >= world.config.sea_level).all()
        assert int(a.height.max()) - int(a.height.min()) <= s.max_relief


def test_some_seed_has_temples():
    assert any(World(seed).structures.instance("desert_temple", rx, rz)
               for seed in SEEDS for rx in range(-6, 6) for rz in range(-6, 6))


def test_temple_excluded_near_villages(world):
    villages = [world.structures.candidate("village", rx, rz)
                for rx, rz, i in _instances(world, "village", 10) if i is not None]
    for _, _, t in _instances(world, "desert_temple"):
        if t is None:
            continue
        tcx, tcz = t.x // 16, t.z // 16
        assert all(max(abs(tcx - vx), abs(tcz - vz)) >= 4 for vx, vz in villages)


def test_strongholds_rings(world):
    cfg = world.config.stronghold
    sh = world.structures.strongholds()
    assert len(sh) == cfg.count
    d = cfg.distance * 16
    first_ring = sorted(np.hypot(s.x, s.z) for s in sh)[: cfg.spread]
    slack = 112 + 16  # 寻找陆地的搜索半径 + 区块取整
    for r in first_ring:
        assert 4 * d - 1.25 * d - slack <= r <= 4 * d + 1.25 * d + slack


def test_spawn_on_land(world):
    x, z = world.find_spawn()
    a = world.generate_area(x, z, 1, 1)
    assert a.terrain[0, 0] not in WATER_TILES
    assert a.height[0, 0] >= world.config.sea_level
