"""陷阱：岩浆（每回合随机、保证有活路）与摔落伤害（同 MC 规则）。"""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from envs import MinecraftLikeGridEnv
from worldgen.navigation import min_damage_to_targets
from worldgen.tiles import Obj, Tile

KW = dict(goal_distance=150, coin_spacing=24, fall_damage=True, lava_density=4.0, reward_shaping="none")


@pytest.fixture(scope="module", params=[7, 2024])
def env(request):
    e = MinecraftLikeGridEnv(world_seed=request.param, **KW)
    e.reset(seed=0)
    return e


def _lava_mask(e):
    q = e.quest
    m = np.zeros_like(q.free)
    for x, z in e.lava:
        m[z - q.box[1], x - q.box[0]] = True
    return m


def test_lava_reproducible_and_changes_per_episode(env):
    env.reset(seed=11)
    a = set(env.lava)
    env.reset(seed=11)
    assert env.lava == a and len(a) > 0
    env.reset(seed=12)
    assert env.lava != a


def test_world_layout_is_fixed():
    e = MinecraftLikeGridEnv(world_seed=7, hazard_layout="world", **KW)
    e.reset(seed=1)
    a = set(e.lava)
    e.reset(seed=2)
    assert e.lava == a and len(a) > 0


def test_lava_placement_rules(env):
    for s in range(20):
        env.reset(seed=s)
        q = env.quest
        lava = _lava_mask(env)
        assert env.quest.start not in env.lava
        assert not any(c in env.lava for c in env.coin_layout)
        assert np.isin(q.objects[lava], (Obj.NONE, Obj.FLOWER)).all()  # 不压在树、结构、传送门上
        wet = q.water.copy()
        wet[1:, :] |= q.water[:-1, :]
        wet[:-1, :] |= q.water[1:, :]
        wet[:, 1:] |= q.water[:, :-1]
        wet[:, :-1] |= q.water[:, 1:]
        assert not (lava & wet).any()  # 岩浆不挨着水


def test_survivable_route_always_exists(env):
    for s in range(30):
        env.reset(seed=s)
        q = env.quest
        d = min_damage_to_targets(q.stand, q.free, q.water, _lava_mask(env), q.portal_mask(), q.max_climb, q.moves)
        assert d[q.start[1] - q.box[1], q.start[0] - q.box[0]] < env.max_health


def _step_towards(e, dst):
    dx, dz = dst[0] - e.pos[0], dst[1] - e.pos[1]
    return e.step(e.moves.index((dx, dz)))


def test_stepping_into_lava_kills():
    e = MinecraftLikeGridEnv(world_seed=7, **{**KW, "reward_shaping": "geodesic"})
    for s in range(50):
        e.reset(seed=s)
        for x, z in sorted(e.lava):
            for dx, dz in e.moves:
                src = (x - dx, z - dz)
                if src not in e.lava and e.can_move(src, (x, z)) and e.quest.distance(*src) < np.inf:
                    e.pos = src  # 直接把智能体放到岩浆旁边
                    obs, r, te, tr, info = _step_towards(e, (x, z))
                    assert te and info["death"] == "lava" and info["health"] == 0 and not info["success"]
                    assert r < 0  # −death_penalty，且塑形项不会因死亡变成正奖励
                    return
    pytest.fail("no lava cell adjacent to a walkable cell")


def test_fall_damage_follows_mc_rule():
    e = MinecraftLikeGridEnv(world_seed=2024, **{**KW, "lava_density": 0.0})
    e.reset(seed=0)
    q = e.quest
    sea = e.world_config.sea_level
    found_land = found_water = False
    for i in range(1, q.stand.shape[0] - 1):
        for j in range(1, q.stand.shape[1] - 1):
            for dx, dz in e.moves:
                drop = int(q.stand[i, j] - q.stand[i + dz, j + dx])
                if drop < 5 or not q.free[i + dz, j + dx]:
                    continue
                into_water = bool(q.water[i + dz, j + dx])
                if (into_water and found_water) or (not into_water and found_land):
                    continue
                e.reset(seed=0)
                e.pos = (q.box[0] + j, q.box[1] + i)
                _, _, _, _, info = e.step(e.moves.index((dx, dz)))
                assert info["damage"] == (0 if into_water else drop - 3)
                assert info["health"] == e.max_health - info["damage"]
                found_water |= into_water
                found_land |= not into_water
            if found_land and found_water:
                return
    assert found_land, "no drop >= 5 found"


def test_fall_can_kill():
    e = MinecraftLikeGridEnv(world_seed=2024, **{**KW, "lava_density": 0.0})
    e.reset(seed=0)
    q = e.quest
    zs, xs = np.nonzero(q.free)
    for i, j in zip(zs, xs):
        for dx, dz in e.moves:
            ni, nj = i + dz, j + dx
            if 0 <= ni < q.stand.shape[0] and 0 <= nj < q.stand.shape[1] and q.free[ni, nj] \
                    and not q.water[ni, nj] and q.stand[i, j] - q.stand[ni, nj] >= 4:
                e.pos = (q.box[0] + int(j), q.box[1] + int(i))
                e.health = 1
                _, r, te, _, info = e.step(e.moves.index((dx, dz)))
                assert te and info["death"] == "fall" and r < 0
                return
    pytest.fail("no damaging drop found")


def test_observation_shows_lava_and_health(env):
    env.reset(seed=3)
    x, z = sorted(env.lava)[0]
    env.pos = (x - 2, z)  # 站在岩浆左边两格
    obs = env._observe()
    r = env.view_radius
    assert obs["terrain"][r, r + 2] == Tile.LAVA
    expected = sum(1 for lx, lz in env.lava if abs(lx - env.pos[0]) <= r and abs(lz - env.pos[1]) <= r)
    assert (obs["terrain"] == Tile.LAVA).sum() == expected
    assert obs["health"][0] == pytest.approx(env.health / env.max_health)


def test_world_pool_and_check_env():
    e = MinecraftLikeGridEnv(world_seeds=[7, 12345], **KW)
    seen = {e.reset(seed=s)[1]["world_seed"] for s in range(6)}
    assert seen <= {7, 12345} and len(seen) == 2
    check_env(MinecraftLikeGridEnv(world_seeds=[7], **KW), skip_render_check=True)


def test_ppo_network_shapes():
    torch = pytest.importorskip("torch")
    from agents.ppo import ActorCritic, encode, stack_obs

    e = MinecraftLikeGridEnv(world_seed=7, **KW)
    obs, _ = e.reset(seed=0)
    img, vec = encode(stack_obs([obs, obs]))
    assert img.shape[0] == 2 and vec.shape == (2, 4)
    net = ActorCritic(e.action_space.n)
    logits, v = net(img, vec)
    assert logits.shape == (2, 4) and v.shape == (2,)
    assert e.action_space.contains(int(net.act(stack_obs([obs]))[0]))
