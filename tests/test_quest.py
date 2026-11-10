"""主线：末地传送门作为终点。"""

import math

import numpy as np
import pytest

from envs import MinecraftLikeGridEnv
from worldgen import Obj, World
from worldgen.quest import nominal_goal_distance, plan_end_portal_quest

SEEDS = [7, 12345, 2024]


@pytest.fixture(scope="module", params=SEEDS)
def env(request):
    e = MinecraftLikeGridEnv(world_seed=request.param)
    e.reset(seed=0)
    return e


def test_portal_room_layout():
    for s in World(7).structures.strongholds()[:3]:
        a = World(7).generate_area(s.x - 4, s.z - 4, 9, 9)
        assert (a.objects == Obj.PORTAL).sum() == 9
        assert (a.objects == Obj.PORTAL_FRAME).sum() == 12
        assert (a.objects[3:6, 3:6] == Obj.PORTAL).all()


def test_quest_is_deterministic():
    a = plan_end_portal_quest(World(12345), World(12345).find_spawn())
    b = plan_end_portal_quest(World(12345), World(12345).find_spawn())
    assert a.goal_pos == b.goal_pos and a.optimal_length == b.optimal_length
    np.testing.assert_array_equal(a.dist, b.dist)


def test_goal_distance_from_start(env):
    q = env.quest
    nominal = nominal_goal_distance(env.world_config)
    assert 0.5 * nominal <= q.straight_distance <= 2.0 * nominal
    assert q.optimal_length >= q.straight_distance / math.sqrt(2) - 1  # 4 邻接下步数 ≥ 曼哈顿 ≥ 直线/√2


def test_goal_distance_knob():
    near = MinecraftLikeGridEnv(world_seed=7, goal_distance=200)
    far = MinecraftLikeGridEnv(world_seed=7, goal_distance=700)
    near.reset(seed=0)
    far.reset(seed=0)
    assert near.quest.straight_distance < far.quest.straight_distance


def test_distance_field_matches_movement_rules(env):
    """距离场里相邻格只能按移动规则递减 1：dist(u) ≤ dist(v) + 1 对每条合法边 u→v 成立。"""
    q = env.quest
    d, s, f = q.dist, q.stand, q.free
    for dx, dz in env.moves:
        u = (slice(max(0, -dz), d.shape[0] - max(0, dz)), slice(max(0, -dx), d.shape[1] - max(0, dx)))
        v = (slice(max(0, dz), d.shape[0] - max(0, -dz)), slice(max(0, dx), d.shape[1] - max(0, -dx)))
        legal = f[v] & (s[v] - s[u] <= env.max_climb) & np.isfinite(d[v])
        assert (d[u][legal] <= d[v][legal] + 1).all()


def test_oracle_reaches_portal_in_optimal_steps(env):
    env.reset(seed=0)
    q = env.quest
    path = q.optimal_path()
    action_of = {m: a for a, m in enumerate(env.moves)}
    total, terminated, info = 0.0, False, {}
    for (x0, z0), (x1, z1) in zip(path, path[1:]):
        _, r, terminated, truncated, info = env.step(action_of[(x1 - x0, z1 - z0)])
        total += r
        assert info["moved"]
        assert not truncated
    assert terminated and info["success"]
    assert info["steps"] == q.optimal_length
    assert total == pytest.approx(env.success_reward + 1.0)  # 默认塑形：整条最短路总和恰为 +1


def test_standing_still_has_zero_shaping(env):
    """撞墙（没移动）时默认奖励为 0：塑形只奖励真实的进展。"""
    q = env.quest
    d, st, fr = q.dist, q.stand, q.free
    x0, z0 = q.box[0], q.box[1]
    for i, j in zip(*np.nonzero(np.isfinite(d) & fr)):
        for a, (dx, dz) in enumerate(env.moves):
            ni, nj = i + dz, j + dx
            if 0 <= ni < d.shape[0] and 0 <= nj < d.shape[1] and (not fr[ni, nj] or st[ni, nj] - st[i, j] > 1):
                start = (x0 + int(j), z0 + int(i))
                env.reset(seed=0, options={"start": start})
                _, r, _, _, info = env.step(a)
                assert not info["moved"] and r == 0.0
                return
    pytest.fail("no blocked move found in quest area")


def test_compass_points_to_goal(env):
    obs, info = env.reset(seed=0)
    gx, gz = info["goal"]
    x, z = info["pos"]
    d = math.hypot(gx - x, gz - z)
    np.testing.assert_allclose(obs["goal"][:2], [(gx - x) / d, (gz - z) / d], atol=1e-6)


@pytest.mark.parametrize("seed", [11, 18])
def test_unreachable_nearest_portal_is_skipped(seed):
    """这两个种子里最近的要塞被高崖围住、走不到：规划器必须跳过它，选下一个可达的。"""
    w = World(seed)
    start = w.find_spawn()
    assert plan_end_portal_quest(w, start, max_candidates=1) is None
    q = plan_end_portal_quest(w, start)
    assert q is not None and math.isfinite(q.optimal_length)
    nearest = min(w.structures.strongholds(), key=lambda s: (s.x - start[0]) ** 2 + (s.z - start[1]) ** 2)
    assert q.goal is not nearest


def test_no_portal_in_bounds_raises():
    with pytest.raises(RuntimeError):
        MinecraftLikeGridEnv(world_seed=7, world_bounds=(-20, -20, 20, 20)).reset(seed=0)


def test_free_roam_task_none():
    e = MinecraftLikeGridEnv(world_seed=7, task=None)
    obs, info = e.reset(seed=0)
    assert e.quest is None and not obs["goal"].any() and "goal" not in info
    _, r, terminated, _, _ = e.step(0)
    assert r == 0.0 and not terminated


def test_only_goal_portal_counts():
    """世界 5008 的出生点旁边有另一个要塞（离起点太近，被跳过不作为目标）：踩它不能算成功。"""
    e = MinecraftLikeGridEnv(world_seed=5008, goal_distance=150)
    e.reset(seed=0)
    others = [s for s in e.world.structures.strongholds() if s is not e.quest.goal
              and math.hypot(s.x - e.pos[0], s.z - e.pos[1]) < 60]
    assert others, "expected a non-goal stronghold near spawn"
    near = others[0]
    for dx, dz in e.moves:
        src = (near.x - dx, near.z - dz)
        if e.can_move(src, (near.x, near.z)):
            e.pos = src
            _, _, terminated, _, info = e.step(e.moves.index((dx, dz)))
            assert info["pos"] == (near.x, near.z) and not info["success"] and not terminated
            return
    pytest.fail("could not step onto the other portal")
