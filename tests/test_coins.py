"""金币（中间奖励）：位置随机但受规则约束，每回合每枚只能捡一次。"""

import numpy as np
import pytest

from agents import TabularQAgent, run_episode
from envs import MinecraftLikeGridEnv
from worldgen.navigation import distance_from_source
from worldgen.quest import route_through
from worldgen.tiles import Obj

SPACING, DETOUR = 12, 4


@pytest.fixture(scope="module", params=[7, 12345, 2024])
def env(request):
    e = MinecraftLikeGridEnv(world_seed=request.param, goal_distance=150, coin_spacing=SPACING,
                             coin_max_detour=DETOUR, reward_shaping="none")
    e.reset(seed=0)
    return e


def _d_from(q, pos):
    x0, z0 = q.box[0], q.box[1]
    return distance_from_source(q.stand, q.free, (pos[1] - z0, pos[0] - x0), q.max_climb, q.moves)


def test_coins_deterministic(env):
    other = MinecraftLikeGridEnv(world_seed=env.world.seed, goal_distance=150, coin_spacing=SPACING,
                                 coin_max_detour=DETOUR)
    other.reset(seed=123)
    assert other.coin_layout == env.coin_layout and len(env.coin_layout) > 0


def test_coin_rules(env):
    q = env.quest
    x0, z0 = q.box[0], q.box[1]
    d_start = _d_from(q, q.start)
    L = q.optimal_length
    prev = q.start
    for x, z in env.coin_layout:
        i, j = z - z0, x - x0
        assert q.free[i, j] and q.objects[i, j] == Obj.NONE and (x, z) != q.start
        assert d_start[i, j] + q.dist[i, j] <= L + DETOUR  # 规则 1：走廊
        assert _d_from(q, prev)[i, j] <= SPACING + DETOUR  # 规则 3：连贯
        prev = (x, z)


def test_route_through_coins_bound(env):
    """按顺序捡完全部金币再进传送门：全部捡到、成功、总步数 ≤ L + (n+1)·detour + n。"""
    env.reset(seed=0)
    q = env.quest
    route = route_through(q, env.coin_layout)
    action_of = {m: a for a, m in enumerate(env.moves)}
    total, info = 0.0, {}
    for (ax, az), (bx, bz) in zip(route, route[1:]):
        _, r, terminated, _, info = env.step(action_of[(bx - ax, bz - az)])
        total += r
    n = len(env.coin_layout)
    assert info["success"] and info["coins_collected"] == n
    assert info["coins_mask"] == (1 << n) - 1
    assert len(route) - 1 <= q.optimal_length + (n + 1) * DETOUR + n
    assert total == pytest.approx(env.success_reward + n * env.coin_reward)


def test_coin_collected_once_and_observed():
    e = MinecraftLikeGridEnv(world_seed=7, goal_distance=150, coin_spacing=SPACING, reward_shaping="none")
    obs, _ = e.reset(seed=0)
    route = route_through(e.quest, e.coin_layout[:1])
    action_of = {m: a for a, m in enumerate(e.moves)}
    rewards = []
    for (ax, az), (bx, bz) in zip(route, route[1:]):
        obs, r, _, _, info = e.step(action_of[(bx - ax, bz - az)])
        rewards.append(r)
        if info["coin_collected"]:
            break
    assert rewards[-1] == pytest.approx(e.coin_reward) and sum(rewards[:-1]) == 0
    r = e.view_radius
    assert obs["coins"][r, r] == 0  # 脚下的金币已消失
    coin = e.pos
    for a, (dx, dz) in enumerate(e.moves):  # 找一个能走过去再走回来的邻格
        nb = (coin[0] + dx, coin[1] + dz)
        if e.can_move(coin, nb) and e.can_move(nb, coin):
            e.step(a)
            _, r2, _, _, info = e.step(action_of[(-dx, -dz)])
            assert info["pos"] == coin and r2 == 0.0
            assert not info["coin_collected"] and info["coins_collected"] == 1
            break
    else:
        pytest.fail("coin cell has no two-way neighbour")
    obs, _ = e.reset(seed=0)  # 新回合金币重新出现
    assert len(e.coins) == len(e.coin_layout)


def test_coins_visible_in_window():
    """coins 观测通道与真实金币位置逐格一致；沿路走向第一枚金币时它会出现在视野中。"""
    e = MinecraftLikeGridEnv(world_seed=7, goal_distance=150, coin_spacing=SPACING)
    obs, _ = e.reset(seed=0)
    r = e.view_radius
    route = route_through(e.quest, e.coin_layout[:1])
    action_of = {m: a for a, m in enumerate(e.moves)}
    seen = False
    for (ax, az), (bx, bz) in zip(route, route[1:]):
        expected = np.zeros_like(obs["coins"])
        for x, z in e.coins:
            if abs(x - e.pos[0]) <= r and abs(z - e.pos[1]) <= r:
                expected[z - e.pos[1] + r, x - e.pos[0] + r] = 1
        np.testing.assert_array_equal(obs["coins"], expected)
        seen |= bool(expected.any())
        obs, *_ = e.step(action_of[(bx - ax, bz - az)])
    assert seen


def test_uphill_segments_get_denser_coins():
    """世界 2024 的路线中段爬一座陡坡：那一段的金币间距应明显小于平地段。"""
    e = MinecraftLikeGridEnv(world_seed=2024, goal_distance=150, coin_spacing=SPACING)
    e.reset(seed=0)
    q = e.quest
    progress = [q.optimal_length - q.distance(*c) for c in e.coin_layout]
    gaps = np.diff([0] + progress)
    assert gaps.min() <= SPACING // 2 + 1 and np.median(gaps) >= SPACING - 2


def test_no_coins_when_disabled():
    e = MinecraftLikeGridEnv(world_seed=7, coins=False)
    obs, info = e.reset(seed=0)
    assert e.coin_layout == [] and not obs["coins"].any() and info["coins_total"] == 0


def test_qlearning_agent_smoke():
    e = MinecraftLikeGridEnv(world_seed=12345, goal_distance=150, coin_spacing=SPACING,
                             reward_shaping="none", max_steps=200)
    agent = TabularQAgent(e.action_space.n, seed=0)
    stats = [run_episode(e, agent, epsilon=1.0, seed=0) for _ in range(5)]
    assert all(s["steps"] == 200 or s["success"] for s in stats)
    assert len(agent.q) > 0
    greedy_a = run_episode(e, agent, epsilon=0.0, learn=False, seed=0)
    greedy_b = run_episode(e, agent, epsilon=0.0, learn=False, seed=0)
    assert len(greedy_a["path"]) > 1 and greedy_b["steps"] > 0
