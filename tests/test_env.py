import gymnasium
import numpy as np
from gymnasium.utils.env_checker import check_env

import envs  # noqa: F401  注册环境
from envs import MinecraftLikeGridEnv
from worldgen.tiles import Tile


def test_check_env():
    check_env(MinecraftLikeGridEnv(), skip_render_check=True)


def test_make_and_render():
    env = gymnasium.make("MinecraftLikeGrid-v0", world_seed=7, render_mode="rgb_array")
    env.reset(seed=0)
    img = env.render()
    assert img.ndim == 3 and img.shape[2] == 3
    env.close()


def _rollout(seed, n=60):
    env = MinecraftLikeGridEnv()
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(0)
    traj = [(info["world_seed"], info["pos"])]
    for _ in range(n):
        obs, *_rest, info = env.step(int(rng.integers(4)))
        traj.append((info["pos"], obs["terrain"].tobytes(), obs["height"].tobytes()))
    return traj


def test_reset_seed_reproducible():
    assert _rollout(3) == _rollout(3)
    assert _rollout(3)[0] != _rollout(4)[0]


def test_world_seed_option_overrides():
    env = MinecraftLikeGridEnv()
    _, info = env.reset(seed=1, options={"world_seed": 12345})
    assert info["world_seed"] == 12345


def test_climb_rule():
    env = MinecraftLikeGridEnv(world_seed=7, task=None)
    env.reset(seed=0)
    world = env.world
    x0, z0 = world.find_spawn()
    a = world.generate_area(x0 - 300, z0 - 300, 600, 600)
    sea = world.config.sea_level
    stand = np.maximum(a.height.astype(int), sea)
    diff = stand[:, 1:] - stand[:, :-1]  # 向东走的高差
    free = a.objects[:, 1:] == 0
    zs, xs = np.nonzero((diff >= 2) & free)
    assert len(zs) > 0
    src = (a.x0 + int(xs[0]), a.z0 + int(zs[0]))
    env.reset(seed=0, options={"start": src})
    _, _, _, _, info = env.step(1)  # 东
    assert info["moved"] is False and info["pos"] == src
    zs, xs = np.nonzero((np.abs(diff) <= 1) & free)
    src = (a.x0 + int(xs[0]), a.z0 + int(zs[0]))
    env.reset(seed=0, options={"start": src})
    _, _, _, _, info = env.step(1)
    assert info["moved"] is True


def test_world_bounds():
    env = MinecraftLikeGridEnv(world_seed=7, world_bounds=(-10, -10, 10, 10), view_radius=5, task=None)
    obs, info = env.reset(seed=0, options={"start": (-10, 0)})
    assert (obs["terrain"][:, :5] == Tile.VOID).all()
    _, _, _, _, info = env.step(3)  # 西：出界
    assert info["pos"] == (-10, 0) and not info["moved"]


def test_reward_fn_hook():
    calls = []

    def reward(env, prev, pos, info):
        calls.append((prev, pos))
        return 1.0 if info["moved"] else -1.0

    env = MinecraftLikeGridEnv(world_seed=7, reward_fn=reward, max_steps=3)
    env.reset(seed=0)
    rewards, truncated = [], False
    while not truncated:
        _, r, _, truncated, _ = env.step(0)
        rewards.append(r)
    assert len(rewards) == 3 and len(calls) == 3 and set(rewards) <= {1.0, -1.0}
