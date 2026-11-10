"""2.5D 网格 RL 环境：智能体在确定性生成的类 Minecraft 世界里移动，主线任务是找到末地传送门。

- 世界由 world_seed 唯一确定；reset(seed=s) 按 gymnasium 语义可复现
- 移动规则（仿 MC，定义在 worldgen/navigation.py）：上坡高差 > max_climb 不能走（只能跳 1 格），
  下坡任意；水面可游泳；树、仙人掌、房屋不可穿过
- 观测：以智能体为中心的 (2r+1)×(2r+1) 局部窗口（地表、物体、群系、相对高度）
  + goal：末影之眼式的指向 [dir_x, dir_z, 归一化距离]
- 主线（task="end_portal"）：终点是离起点最近且可达的要塞传送门，约在 goal_distance 方块外；
  踩进传送门 → terminated，获得 success_reward
- 金币（coins=True）：沿近似最优路线每隔 coin_spacing 步放一枚（上坡路段加密为一半），每枚最多绕远 coin_max_detour 步，
  捡到 +coin_reward，每回合每枚只能捡一次；
  观测 coins 通道标出视野内尚未捡起的金币
- 奖励 = success_reward·[到达] + coin_reward·[捡到金币] + 势函数塑形 shaping_scale·(γΦ(s')−Φ(s)) − step_penalty，Φ = −到传送门的最短步数。
  shaping_scale 默认 1/最优步数，γ 默认 1：整条最短路的塑形总和恰为 +1，原地不动为 0。
  若要严格满足折扣意义下的策略不变性，把 gamma 设成智能体的折扣因子。传入 reward_fn 则完全由它决定
- 陷阱（考验应变能力）：
  - 摔落伤害（fall_damage=True）：同 MC，下落超过 safe_fall(3) 格扣 (下落−3) 点血，落进水里不扣；生命值 max_health(20)
  - 岩浆池（lava_density>0，每 100 步路线约几个池）：踩进去即死。hazard_layout="episode" 时每回合重新随机布置，
    "world" 时由世界与起点决定；布置保证存在一条不踩岩浆、摔伤总和 < 生命值的路线（worldgen/hazards.py）
  - 死亡 → terminated，奖励 −death_penalty；观测 terrain 里岩浆显示为 Tile.LAVA，health 给出剩余生命比例
  - 死亡时塑形项取死亡位置的势函数而不是 0，否则"离终点越远死得越赚"
- 多世界训练：world_seeds=[...] 时每回合从列表里按 np_random 抽一个世界（世界对象缓存复用）
"""

from __future__ import annotations

import dataclasses
import math
from collections import OrderedDict
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from worldgen import Biome, Obj, Tile, World, WorldConfig
from worldgen.hazards import place_lava
from worldgen.navigation import MOVES_4, MOVES_8, SAFE_FALL, can_step, fall_damage, min_damage_to_targets
from worldgen.quest import EndPortalQuest, place_coins, plan_end_portal_quest, stronghold_distance_for
from worldgen.render import terrain_rgb
from worldgen.rng import hash_ints, normalize_seed

HEIGHT_CLIP = 64.0
GOAL_DISTANCE_NORM = 1024.0  # 观测中距离的归一化尺度

RewardFn = Callable[["MinecraftLikeGridEnv", tuple[int, int], tuple[int, int], dict], float]


class MinecraftLikeGridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        world_seed: int | str | None = None,
        world_config: WorldConfig | None = None,
        task: str | None = "end_portal",
        goal_distance: float | None = None,
        goal_hint: str = "direction",
        view_radius: int = 7,
        max_steps: int | None = None,
        world_bounds: tuple[int, int, int, int] | None = None,
        diagonal: bool = False,
        max_climb: int = 1,
        success_reward: float = 1.0,
        reward_shaping: str = "geodesic",
        shaping_scale: float | None = None,
        gamma: float = 1.0,
        step_penalty: float = 0.0,
        coins: bool = True,
        coin_spacing: int = 24,
        coin_reward: float = 0.1,
        coin_max_detour: int = 4,
        fall_damage: bool = False,
        max_health: int = 20,
        safe_fall: int = SAFE_FALL,
        lava_density: float = 0.0,
        lava_pool_size: tuple[int, int] = (3, 10),
        hazard_layout: str = "episode",
        death_penalty: float = 1.0,
        world_seeds: list | None = None,
        world_cache_size: int = 64,
        render_mode: str | None = None,
        reward_fn: RewardFn | None = None,
    ):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        assert task in (None, "end_portal"), f"unknown task {task!r}"
        assert goal_hint in ("none", "direction", "direction_distance")
        assert reward_shaping in ("none", "geodesic")
        assert hazard_layout in ("episode", "world")
        config = world_config or WorldConfig()
        if goal_distance is not None:
            sh = dataclasses.replace(config.stronghold,
                                     distance=stronghold_distance_for(goal_distance, config.chunk_size))
            config = dataclasses.replace(config, stronghold=sh)
        self.fixed_world_seed = world_seed
        self.world_config = config
        self.task = task
        self.goal_hint = goal_hint
        self.view_radius = view_radius
        self.max_steps_arg = max_steps
        self.world_bounds = world_bounds  # (x0, z0, x1, z1)，None 表示无限世界
        self.max_climb = max_climb
        self.success_reward = success_reward
        self.reward_shaping = reward_shaping
        self.shaping_scale = shaping_scale
        self.gamma = gamma
        self.step_penalty = step_penalty
        self.use_coins = coins
        self.coin_spacing = coin_spacing
        self.coin_reward = coin_reward
        self.coin_max_detour = coin_max_detour
        self.fall_damage = fall_damage
        self.max_health = max_health
        self.safe_fall = safe_fall
        self.lava_density = lava_density
        self.lava_pool_size = lava_pool_size
        self.hazard_layout = hazard_layout
        self.death_penalty = death_penalty
        self.world_seeds = list(world_seeds) if world_seeds is not None else None
        self.world_cache_size = world_cache_size
        self.render_mode = render_mode
        self.reward_fn = reward_fn
        self.moves = MOVES_8 if diagonal else MOVES_4

        size = 2 * view_radius + 1
        self.action_space = spaces.Discrete(len(self.moves))
        self.observation_space = spaces.Dict(
            {
                "terrain": spaces.Box(0, len(Tile) - 1, (size, size), np.uint8),
                "objects": spaces.Box(0, len(Obj) - 1, (size, size), np.uint8),
                "biome": spaces.Box(0, len(Biome) - 1, (size, size), np.uint8),
                "height": spaces.Box(-HEIGHT_CLIP, HEIGHT_CLIP, (size, size), np.float32),
                "coins": spaces.Box(0, 1, (size, size), np.uint8),
                "goal": spaces.Box(-1.0, 1.0, (3,), np.float32),
                "health": spaces.Box(0.0, 1.0, (1,), np.float32),
            }
        )

        self.world: World | None = None
        self._worlds: OrderedDict[int, World] = OrderedDict()
        self.lava: set[tuple[int, int]] = set()
        self.health = max_health
        self.quest: EndPortalQuest | None = None
        self._quests: OrderedDict[tuple, tuple[EndPortalQuest, list] | None] = OrderedDict()
        self.coin_layout: list[tuple[int, int]] = []
        self.coins: set[tuple[int, int]] = set()
        self.coins_total = 0
        self.pos = (0, 0)
        self.steps = 0
        self.max_steps = max_steps or 1000
        self._fig = None

    # ---- 世界与地形查询 ---------------------------------------------------
    def _ensure_world(self, world_seed: int | str) -> None:
        """同一个世界种子复用 World（保留区块缓存）。"""
        seed = normalize_seed(world_seed)
        if self.world is not None and self.world.seed == seed:
            return
        world = self._worlds.pop(seed, None) or World(seed, self.world_config)
        self._worlds[seed] = world
        if len(self._worlds) > self.world_cache_size:
            self._worlds.popitem(last=False)
        self.world = world

    def _in_bounds(self, x: int, z: int) -> bool:
        if self.world_bounds is None:
            return True
        x0, z0, x1, z1 = self.world_bounds
        return x0 <= x < x1 and z0 <= z < z1

    def _cell(self, x: int, z: int):
        return self.world.get_area(x, z, 1, 1)

    def stand_height(self, x: int, z: int) -> int:
        """站立高度：水面按海平面计（游泳）。"""
        return max(int(self._cell(x, z).height[0, 0]), self.world_config.sea_level)

    def can_move(self, src: tuple[int, int], dst: tuple[int, int]) -> bool:
        if not self._in_bounds(*dst):
            return False
        obj = int(self._cell(*dst).objects[0, 0])
        return can_step(self.stand_height(*src), self.stand_height(*dst), obj, self.max_climb)

    # ---- 主线 ----------------------------------------------------------------
    def _plan_quest(self, start: tuple[int, int]) -> tuple[EndPortalQuest, list] | None:
        """规划终点与金币布局；只依赖 (世界, 起点, 规则)，结果缓存。"""
        key = (self.world.seed, start, self.max_climb, self.moves, self.world_bounds,
               self.use_coins, self.coin_spacing, self.coin_max_detour,
               self.fall_damage, self.max_health, self.safe_fall)
        if key not in self._quests:
            quest = plan_end_portal_quest(self.world, start, self.max_climb, self.moves, bounds=self.world_bounds)
            if quest is not None and self.fall_damage and not self._survivable(quest, np.zeros_like(quest.free)):
                quest = None  # 连一条摔不死的路都没有：当作不可达
            coins = []
            if quest is not None and self.use_coins:
                coins = place_coins(quest, self.world.seed, self.coin_spacing, self.coin_max_detour)
            self._quests[key] = None if quest is None else (quest, coins)
            if len(self._quests) > 64:
                self._quests.popitem(last=False)
        return self._quests[key]

    def _survivable(self, quest: EndPortalQuest, lava: np.ndarray) -> bool:
        dmg = min_damage_to_targets(quest.stand, quest.free, quest.water, lava, quest.portal_mask(),
                                    quest.max_climb, quest.moves, safe_fall=self.safe_fall)
        return dmg[quest.start[1] - quest.box[1], quest.start[0] - quest.box[0]] < self.max_health

    def _place_hazards(self) -> set[tuple[int, int]]:
        if self.quest is None or self.lava_density <= 0:
            return set()
        n_pools = int(round(self.lava_density * self.quest.optimal_length / 100.0))
        if self.hazard_layout == "episode":
            seed = int(self.np_random.integers(0, 2**63 - 1))
        else:
            seed = hash_ints(self.world.seed, *self.quest.start)
        return place_lava(self.quest, seed, n_pools, self.coin_layout, *self.lava_pool_size,
                          max_health=self.max_health if self.fall_damage else None, safe_fall=self.safe_fall)

    def at_goal(self) -> bool:
        """只有目标要塞的传送门算到达——附近别的要塞（离起点太近被跳过的）不算，否则会被抄近路。"""
        return self.quest is not None and self.pos in self._goal_cells

    @property
    def _goal_cells(self) -> frozenset:
        if getattr(self, "_goal_cells_for", None) is not self.quest:
            self._goal_cells_cache = frozenset((x, z) for x, z, o in self.quest.goal.cells if o == Obj.PORTAL)
            self._goal_cells_for = self.quest
        return self._goal_cells_cache

    def _potential(self, pos: tuple[int, int]) -> float:
        d = self.quest.distance(*pos)
        if not math.isfinite(d):  # 离开距离场范围或掉进爬不出来的坑：退化为直线距离
            gx, gz = self.quest.goal_pos
            d = math.hypot(gx - pos[0], gz - pos[1]) + self.quest.optimal_length
        return -d

    # ---- gymnasium API ----------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        if "world_seed" in options:
            seeds = [options["world_seed"]]
        elif self.fixed_world_seed is not None:
            seeds = [self.fixed_world_seed]
        elif self.world_seeds is not None:  # 世界池：主线不可达就从池里重抽（最多 8 次）
            seeds = (self.world_seeds[int(self.np_random.integers(len(self.world_seeds)))] for _ in range(8))
        else:  # 随机世界：若主线不可达就换一个世界（最多 8 次），仍由 np_random 决定，可复现
            seeds = (int(self.np_random.integers(0, 2**63 - 1)) for _ in range(8))

        self.quest = None
        planned = None
        for world_seed in seeds:
            self._ensure_world(world_seed)
            self.pos = tuple(options.get("start", self.world.find_spawn()))
            if self.task is None:
                break
            planned = self._plan_quest(self.pos)
            if planned is not None:
                break
        self.quest = planned[0] if planned else None
        self.coin_layout = list(planned[1]) if planned else []  # 本世界本起点的全部金币，按进度排序
        self.coins = set(self.coin_layout)  # 本回合尚未捡起的金币
        self.coins_total = len(self.coin_layout)
        self.health = self.max_health
        if self.task is not None and self.quest is None:
            raise RuntimeError(
                f"no reachable end portal from {self.pos} in world {self.world.seed} "
                f"(bounds={self.world_bounds}); try another world_seed/start or a larger max_climb"
            )

        self.lava = self._place_hazards()
        self.steps = 0
        if self.max_steps_arg is not None:
            self.max_steps = self.max_steps_arg
        elif self.quest is not None:
            self.max_steps = max(500, 4 * self.quest.optimal_length)
        else:
            self.max_steps = 1000
        obs = self._observe()
        info = self._info(moved=False, success=False, coin=False, damage=0, death=None)
        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(self, action):
        assert self.action_space.contains(action), f"invalid action {action!r}"
        dx, dz = self.moves[int(action)]
        prev = self.pos
        dst = (prev[0] + dx, prev[1] + dz)
        moved = self.can_move(prev, dst)
        damage, death = 0, None
        if moved:
            if self.fall_damage:
                into_water = int(self._cell(*dst).height[0, 0]) < self.world_config.sea_level
                damage = fall_damage(self.stand_height(*prev) - self.stand_height(*dst), into_water, self.safe_fall)
            self.pos = dst
        self.steps += 1
        if self.pos in self.lava:
            damage, death = self.health, "lava"
        self.health = max(0, self.health - damage)
        if self.health <= 0 and death is None:
            death = "fall"

        coin = death is None and self.pos in self.coins
        if coin:
            self.coins.discard(self.pos)
        success = death is None and self.quest is not None and self.at_goal()
        info = self._info(moved=moved, success=success, coin=coin, damage=damage, death=death)
        if self.reward_fn is not None:
            reward = float(self.reward_fn(self, prev, self.pos, info))
        elif self.quest is not None:
            reward = -self.step_penalty
            if coin:
                reward += self.coin_reward
            if success:
                reward += self.success_reward
            if death is not None:
                reward -= self.death_penalty
            if self.reward_shaping == "geodesic":
                scale = self.shaping_scale if self.shaping_scale is not None else 1.0 / max(1, self.quest.optimal_length)
                next_phi = 0.0 if success else self._potential(self.pos)  # 死亡不归零，见模块说明
                reward += scale * (self.gamma * next_phi - self._potential(prev))
        else:
            reward = 0.0
        terminated = success or death is not None
        truncated = not terminated and self.steps >= self.max_steps
        if self.render_mode == "human":
            self.render()
        return self._observe(), reward, terminated, truncated, info

    def _window(self):
        r = self.view_radius
        return self.world.get_area(self.pos[0] - r, self.pos[1] - r, 2 * r + 1, 2 * r + 1)

    def _goal_obs(self) -> np.ndarray:
        out = np.zeros(3, dtype=np.float32)
        if self.quest is None or self.goal_hint == "none":
            return out
        gx, gz = self.quest.goal_pos
        dx, dz = gx - self.pos[0], gz - self.pos[1]
        d = math.hypot(dx, dz)
        if d > 0:
            out[0], out[1] = dx / d, dz / d
        if self.goal_hint == "direction_distance":
            out[2] = min(d / GOAL_DISTANCE_NORM, 1.0)
        return out

    def _coin_mask(self, x0: int, z0: int, shape: tuple[int, int]) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        for x, z in self.coins:
            i, j = z - z0, x - x0
            if 0 <= i < shape[0] and 0 <= j < shape[1]:
                mask[i, j] = 1
        return mask

    def _lava_mask(self, x0: int, z0: int, shape: tuple[int, int]) -> np.ndarray:
        mask = np.zeros(shape, dtype=bool)
        if not self.lava:
            return mask
        for x, z in self.lava:
            i, j = z - z0, x - x0
            if 0 <= i < shape[0] and 0 <= j < shape[1]:
                mask[i, j] = True
        return mask

    def _observe(self) -> dict[str, np.ndarray]:
        area = self._window()
        sea = self.world_config.sea_level
        stand = np.maximum(area.height.astype(np.float32), sea)
        rel = np.clip(stand - self.stand_height(*self.pos), -HEIGHT_CLIP, HEIGHT_CLIP).astype(np.float32)
        terrain = area.terrain.astype(np.uint8)
        terrain[self._lava_mask(area.x0, area.z0, area.shape)] = Tile.LAVA
        obs = {
            "terrain": terrain,
            "objects": area.objects.astype(np.uint8),
            "biome": area.biome.astype(np.uint8),
            "height": rel,
            "coins": self._coin_mask(area.x0, area.z0, area.shape),
            "goal": self._goal_obs(),
            "health": np.array([self.health / self.max_health], dtype=np.float32),
        }
        if self.world_bounds is not None:
            r = self.view_radius
            xs = np.arange(self.pos[0] - r, self.pos[0] + r + 1)
            zs = np.arange(self.pos[1] - r, self.pos[1] + r + 1)
            x0, z0, x1, z1 = self.world_bounds
            outside = ~(((zs >= z0) & (zs < z1))[:, None] & ((xs >= x0) & (xs < x1))[None, :])
            obs["terrain"][outside] = Tile.VOID
            obs["objects"][outside] = Obj.NONE
            obs["biome"][outside] = 0
            obs["height"][outside] = 0.0
        return obs

    def _info(self, moved: bool, success: bool, coin: bool, damage: int, death: str | None) -> dict[str, Any]:
        x, z = self.pos
        col = self._cell(x, z)
        info = {
            "pos": self.pos,
            "height": int(col.height[0, 0]),
            "biome": Biome(int(col.biome[0, 0])).name.lower(),
            "world_seed": self.world.seed,
            "moved": moved,
            "steps": self.steps,
            "structures_in_view": [(s.kind, s.x, s.z) for s in self._window().structures],
            "success": success,
            "health": self.health,
            "damage": damage,
            "death": death,  # None / "lava" / "fall"
            "lava_cells": len(self.lava),
            "coin_collected": coin,
            "coins_collected": self.coins_total - len(self.coins),
            "coins_total": self.coins_total,
            # 第 k 位 = coin_layout[k] 已被捡起。(pos, coins_mask) 才是完整的马尔可夫状态
            "coins_mask": sum(1 << k for k, c in enumerate(self.coin_layout) if c not in self.coins),
        }
        if self.quest is not None:
            gx, gz = self.quest.goal_pos
            info.update(
                goal=self.quest.goal_pos,
                goal_distance=math.hypot(gx - x, gz - z),  # 直线距离
                goal_path_length=self.quest.distance(x, z),  # 最短步数（inf = 当前位置已走不到）
                optimal_length=self.quest.optimal_length,  # 本回合起点的最短步数，用于 SPL 等评估
            )
        return info

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("render() called without render_mode")
            return None
        area = self._window()
        img = terrain_rgb(area)
        img[self._lava_mask(area.x0, area.z0, area.shape)] = (235, 95, 15)
        img[self._coin_mask(area.x0, area.z0, area.shape) == 1] = (255, 215, 0)
        r = self.view_radius
        img[r, r] = (255, 0, 0)
        img = np.repeat(np.repeat(img, 16, axis=0), 16, axis=1)
        if self.quest is not None:  # 在边缘画一个指向终点的标记（末影之眼）
            g = self._goal_obs()
            c = (2 * r + 1) * 8
            px = int(c + g[0] * (c - 6))
            pz = int(c + g[1] * (c - 6))
            img[max(pz - 4, 0):pz + 4, max(px - 4, 0):px + 4] = (160, 40, 200)
        if self.render_mode == "rgb_array":
            return img
        import matplotlib.pyplot as plt

        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(5, 5))
            self._im = self._ax.imshow(img)
            self._ax.axis("off")
        else:
            self._im.set_data(img)
        self._ax.set_title(f"{self.pos}  {Biome(int(self._cell(*self.pos).biome[0, 0])).name.lower()}")
        self._fig.canvas.draw_idle()
        plt.pause(1.0 / self.metadata["render_fps"])
        return None

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None
