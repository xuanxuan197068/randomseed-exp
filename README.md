# randomseed-exp：类 Minecraft 的可复现世界生成器 + RL 网格环境

同一个种子永远生成同一个世界；不同种子的世界各不相同；但每个世界都满足一组硬规则
（群系过渡合理、结构有最小间距且只出现在合法地形、树不长在水里、出生点在陆地……）。

## Minecraft 如何兼顾"随机"与"规则"

| 层次 | 做法 | 随机性来自 | 规则性来自 | 本项目对应 |
|---|---|---|---|---|
| 种子 | 所有随机数都由 `hash(世界种子, 坐标, salt)` 派生，没有全局随机状态 | 种子 | 纯函数 → 与生成顺序无关、可无限延伸 | `worldgen/rng.py`（Xoroshiro128++、位置哈希） |
| 相干噪声 | 多倍频 Perlin（fBm），两路叠加 | 种子打乱梯度表 | 连续平滑，相邻格数值相近 | `worldgen/noise.py` |
| 气候参数 | 温度、湿度、大陆性、侵蚀、怪异度 5 个独立噪声 + 峰谷 PV | 每个参数独立 salt | 群系按参数空间查表；温度连续 ⇒ 冷热群系不会相邻 | `worldgen/climate.py`、`worldgen/biomes.py` |
| 地形形状 | 嵌套三次样条：大陆性 → 侵蚀 → 峰谷 → 高度 | 输入是噪声 | 样条节点是写死的设计：内陆高、低侵蚀多山、谷底低 | `worldgen/spline.py` |
| 河流 | 怪异度 ≈ 0 的谷底开凿 | 噪声零等值线的位置 | 等值线连续 ⇒ 河流成带、流向海岸 | `TerrainGenerator.river_strength` |
| 表面规则 | 有序 if-else（水下 → 海滩 → 雪线 → 陡坡 → 群系顶层） | 无 | 完全确定 | `default_surface_rules` |
| 结构 | 网格分区 + 区内随机偏移（spacing/separation）+ 群系/地形校验 + 互斥距离；要塞用同心环 | 区内偏移 | 最小间距、合法群系、平坦陆地、固定数量 | `worldgen/structures.py` |
| 装饰 | 每区块独立随机源 + 放置修饰链（次数 → 位置 → 群系概率 → 地表 → 坡度 → 间距） | 位置与概率 | 只放在合法地表；仙人掌四周必须空 | `worldgen/features.py` |
| 出生点 | 从原点向外找最近的合适陆地 | — | 必在陆地 | `World.find_spawn` |
| 主线终点 | 要塞按同心环分布在出生点周围；末影之眼指向最近的要塞，里面有末地传送门 | 环上的角度、半径抖动 | 数量固定、距离有界、建在平坦陆地；**本项目额外保证按移动规则可达** | `worldgen/quest.py`、`worldgen/navigation.py` |

一句话：**随机性只来自"种子 + 坐标"的哈希；规则性来自"连续噪声 + 参数空间映射 + 写死的样条/规则表 + 网格化放置约束"**。
每一步都是世界坐标的纯函数，所以可复现，并且与区块加载顺序无关。

## 安装

系统 Python 是 externally-managed，项目使用继承系统包的虚拟环境：

```bash
python3 -m venv --system-site-packages .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

### 在新机器 / GPU 服务器上

`requirements-lock.txt` 固定了与开发机相同的版本，`scripts/setup_env.sh` 用 uv 建 Python 3.12 的 `.venv` 并装 CUDA 12.6 版 torch
（NVIDIA 驱动 ≥ 525 即可；纯 CPU 机器用 `TORCH_VARIANT=cpu`）：

```bash
bash scripts/setup_env.sh
```

装好后用 `scripts/check_repro.py` 确认同一种子在这台机器上生成的世界与开发机逐位相同（seed 12345 的摘要应为 `7abab20ff4e418c0`）。

注意：`agents/ppo.py` 目前全部在 CPU 上计算。这个任务的瓶颈是环境步进（Python 里的地形查询），而不是网络——
网络很小，GPU 帮助有限；服务器上更有用的是多开环境进程（`--num-envs`）。

## 用法

渲染一块区域（群系图 / 地表+结构 / 高度图）到 `out/`：

```bash
.venv/bin/python scripts/render_world.py --seed 12345 --size 1024
```

检查可复现性（同种子、乱序区块、新进程、不同种子）：

```bash
.venv/bin/python scripts/check_repro.py --seed 12345
```

运行测试：

```bash
.venv/bin/python -m pytest -q
```

### 作为库

```python
from worldgen import World, Biome

world = World("hello world")            # 整数或字符串种子（字符串规则同 MC）
area = world.generate_area(-256, -256, 512, 512)   # 一次性向量化生成
area.height, area.terrain, area.biome, area.objects, area.structures
chunk = world.get_chunk(3, -2)          # 16×16 区块（LRU 缓存）
world.find_spawn()
```

### 主线任务：找到末地传送门

默认任务 `task="end_portal"`，对应 MC 的主线"找要塞 → 进末地传送门"：

- **终点**：要塞的传送门房间（9×9 石砖地面，中央 3×3 末地传送门，周围 12 个框架，同 MC 布局）。
  要塞沿同心环分布，第一环 3 个，名义半径 `4 × StrongholdConfig.distance` 区块（默认 384 方块，±31% 随机）。
- **距离控制**：`goal_distance=600` 会把环半径调到约 600 方块；比名义半径一半还近的要塞会被跳过（出生点可能因原点是海而偏离原点）。
- **可达性保证**：按环境的移动规则（上坡 ≤ 1 格、障碍不可穿过）建有向图，在反向图上从传送门做 BFS。
  离起点最近的要塞走不到时自动换下一个（种子 11、18 就是这种情况）；随机世界下最多换 8 个世界，固定世界则报错。
- **观测 `goal`**：末影之眼式的提示 `[dir_x, dir_z, 距离/1024]`。`goal_hint="direction"`（默认）只给方向，`"direction_distance"` 加上距离，`"none"` 全为 0（纯探索）。
- **奖励**：踩进传送门 `+success_reward`（默认 1）并 `terminated=True`；默认还有基于**最短步数**的势函数塑形，
  整条最短路总和恰为 +1，撞墙或原地不动为 0；`reward_shaping="none"` 关闭，`step_penalty` 可加时间惩罚。
- **评估**：`info` 中有 `optimal_length`（本回合最短步数）、`goal_path_length`（当前位置的剩余最短步数）、`success`，可直接算 SPL。
- `max_steps` 默认 `max(500, 4 × 最短步数)`。

画出起点、终点和最短路径，并让"神谕"智能体沿最短路径走一遍：

```bash
.venv/bin/python scripts/quest_demo.py --seed 12345
```

```bash
.venv/bin/python scripts/quest_demo.py --seed 7 --goal-distance 800
```

`task=None` 回到无目标的自由漫游（奖励为 0，`goal` 观测全为 0）。

### 金币：缓解奖励稀疏的中间奖励

只有终点奖励时，几百步外的传送门几乎不可能被随机探索碰到。`coins=True`（默认）沿途放金币，
捡到 `+coin_reward`（默认 0.1），每回合每枚只能捡一次，观测 `coins` 通道标出视野内的金币。
金币位置随机，但受规则约束（`worldgen/quest.py: place_coins`）：

1. **走廊**：`d_start(c) + d_goal(c) ≤ L + coin_max_detour` —— 捡它最多绕远几步；它从起点可达、捡完一定还能走到传送门
2. **进度分层**：按路程每 `coin_spacing` 步一层，每层在所有合格格子里按种子随机选一枚（不偏向某一条最短路）
3. **上坡加密**：接下来一段若一半以上是上坡，间距减半 —— "下坡随意、上坡最多 1 格"的规则会把随机探索推回坡底，上坡处最需要引导
4. **连贯**：相邻金币不超过 `间距 + detour` 步，按顺序捡完全部金币的总路程 ≤ `L + (n+1)·detour + n`

`info` 里有 `coin_collected`、`coins_collected`、`coins_total`，以及 `coins_mask`（哪些金币已捡的位掩码）。
**表格类智能体请用 `(pos, coins_mask)` 作状态**：只用"捡了几枚"会让状态不满足马尔可夫性——某回合跳过一枚金币后，
同一个 (位置, 数量) 学到指向那枚金币的价值，之后它已不在，智能体会原地打转追一个不存在的奖励。

### 实验：表格 Q-learning，稀疏 vs 金币

`agents/qlearning.py` 是一个最简单的智能体：表格 Q-learning，状态 `(x, z, coins_mask)`，ε-greedy，
回合结束后倒序回放整条轨迹做更新。两组都关闭最短路塑形，只比较金币本身：

```bash
.venv/bin/python scripts/train_qlearning.py --seed 7 --episodes 600
```

3 个世界 × 3 个智能体种子，终点约 150 方块外，每组 600 回合：

| 世界 | 最短步数 | 只有终点奖励 | 有金币：贪心策略 |
|---|---|---|---|
| 7 | 134 | 0/3 成功（600 回合一次都没碰到终点） | 3/3 成功，152–168 步 |
| 12345 | 103 | 0/3 | 3/3 成功，118–151 步 |
| 2024（中途爬陡坡） | 211 | 0/3 | 3/3 成功，284–313 步 |

#### 金币能稀疏到什么程度？

```bash
.venv/bin/python scripts/sweep_coin_spacing.py --spacings 12 24 36 48 72 0 --episodes 600
```

扫描金币间距（3 个世界 × 3 个智能体种子；600 和 2000 回合各跑一遍）。决定成败的是**最难的那一段间隔**，而不是平均间距：

| 最难一段 | 600 回合能学会 | 2000 回合能学会 | 学不会 |
|---|---|---|---|
| 平地 | ≤ ~37 步 | ≤ ~54 步（世界 12345 只有 1 枚金币也学会了） | ≥ ~62 步 |
| 以上坡为主 | ≤ 8 步 | ≤ 8 步（12 步时 3 个种子里只有 1 个成功） | ≥ 18 步 |

原因在探索方式：ε-greedy 在没有奖励信号的区域就是随机游走，能"碰到"的距离有限；
而"下坡随意、上坡受限"的移动规则让随机游走往坡底漂，上坡能覆盖的距离小一个数量级。
所以金币可以稀疏，但平地间距别超过训练预算下的探索半径，上坡段要密得多——这正是 `place_coins` 上坡加密规则的依据。

表格智能体记住的是"这张地图"，换个世界要重新学；它用来验证奖励设计本身是否让任务可学。
要泛化到新世界，需要用观测（局部视野 + `goal` 指向 + `coins` 通道）训练神经网络策略（如 PPO + CNN）。

### 陷阱：岩浆与摔落伤害

默认关闭；打开后考验智能体看到危险时的应变能力：

```python
env = MinecraftLikeGridEnv(fall_damage=True, lava_density=4.0, hazard_layout="episode")
```

- **摔落伤害**（`fall_damage=True`）：同 MC，下落超过 3 格扣 `下落 − 3` 点血，落进水里不扣；生命值 20，归零即死。"下坡随意"不再免费，悬崖成了真正的危险。
- **岩浆池**（`lava_density`：每 100 步路线约几个池）：踩进去即死。地形由世界种子决定、永不改变，
  岩浆是回合级的一层：`hazard_layout="episode"` 时每回合按回合种子重新布置（智能体背不了答案），`"world"` 时固定。
  布置规则（`worldgen/hazards.py`）：
  1. 池心只在路线附近的带状区域内，离起点、传送门至少 6 步
  2. 从池心随机长出 3–10 格的不规则块；只铺在干燥陆地上，不压金币、树和结构；同 MC，岩浆不挨着水
  3. **保证有活路**：在"不踩岩浆"的图上按摔落伤害做 Dijkstra，到传送门的最小总伤害必须 < 生命值，否则逆序撤掉岩浆池
- 死亡 → `terminated`，奖励 `−death_penalty`（默认 1），`info["death"]` 为 `"lava"` 或 `"fall"`。
  开着最短路塑形时，死亡位置的势函数不归零——否则"离终点越远死得越赚"。
- 观测：岩浆在 `terrain` 里显示为 `Tile.LAVA`，新增 `health`（剩余生命比例）。
- 多世界训练：`world_seeds=[...]` 每回合从列表里抽一个世界，世界对象缓存复用。
- 只有**目标要塞**的传送门算到达。出生点附近可能有别的要塞（离起点太近被跳过），踩它不算——否则智能体会学会抄近路。

### 实验：PPO + CNN 面对随机岩浆的应变能力

```bash
.venv/bin/python scripts/train_ppo.py --steps 5000000     # 约 40 分钟（32 核 CPU）
```

```bash
.venv/bin/python scripts/eval_ppo.py --episodes 200 --worlds 20
```

训练：48 个世界，终点约 150 方块，每回合岩浆重新随机（`lava_density=4`），摔落伤害开启；奖励 = 成功 +1、金币 +0.1、
死亡 −1 + 最短路塑形。塑形用的最短路**无视岩浆**，一直把智能体往直线拉，绕开岩浆只能从死亡惩罚和观测里学。
不加塑形时，48 个世界下 PPO 前 60 万步成功率始终 ≈ 1%（随机策略几乎碰不到 24 步外的第一枚金币）；加上后 20 万步即达 90%。

评估（每格 200 回合 = 20 个世界 × 10 个**训练中从未出现过**的岩浆布局）：

| 测试集 | 只看指向（无视岩浆） | 手写规则（绝不踩岩浆） | PPO |
|---|---|---|---|
| A 训练世界·新岩浆 | 成功 65%，岩浆死 35% | 98%，0% | **100%，0%** |
| B 新世界·岩浆 | 72%，24% | 96%，0% | 94%，0% |
| C 新世界·2 倍岩浆（比训练更难） | 41%，56% | 92%，0% | 94%，0% |
| D 新世界·无岩浆 | 96%，0% | 96%，0% | 94%，0% |
| 路径效率 SPL（B / C） | 0.56 / 0.32 | 0.73 / 0.68 | **0.88 / 0.86** |

- **岩浆确实致命**：只看指向走，24%–56% 的回合死在岩浆里。
- **PPO 学会了应变**：在没见过的岩浆布局、没见过的世界、甚至 2 倍密度下，岩浆死亡都是 0%，且路径比手写规则短得多。
  更细的指标：在 B+C 的 400 回合里，"最贴合指向的那一步正好是岩浆"的局面，手写规则遇到 4278 次（反复在岩浆边打转），
  PPO 只遇到 11 次且全部避开——它用 15×15 视野**提前**绕开，而不是走到岩浆边才拐。
- **背地图不行**：表格 Q-learning 在世界 1000 上学 600 回合（每回合岩浆不同），新布局上成功 0%（岩浆死 24%，超时 76%）。
- **失败几乎都来自地形而不是岩浆**：新世界上 6% 超时，集中在少数世界、且同一世界不论岩浆怎么摆都卡在同一处：
  - 世界 5015：走进河谷，"下坡随意、上坡最多 1 格"让它下得去上不来——规则自然形成的陷阱；无记忆的策略找不到出口
  - 世界 5008：出生点旁有一个非目标的要塞，智能体在那附近徘徊——训练里"看到传送门 ≈ 快到终点"，诱饵让它困惑
- **摔落伤害在当前地形里几乎不起作用**：地形是平滑丘陵，任务区域内只有 0.01% 的相邻格高差 ≥ 4，最短路上最大落差 3 格，
  所有智能体的摔死率都是 0%。机制按 MC 规则实现并有测试，但要让它成为真正的挑战，需要地形里有悬崖或峡谷。

### 作为 RL 环境

```python
import gymnasium, envs   # import envs 会注册 MinecraftLikeGrid-v0

env = gymnasium.make("MinecraftLikeGrid-v0", goal_distance=400)
obs, info = env.reset(seed=0)                               # 世界种子由 seed 派生
obs, info = env.reset(seed=0, options={"world_seed": 12345}) # 或者指定世界
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

- 动作：`Discrete(4)` 北/东/南/西（`diagonal=True` 时 8 向）
- 观测：`terrain` / `objects` / `biome`（uint8）、相对高度 `height`（float32）、`coins`（uint8），均为 (2r+1)² 局部窗口；`goal`（3,) 指向终点；`health`（1,）剩余生命比例
- 移动规则：上坡高差 > `max_climb`(默认 1) 不能走；水面可游泳；树、仙人掌、房屋不可穿过；`world_bounds` 外不可走
- 奖励：见上方主线；传入 `reward_fn(env, prev_pos, pos, info) -> float` 可完全替换
- `info`：位置、高度、群系、世界种子、是否移动成功、视野内结构，以及主线的 `goal` / `goal_distance` / `goal_path_length` / `optimal_length` / `success`

## 与 Minecraft 的差异

- 2.5D 高度图，没有 3D 密度函数和洞穴
- 气候噪声波长约为 MC 的 1/3，结构间距按比例缩小（村庄 16/5、沙漠神殿 12/4 区块；MC 原版为 34/8、32/8，salt 保留原值），便于在 RL 尺寸的地图里看到多种群系与结构
- 群系表是 MC `OverworldBiomeBuilder` 的简化版（14 个群系）
- `continentalness_bias` 让陆地比 MC 多，便于智能体行走

所有参数都在 `worldgen/config.py` 的 `WorldConfig` 中，可以 `to_dict()` / `from_dict()` 存成 JSON。
同一个 (seed, WorldConfig) 生成同一个世界；改配置等于换了一套"物理规则"。
