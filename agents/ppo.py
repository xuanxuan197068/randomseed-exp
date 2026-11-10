"""PPO + CNN：看着局部观测做决策的智能体，用来测试面对随机陷阱的应变能力。

和表格 Q-learning 的根本区别：策略的输入是观测而不是坐标——
- 15×15 局部视野：地表 one-hot（含岩浆）、物体 one-hot、相对高度、金币
- 向量：末影之眼方向 goal(3)、生命值 health(1)
所以它能对**没见过的**岩浆布局、没见过的世界做出反应，而不是背地图。

实现参照 CleanRL 的 PPO（GAE、clip、价值损失、熵正则），向量环境用 gymnasium 的
AsyncVectorEnv（SAME_STEP 自动重置）；超时截断时用最终观测的价值自举，而不是当成终止。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from worldgen.tiles import Obj, Tile

N_TILE, N_OBJ = len(Tile), len(Obj)
IN_CHANNELS = N_TILE + N_OBJ + 2  # + 相对高度 + 金币
VEC_DIM = 4  # goal(3) + health(1)
HEIGHT_SCALE = 8.0  # 相对高度裁剪到 ±8 再归一化：能分清"能爬的 1 格"和"会摔伤的 4 格"


def encode(obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """批量观测 dict（每项第 0 维是 batch）→ (图像 [B,C,H,W], 向量 [B,4])。"""
    terrain = torch.as_tensor(np.asarray(obs["terrain"]), dtype=torch.long)
    objects = torch.as_tensor(np.asarray(obs["objects"]), dtype=torch.long)
    height = torch.as_tensor(np.asarray(obs["height"]), dtype=torch.float32)
    coins = torch.as_tensor(np.asarray(obs["coins"]), dtype=torch.float32)
    img = torch.cat([
        nn.functional.one_hot(terrain, N_TILE).permute(0, 3, 1, 2).float(),
        nn.functional.one_hot(objects, N_OBJ).permute(0, 3, 1, 2).float(),
        (height.clamp(-HEIGHT_SCALE, HEIGHT_SCALE) / HEIGHT_SCALE).unsqueeze(1),
        coins.unsqueeze(1),
    ], dim=1)
    vec = torch.cat([torch.as_tensor(np.asarray(obs["goal"]), dtype=torch.float32),
                     torch.as_tensor(np.asarray(obs["health"]), dtype=torch.float32)], dim=1)
    return img, vec


def stack_obs(list_of_obs: list[dict]) -> dict:
    return {k: np.stack([o[k] for o in list_of_obs]) for k in list_of_obs[0]}


def _layer(m: nn.Module, std: float = np.sqrt(2)) -> nn.Module:
    nn.init.orthogonal_(m.weight, std)
    nn.init.constant_(m.bias, 0.0)
    return m


class ActorCritic(nn.Module):
    def __init__(self, n_actions: int, view: int = 15):
        super().__init__()
        self.cnn = nn.Sequential(
            _layer(nn.Conv2d(IN_CHANNELS, 32, 3, padding=1)), nn.ReLU(),
            _layer(nn.Conv2d(32, 64, 3, stride=2, padding=1)), nn.ReLU(),
            _layer(nn.Conv2d(64, 64, 3, stride=2, padding=1)), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, IN_CHANNELS, view, view)).shape[1]
        self.trunk = nn.Sequential(_layer(nn.Linear(n_flat + VEC_DIM, 256)), nn.ReLU())
        self.pi = _layer(nn.Linear(256, n_actions), std=0.01)
        self.v = _layer(nn.Linear(256, 1), std=1.0)

    def forward(self, img: torch.Tensor, vec: torch.Tensor):
        h = self.trunk(torch.cat([self.cnn(img), vec], dim=1))
        return self.pi(h), self.v(h).squeeze(-1)

    def act(self, obs: dict, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            logits, _ = self(*encode(obs))
            if deterministic:
                return logits.argmax(-1).numpy()
            return torch.distributions.Categorical(logits=logits).sample().numpy()


@dataclass
class PPOConfig:
    total_steps: int = 5_000_000
    num_envs: int = 16
    num_steps: int = 256
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epochs: int = 4
    minibatches: int = 8
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 0
    torch_threads: int = 8


def train_ppo(make_env, cfg: PPOConfig, log=print, on_update=None) -> tuple[ActorCritic, list[dict]]:
    """make_env(i) → 第 i 个子环境的构造函数。返回训练好的网络与每个回合的统计。"""
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.torch_threads)
    envs = gym.vector.AsyncVectorEnv([make_env(i) for i in range(cfg.num_envs)],
                                     autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    n_actions = envs.single_action_space.n
    net = ActorCritic(n_actions, view=envs.single_observation_space["terrain"].shape[0])
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)

    T, N = cfg.num_steps, cfg.num_envs
    batch = T * N
    n_updates = cfg.total_steps // batch
    obs, _ = envs.reset(seed=cfg.seed)
    episodes: list[dict] = []
    ep_return = np.zeros(N)
    t0 = time.time()

    for update in range(n_updates):
        frac = 1.0 - update / n_updates
        for g in opt.param_groups:
            g["lr"] = cfg.lr * frac
        buf_obs, buf_act, buf_logp, buf_val, buf_rew, buf_done = [], [], [], [], [], []
        for _ in range(T):
            img, vec = encode(obs)
            with torch.no_grad():
                logits, value = net(img, vec)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            next_obs, reward, terminated, truncated, info = envs.step(action.numpy())
            reward = reward.astype(np.float64).copy()
            done = terminated | truncated
            if truncated.any():  # 截断：用最终观测的价值自举
                idx = np.flatnonzero(truncated & ~terminated)
                if len(idx):
                    final = stack_obs([info["final_obs"][k] for k in idx])
                    with torch.no_grad():
                        _, v_final = net(*encode(final))
                    reward[idx] += cfg.gamma * v_final.numpy()
            ep_return += reward
            for k in np.flatnonzero(done):
                fi = info["final_info"]
                episodes.append({
                    "success": bool(fi["success"][k]), "death": fi["death"][k],
                    "steps": int(fi["steps"][k]), "optimal": int(fi["optimal_length"][k]),
                    "coins": int(fi["coins_collected"][k]), "return": float(ep_return[k]),
                })
                ep_return[k] = 0.0
            buf_obs.append((img, vec))
            buf_act.append(action)
            buf_logp.append(dist.log_prob(action))
            buf_val.append(value)
            buf_rew.append(torch.as_tensor(reward, dtype=torch.float32))
            buf_done.append(torch.as_tensor(done, dtype=torch.float32))
            obs = next_obs

        with torch.no_grad():
            _, next_value = net(*encode(obs))
        values = torch.stack(buf_val)
        rewards = torch.stack(buf_rew)
        dones = torch.stack(buf_done)
        adv = torch.zeros_like(rewards)
        last = torch.zeros(N)
        for t in reversed(range(T)):
            nv = next_value if t == T - 1 else values[t + 1]
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + cfg.gamma * nv * nonterminal - values[t]
            last = delta + cfg.gamma * cfg.gae_lambda * nonterminal * last
            adv[t] = last
        returns = adv + values

        b_img = torch.cat([o[0] for o in buf_obs])
        b_vec = torch.cat([o[1] for o in buf_obs])
        b_act = torch.cat(buf_act)
        b_logp = torch.cat(buf_logp)
        b_adv = adv.reshape(-1)
        b_ret = returns.reshape(-1)
        mb = batch // cfg.minibatches
        for _ in range(cfg.epochs):
            perm = torch.randperm(batch)
            for s in range(0, batch, mb):
                i = perm[s:s + mb]
                logits, v = net(b_img[i], b_vec[i])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = (dist.log_prob(b_act[i]) - b_logp[i]).exp()
                a = b_adv[i]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg = torch.max(-a * ratio, -a * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                vl = 0.5 * ((v - b_ret[i]) ** 2).mean()
                loss = pg + cfg.vf_coef * vl - cfg.ent_coef * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                opt.step()

        steps = (update + 1) * batch
        recent = episodes[-200:]
        if recent and (update % 10 == 0 or update == n_updates - 1):
            sr = np.mean([e["success"] for e in recent])
            lava = np.mean([e["death"] == "lava" for e in recent])
            fall = np.mean([e["death"] == "fall" for e in recent])
            log(f"  steps {steps:9,d}  episodes {len(episodes):6d}  success {sr:.2f}  "
                f"lava {lava:.2f}  fall {fall:.2f}  return {np.mean([e['return'] for e in recent]):6.2f}  "
                f"({steps / (time.time() - t0):,.0f} steps/s)")
        if on_update is not None:
            on_update(update, net, episodes)
    envs.close()
    return net, episodes


def save(net: ActorCritic, cfg: PPOConfig, env_kwargs: dict, path: str) -> None:
    torch.save({"state_dict": net.state_dict(), "config": asdict(cfg), "env_kwargs": env_kwargs,
                "n_actions": net.pi.out_features}, path)


def load(path: str) -> tuple[ActorCritic, dict]:
    ckpt = torch.load(path, weights_only=False)
    view = 2 * ckpt["env_kwargs"].get("view_radius", 7) + 1
    net = ActorCritic(ckpt["n_actions"], view=view)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt
