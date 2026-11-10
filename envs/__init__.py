import gymnasium

from .grid_env import MinecraftLikeGridEnv

if "MinecraftLikeGrid-v0" not in gymnasium.registry:
    gymnasium.register(id="MinecraftLikeGrid-v0", entry_point="envs.grid_env:MinecraftLikeGridEnv")

__all__ = ["MinecraftLikeGridEnv"]
