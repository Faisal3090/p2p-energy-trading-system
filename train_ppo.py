"""PPO Training Script for KLS VDIT P2P Energy Trading Agent.

Trains two PPO agents using stable-baselines3:
1. Base model (single neighbor, scale_hotel=1.0)
2. Neighborhood model (aggregated neighborhood, scale_hotel=20.0)
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from EnergyEnv import EnergyTradingEnv

STEPS = 300_000
SEED = 42


def train_base_model() -> None:
    """Train a PPO agent on the single-neighbor environment."""
    print("⚡ Initializing Base Single-Neighbor Environment (scale_hotel=1.0)...")
    env_base = EnergyTradingEnv("kls_vdit_hourly_market.csv", scale_hotel=1.0)
    env_base = Monitor(env_base)

    model_base = PPO(
        "MlpPolicy", env_base, verbose=1,
        learning_rate=0.0003, seed=SEED,
        tensorboard_log="./ppo_tensorboard/",
    )

    print(f"🧠 Training Base AI Agent ({STEPS:,} steps)...")
    model_base.learn(total_timesteps=STEPS)
    model_base.save("ppo_energy_agent_1M_steps")
    print("✅ Base Model Saved!")


def train_neighborhood_model() -> None:
    """Train a PPO agent on the aggregated neighborhood environment."""
    print("\n⚡ Initializing Aggregated Neighborhood Environment (scale_hotel=20.0)...")
    env_neigh = EnergyTradingEnv("kls_vdit_hourly_market.csv", scale_hotel=20.0)
    env_neigh = Monitor(env_neigh)

    model_neigh = PPO(
        "MlpPolicy", env_neigh, verbose=1,
        learning_rate=0.0003, seed=SEED,
        tensorboard_log="./ppo_tensorboard/",
    )

    print(f"🧠 Training Neighborhood AI Agent ({STEPS:,} steps)...")
    model_neigh.learn(total_timesteps=STEPS)
    model_neigh.save("ppo_energy_agent_Neighborhood_v1")
    print("✅ Neighborhood Model Saved!")


if __name__ == "__main__":
    train_base_model()
    train_neighborhood_model()