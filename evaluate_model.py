"""Evaluate a trained PPO agent on the KLS VDIT P2P Energy Trading environment.

Runs a full 7-year (61,368 step) simulation using the trained
neighborhood-scale model and prints financial impact and action distribution.
"""

import os
import sys

from stable_baselines3 import PPO
from EnergyEnv import EnergyTradingEnv

MODEL_PATH = "ppo_energy_agent_Neighborhood_v1"
CSV_PATH = "kls_vdit_hourly_market.csv"
SCALE_HOTEL = 20.0


def evaluate() -> None:
    """Run full evaluation of the trained PPO agent."""
    # Validate model file exists
    model_file = MODEL_PATH + ".zip"
    if not os.path.isfile(model_file):
        print(f"❌ Model file not found: {model_file}")
        print("   Run train_ppo.py first to train a model.")
        sys.exit(1)

    print("⚡ Loading Environment with Neighborhood Scale...")
    env = EnergyTradingEnv(CSV_PATH, scale_hotel=SCALE_HOTEL)

    print("🧠 Loading Trained Neighborhood AI Model...")
    model = PPO.load(MODEL_PATH)

    obs, info = env.reset()
    done = False
    truncated = False

    total_reward = 0.0
    action_counts = {0: 0, 1: 0, 2: 0}  # 0: USE, 1: STORE, 2: TRADE

    print("▶️ Running 7-Year Simulation with Trained AI...")
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        action_counts[int(action)] += 1
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

    # Print results
    print("\n" + "=" * 50)
    print("🏆 AI STRATEGY RESULTS (7-YEAR SIMULATION)")
    print("=" * 50)
    print(f"Total Financial Impact: ₹{total_reward:,.2f}")
    print("\nAction Distribution:")
    total_actions = sum(action_counts.values())
    print(f"USE   (Self-Consume) : {action_counts[0]} times ({(action_counts[0]/total_actions)*100:.1f}%)")
    print(f"STORE (Battery)      : {action_counts[1]} times ({(action_counts[1]/total_actions)*100:.1f}%)")
    print(f"TRADE (Sell to Peer) : {action_counts[2]} times ({(action_counts[2]/total_actions)*100:.1f}%)")

    # Episode summary
    summary = env.get_episode_summary()
    print("\nEpisode Summary:")
    for k, v in summary.items():
        print(f"  {k:<30} {v}")
    print("=" * 50)


if __name__ == "__main__":
    evaluate()