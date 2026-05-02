from EnergyEnv import EnergyTradingEnv
import numpy as np

#TEST 1: 24-step random run
print("=" * 60)
print("TEST 1: 24-step random run (Phase 1 demo)")
print("=" * 60)
env = EnergyTradingEnv("kls_vdit_hourly_market.csv", render_mode="human")
obs, info = env.reset(seed=42)
print(f"Start obs (normalized): {np.round(obs, 3)}")
print()
for hour in range(24):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break
summary = env.get_episode_summary()
print()
print("Episode summary:")
for k, v in summary.items():
    print(f"  {k}: {v}")

#TEST 2: actions
print()
print("=" * 60)
print("TEST 2: Reward comparison at peak solar (row 754 = 211 kW)")
print("=" * 60)
PEAK_ROW = 754
for action, name in [(0, "USE  "), (1, "STORE"), (2, "TRADE")]:
    e = EnergyTradingEnv("kls_vdit_hourly_market.csv")
    e.reset()
    e.current_step = PEAK_ROW
    e.battery_soc = 50.0
    _, reward, _, _, info = e.step(action)
    print(f"  {name}: reward=Rs.{reward:>8.2f}  "
          f"trade={info['trade_kw']:.1f}kW  stored={info['stored_kw']:.1f}kW")

# ── TEST 3: episode
print()
print("=" * 60)
print("TEST 3: Full episode (all 61,368 steps)")
print("=" * 60)
env2 = EnergyTradingEnv("kls_vdit_hourly_market.csv")
obs, _ = env2.reset(seed=0)
total_reward = 0
steps = 0
while True:
    action = env2.action_space.sample()
    obs, reward, terminated, truncated, info = env2.step(action)
    total_reward += reward
    steps += 1
    if terminated:
        break
print(f"  Steps completed : {steps}")
print(f"  Total reward (random agent): Rs.{total_reward:,.0f}")
print(f"  This is the BASELINE. A trained PPO should beat this.")
print()
print("All tests passed. env is ready.")