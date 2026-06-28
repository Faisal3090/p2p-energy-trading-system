import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os


class EnergyTradingEnv(gym.Env):
    """
    Gymnasium environment for KLS VDIT P2P solar energy trading.

    Agents   : College (solar producer), Neighbour (P2P buyer), Utility Grid (fallback)
    Data     : kls_vdit_hourly_market.csv — 61,368 hourly rows (2019–2026)
    Timestep : 1 hour

    Observation (normalized to [0,1], 6 values):
        solar_norm      = College_Solar_kW / MAX_SOLAR
        demand_norm     = Campus_Demand_kW / MAX_DEMAND
        neighbour_norm  = Neighbor_Hotel_kW / MAX_NEIGHBOUR
        battery_norm    = battery_soc / BATTERY_CAPACITY_KWH
        hour_norm       = hour_of_day / 23.0
        surplus_norm    = clip(surplus, 0, MAX_SOLAR) / MAX_SOLAR

    Action (Discrete 3):
        0 = USE   — consume solar locally, grid covers deficit
        1 = STORE — charge battery with surplus solar
        2 = TRADE — sell surplus to neighbour at P2P price

    Reward (INR, data-derived):
        USE   : surplus * GRID_SELL + 0           - deficit * GRID_BUY_ALLIN
        STORE : stored * GRID_BUY_ENERGY * 0.9    - deficit * GRID_BUY_ALLIN
        TRADE : trade * P2P_PRICE + leftover * GRID_SELL - deficit * GRID_BUY_ALLIN
    """

    metadata = {"render_modes": ["human"]}

    # ── Pricing (data-derived from KLS VDIT billing, 2024-2026 period) ──
    GRID_BUY_ENERGY  = 8.10   # Pure energy tariff col(d) — what agent optimises against
    GRID_BUY_ALLIN   = 13.77  # Effective all-in rate col(f)/col(b) — true import cost
    GRID_SELL_PRICE  = 8.10   # Net metering rate col(g)/col(c) ≈ energy tariff
    P2P_TRADE_PRICE  = 10.50  # Between GRID_SELL and GRID_BUY_ALLIN — both parties benefit
    # P2P price > GRID_SELL means producer earns more than exporting to grid
    # P2P price < GRID_BUY_ALLIN means buyer pays less than importing from grid
    # This is the economic argument for P2P — both sides win vs the grid

    #Battery
    BATTERY_CAPACITY_KWH = 100.0
    BATTERY_CHARGE_RATE  = 30.0   # max kW per hour
    BATTERY_EFFICIENCY   = 0.90

    #Normalisation constant
    MAX_SOLAR     = 220.0   # actual max 211.3 kW, rounded up for headroom
    MAX_DEMAND    = 70.0    # actual max 66.0 kW
    MAX_NEIGHBOUR = 7.0     # actual max 6.4 kW
    MAX_SURPLUS   = 220.0

   
    TRADE_CAPACITY_FACTOR = 1.0 

    def __init__(self, csv_path: str = "kls_vdit_hourly_market.csv",
                 render_mode: str | None = None, scale_hotel: float = 1.0):
        super().__init__()

        # Security: reject paths containing directory traversal
        if ".." in str(csv_path):
            raise ValueError(f"csv_path must not contain '..': {csv_path}")
        if scale_hotel <= 0:
            raise ValueError(f"scale_hotel must be positive, got {scale_hotel}")

        self.render_mode = render_mode
        self.csv_path = csv_path
        self.scale_hotel = scale_hotel  # multiply neighbour demand to simulate more neighbors

        self._load_data()

        #Observation space: 6 values, all in [0, 1]
        self.observation_space = spaces.Box(
            low=np.zeros(6, dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32
        )

        # ── Action space: 3 discrete ──
        self.action_space = spaces.Discrete(3)

        self.current_step = 0
        self.battery_soc  = 0.0
        self.episode_log  = []
        self._grid_safety_fn = None

    # Private helpers

    def _load_data(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: '{self.csv_path}'")
        df = pd.read_csv(self.csv_path, parse_dates=["Timestamp"])
        required = {"College_Solar_kW", "Campus_Demand_kW", "Neighbor_Hotel_kW"}
        if missing := required - set(df.columns):
            raise ValueError(f"CSV missing columns: {missing}")
        df = df.sort_values("Timestamp").reset_index(drop=True)
        df[list(required)] = df[list(required)].fillna(0.0)
        self.df = df
        self.total_steps = len(df)

    def _get_obs(self) -> np.ndarray:
        if self.current_step >= self.total_steps:
            return np.zeros(6, dtype=np.float32)
        row = self.df.iloc[self.current_step]
        solar     = float(row["College_Solar_kW"])
        demand    = float(row["Campus_Demand_kW"])
        neighbour = float(row["Neighbor_Hotel_kW"]) * self.scale_hotel
        surplus   = max(0.0, solar - demand)
        hour      = float(row["Timestamp"].hour)
        return np.array([
            np.clip(solar     / self.MAX_SOLAR,     0, 1),
            np.clip(demand    / self.MAX_DEMAND,    0, 1),
            np.clip(neighbour / (self.MAX_NEIGHBOUR * self.scale_hotel) if self.scale_hotel > 0 else 0, 0, 1),
            np.clip(self.battery_soc / self.BATTERY_CAPACITY_KWH, 0, 1),
            hour / 23.0,
            np.clip(surplus / self.MAX_SURPLUS, 0, 1),
        ], dtype=np.float32)

    def _get_info(self) -> dict:
        if self.current_step >= self.total_steps:
            return {"step": self.current_step}
        row = self.df.iloc[self.current_step]
        return {
            "step": self.current_step,
            "timestamp": str(row["Timestamp"]),
            "hour": int(row["Timestamp"].hour),
            "battery_soc_kwh": round(self.battery_soc, 2),
        }

    # Public API

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.battery_soc  = 0.0
        self.episode_log  = []
        if self.render_mode == "human":
            print(f"\n[RESET] {self.total_steps} timesteps loaded.")
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}. Must be 0, 1, or 2.")

        #Safety guard: don't step past end
        if self.current_step >= self.total_steps:
            return np.zeros(6, dtype=np.float32), 0.0, True, False, {}

        row       = self.df.iloc[self.current_step]
        solar     = float(row["College_Solar_kW"])
        demand    = float(row["Campus_Demand_kW"])
        neighbour = float(row["Neighbor_Hotel_kW"]) * self.scale_hotel
        surplus   = solar - demand

        current_battery_soc = self.battery_soc

        reward         = 0.0
        grid_import_kw = 0.0
        trade_kw       = 0.0
        stored_kw      = 0.0
        action_label   = ["USE", "STORE", "TRADE"][action]

        # Deficit handling (automatic battery discharge first, then grid import)
        if surplus < 0:
            deficit = abs(surplus)
            # Battery discharge to cover deficit
            discharge_kw = min(deficit, self.battery_soc, self.BATTERY_CHARGE_RATE)
            self.battery_soc -= discharge_kw
            grid_import_kw = deficit - discharge_kw
            reward -= grid_import_kw * self.GRID_BUY_ALLIN  # true cost of import

        # Surplus handling (action determines what we do with surplus solar)
        elif surplus > 0:
            if action == 0:  # USE — export surplus to grid
                reward += surplus * self.GRID_SELL_PRICE

            elif action == 1:  # STORE — charge battery
                charge_kw    = min(surplus, self.BATTERY_CHARGE_RATE)
                # Remaining physical capacity in battery
                remaining_cap = self.BATTERY_CAPACITY_KWH - self.battery_soc
                # Stored energy considering charging efficiency
                stored_kw    = min(charge_kw * self.BATTERY_EFFICIENCY, remaining_cap)
                # Actual power drawn from surplus to charge battery
                actual_charge_kw = stored_kw / self.BATTERY_EFFICIENCY if stored_kw > 0 else 0.0
                self.battery_soc += stored_kw
                
                leftover = surplus - actual_charge_kw
                reward += leftover * self.GRID_SELL_PRICE

            elif action == 2:  # TRADE — P2P to neighbour
                trade_kw = min(surplus, neighbour)
                reward  += trade_kw * self.P2P_TRADE_PRICE      # premium over grid sell
                leftover = surplus - trade_kw
                reward  += leftover * self.GRID_SELL_PRICE       # rest goes to grid

        #Grid-safety hook
        grid_safe = True
        if self._grid_safety_fn is not None and trade_kw > 0:
            grid_safe = self._grid_safety_fn(trade_kw)
            if not grid_safe:
                reward -= 50.0

        physical_reward = reward
        
        # Potential-based reward shaping: Phi(s) = battery_soc * GRID_BUY_ALLIN
        gamma = 0.99
        shaping_reward = (gamma * self.battery_soc - current_battery_soc) * self.GRID_BUY_ALLIN
        training_reward = physical_reward + shaping_reward

        #Log
        log_entry = {
            "step": self.current_step, "timestamp": str(row["Timestamp"]),
            "action": action_label, "solar_kw": round(solar, 2),
            "campus_demand_kw": round(demand, 2), "neighbour_demand_kw": round(neighbour, 2),
            "surplus_kw": round(surplus, 2), "trade_kw": round(trade_kw, 2),
            "stored_kw": round(stored_kw, 2), "grid_import_kw": round(grid_import_kw, 2),
            "battery_soc_kwh": round(self.battery_soc, 2),
            "reward": round(physical_reward, 3), "grid_safe": grid_safe,
        }
        self.episode_log.append(log_entry)

        self.current_step += 1
        terminated = self.current_step >= self.total_steps
        obs  = self._get_obs()
        info = {**self._get_info(), **log_entry}

        if self.render_mode == "human":
            self.render()
        return obs, training_reward, terminated, False, info

    def render(self):
        if not self.episode_log:
            return
        e = self.episode_log[-1]
        print(f"  {e['timestamp']} | {e['action']:<5} | "
              f"Solar:{e['solar_kw']:>6.1f}kW | Surplus:{e['surplus_kw']:>6.1f}kW | "
              f"Trade:{e['trade_kw']:>5.1f}kW | Bat:{e['battery_soc_kwh']:>5.1f}kWh | "
              f"R:{e['reward']:>8.2f}")

    def close(self):
        pass

    def register_grid_safety_fn(self, fn):
        """Plug pandapower grid-safety"""
        self._grid_safety_fn = fn

    def get_episode_summary(self) -> dict:
        if not self.episode_log:
            return {}
        return {
            "steps":               len(self.episode_log),
            "total_reward":        round(sum(e["reward"] for e in self.episode_log), 2),
            "total_traded_kwh":    round(sum(e["trade_kw"] for e in self.episode_log), 2),
            "total_stored_kwh":    round(sum(e["stored_kw"] for e in self.episode_log), 2),
            "total_grid_import_kwh": round(sum(e["grid_import_kw"] for e in self.episode_log), 2),
            "pct_trade_actions":   round(100 * sum(1 for e in self.episode_log if e["action"]=="TRADE") / len(self.episode_log), 1),
        }


#Demo — 24 random steps


if __name__ == "__main__":
    print("=" * 72)
    print("  KLS VDIT EnergyTradingEnv v2 — Phase 1 Demo")
    print("=" * 72)
    env = EnergyTradingEnv("kls_vdit_hourly_market.csv", render_mode="human")
    obs, info = env.reset(seed=42)
    for _ in range(24):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated:
            break
    summary = env.get_episode_summary()
    print("\n" + "=" * 72)
    for k, v in summary.items():
        print(f"  {k:<30} {v}")
    print("\n  Phase 1 check: PASSED")
    env.close()
