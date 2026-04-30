import pandas as pd
import matplotlib.pyplot as plt

# File paths
rew_file_sac = "csv_files/run-SAC_5-tag-rollout_ep_rew_mean.csv"
rew_file_ppo = "csv_files/run-PPO_10-tag-rollout_ep_rew_mean.csv"

# Load data
rew_df_sac = pd.read_csv(rew_file_sac)
rew_df_ppo = pd.read_csv(rew_file_ppo)

# Extract columns (TensorBoard CSV format)
steps_rew_sac = rew_df_sac["Step"]
values_rew_sac = rew_df_sac["Value"]

steps_rew_ppo = rew_df_ppo["Step"]
values_rew_ppo = rew_df_ppo["Value"]

print(values_rew_sac.max())
print(values_rew_ppo.max())

# Plot Reward vs Timesteps
plt.figure(figsize=(8,5))
plt.plot(steps_rew_sac, values_rew_sac, label="SAC")
plt.plot(steps_rew_ppo, values_rew_ppo, label="PPO")
plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.title("Reward Curves")
plt.legend()

plt.grid()
plt.tight_layout()
plt.show()
