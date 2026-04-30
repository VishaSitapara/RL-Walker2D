import pandas as pd
import matplotlib.pyplot as plt

# File paths
rew_file = "csv_files/run-PPO_10-tag-rollout_ep_rew_mean.csv"
# len_file = "csv_files/run-PPO_7-tag-rollout_ep_len_mean.csv"

# Load data
rew_df = pd.read_csv(rew_file)
# len_df = pd.read_csv(len_file)

# Extract columns (TensorBoard CSV format)
steps_rew = rew_df["Step"]
values_rew = rew_df["Value"]

print(values_rew.max())

# steps_len = len_df["Step"]
# values_len = len_df["Value"]

# Plot Reward vs Timesteps
plt.figure(figsize=(8,5))
plt.plot(steps_rew, values_rew)
plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.title("PPO Reward Curve")
plt.grid()
plt.tight_layout()

# # Plot Episode Length vs Timesteps
# plt.figure(figsize=(8,5))
# plt.plot(steps_len, values_len)
# plt.xlabel("Timesteps")
# plt.ylabel("Episode Length")
# plt.title("PPO Episode Length Curve")
# plt.grid()
# plt.tight_layout()
plt.show()
