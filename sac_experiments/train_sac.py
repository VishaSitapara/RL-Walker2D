import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

def make_env():
    return Monitor(gym.make("Walker2d-v5"))

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("logs/sac", exist_ok=True)

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=False)

# SAC model
model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=1_000_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto",   
    verbose=1,
    tensorboard_log="./logs/sac/"
)

# Train
TIMESTEPS = 2_000_000
model.learn(total_timesteps=TIMESTEPS)

# Save
model.save("models/sac_walker_2000k")
env.save("models/sac_vec_normalize.pkl")

print("SAC Training complete.")