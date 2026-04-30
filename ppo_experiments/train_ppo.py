import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from utils import create_dirs

create_dirs()

# Create vectorized environment
env = make_vec_env("Walker2d-v5", n_envs=1)

# Normalize observations only
env = VecNormalize(env, norm_obs=True, norm_reward=False)

# Initialize PPO model
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.001,  # smaller entropy coefficient
    verbose=1,
    tensorboard_log="./logs/ppo/"
)

# Train model
TIMESTEPS = 2_000_000
model.learn(total_timesteps=TIMESTEPS)

# Save model + normalization stats
model.save("models/ppo_walker_2000k_new")
env.save("models/ppo_vec_normalize.pkl")

print("PPO Training complete.")