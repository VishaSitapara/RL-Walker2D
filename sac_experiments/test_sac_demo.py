# Note- This test code file is the same as the test_sac_save.py file, except that it will showing a 'live render' instead of saving the rendered frames as a video.

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Create env with live render
env = DummyVecEnv([lambda: gym.make("Walker2d-v5", render_mode="human")])

# Load normalization stats
env = VecNormalize.load("models/sac_vec_normalize.pkl", env)
env.training = False
env.norm_reward = False

# Load model
model = SAC.load("models/sac_walker_2000k")

obs = env.reset()

for _ in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    # Access underlying mujoco env
    base_env = env.venv.envs[0].unwrapped

    # Get torso x-position
    torso_x = base_env.data.qpos[0]

    # Move camera to follow torso
    viewer = base_env.mujoco_renderer.viewer
    viewer.cam.lookat[0] = torso_x
    viewer.cam.lookat[1] = 0
    viewer.cam.lookat[2] = 1.0

    if done:
        obs = env.reset()

env.close()
print("SAC live tracking test finished!")