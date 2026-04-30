# Note- This test code file is the same as the test_sac_demo.py file, except that it will save the rendered frames as a 'video' instead of showing a live render.

import gymnasium as gym
import cv2
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Create env for frame rendering
env = DummyVecEnv([lambda: gym.make("Walker2d-v5", render_mode="rgb_array")])

# Load normalization statistics
env = VecNormalize.load("models/sac_vec_normalize.pkl", env)
env.training = False
env.norm_reward = False

# Load trained model
model = SAC.load("models/sac_walker_2000k")

obs = env.reset()

frames = []

for _ in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    # Capture rendered frame
    frame = env.render()
    frames.append(frame)

    if done:
        obs = env.reset()

env.close()

height, width, _ = frames[0].shape

video = cv2.VideoWriter(
    "videos/sac_walker_test_2000k.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    30,
    (width, height)
)

for frame in frames:
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video.write(frame)

video.release()

print("SAC video saved successfully!")