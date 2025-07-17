"""This script is used to test rendering capabilities in a headless Docker environment."""

import gymnasium as gym
import gymnasium_robotics
import imageio

env = gym.make("FetchReach-v4", render_mode="rgb_array")
obs, _ = env.reset()
frames = []

for _ in range(100):
    obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
    frames.append(env.render())
    if terminated or truncated:
        break

env.close()
imageio.mimsave("test_headless_video.mp4", frames, fps=30)
print("🎉 Render test complete. Saved as test_headless_video.mp4")