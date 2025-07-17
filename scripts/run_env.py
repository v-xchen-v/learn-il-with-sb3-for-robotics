"""This script is used to test env is working correctly."""

import gymnasium as gym
import gymnasium_robotics
import imageio

# env_ids = list(gym.envs.registry.keys())
# for env_id in sorted(env_ids):
#     print(env_id)

env = gym.make("FetchPickAndPlaceDense-v4", render_mode="rgb_array")
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