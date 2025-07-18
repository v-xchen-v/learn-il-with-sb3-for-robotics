import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import gymnasium
import gymnasium.wrappers
import types

# # Hack to patch missing monitoring module
# gymnasium.wrappers.monitoring = types.SimpleNamespace()

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from utils.video import record_video

def load_model(model_path):
    if "ppo" in model_path.lower():
        return PPO.load(model_path)
    elif "sac" in model_path.lower():
        return SAC.load(model_path)
    elif "td3" in model_path.lower():
        return TD3.load(model_path)
    else:
        raise ValueError("Unsupported algorithm in model_path.")

def run_policy(env, model, n_episodes=10, record=False, video_dir=None):
    successes = []
    all_infos = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        success = False
        frames = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if record:
                frame = env.render()
                frames.append(frame)

            if info.get("is_success") is not None:
                success = info["is_success"]

        successes.append(success)
        all_infos.append(info)

        if record and video_dir is not None:
            video_path = os.path.join(video_dir, f"ep_{ep:03d}.mp4")
            record_video(frames, video_path)

    return successes, all_infos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="FetchPickAndPlaceDense-v4", help="Environment ID, e.g. FetchPush-v2")
    parser.add_argument("--model_path", default="experts/FetchPickAndPlaceDense-v4/model_ppo.zip", type=str,  help="Path to trained SB3 model")
    parser.add_argument("--n_episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--record_video", action="store_true", default=True, help="Record rollout videos")
    parser.add_argument("--video_dir", type=str, default="data/videos/", help="Directory to save videos")
    args = parser.parse_args()

    os.makedirs(args.video_dir, exist_ok=True)

    env = gym.make(args.env, render_mode="rgb_array" if args.record_video else None)
    model = load_model(args.model_path)

    print(f"Running expert policy for {args.n_episodes} episodes...")
    successes, _ = run_policy(env, model, args.n_episodes, args.record_video, args.video_dir)

    success_rate = np.mean(successes)
    print(f"✅ Success rate: {success_rate*100:.2f}% over {args.n_episodes} episodes")

    env.close()


if __name__ == "__main__":
    main()
