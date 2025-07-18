"""
Visualizes expert or imitation learning trajectories saved in either:
- raw format (.pkl with keys: "obs", "act", etc.)
- imitation format (.npz with keys: "observations", "actions", etc.)

It will:
- Render the environment using env.render() (requires render_mode="human" or "rgb_array" if needed)
- Replay each trajectory sequentially
- Support both .pkl and .npz formats
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import argparse
import pickle
import numpy as np
import gymnasium as gym
import gymnasium_robotics
from time import sleep
from utils.video import record_video

SEED = 42  # For reproducibility, same env.reset seed and model predict with deterministic=True to make the env initial state is same when reset

def load_trajectories(path):
    if path.endswith(".pkl") and "raw" in path:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, list):
            print("🟢 Loaded raw trajectory format")
            return "raw", data
        else:
            raise ValueError("Invalid .pkl format: expected list of trajectories")

    elif path.endswith(".pkl") and "imitation" in path:
        with open(path, "rb") as f:
            data = pickle.load(f)
        print("🟢 Loaded imitation trajectory format")
        return "imitation", data
    else:
        raise ValueError("Unsupported file format. Use .pkl")


def visualize_raw_trajectories(env, traj_list, sleep_time=0.05, record=False, video_dir=None):
    env.reset(seed=SEED)  # to make sure reproducibility
    for idx, traj in enumerate(traj_list):
        obs_seq = traj["obs"]
        act_seq = traj["act"]
        frames = []
        print(f"🎬 Playing raw trajectory {idx+1}/{len(traj_list)} (length={len(obs_seq)})")

        obs, _ = env.reset()
        for i in range(len(obs_seq)):
            frame = env.render()
            if record:
                frames.append(frame)
            obs = obs_seq[1:][i]
            action = act_seq[i]
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
            sleep(sleep_time)
            
        if record and video_dir is not None:
            video_path = os.path.join(video_dir, f"ep_{idx:03d}.mp4")
            record_video(frames, video_path)
            print(f"📹 Recorded video for trajectory {idx+1} at {video_path}")
    env.close()


def visualize_imitation_trajectory(env, obs_array, act_array, episode_length=50, sleep_time=0.05, record=False, video_dir=None):
    num_frames = len(obs_array)
    num_episodes = num_frames // episode_length
    print(f"🟢 Playing {num_episodes} episodes from imitation trajectory")

    for ep in range(num_episodes):
        start = ep * episode_length
        end = start + episode_length
        frames = []
        obs, _ = env.reset()

        print(f"🎬 Playing imitation trajectory {ep+1}/{num_episodes}")
        for i in range(start, end):
            frame = env.render()
            if record:
                frames.append(frame)
            obs = obs_array[i]
            action = act_array[i]
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
            sleep(sleep_time)
            
        if record and video_dir is not None:
            video_path = os.path.join(video_dir, f"ep_{ep:03d}.mp4")
            record_video(frames, video_path)
            
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="FetchPickAndPlaceDense-v4", help="Environment ID")
    parser.add_argument("--traj_path", type=str, default="expert_demos/FetchPickAndPlaceDense-v4/raw/expert_10.pkl", help="Path to trajectory file (.pkl or .npz)")
    parser.add_argument("--episode_length", type=int, default=50, help="Step length per episode for imitation format")
    parser.add_argument("--sleep", type=float, default=0.05, help="Sleep time between steps")
    parser.add_argument("--record", action="store_true", default=True, help="Record video of the trajectory")
    parser.add_argument("--video_dir", type=str, default="data/videos/visualize_traj", help="Directory to save videos if recording is enabled")
    args = parser.parse_args()

    format, data = load_trajectories(args.traj_path)
    env = gym.make(args.env, render_mode="rgb_array" if args.record else None)

    if format == "raw":
        visualize_raw_trajectories(env, data, sleep_time=args.sleep, record=args.record, video_dir=args.video_dir)
    elif format == "imitation":
        traj_dict = {
            "obs": np.array([step['observation'] for traj in data for step in traj.obs]),
            "acts": np.array([traj.acts for traj in data]),
        }
        obs_arr = traj_dict["obs"]
        act_arr = traj_dict["acts"]
        visualize_imitation_trajectory(env, obs_arr, act_arr, args.episode_length, sleep_time=args.sleep, record=args.record, video_dir=args.video_dir)


if __name__ == "__main__":
    main()
