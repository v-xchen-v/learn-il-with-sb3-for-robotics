import os
import argparse
import pickle
import numpy as np
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import PPO, SAC, TD3
from imitation.data.types import Trajectory
SEED = 42 # For reproducibility, otherwise you can not visualize the same trajectory in env

def load_model(model_path):
    if "ppo" in model_path.lower():
        return PPO.load(model_path)
    elif "sac" in model_path.lower():
        return SAC.load(model_path)
    elif "td3" in model_path.lower():
        return TD3.load(model_path)
    else:
        raise ValueError("Unsupported algorithm or model filename.")


def collect_trajectories(env, model, n_episodes=100, require_success=True):
    raw_trajectories = []
    imitation_trajectories = []

    attempt = 0
    collected = 0

    env.reset(seed=SEED) # to make sure reproducibility
    
    while collected < n_episodes:
        obs_list, act_list, reward_list, done_list, info_list = [], [], [], [], []

        obs, _ = env.reset() # obs includes observation, target, and achieved goal

        done = False
        info = {}

        # add init obs
        obs_list.append(obs)
        
        # iterate until done
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            obs_list.append(obs)
            act_list.append(action)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

            obs = next_obs

        if not require_success or info.get("is_success", True):
            print(f"✅ Collected {collected + 1}/{n_episodes}")
            collected += 1

            raw_trajectories.append({
                "obs": np.array(obs_list),
                "act": np.array(act_list),
                "rew": np.array(reward_list),
                "done": np.array(done_list),
                "info": info_list,
            })

            imitation_trajectories.append(Trajectory(
                obs=np.array(obs_list),
                acts=np.array(act_list),
                # reward=np.array(reward_list), # If you want to include rewards, use TrajectoryWithRew data type instead
                infos=info_list,
                terminal=np.array(done_list),
            ))
        else:
            print("❌ Skipped: not successful")

        attempt += 1
        if attempt > 5 * n_episodes:
            print("⚠️ Too many failed attempts — exiting")
            break

    return raw_trajectories, imitation_trajectories


def save_pickle(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


# def save_npz(data_dict, path):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     np.savez_compressed(path, **data_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="FetchPickAndPlaceDense-v4", help="Environment ID")
    parser.add_argument("--model_path", type=str, default="experts/FetchPickAndPlaceDense-v4/model_ppo.zip", help="Trained SB3 model path")
    parser.add_argument("--n_episodes", type=int, default=1000, help="Number of successful episodes to collect")
    parser.add_argument("--require_success", action="store_true", default=True, help="Only save successful episodes")

    # parser.add_argument("--save_raw", type=str, default="expert_demos/FetchPickAndPlaceDense-v4/raw/expert_10.pkl", help="Path to save raw format data (.pkl)")
    # parser.add_argument("--save_imitation", type=str, default="expert_demos/FetchPickAndPlaceDense-v4/imitation_format/expert_10.pkl", help="Path to save imitation format (.pkl)")

    args = parser.parse_args()

    # if not args.save_raw and not args.save_imitation:
    #     print("❌ You must specify at least --save_raw or --save_imitation")
    #     return

    env = gym.make(args.env)
    model = load_model(args.model_path)

    print(f"Collecting expert data from: {args.model_path}")
    raw_data, imitation_data = collect_trajectories(
        env, model,
        n_episodes=args.n_episodes,
        require_success=args.require_success
    )

    # if args.save_raw:
    save_raw = f"expert_demos/{args.env}/raw/expert_{args.n_episodes}.pkl"
    save_pickle(raw_data, save_raw)
    print(f"📦 Raw data saved to {save_raw}")

    # if args.save_imitation:
    save_imitation = f"expert_demos/{args.env}/imitation_format/expert_{args.n_episodes}.pkl"
    save_pickle(imitation_data, save_imitation)
    print(f"📦 Imitation data saved to {save_imitation}")
    print("✅ Data collection complete.")


if __name__ == "__main__":
    main()
