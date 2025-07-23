import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.util.util import make_vec_env
# import gymnasium.wrappers
import gymnasium_robotics
from imitation.policies.serialize import load_policy
from utils.video import record_video
from algorithms.bc.utils.save import create_output_dir
from utils.config import load_config
import argparse
seed = 42

NORMALIZE_OBS= False  # Whether to normalize observations

    
rng = np.random.default_rng(seed)
def evaluate_il_policy(
    policy_path,
    env_id="FetchPickAndPlaceDense-v4",
    n_eval_episodes=100,
    render=False,
    seed=42,
    eval_result_dir=None,
):
    
    # Create environment
    # env = make_vec_env(env_id, n_envs=1, rng=rng)
    def make_flat_env():
        env = gym.make(env_id, render_mode="rgb_array")
        from gymnasium.wrappers import FlattenObservation
        from stable_baselines3.common.monitor import Monitor
        env = FlattenObservation(env)
        env = Monitor(env)
        return env
    flat_env = make_flat_env()
    # Load trained policy
    import torch

    # from imitation.util.networks import make_policy
    # Recreate policy architecture
    # 2. Recreate the exact policy architecture used in BC
    # FeedForward32Policy is the default in BC
    from imitation.policies.base import FeedForward32Policy
    from stable_baselines3 import PPO
    # policy = FeedForward32Policy(
    #     observation_space=env.observation_space,
    #     action_space=env.action_space,
    #     lr_schedule=lambda _: 1e-3,
    # )

    # # 3. Load weights
    # policy.load_state_dict(torch.load("bc_policy.pt"))
    # policy.eval()
    
    ppo_model = PPO(
        FeedForward32Policy,
        flat_env,
        verbose=1,
        # policy_kwargs={"net_arch": [dict(pi=[64, 64], vf=[64, 64])]},
        # tensorboard_log = out_paths["logs"]
    )

    # Load weights from BC into PPO
    ppo_model.policy.load_state_dict(torch.load(policy_path))

    # # Load saved weights
    # policy.load_state_dict(torch.load("bc_policy.pt"))
    # policy.eval()
    episode_rewards = []
    success_count = 0
    def flatten_obs(obs):
        """
        Flattens dict observation into a single numpy array or batch.
        Works for both single env and VecEnv.
        """
        if isinstance(obs, dict):
            if isinstance(obs["observation"], np.ndarray) and obs["observation"].ndim == 2:
                # Vectorized env, shape (batch_size, dim)
                return np.concatenate([
                    obs["observation"],
                    obs["achieved_goal"],
                    obs["desired_goal"]
                ], axis=1)
            else:
                # Single env, shape (dim,)
                return np.concatenate([
                    obs["observation"],
                    obs["achieved_goal"],
                    obs["desired_goal"]
                ])
        return obs
    for i in range(n_eval_episodes):
        if NORMALIZE_OBS:
            from imitation.util.networks import RunningNorm

            # Load normalization stats
            obs_dim = flat_env.observation_space.shape[0]  # usually 25 for FetchPickAndPlace
            obs_norm = RunningNorm(obs_dim)
            obs_norm.load_state_dict(torch.load("normalization/obs_norm.pt"))
            obs_norm.eval()  # disable updates
            
        frames = []
        obs, _ = flat_env.reset()
        done = False
        total_reward = 0
        while not done:
            if NORMALIZE_OBS:
                # Normalize obs
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                normed_obs = obs_norm(obs_tensor).numpy()
                obs = normed_obs
            action, _ = ppo_model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = flat_env.step(action)
            done = terminated or truncated
            total_reward += reward
            if render:
                frame = flat_env.render()
                frames.append(frame)
        episode_rewards.append(total_reward)
        if info.get("is_success", False):
            success_count += 1
        if render:
            record_video(frames, f"{eval_result_dir['video_il_eval']}/episode_{i}.mp4", fps=30)

    avg_reward = np.mean(episode_rewards)
    success_rate = success_count / n_eval_episodes

    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.2%}")

    flat_env.close()
    return avg_reward, success_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", type=str,
                        help="Path to trained imitation policy (e.g., models/bc_policy.pt)")
    parser.add_argument("--env_id", type=str)
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true", default=True, help="Render the environment during evaluation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    exp_config = load_config("configs/FetchPickAndPlaceDense-v4/bc/bc_exp_02.yml")
    env_name = exp_config["env_name"]
    experiment_id = exp_config["experiment_id"]
    output_dirs = create_output_dir(env_name, experiment_id)
    print(f"Evaluating IL policy on {args.env_id} with {args.n_eval_episodes} episodes...")
    print(f"Output directory: {output_dirs['eval']}")
    policy_path = args.policy_path if args.policy_path else f"{output_dirs['model']}/bc_policy.pt"
    if not os.path.exists(policy_path):
        print(f"Policy file {policy_path} does not exist. Please check the path.")
        sys.exit(1)
    print(f"Using policy from: {policy_path}")
    print(f"Environment ID: {args.env_id}")
    print(f"Number of evaluation episodes: {args.n_eval_episodes}")
    print(f"Render mode: {'enabled' if args.render else 'disabled'}")

    evaluate_il_policy(
        policy_path=policy_path,
        env_id=env_name,
        n_eval_episodes=args.n_eval_episodes,
        render=args.render,
        seed=args.seed,
        eval_result_dir=output_dirs
    )
