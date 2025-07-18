import os
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.util.util import make_vec_env
# import gymnasium.wrappers
import gymnasium_robotics
from imitation.policies.serialize import load_policy
import argparse
seed = 42
rng = np.random.default_rng(seed)
def evaluate_il_policy(
    policy_path,
    env_id="FetchPickAndPlaceDense-v4",
    n_eval_episodes=20,
    render=False,
    seed=42,
):
    
    # Create environment
    # env = make_vec_env(env_id, n_envs=1, rng=rng)
    def make_flat_env():
        env = gym.make("FetchPickAndPlaceDense-v4")
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
    ppo_model.policy.load_state_dict(torch.load("bc_policy.pt"))

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
        obs, _ = flat_env.reset()
        done = False
        total_reward = 0
        while not done:
            # fl_obs = flatten_obs(obs)
            action, _ = ppo_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = flat_env.step(action)
            done = terminated or truncated
            total_reward += reward
            if render:
                flat_env.render()
        episode_rewards.append(total_reward)
        if info.get("is_success", False):
            success_count += 1

    avg_reward = np.mean(episode_rewards)
    success_rate = success_count / n_eval_episodes

    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.2%}")

    flat_env.close()
    return avg_reward, success_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", type=str, default="models/bc_fetchpickandplacedense_policy",
                        help="Path to trained imitation policy (e.g., models/bc_policy.pt)")
    parser.add_argument("--env_id", type=str, default="FetchPickAndPlaceDense-v4")
    parser.add_argument("--n_eval_episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluate_il_policy(
        policy_path=args.policy_path,
        env_id=args.env_id,
        n_eval_episodes=args.n_eval_episodes,
        render=args.render,
        seed=args.seed,
    )
