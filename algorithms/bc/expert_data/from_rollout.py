# algorithms/bc/expert_data/from_rollout.py

import numpy as np
from typing import Optional, List
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data.types import Trajectory, Transitions
from imitation.data import rollout
import gymnasium as gym
import gymnasium_robotics


def generate_rollout_transitions(
    env_id: str = "FetchPickAndPlaceDense-v4",
    expert_path: str = "experts/FetchPickAndPlaceDense-v4/model_ppo.zip",
    min_episodes: int = 5,
    seed: Optional[int] = None,
) -> Transitions:
    """
    Generates transitions using an expert policy via rollouts.

    Args:
        env_id (str): Gymnasium environment ID.
        expert_path (str): Path to the expert policy checkpoint.
        min_episodes (int): Number of episodes to collect.
        seed (Optional[int]): RNG seed for reproducibility.

    Returns:
        Transitions: Flattened transitions object from rollouts.
    """
    rng = np.random.default_rng(seed)
    
    env = make_vec_env(
        env_id,
        rng=rng,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
    )

    expert = load_policy("ppo", path=expert_path, venv=env)

    rollouts: List[Trajectory] = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=min_episodes),
        rng=rng,
    )

    transitions: Transitions = rollout.flatten_trajectories(rollouts)

    return transitions

# Usage example:
if __name__ == "__main__":
    transitions = generate_rollout_transitions(
        env_id="FetchPickAndPlaceDense-v4",
        expert_path="experts/FetchPickAndPlaceDense-v4/model_ppo.zip",
        min_episodes=5,
        seed=42,
    )
    print(f"Generated {len(transitions)} transitions from rollouts.")
    print(f"Transitions contain arrays for: {', '.join(transitions.__dict__.keys())}.")
    # You can now use `transitions` for training or evaluation.