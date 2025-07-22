# algorithms/bc/expert_data/from_file.py

import pickle
import numpy as np
from typing import List
from imitation.data.types import Trajectory, DictObs


def wrap_dict_obs(obs_dict_list):
    """Convert each dict to proper np.arrays and wrap with DictObs."""
    return DictObs.from_obs_list([
        {k: np.asarray(v, dtype=np.float32) for k, v in step.items()}
        for step in obs_dict_list
    ])


def load_expert_from_file(
    path: str,
    seed: int = 42,
) -> List[Trajectory]:
    """
    Loads expert trajectories from a pickled file and wraps observations properly.

    Args:
        path (str): Path to the .pkl file containing a list of Trajectories.
        seed (int): Random seed (only used for reproducibility hints).

    Returns:
        List[Trajectory]: List of Trajectory objects with DictObs observations.
    """
    with open(path, "rb") as f:
        trajectories = pickle.load(f)

    rng = np.random.default_rng(seed)

    fixed_trajectories = []
    for traj in trajectories:
        obs = wrap_dict_obs(traj.obs)

        # Imitation expects obs length = acts length + 1
        if len(obs) != len(traj.acts) + 1:
            raise ValueError("Each trajectory must have len(obs) == len(acts) + 1")

        fixed_trajectories.append(Trajectory(
            obs=obs,
            acts=np.array(traj.acts, dtype=np.float32),
            infos=traj.infos if traj.infos is not None else [{}] * len(traj.acts),
            terminal=True,
        ))

    return fixed_trajectories


# Optional CLI/test block
if __name__ == "__main__":
    path = "expert_demos/FetchPickAndPlaceDense-v4/imitation/expert_2000.pkl"
    fixed_trajectories = load_expert_from_file(path)
    print(f"✅ Loaded {len(fixed_trajectories)} expert trajectories from {path}")
