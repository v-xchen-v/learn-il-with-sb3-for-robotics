from imitation.algorithms.bc import BC
from imitation.data.types import Trajectory
from imitation.util.util import make_vec_env
import pickle
import gymnasium as gym
import gymnasium_robotics
import numpy as np
# 1. Load expert data
with open("expert_demos/FetchPickAndPlaceDense-v4/imitation/expert_2000.pkl", "rb") as f:
    trajectories = pickle.load(f)
seed = 42
rng = np.random.default_rng(seed)
# NORMALIZE_OBS = False  # Whether to normalize observations


# 2. Create environment
def make_flat_env():
    env = gym.make("FetchPickAndPlaceDense-v4")
    from gymnasium.wrappers import FlattenObservation
    from stable_baselines3.common.monitor import Monitor
    # env = FlattenObservation(env)
    # env = Monitor(env)
    return env
venv = make_flat_env()

# venv = make_vec_env("FetchPickAndPlaceDense-v4", rng=rng)
def flatten_obs_dict_list(obs_dict_list):
    return np.array([
        np.concatenate([v for v in step.values()])
        for step in obs_dict_list
    ], dtype=np.float32)
# def split_obs_dict_list(obs_dict_list):
#     # Returns a dict of arrays: each key maps to a (T, dim) array
#     return [
#         {k: np.asarray(v) for k, v in step.items()}
#         for step in obs_dict_list
#     ]
# Convert each trajectory

def ensure_obs_dict_format(obs_dict_list):
    """Ensure each step in the list is a dict of np.float32 arrays."""
    return [
        {k: np.asarray(v, dtype=np.float32) for k, v in step.items()}
        for step in obs_dict_list
    ]
from imitation.data.types import DictObs, Trajectory

def wrap_dict_obs(obs_dict_list):
    # Convert each dict to proper np.arrays and wrap with DictObs
    return DictObs.from_obs_list([
        {k: np.asarray(v, dtype=np.float32) for k, v in step.items()}
        for step in obs_dict_list
    ])
    
fixed_trajectories = []
for traj in trajectories:
    # obs = flatten_obs_dict_list(traj.obs)
    # obs = split_obs_dict_list(traj.obs)
    obs = wrap_dict_obs(traj.obs)

    # Ensure obs and acts have matching length
    if not len(obs) == len(traj.acts) + 1:
        raise ValueError("Observation length should be one more than action length.")

    # if NORMALIZE_OBS:
    #     from imitation.util.networks import RunningNorm
    #     obs_dim = obs.shape[1]
    #     obs_norm = RunningNorm(obs_dim)
    #     import torch
    #     obs_tensor = torch.tensor(obs, dtype=torch.float32)
    #     obs_norm.eval()
    #     normed_obs = obs_norm(obs_tensor).numpy()

    fixed_trajectories.append(Trajectory(
        # obs=normed_obs if NORMALIZE_OBS else obs,
        obs = obs,
        acts=np.array(traj.acts, dtype=np.float32),
        # rews=traj.rews if traj.rews is not None else None,
        infos=traj.infos if traj.infos is not None else [{}] * len(traj.acts),
        terminal=True,
    ))
    
        
# 3. Initialize BC trainer
bc_trainer = BC(
    observation_space=venv.observation_space,
    action_space=venv.action_space,
    demonstrations=fixed_trajectories,
    rng=rng
)

# 4. Train
bc_trainer.train(n_epochs=1)
# Save policy class info (optional, if you may switch policies)
with open("bc_policy_class.txt", "w") as f:
    f.write(bc_trainer.policy.__class__.__name__)
# 5. Save model
bc_trainer.policy.save("models/bc_fetchpickandplacedense_policy")

import torch
# Save only the policy
torch.save(bc_trainer.policy.state_dict(), "bc_policy.pt")

# Optionally save the whole trainer (state, optimizer, etc.)
torch.save(bc_trainer, "bc_trainer_full.pt")

# if NORMALIZE_OBS:
#     # You'll need this normalization module later for evaluation
#     import os
#     os.makedirs("normalization", exist_ok=True)
#     torch.save(obs_norm.state_dict(), "normalization/obs_norm.pt")
  
from stable_baselines3.common.evaluation import evaluate_policy  
reward, _ = evaluate_policy(bc_trainer.policy, venv, 10)
print("Reward from BC policy:", reward)