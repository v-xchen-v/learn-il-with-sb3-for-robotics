import numpy as np
import gymnasium as gym
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
import gymnasium_robotics
from imitation.data.wrappers import RolloutInfoWrapper

import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

class DictFlattenObservation(ObservationWrapper):
    def __init__(self, env, keys=("observation", "achieved_goal", "desired_goal")):
        super().__init__(env)
        self.keys = keys
        flat_dim = sum(env.observation_space[k].shape[0] for k in keys)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32)

    def observation(self, observation):
        return np.concatenate([observation[k] for k in self.keys], axis=-1)
def make_env_with_flat_obs():
    def thunk():
        env = gym.make("FetchPickAndPlaceDense-v4")
        env = DictFlattenObservation(env)  # ✅ Apply the wrapper
        return env
    return thunk

# Env builder
def make_env():
    env = gym.make("FetchPickAndPlaceDense-v4")
    # env = DictFlattenObservation(env)
    env = RolloutInfoWrapper(env)  # optional for imitation's rollout
    return env

# VecEnv for DAgger
from stable_baselines3.common.vec_env import DummyVecEnv
env = DummyVecEnv([make_env])  # <== ✅ This is the proper wrapper
# env = make_vec_env(
#     make_env_with_flat_obs(),
#     rng=np.random.default_rng(),
#     n_envs=1,
#     post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # optional
# )

def make_flattened_env():
    env = gym.make("FetchPickAndPlaceDense-v4")
    env = DictFlattenObservation(env)
    env = RolloutInfoWrapper(env)
    return env

dagger_env = DummyVecEnv([make_flattened_env])

expert = load_policy(
    "ppo",
    path="experts/FetchPickAndPlaceDense-v4/model_ppo.zip",  # Path to your trained PPO model
    venv=env,
)




import tempfile

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer

bc_trainer = bc.BC(
    observation_space=dagger_env.observation_space,
    action_space=dagger_env.action_space,
    rng=np.random.default_rng(),
)

class UnflattenWrapper:
    def __init__(self, expert_policy, obs_keys=("observation", "achieved_goal", "desired_goal"),
                 obs_dims=(25, 3, 3)):
        self.expert_policy = expert_policy
        self.obs_keys = obs_keys
        self.obs_dims = obs_dims

    def predict(self, flat_obs, *args, **kwargs):
        # Handle batched or single obs
        if len(flat_obs.shape) == 1:
            flat_obs = flat_obs[None, :]  # add batch dim

        dict_obs = {}
        start = 0
        for k, dim in zip(self.obs_keys, self.obs_dims):
            end = start + dim
            dict_obs[k] = flat_obs[:, start:end]
            start = end

        return self.expert_policy
wrapped_expert = UnflattenWrapper(
    expert_policy=expert,
    obs_keys=("observation", "achieved_goal", "desired_goal"),
    obs_dims=(25, 3, 3),
)

import numpy as np
from gymnasium.spaces import Box

import numpy as np
from gymnasium.spaces import Box

class FlattenExpertWrapper:
    """
    Wraps a flat-observation expert policy to accept dict observations.

    Makes the wrapper callable so it can be used as `expert_policy` in SimpleDAggerTrainer.
    """
    def __init__(self, expert_policy, original_dict_space, keys=("observation", "achieved_goal", "desired_goal")):
        self.expert_policy = expert_policy
        self.keys = keys

        # Construct flat observation space
        low = np.concatenate([original_dict_space[k].low.flatten() for k in keys], axis=-1)
        high = np.concatenate([original_dict_space[k].high.flatten() for k in keys], axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)
        self.action_space = expert_policy.action_space

    def predict(self, obs, *args, **kwargs):
        # if isinstance(obs, dict):
        #     flat_obs = np.concatenate([obs[k].ravel() for k in self.keys], axis=-1)
        # elif isinstance(obs, np.ndarray) and not isinstance(obs[0], dict):
        #     # flat_obs = np.array([
        #     #     np.concatenate([o[k].ravel() for k in self.keys], axis=-1)
        #     #     for o in obs
        #     # ])
        #     return obs
        # else:
        #     raise ValueError("Unsupported observation format for flattening.")
        return self.expert_policy.predict(obs, *args, **kwargs)

    def __call__(self, obs, *args, **kwargs):
        return self.predict(obs, *args, **kwargs)

# Wrap expert trained on flat obs to handle dict env
flattened_expert = FlattenExpertWrapper(
    expert_policy=expert,
    original_dict_space=env.observation_space,
    keys=("observation", "achieved_goal", "desired_goal")
)

class PatchedSimpleDAggerTrainer(SimpleDAggerTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ✅ fix the root cause early
        self._deterministic_policy = None
        
with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
    print(tmpdir)
    dagger_trainer = PatchedSimpleDAggerTrainer(
        venv=dagger_env,
        scratch_dir=tmpdir,
        expert_policy=flattened_expert,
        bc_trainer=bc_trainer,
        rng=np.random.default_rng(),
    )
    # ✅ Avoid triggering the internal assert by disabling the flag after creation
    # ✅ Force this to None to avoid the ValueError
    # dagger_trainer._deterministic_policy = None
    dagger_trainer.train(2000)
    
from stable_baselines3.common.evaluation import evaluate_policy

reward, _ = evaluate_policy(dagger_trainer.policy, env, 20)
print(reward)


import torch
# Save only the policy
torch.save(bc_trainer.policy.state_dict(), "dagger_policy.pt")
