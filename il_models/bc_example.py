import numpy as np
import gymnasium as gym
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper
import gymnasium_robotics

env = make_vec_env(
    "FetchPickAndPlaceDense-v4",
    rng=np.random.default_rng(),
    post_wrappers=[
        lambda env, _: RolloutInfoWrapper(env)
    ],  # needed for computing rollouts later
)

# load from stable-baselines3 model
expert = load_policy(
    "ppo",
    path="experts/FetchPickAndPlaceDense-v4/model_ppo.zip",  # Path to your trained PPO model
    venv=env,
)

from imitation.data import rollout

rng = np.random.default_rng()
rollouts = rollout.rollout(
    expert,
    env,
    rollout.make_sample_until(min_timesteps=None, min_episodes=5),
    rng=rng,
)
transitions = rollout.flatten_trajectories(rollouts)
print(
    f"""The `rollout` function generated a list of {len(rollouts)} {type(rollouts[0])}.
After flattening, this list is turned into a {type(transitions)} object containing {len(transitions)} transitions.
The transitions object contains arrays for: {', '.join(transitions.__dict__.keys())}."
"""
)

from imitation.algorithms import bc

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
)

from stable_baselines3.common.evaluation import evaluate_policy
reward_before_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
print(f"Reward before training: {reward_before_training}")

bc_trainer.train(n_epochs=1)
reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
print(f"Reward after training: {reward_after_training}")

# Save the trained policy
bc_trainer.policy.save("models/bc_fetchpickandplacedense_policy")
import torch
# Save only the policy
torch.save(bc_trainer.policy.state_dict(), "bc_policy.pt")
# Optionally save the whole trainer (state, optimizer, etc.)
torch.save(bc_trainer, "bc_trainer_full.pt")
