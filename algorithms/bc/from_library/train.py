from algorithms.bc.from_library.expert_data.from_file import load_expert_data_from_file
from algorithms.bc.from_library.expert_data.from_rollout import generate_rollout_transitions
from algorithms.bc.utils.save import save_bc_model, create_output_dir, save_config
from utils.config import load_config

import gymnasium as gym
import gymnasium_robotics
from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.logger import configure
import numpy as np

exp_config = load_config("configs/FetchPickAndPlaceDense-v4/bc/bc_exp_02.yml")
env_name = exp_config["env_name"]
experiment_id = exp_config["experiment_id"]
task = exp_config["env_name"]
output_dirs = create_output_dir(task, experiment_id)
n_epochs = exp_config["n_epochs"]
data_source = exp_config["data_source"]
if data_source == "expert_file":
    # Load expert data from file
    expert_data_path = exp_config["expert_data_path"]
    transitions = load_expert_data_from_file(expert_data_path)
# elif data_source == "rollout":
#     # Generate transitions from a rollout of an expert policy
#     expert_path = exp_config["expert_policy_path"]
#     transitions = generate_rollout_transitions(
#         env_id=env_name,
#         expert_path=expert_path,
#         min_episodes=exp_config["rollout"]["n_episodes"],
#         seed=exp_config["rollout"]["seed"],
#     )
else:
    raise ValueError(f"Unknown data source: {data_source}")

env = make_vec_env(
    env_name,
    rng=np.random.default_rng(),
    post_wrappers=[
        lambda env, _: RolloutInfoWrapper(env)
    ],  # needed for computing rollouts later
)

# expert = load_policy(
#     "ppo",
#     path="experts/FetchPickAndPlaceDense-v4/model_ppo.zip",  # Path to your trained PPO model
#     venv=env,
# )

# transitions = generate_rollout_transitions(
#     env_id="FetchPickAndPlaceDense-v4",
#     expert_path="experts/FetchPickAndPlaceDense-v4/model_ppo.zip",
#     min_episodes=5,
#     seed=42,
# )

# load expert data from file
expert_data_path = "expert_demos/FetchPickAndPlaceDense-v4/imitation_format/expert_2000.pkl"
transitions = load_expert_data_from_file(expert_data_path)


rng = np.random.default_rng(seed=42)
from imitation.algorithms import bc
logger = configure(
    output_dirs["tb"],  # Output dir for TensorBoard logs
    ["tensorboard", "stdout"],    # Log to TensorBoard and console
)

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
    custom_logger=logger,
)


from stable_baselines3.common.evaluation import evaluate_policy
reward_before_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
print(f"Reward before training: {reward_before_training}")

bc_trainer.train(n_epochs=n_epochs)
reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
print(f"Reward after training: {reward_after_training}")

# Save the experiment configuration
save_config(exp_config, output_dirs["logs"], filename="config.yaml")

# Save the trained policy
save_bc_model(
    bc_trainer,
    f"{output_dirs['model']}/bc_fetchpickandplacedense_policy",
    policy_weight_path=f"{output_dirs['model']}/bc_policy.pt",
    trainer_state_path=f"{output_dirs['model']}/bc_trainer_full.pt",
)
