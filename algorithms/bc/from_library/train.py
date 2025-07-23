import os
import numpy as np
import gymnasium as gym
import gymnasium_robotics
from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.algorithms import bc
from imitation.util.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy

from algorithms.bc.from_library.expert_data.from_file import load_expert_data_from_file
from algorithms.bc.from_library.expert_data.from_rollout import generate_rollout_transitions
from algorithms.bc.utils.save import save_bc_model, create_output_dir, save_config
from utils.config import load_config


def load_transitions(config):
    data_source = config["data_source"]
    if data_source == "from_local_file":
        return load_expert_data_from_file(config["expert_data_path"])
    elif data_source == "from_rollout":
        return generate_rollout_transitions(
            env_id=config["env_name"],
            expert_path=config["expert_policy_path"],
            min_episodes=config["rollout"]["n_episodes"],
            seed=config["rollout"]["seed"],
        )
    else:
        raise ValueError(f"Unknown data source: {data_source}")


def create_env(env_name, seed=42):
    rng = np.random.default_rng(seed)
    return make_vec_env(
        env_name,
        rng=rng,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
    )


def setup_logger(log_dir):
    return configure(log_dir, ["tensorboard", "stdout"])


def train_bc(env, transitions, config, output_dirs):
    logger = setup_logger(output_dirs["tb"])
    rng = np.random.default_rng(seed=42)
    
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
        custom_logger=logger,
        batch_size=config["batch_size"],
    )

    reward_before, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=10)
    print(f"Reward before training: {reward_before:.2f}")

    bc_trainer.train(n_epochs=config["n_epochs"])

    reward_after, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=10)
    print(f"Reward after training: {reward_after:.2f}")

    return bc_trainer


def train_bc_main(config_path):
    # === Load config ===
    config = load_config(config_path)
    env_name = config["env_name"]
    experiment_id = config["experiment_id"]

    # === Setup ===
    output_dirs = create_output_dir(env_name, experiment_id)
    transitions = load_transitions(config)
    env = create_env(env_name)

    # === Train ===
    bc_trainer = train_bc(env, transitions, config, output_dirs)

    # === Save ===
    save_config(config, output_dirs["logs"], filename="config.yaml")
    save_bc_model(
        bc_trainer,
        f"{output_dirs['model']}/policy.pt",
        policy_weight_path=f"{output_dirs['model']}/policy_state_dict.pt",
        trainer_state_path=f"{output_dirs['model']}/trainer.pt",
    )


if __name__ == "__main__":
    train_bc_main("configs/FetchPickAndPlaceDense-v4/bc/bc_exp_02.yml")
