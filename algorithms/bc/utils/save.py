# algorithms/bc/utils/save.py

import torch
from imitation.algorithms.bc import BC
from omegaconf import OmegaConf

def save_bc_model(
    bc_trainer: BC,
    save_path: str = "models/bc_policy",
    policy_weight_path: str = "bc_policy.pt",
    trainer_state_path: str = "bc_trainer_full.pt",
):
    """
    Saves a trained BC policy and trainer.

    Args:
        bc_trainer (BC): The BC trainer object.
        save_path (str): Path to save the full policy object (includes architecture).
        policy_weight_path (str): Path to save only the policy weights.
        trainer_state_path (str): Path to save the full trainer (includes optimizer, etc.).
    """
    print("💾 Saving trained policy...")
    bc_trainer.policy.save(save_path)
    torch.save(bc_trainer.policy.state_dict(), policy_weight_path)
    torch.save(bc_trainer, trainer_state_path)
    print("✅ Policy and trainer saved.")

def save_config(cfg, save_dir: str, filename: str = "config.yml"):
    """
    Saves the given OmegaConf config to a YAML file.

    Args:
        cfg (OmegaConf): The resolved config to save.
        save_dir (str): Directory where the config will be saved.
        filename (str): YAML file name. Default: "config.yml".
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    OmegaConf.save(config=cfg, f=path)
    print(f"✅ Saved config to {path}")

import os

"""
runs/
└── FetchPickAndPlaceDense-v4/
    └── bc_2025-07-22_seed42/
        ├── model/
        │   ├── bc_policy.pt
        │   ├── bc_trainer_full.pt
        │   └── policy/
        │       └── policy.pkl
        ├── eval/
        │   ├── metrics.json
        │   └── video_il_eval/
        │       ├── episode_0.mp4
        │       ├── episode_1.mp4
        ├── replay/
        │   ├── rollout_rl/
        │   │   ├── episode_0.mp4
        │   │   └── episode_1.mp4
        │   └── replay_trajectories/
        │       ├── demo_0.mp4
        │       └── demo_1.mp4
        └── logs/
            ├── tb/
            └── config.yml
"""
def create_output_dir(task_name: str, experiment_id: str) -> dict:
    """
    Creates and returns paths for a given experiment output folder.

    Args:
        task (str): Name of the task, e.g., "FetchPickAndPlaceDense-v4".
        experiment_id (str): Unique identifier for the experiment, e.g., "exp01_bc_seed42".

    Returns:
        dict: Dictionary of key output directories.
    """
    base = os.path.join("runs", task_name, experiment_id)

    paths = {
        "base": base,
        "model": os.path.join(base, "model"),
        "policy": os.path.join(base, "model/policy"),
        "logs": os.path.join(base, "logs"),
        "tb": os.path.join(base, "logs/tb"),
        "eval": os.path.join(base, "eval"),
        "config": os.path.join(base, "logs/config.yaml"),
        "video_il_eval": os.path.join(base, "eval/video_il_eval"),
        "video_rl_rollout": os.path.join(base, "replay/rollout_rl"),
        "video_demo_replay": os.path.join(base, "replay/replay_trajectories"),
    }

    for path in paths.values():
        if "config.yaml" not in path:  # skip creating file path
            os.makedirs(path, exist_ok=True)

    return paths