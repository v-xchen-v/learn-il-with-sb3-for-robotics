# algorithms/bc/utils/save.py

import torch
from imitation.algorithms.bc import BC

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

import os

"""
trained_models/
└── FetchPickAndPlaceDense-v4/
    └── bc_2025-07-22_seed42/
        ├── model/
        │   ├── bc_policy.pt
        │   ├── bc_trainer_full.pt
        │   └── policy/
        │       └── policy.pkl
        ├── eval/
        │   ├── returns.json
        │   └── video.mp4
        └── logs/
            ├── tb/
            └── config.yaml
"""
def create_output_dir(task: str, experiment_id: str) -> dict:
    """
    Creates and returns paths for a given experiment output folder.

    Args:
        task (str): Name of the task, e.g., "FetchPickAndPlaceDense-v4".
        experiment_id (str): Unique identifier for the experiment, e.g., "exp01_bc_seed42".

    Returns:
        dict: Dictionary of key output directories.
    """
    base = os.path.join("trained_models", task, experiment_id)

    paths = {
        "base": base,
        "model": os.path.join(base, "model"),
        "policy": os.path.join(base, "model/policy"),
        "logs": os.path.join(base, "logs"),
        "tb": os.path.join(base, "logs/tb"),
        "eval": os.path.join(base, "eval"),
        "config": os.path.join(base, "logs/config.yaml"),
    }

    for path in paths.values():
        if "config.yaml" not in path:  # skip creating file path
            os.makedirs(path, exist_ok=True)

    return paths