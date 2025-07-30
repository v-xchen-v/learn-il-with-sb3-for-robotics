import sys
import os
import argparse

# Ensure project root is in PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


def parse_args():
    parser = argparse.ArgumentParser(description="Train imitation learning algorithms.")
    parser.add_argument(
        "--task_config",
        type=str,
        required=True,
        help="Path to the task-specific YAML configuration file.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["bc"],
        default="bc",
        help="Imitation learning algorithm to use. Currently supported: [bc]",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.algorithm == "bc":
        from algorithms.bc.from_library.train import main as train_bc_main
        train_bc_main(config_path=args.task_config)
    else:
        raise NotImplementedError(f"Algorithm {args.algorithm} is not implemented.")


if __name__ == "__main__":
    main()
