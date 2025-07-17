# learn-il-with-sb3-robotics
A hands-on playground for exploring **Imitation Learning (IL)** using:
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [imitation](https://github.com/HumanCompatibleAI/imitation)
- [gymnasium-robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics)

This repo helps you:
- ✅ Collect expert demonstrations using SB3-trained policies
- ✅ Train imitation agents using Behavior Cloning (BC), DAgger, GAIL, AIRL
- ✅ Evaluate learned policies on standard robotic manipulation tasks


---

## 📦 Features

- ✅ Expert policy generation via PPO/SAC
- ✅ Demo collection in `.pkl` or `.npz` format
- ✅ Training pipelines using the `imitation` library
- ✅ Evaluation and visualization utilities
- ✅ Preconfigured setups for FetchReach, FetchPush, and more

---

## 🚀 Getting Started

### Recommended: Use GPU Docker Container (Preconfigured)

```bash
# 1. Build the image
docker build -f Dockerfile.gpu -t learn_il_sb3 .

# 2. Start the container in background (GPU required)
./docker/start.sh

# 3. Attach to the running container
./docker/into.sh
```

✅ This launches an interactive shell with the sb3_il conda environment activated and MuJoCo preinstalled.

✅ It use MUJOCO_GL=osmesa for CPU-based rendering as default, if you need using EGL (GPU rendering), please let me know.

### 🧪 Test setup (inside container)

```bash
python ./docker/test_setup.py
```

### 🧰 Run IL Pipeline
```bash
# 1. Collect expert demonstrations
python scripts/collect_expert.py --task FetchReach-v2 --algo ppo --save-path data/fetchreach_demos.pkl

# 2. Train a Behavior Cloning (BC) agent
python scripts/train_bc.py --config configs/fetchreach_bc.json

# 3. Evaluate the learned policy
python scripts/evaluate_policy.py --policy-path logs/bc_fetchreach.zip --env FetchReach-v2
```