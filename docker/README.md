# Learn IL with SB3 for Robotics – GPU Docker (Recommended)

## 📦 About This Image

✅ **Recommended GPU-accelerated Dockerfile** for the `learn_il_with_sb3_for_robotics` project.
This image is designed to help you **train and experiment with Imitation Learning (IL)** using **Stable-Baselines3 (SB3)** and related tools, with support for **Imitation**, **MuJoCo**, **Gymnasium Robotics**, and **PyTorch**, in a reproducible and ready-to-use Conda environment.

---

## ✅ Key Features

* ✅ Based on `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04`
* ✅ Miniconda + Python 3.10
* ✅ PyTorch 2.0.1 + torchvision (CUDA 11.8 compatible)
* ✅ Stable-Baselines3 (SB3) ==2.2.1
* ✅ `imitaiton`, `mujoco`, `gymnasium`, and `gymnasium-robotics` pre-installed and configured
* ✅ Headless MuJoCo rendering support with fallback from EGL to OSMesa(default)
* ✅ Starts in interactive Conda shell with environment `sb3_il`
* ✅ Allocates shared memory with `--shm-size=16g` (important for MuJoCo and PyTorch)
* ✅ Includes `start.sh` and `into.sh` scripts for easy container management

---

## 🧱 Build the Image

```bash
docker build -f Dockerfile.gpu -t il_sb3 .
```

---

## 🚀 Run the Container (Background)

```bash
./docker/start.sh
```

This script will:

* Start the container in background (`-d`)
* Use GPU (`--gpus all`)
* Allocate 16GB shared memory (`--shm-size=16g`)
* Mount the current directory to `/workspace` in the container

---

## 🔁 Attach to the Running Container

```bash
./docker/into.sh
```

This opens an **interactive shell** with the `sb3_il` Conda environment activated.

---

## 🧪 Test GPU and MuJoCo Setup (Optional)

Once inside the container:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -c "import mujoco; print(mujoco.__version__)"
```

---

## 📂 Folder Structure

```
.
├── Dockerfile.gpu              # GPU-ready Dockerfile for SB3 + IL
├── docker/
│   ├── start.sh                # Starts the container in background
│   ├── into.sh                 # Attaches to the running container
|   ├── README.md                   # This file
```

---

## 💡 Requirements

* NVIDIA GPU with drivers installed
* NVIDIA Container Toolkit:

  ```bash
  sudo apt install -y nvidia-container-toolkit
  sudo systemctl restart docker
  ```

---

## 📌 Notes

* You can modify `start.sh` to mount additional folders (e.g., datasets, logs).
* The container will start in `/workspace`, which maps to your project root.
* Conda environment name: `sb3_il`

---
