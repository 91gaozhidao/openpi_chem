# DBP-Pi0: End-to-End Robot Control Policy with Drift-Based Policy (DBP)

## 1. Overview
This repository builds upon the Physical Intelligence [OpenPI](https://github.com/Physical-Intelligence/openpi) framework. The primary contribution of this codebase is the integration of our proposed **Drift-Based Policy (DBP)**. By replacing the native Denoising Diffusion Probabilistic Model (DDPM) and Flow Matching paradigms in the Pi0 architecture with the DBP objective, this modification significantly improves the sample efficiency, robustness, and mathematical stability of online end-to-end robot control training.

## 2. Core Architecture Modifications

- **Objective Function Redesign (`src/openpi/models_pytorch/DBP_loss.py`)**: 
  The standard noise matching loss in the Pi0 `forward()` pass has been completely substituted with our mathematically rigorous `compute_dbp_loss()` implementation. The `sample_actions()` function has also been correspondingly updated to support both single-step and multi-step DBP inference generation.
  
- **Hyperparameter Integration (`src/openpi/models/pi0_config.py`)**: 
  The `Pi0Config` module has been extended to include hyperparameters specific to Drifting Loss, including `gen_per_label` (default: 4), `per_timestep_loss`, and the `temperatures` schedule.

- **Dependency Standardization (`pyproject.toml`)**: 
  To resolve underlying package conflicts (e.g., `ModuleNotFoundError`), the environment relies on a rigidly locked `lerobot==0.4.1` with a forced override constraint of `numpy<2.0.0`.

- **PyTorch Distributed Support**: 
  While OpenPI originally emphasized JAX, this pipeline is optimized for PyTorch Distributed Data Parallel (DDP) training. Pre-trained JAX Hugging Face checkpoints must be converted and stored in `pi0_base_pytorch/` prior to native PyTorch execution.

---

## 3. Environment Configuration Guidelines

Robust execution of the multi-process DataLoader in PyTorch relies heavily on deterministic environment variables and proper dynamic linking, specifically for the `torchcodec` video decoding backend. 

### 3.1 Base Environment Setup
It is recommended to deploy the system within an isolated Conda environment:
```bash
conda create -n driftingpi python=3.10 -y
conda activate driftingpi

# Critical Step: Install specifically versioned ffmpeg via conda-forge to ensure compatibility with torchcodec
conda install -c conda-forge "ffmpeg=7.0.2" -y

# Synchronize dependencies strictly utilizing uv
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

### 3.2 Dynamic Library Pathing
Due to inter-process isolation boundaries in Python multiprocessing, the `LD_LIBRARY_PATH` must be explicitly exported in the active shell to propagate the FFmpeg shared libraries to spawned worker processes. Failure to do so will result in `Could not load libtorchcodec` errors.
```bash
export LD_LIBRARY_PATH=/data1/workspace/gaoyuxuan/miniforge3/envs/driftingpi/lib:$LD_LIBRARY_PATH
```

---

## 4. Training Protocol

### 4.1 Data Preprocessing & Normalization
The dataset must comply with the LeRobot `v2.1` structural format. Normalization statistics based on observation features must be computed prior to training:
```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync scripts/compute_norm_stats.py --config-name pi0_chem
```

### 4.2 Architectural Configuration for Dual vs. Single-Arm Manipulators
Depending on whether the target task involves dual-arm collaboration or single-arm operation, explicit modifications to the policy network architecture and the training configuration are required:

- **Dual-Arm Configuration**:
  - **`openpi/src/openpi/policies/chem_policy.py`** (Retain bilateral visual inputs):
    ![Dual Arm Input Configuration](./assets/dual_policy_input.png)
    ![Dual Arm Output Configuration](./assets/dual_policy_output.png)
  - **`openpi/src/openpi/training/config.py`** (Configure `left_wrist` dictionary keys):
    ![Dual Arm Config](./assets/config_left.png)

- **Single-Arm Configuration (Right Arm Only)**:
  - **`openpi/src/openpi/policies/chem_policy.py`** (Prune left-arm nodes):
    ![Single Arm Input Configuration](./assets/right_policy_input.png)
    ![Single Arm Output Configuration](./assets/right_policy_output.png)
  - **`openpi/src/openpi/training/config.py`** (Remove left-arm configuration entries):
    ![Single Arm Config](./assets/config_no_left.png)

- **Optimization Hyperparameters**:
  Refer to the following diagram for standard adjustments to learning rates, batch schedules, and step intervals:
  ![Configuration Overview](./assets/config.png)

### 4.3 Distributed Training Execution
To circumvent arbitrary signal termination (e.g., `SIGHUP` and `SIGPIPE`) caused by broken pseudoterminal (PTY) connections in SSH instances, relying solely on `nohup` is insufficient against PyTorch elastic launcher's process group management mechanisms. 

It is strictly formalized to initialize training within a persistent virtual terminal multiplexer such as `tmux`.

```bash
# Provide native torchrun access isolated from environment-stripping abstraction layers
env CUDA_VISIBLE_DEVICES=1 torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    scripts/train_pytorch.py pi0_chem \
    --exp_name pour20260407_drifting \
    --overwrite 
```

---

## 5. Deployment and Serving Inference

To instantiate the trained PyTorch policies for real-world execution or simulated evaluations, employ the server-client inference architecture provided below:

```bash
# Instantiate the policy inference server
nohup env CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=.85 \
    uv run --no-sync scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config=pi0_chem \
    --policy.dir=checkpoints/pi0_chem/pour20260407_drifting/latest_ckpt_step \
    > output_serve.log 2>&1 &
```
