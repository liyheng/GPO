# Guided Policy Optimization
This repository provides an implementation of Guided Policy Optimization (GPO), as introduced in our paper: [Guided Policy Optimization under Partial Observability](http://arxiv.org/abs/2505.15418).
Our implementation is built upon Brax (https://github.com/google/brax), utilizing its environment and single-file PPO implementation as a foundation.

## 📁 Repository Structure

Structure of the repository
- `envs/`:  Partially observable environments with injected noise, modified from Brax.
- `scripts/`: Training scripts for each environment, using hyperparameters from the GPO paper.
- `GPO.py`: Implementation of Guided Policy Optimization, including both GPO-Penalty and GPO-Clip variants.
- `config.py`: Configuration files for training.
- `wrappers.py`: Environment wrappers for partial observability and preprocessing.

## 🛠️ Installation
**1. Create a new conda environment**
```
conda create -n your_env_name python=3.10
conda activate your_env_name
```
**2. Install Brax**

Clone the [Brax repository](https://github.com/google/brax), navigate to the root directory, and run:
```
pip install --upgrade pip
pip install -e .
```
**3. Install Jax**
JAX installation depends on your hardware (CPU, GPU, or TPU). Please refer to the [official JAX installation guide](https://github.com/jax-ml/jax#installation) and follow the instructions that match your setup.

> **Note**: Although a `requirements.txt` file is provided, it may contain redundant packages. For best results, we recommend following the official installation guides for [Brax](https://github.com/google/brax) and [JAX](https://github.com/jax-ml/jax#installation).

## 🚀 Training
To train an agent (e.g., on the Ant environment), run:
```
cd scripts
chmod +x ./train_ant.sh
./train_ant.sh
```
Training logs and results will be saved in the directory structure: `results/env_name/seed/`. You can view progress using **TensorBoard**.

## 📖 Citation
If you find this repository or GPO useful in your research, please consider citing our paper:
```
@article{li2025guided,
  title={Guided Policy Optimization under Partial Observability},
  author={Li, Yueheng and Xie, Guangming and Lu, Zongqing},
  journal={arXiv preprint arXiv:2505.15418},
  year={2025}
}
```
