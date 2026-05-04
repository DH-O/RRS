# RRS

Code for the paper "Quality-Aware Exploration Budget Allocation for Cooperative Multi-Agent Reinforcement Learning".

## Install

```bash
pip install -r requirements.txt
pip install -e .            # registers bundled jaxmarl, core, coin_modules

# Expose the nvcc binaries shipped by nvidia-cuda-nvcc-cu12 so JAX can JIT-compile
# PTX. setup_jax.py wires the same path into XLA_FLAGS, but ptxas itself still
# has to be on PATH.
export PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cuda_nvcc/bin')"):$PATH

python setup_jax.py
```

Requires JAX with CUDA 12. The `jaxmarl` package shipped in this repository is a
fork that adds the custom MPE / SMAX / MABrax environments used in the paper
(`MPE_simple_corridor_v3`, `MPE_simple_tag_scripted_prey_6v2`,
`SMAX_27m_vs_30m`, `SMAX_3s5z_vs_3s6z`, `ant_4x2`, `ant_ball_4x2`,
`halfcheetah_6x1`). Do not install `jaxmarl` from PyPI -- the bundled version
must be used.

## Run

Main method (RCB + RSQ):

```bash
python algorithm/MAPPO/mappo_rcb_rsq.py --config-name rcb_rsq_corridor SEED=0
python algorithm/MAPPO/mappo_rcb_rsq_brax.py --config-name rcb_rsq_ant SEED=0
```

| Env          | Script                                  | Config                          |
|--------------|-----------------------------------------|---------------------------------|
| MPE-corridor | `algorithm/MAPPO/mappo_rcb_rsq.py`      | `rcb_rsq_corridor`              |
| MPE-tag      | `algorithm/MAPPO/mappo_rcb_rsq.py`      | `rcb_rsq_tag`                   |
| SMAX-3s5z    | `algorithm/MAPPO/mappo_rcb_rsq.py`      | `rcb_rsq_smax_3s5z`             |
| SMAX-27m     | `algorithm/MAPPO/mappo_rcb_rsq.py`      | `rcb_rsq_smax_27m`              |
| ant_4x2      | `algorithm/MAPPO/mappo_rcb_rsq_brax.py` | `rcb_rsq_ant`                   |
| ant_ball     | `algorithm/MAPPO/mappo_rcb_rsq_brax.py` | `rcb_rsq_ant_ball`              |
| halfcheetah  | `algorithm/MAPPO/mappo_rcb_rsq_brax.py` | `rcb_rsq_halfcheetah`           |

Baselines (Linear, Lagrangian, MAVEN, COIN, IPPO, MAPPO) are in `algorithm/`.

Set `WANDB_MODE=disabled` in the YAML or as an override to skip Weights and Biases logging.

## Layout

```
algorithm/     trainers and per-environment configs
core/          RSQ, Successor Distance, networks, advantage utilities
coin_modules/  COIN baseline modules
jaxmarl/       bundled JaxMARL fork with custom MPE/SMAX/MABrax envs
setup_jax.py   XLA flag setup (deterministic ops, CUDA data dir)
pyproject.toml package definition (registers jaxmarl, core, coin_modules)
```
