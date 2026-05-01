# RRS

Code for the paper "Quality-Aware Exploration Budget Allocation for Cooperative Multi-Agent Reinforcement Learning".

## Install

```bash
pip install -r requirements.txt
python setup_jax.py
```

Requires JAX with CUDA and JaxMARL.

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
algorithm/   trainers and per-environment configs
core/        RSQ, Successor Distance, networks, advantage utilities
coin_modules/  COIN baseline modules
scripts/     plot scripts (require local logs)
```
