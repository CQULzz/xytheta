# xytheta_transa

Isaac Lab multi-car LiDAR exploration environment.

This repository contains an Isaac Lab extension for simulating multiple planar cars in a bounded arena with obstacles and LiDAR sensors. It is intended for iterative development of exploration rewards, random or guided rollout scripts, reinforcement learning experiments, and reproducible experiment records.

## Project Goals

- Build a reusable Isaac Lab environment for multi-car planar exploration.
- Compare different action policies, reward functions, and LiDAR configurations.
- Record exploration area over time for evaluation.
- Keep code, experiments, and project management documents clear enough for future human or AI-assisted development.

## Current Features

- `Xytheta-Transa-v0.0`
  - Three planar cars with LiDAR sensors.
  - Simple wall and obstacle arena.
  - Adds exploration reward based on newly LiDAR-covered, previously unexplored area.
  - Adds wall proximity penalty and stronger termination penalty.
  - Random, fixed demo, reward-guided, and zero-action rollout scripts.
- `Xytheta-Transa-Offroad-v0.0`
  - One articulation-based four-wheel differential-drive car.
  - Fixed 15 m x 15 m generated off-road terrain.
  - Terrain includes rough height fields, rocks, discrete obstacles, cones, and low stairs.
  - 16-channel LiDAR scans terrain and obstacles.
  - Uses single-car exploration area as the current data-collection metric.
- Exploration CSV logging in `scripts/exploration_agent.py`.
- Optional per-environment CSV splitting for vectorized runs.
- Version notes in `VerDairy`.

## Environment Dependencies

This project is an Isaac Lab extension. It assumes:

- Ubuntu/Linux development environment.
- Isaac Sim and Isaac Lab installed.
- A Python environment that can import Isaac Lab and Isaac Sim modules.
- Recommended local conda environment name: `env_isaaclab`.

The project does not vendor Isaac Lab or Isaac Sim. Install those separately following the official Isaac Lab installation guide.

## Installation

From the repository root:

```bash
conda activate env_isaaclab
python -m pip install -e source/xytheta_transa
```

For local scripts, also set:

```bash
export PYTHONPATH=$PWD/source/xytheta_transa:$PYTHONPATH
```

## Quick Start

List available tasks:

```bash
python scripts/list_envs.py
```

Run the current environment with random actions:

```bash
python scripts/exploration_agent.py \
  --task Xytheta-Transa-v0.0 \
  --num_envs 1 \
  --disable_fabric
```

Run the current environment with reward-guided exploration:

```bash
python scripts/exploration_agent.py \
  --task Xytheta-Transa-v0.0 \
  --num_envs 1 \
  --disable_fabric \
  --greedy \
  --exploration_csv logs/exploration/v0_0_reward_guided.csv
```

Run the off-road data-collection environment:

```bash
python scripts/exploration_agent.py \
  --task Xytheta-Transa-Offroad-v0.0 \
  --num_envs 1 \
  --disable_fabric \
  --greedy \
  --exploration_csv logs/exploration/offroad_demo.csv
```

Run a short off-road headless smoke test:

```bash
python scripts/exploration_agent.py \
  --task Xytheta-Transa-Offroad-v0.0 \
  --num_envs 1 \
  --headless \
  --disable_fabric \
  --max_steps 50
```

Run a headless short smoke test:

```bash
python scripts/exploration_agent.py \
  --task Xytheta-Transa-v0.0 \
  --num_envs 1 \
  --headless \
  --disable_fabric \
  --max_steps 100
```

Split exploration CSV output by environment:

```bash
python scripts/exploration_agent.py \
  --task Xytheta-Transa-v0.0 \
  --num_envs 8 \
  --disable_fabric \
  --split_exploration_csv_by_env \
  --exploration_csv logs/exploration/my_run.csv
```

This creates files such as `my_run_env0.csv`, `my_run_env1.csv`, and so on.

`scripts/random_agent.py` is kept only as a backward-compatible wrapper for older commands.

## Directory Structure

```text
.
├── scripts/                         # Rollout, training, play, and utility scripts
├── source/xytheta_transa/            # Isaac Lab extension source
├── docs/                            # Project management and experiment documentation
├── .github/ISSUE_TEMPLATE/           # GitHub issue templates
├── VerDairy                         # Project version diary
├── ROADMAP.md                       # Development roadmap
├── CHANGELOG.md                     # Change history
├── TODO.md                          # Current task list
└── README.md                        # Project overview and quick start
```

Generated outputs such as logs, checkpoints, models, videos, and large datasets should not be committed.

## Development Plan

See:

- `ROADMAP.md` for staged goals.
- `TODO.md` for current task tracking.
- `CHANGELOG.md` for user-facing changes.
- `docs/project-management.md` for branch, commit, release, and experiment rules.
- `docs/experiment-log-template.md` for reproducible experiment notes.

## Notes

- Keep `main` stable and runnable.
- Use `dev` for integration work.
- Use `feature/*`, `fix/*`, and `experiment/*` branches for isolated changes.
- Use tags/releases for stable versions instead of long-lived `version1/version2/version3` branches.
- Do not commit large experiment artifacts, model weights, training logs, or private credentials.
