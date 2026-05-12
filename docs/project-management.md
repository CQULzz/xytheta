# Project Management Guide

This document defines how to manage branches, commits, releases, experiments, and large files for this repository.

## Branch Strategy

### `main`

Stable branch. Only merge code that is verified, runnable, and documented.

### `dev`

Development integration branch. Feature, fix, and experiment branches should merge here first.

### `feature/xxx`

Use for concrete feature work.

Examples:

- `feature/data-collector`
- `feature/random-env-generator`
- `feature/config-system`

### `fix/xxx`

Use for bug fixes.

Examples:

- `fix/install-error`
- `fix/path-loading-bug`
- `fix/simulation-reset`

### `experiment/xxx`

Use for exploratory work that may not be merged.

Examples:

- `experiment/new-reward-function`
- `experiment/lidar-resolution-test`
- `experiment/ppo-baseline`

### `release/x.x.x`

Optional. Use only when a release needs final stabilization. For a lightweight project, it is acceptable to release directly from `main` with a tag.

## When To Create A Branch

- Create `feature/*` before adding a new capability.
- Create `fix/*` before fixing a specific bug.
- Create `experiment/*` before trying uncertain algorithms, reward functions, or parameter sweeps.
- Avoid long-lived branches named `version1`, `version2`, or `final`.

## Merge Rules

- Merge feature and fix branches into `dev` first.
- Run smoke tests before merging `dev` into `main`.
- Merge to `main` only when the code is stable enough for others to run.
- Keep PR descriptions clear: what changed, how it was tested, and what risks remain.

## Commit Message Rules

Use concise English commit messages:

- `Add xxx`
- `Fix xxx`
- `Update xxx`
- `Refactor xxx`
- `Remove xxx`
- `Document xxx`
- `Test xxx`

Examples:

- `Add random environment generator`
- `Fix config loading error`
- `Update README installation guide`
- `Refactor data collection pipeline`
- `Document branch strategy`
- `Test simulation reset logic`

Avoid vague messages:

- `update`
- `test`
- `改了一下`
- `临时修改`
- `最终版`
- `最终最终版`

## Versioning Rules

Use semantic versioning:

- `v0.1.0`: initial runnable version
- `v0.2.0`: important new feature
- `v0.2.1`: small fix
- `v1.0.0`: stable milestone

Rules:

- Use patch versions for small fixes.
- Use minor versions for complete new features.
- Use major versions for stable milestones or large structural changes.
- Update `CHANGELOG.md` before each tag.
- Tag stable versions from `main`.

Example:

```bash
git checkout main
git pull
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

## Experiment Records

Use `docs/experiment-log-template.md` for simulation, training, reward, and parameter comparison experiments.

Each experiment should record:

- commit hash
- branch
- tag if available
- command
- key config
- input data
- output files
- metrics
- observations
- next steps

Recommended output locations:

- `logs/`
- `runs/`
- `outputs/`
- `results/`

These directories should not be committed. Only commit small summaries, selected figures, or documentation when needed.

## Large File Policy

Do not commit:

- model weights
- checkpoints
- training logs
- simulation videos
- raw datasets
- large CSV/NPY/NPZ files
- private credentials or tokens

If small example data is necessary, place it under `sample_data/` or `assets/` and document why it is safe to commit.

Before committing, check:

```bash
git status --short
git diff --stat
```

If a large file was accidentally staged:

```bash
git restore --staged path/to/file
```

## Recommended First-Time Git Setup

If the repository has no remote:

```bash
git remote add origin https://github.com/CQULzz/xytheta.git
```

Recommended branch setup:

```bash
git branch -M main
git checkout -b dev
```

Do not push until the working tree is reviewed.
