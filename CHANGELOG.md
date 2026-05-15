# Changelog

This project follows a lightweight Keep a Changelog style.

## [Unreleased]

### Added

- Added `Xytheta-Transa-Offroad-v0.0` with fixed generated 15 m x 15 m off-road terrain.
- Added a project-local four-wheel differential-drive URDF car asset.
- Added `DifferentialWheelVelocityAction` for articulation wheel velocity control.
- Added 16-channel off-road LiDAR over terrain and irregular obstacles.

### Changed

- Updated `scripts/random_agent.py` to discover cars, LiDAR sensors, bounds, and LiDAR range from each task config.
- Updated README and `VerDairy` with off-road environment usage and version notes.

### Fixed

- Kept planar projection logic limited to old rigid-body cars so articulation vehicles can roll and pitch on terrain.

## [v0.0] - 2026-05-12

### Added

- Added Isaac Lab extension project structure.
- Added `Xytheta-Transa-v0.0` environment.
- Added exploration reward based on newly LiDAR-covered, previously unexplored area.
- Added wall proximity penalty and stronger termination penalty.
- Added random, demo, and reward-guided rollout support.
- Added exploration CSV logging support.
- Added `VerDairy` version diary.
- Added project management documents, GitHub issue templates, and pull request template.

### Changed

- Replaced the default Isaac Lab template README with a project-specific README.
- Expanded `.gitignore` for experiment outputs, model files, logs, datasets, and credentials.
