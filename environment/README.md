# SAVR Environment

Phase 1 uses a project-local Micromamba environment. Nothing is installed into the user account, shell configuration, or operating system.

Tracked inputs:

- `phase1-conda.yml`: initial Conda-layer requirements
- `libero-config.yaml`: noninteractive, project-local LIBERO paths
- `../scripts/setup_phase1_environment.sh`: pinned source and pip installation procedure
- `../scripts/verify_phase1_environment.py`: import and CPU-render verification

Generated inventories are written under `environment/locks/` and intentionally ignored because they are created on TITAN. The Phase 1 report records their hashes and the exact environment commit.

Do not run the setup script outside `/home/ved/SAVR`.
