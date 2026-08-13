#!/usr/bin/env python3
"""Invoke canonical LIBERO configuration for the frozen V10 identity."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> int:
    from savr.acr.v5_d_v10_adapter import install_v10_adapters
    from savr.acr.v5_d_v10_runtime import load_v10

    install_v10_adapters()
    if load_v10(ROOT)["current_authorization"].get("gpu_inspection_or_selection") is not True:
        raise SystemExit("V10 GPU execution is not authorized")
    import prepare_acr_v5_d_libero_config as preparation

    return preparation.main()


if __name__ == "__main__":
    raise SystemExit(main())
