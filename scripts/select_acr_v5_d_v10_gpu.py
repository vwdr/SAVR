#!/usr/bin/env python3
"""Aggregate-only V5-D selection wrapper for V10 after future authorization."""

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
        raise SystemExit("V10 GPU inspection or selection is not authorized")
    import select_acr_v5_d_gpu as selector

    return selector.main()


if __name__ == "__main__":
    raise SystemExit(main())
