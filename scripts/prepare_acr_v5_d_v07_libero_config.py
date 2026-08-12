#!/usr/bin/env python3
"""Invoke canonical LIBERO configuration for the frozen V07 identity."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> int:
    from savr.acr.v5_d_v07_adapter import install_v07_adapters

    install_v07_adapters()
    import prepare_acr_v5_d_libero_config as preparation

    return preparation.main()


if __name__ == "__main__":
    raise SystemExit(main())
