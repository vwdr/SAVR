#!/usr/bin/env python3
"""Invoke the frozen V5-D runner through the isolated V04 adapters."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

def main() -> int:
    from savr.acr.v5_d_v04_adapter import install_v04_adapters

    install_v04_adapters()
    import run_acr_v5_d as runner

    runner.V03_RUN_ID = "acr-v5d-real-tensor-feasibility-v04"
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
