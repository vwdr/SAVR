"""Fail-closed whole-attempt inference-mode lifecycle for V08."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any


class V08InferenceLifecycle:
    """Enter inference mode after the transition gate and restore it on exit."""

    def __init__(self, *, torch_module: Any, transition: Callable[[str], dict[str, Any]]) -> None:
        self._torch = torch_module
        self._transition = transition
        self._guard: Any = None
        self._entry: dict[str, Any] | None = None
        self._exit: dict[str, Any] | None = None

    def snapshot_and_enter(self, physical_id: str) -> dict[str, Any]:
        sample = self._transition(physical_id)
        if self._guard is not None:
            return sample
        before = {
            "grad_enabled": bool(self._torch.is_grad_enabled()),
            "inference_mode_enabled": bool(self._torch.is_inference_mode_enabled()),
        }
        if before["inference_mode_enabled"]:
            raise RuntimeError("V08 refuses an inherited inference-mode context")
        self._guard = self._torch.inference_mode()
        self._guard.__enter__()
        after = {
            "grad_enabled": bool(self._torch.is_grad_enabled()),
            "inference_mode_enabled": bool(self._torch.is_inference_mode_enabled()),
        }
        if after != {"grad_enabled": False, "inference_mode_enabled": True}:
            self.close()
            raise RuntimeError("V08 failed to establish exact inference semantics")
        self._entry = {"before": before, "after": after}
        return sample

    def close(self) -> None:
        if self._guard is None:
            return
        guard = self._guard
        self._guard = None
        guard.__exit__(None, None, None)
        self._exit = {
            "grad_enabled": bool(self._torch.is_grad_enabled()),
            "inference_mode_enabled": bool(self._torch.is_inference_mode_enabled()),
        }
        expected = self._entry["before"] if self._entry is not None else None
        if self._exit != expected:
            raise RuntimeError("V08 did not restore prior thread-local grad state")

    def active_attestation(self) -> dict[str, Any]:
        return {
            "entered": self._entry is not None,
            "grad_enabled": bool(self._torch.is_grad_enabled()),
            "inference_mode_enabled": bool(self._torch.is_inference_mode_enabled()),
        }

    def lifecycle_record(self) -> dict[str, Any]:
        return {
            "entry": deepcopy(self._entry),
            "exit": deepcopy(self._exit),
            "restored": self._entry is not None and self._exit == self._entry["before"],
        }
