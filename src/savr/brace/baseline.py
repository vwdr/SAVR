"""Isolated, minimal corrections for the released VLA-Cache evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from savr.brace.b1 import semantic_sha256
from savr.brace.types import B2ValidationError


OFFICIAL_PREVIOUS_BLOCK = (
    "            replay_images.append(img)\n"
    "            replay_images_wrist.append(img_wrist)\n"
    + "            \n"
    + "            # Save previous image\n"
    "            if prev_img is None:\n"
    "                prev_img = img\n"
    "                prev_img_wrist = img_wrist\n"
    "            else:\n"
    "                prev_img = replay_images[-1]\n"
    "                prev_img_wrist = replay_images_wrist[-1]\n"
    + "            \n"
    + '            observation["prev_images"] = [prev_img, prev_img_wrist]\n'
)

CORRECTED_PREVIOUS_BLOCK = '''            # Cache comparisons must use the images that produced last_caches.
            if prev_img is None:
                observation["prev_images"] = [img, img_wrist]
            else:
                observation["prev_images"] = [prev_img, prev_img_wrist]
            replay_images.append(img)
            replay_images_wrist.append(img_wrist)
'''

OFFICIAL_ACTION_BLOCK = '''                replay_images_heatmap.append(result_image[0])
                replay_images_wrist_heatmap.append(result_image[1])
'''

CORRECTED_ACTION_BLOCK = '''                replay_images_heatmap.append(result_image[0])
                replay_images_wrist_heatmap.append(result_image[1])
                prev_img = img
                prev_img_wrist = img_wrist
'''

OFFICIAL_ERROR_BLOCK = (
    "    except Exception as e:\n"
    '        log_message(f"Episode error: {e}", log_file)\n'
    + "        \n"
)

CORRECTED_ERROR_BLOCK = (
    "    except Exception as e:\n"
    '        log_message(f"Episode error: {e}", log_file)\n'
    "        raise\n"
    + "        \n"
)


def corrected_evaluator_source(source: str) -> str:
    """Apply only previous-cache-source semantics and fail-closed propagation."""

    replacements = (
        (OFFICIAL_PREVIOUS_BLOCK, CORRECTED_PREVIOUS_BLOCK),
        (OFFICIAL_ACTION_BLOCK, CORRECTED_ACTION_BLOCK),
        (OFFICIAL_ERROR_BLOCK, CORRECTED_ERROR_BLOCK),
    )
    corrected = source
    for old, new in replacements:
        if corrected.count(old) != 1:
            raise B2ValidationError("pinned evaluator no longer matches the reviewed correction")
        corrected = corrected.replace(old, new)
    if corrected == source:
        raise B2ValidationError("evaluator correction was not applied")
    return corrected


def correction_manifest(source: str) -> dict[str, Any]:
    corrected = corrected_evaluator_source(source)
    return {
        "correction_name": "faithfully-corrected-vla-cache",
        "original_sha256": semantic_sha256(source),
        "corrected_sha256": semantic_sha256(corrected),
        "previous_cache_source_fixed": OFFICIAL_PREVIOUS_BLOCK not in corrected,
        "episode_errors_propagate": CORRECTED_ERROR_BLOCK in corrected,
        "algorithm_configuration_unchanged": True,
    }


@dataclass
class FrameHistory:
    """Tracks only frames that actually produced the live VLA cache."""

    primary: Any | None = None
    wrist: Any | None = None

    def attach(self, primary: Any, wrist: Any) -> tuple[Any, Any]:
        if self.primary is None:
            return primary, wrist
        return self.primary, self.wrist

    def commit_cache_source(self, primary: Any, wrist: Any) -> None:
        self.primary, self.wrist = primary, wrist


def propagate_episode_error(operation: Callable[[], Any]) -> Any:
    """Make explicit that evaluator errors cannot become ordinary failures."""

    return operation()


def execute_p0_or_profile(
    *,
    enabled: bool,
    dense_forward: Callable[[], Any],
    accelerated_forward: Callable[[], Any],
) -> Any:
    """P0/disabled mode is the unchanged dense oracle, with no profile side effect."""

    if not enabled:
        return dense_forward()
    return accelerated_forward()
