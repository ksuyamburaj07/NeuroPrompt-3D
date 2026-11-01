"""Verified display-only focus for frozen refinement removals."""

import numpy as np

from app.backend.services.frozen_slice_service import (
    FrozenSliceError,
    _load_mask_arrays,
)
from app.backend.services.frozen_test_service import (
    load_verified_evaluation,
)

EXPECTED_SHAPE_ZYX = (182, 218, 182)


def get_verified_removed_focus(case_index: int) -> dict:
    """Return a real removed-mask voxel, or explicit unavailability."""
    if (
        isinstance(case_index, bool)
        or not isinstance(case_index, int)
        or not 1 <= case_index <= 129
    ):
        raise ValueError("Invalid frozen case index.")

    evaluation = load_verified_evaluation()
    _, masks = _load_mask_arrays(case_index, evaluation)

    baseline = masks["baseline"]
    final = masks["final"]
    removed = masks["removed"]

    if any(
        array.shape != EXPECTED_SHAPE_ZYX
        for array in (baseline, final, removed)
    ):
        raise FrozenSliceError("Unexpected frozen mask geometry.")

    baseline_fg = baseline > 0
    final_fg = final > 0
    removed_fg = removed > 0

    if np.any(final_fg & ~baseline_fg):
        raise FrozenSliceError(
            "Frozen final mask contains additions."
        )

    expected_removed = baseline_fg & ~final_fg

    if not np.array_equal(removed_fg, expected_removed):
        raise FrozenSliceError(
            "Frozen removal mask is inconsistent."
        )

    removed_voxels = int(np.count_nonzero(removed_fg))

    if removed_voxels == 0:
        return {
            "case_index": case_index,
            "available": False,
            "coordinates": None,
            "removed_voxels": 0,
            "mask_source": "removed",
            "coordinate_system": "native_voxel_xyz",
            "selection": "unavailable_no_removed_voxels",
        }

    positions = np.argwhere(removed_fg)
    centroid = positions.mean(axis=0)

    distances = np.sum(
        (positions - centroid) ** 2,
        axis=1,
    )

    # Stable ZYX iteration order and first-entry tie handling.
    z, y, x = map(
        int,
        positions[np.argmin(distances)],
    )

    if not removed_fg[z, y, x]:
        raise FrozenSliceError(
            "Selected focus is outside the removed region."
        )

    return {
        "case_index": case_index,
        "available": True,
        "coordinates": {"x": x, "y": y, "z": z},
        "removed_voxels": removed_voxels,
        "mask_source": "removed",
        "coordinate_system": "native_voxel_xyz",
        "selection": "foreground_nearest_removed_centroid",
    }
