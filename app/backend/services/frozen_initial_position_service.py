"""Read-only native-voxel positioning from frozen predictions."""

import numpy as np

from app.backend.services.frozen_slice_service import (
    FrozenSliceError,
    _load_mask_arrays,
)
from app.backend.services.frozen_test_service import (
    load_verified_evaluation,
)

EXPECTED_SHAPE_ZYX = (182, 218, 182)


def get_verified_initial_position(case_index: int) -> dict:
    """Select a deterministic display voxel without using ground truth."""
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

    if (
        baseline.shape != EXPECTED_SHAPE_ZYX
        or final.shape != EXPECTED_SHAPE_ZYX
    ):
        raise FrozenSliceError("Unexpected frozen mask geometry.")

    if np.any(final):
        chosen = final
        source = "final"
    elif np.any(baseline):
        chosen = baseline
        source = "baseline"
    else:
        chosen = None
        source = "volume_center"

    if chosen is None:
        z, y, x = (91, 109, 91)
    else:
        positions = np.argwhere(chosen > 0)
        centroid = positions.mean(axis=0)
        distances = np.sum((positions - centroid) ** 2, axis=1)

        # np.argwhere uses a stable ZYX iteration order.
        # np.argmin chooses the first entry in an exact tie.
        z, y, x = map(int, positions[np.argmin(distances)])

        if not chosen[z, y, x] > 0:
            raise FrozenSliceError("Selected voxel is not foreground.")

    return {
        "case_index": case_index,
        "coordinates": {"x": x, "y": y, "z": z},
        "mask_source": source,
        "coordinate_system": "native_voxel_xyz",
        "selection": (
            "volume_center_fallback"
            if chosen is None
            else "foreground_nearest_mask_centroid"
        ),
    }
