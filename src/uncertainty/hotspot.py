"""Frozen automatic uncertainty-hotspot selection.

Internal NeuroPrompt coordinates in this module are ZYX.

Frozen hotspot:
    MRI foreground
    intersect
    physical <=10 mm deterministic coarse-mask boundary shell

The selected hotspot is the maximum predictive variance in that region.
np.argmax over the C-contiguous ZYX array provides the frozen
deterministic C-order / lexicographic tie behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt


BOUNDARY_SHELL_MM = 10.0


class HotspotIntegrityError(ValueError):
    """Hard failure for malformed/non-finite hotspot inputs."""


@dataclass(frozen=True)
class HotspotResult:
    """Result of ground-truth-free frozen hotspot selection."""

    coordinate_zyx: Optional[
        tuple[int, int, int]
    ]

    predictive_variance: Optional[
        float
    ]

    candidate_count: int

    abstention_condition: Optional[
        str
    ]

    @property
    def available(
        self,
    ) -> bool:
        return (
            self.coordinate_zyx
            is not None
        )


def _spacing3(
    spacing_zyx_mm,
) -> tuple[float, float, float]:
    spacing = np.asarray(
        spacing_zyx_mm,
        dtype=np.float64,
    )

    if spacing.shape != (3,):
        raise HotspotIntegrityError(
            "spacing_zyx_mm must contain exactly 3 values."
        )

    if not np.isfinite(
        spacing
    ).all():
        raise HotspotIntegrityError(
            "spacing_zyx_mm must be finite."
        )

    if np.any(
        spacing <= 0.0
    ):
        raise HotspotIntegrityError(
            "spacing_zyx_mm values must be positive."
        )

    return tuple(
        float(value)
        for value in spacing
    )


def physical_boundary_shell_zyx(
    coarse_mask_zyx: np.ndarray,
    spacing_zyx_mm,
    shell_mm: float = BOUNDARY_SHELL_MM,
) -> np.ndarray:
    """Frozen two-sided physical deterministic-mask boundary shell."""

    coarse = np.asarray(
        coarse_mask_zyx
    ).astype(
        bool
    )

    if coarse.ndim != 3:
        raise HotspotIntegrityError(
            "coarse_mask_zyx must be 3D."
        )

    spacing = _spacing3(
        spacing_zyx_mm
    )

    shell = float(
        shell_mm
    )

    if (
        not math.isfinite(
            shell
        )
        or shell < 0.0
    ):
        raise HotspotIntegrityError(
            "shell_mm must be finite and non-negative."
        )

    inside_distance = (
        distance_transform_edt(
            coarse,
            sampling=spacing,
        )
    )

    outside_distance = (
        distance_transform_edt(
            np.logical_not(
                coarse
            ),
            sampling=spacing,
        )
    )

    return np.logical_or(
        np.logical_and(
            coarse,
            inside_distance
            <= shell,
        ),
        np.logical_and(
            np.logical_not(
                coarse
            ),
            outside_distance
            <= shell,
        ),
    )


def select_uncertainty_hotspot(
    mri_czyx: np.ndarray,
    coarse_mask_zyx: np.ndarray,
    predictive_variance_zyx: np.ndarray,
    spacing_zyx_mm,
) -> HotspotResult:
    """Select the frozen automatic uncertainty hotspot.

    Semantic valid-case failures are represented explicitly in the
    returned result so the final orchestrator can map them to the frozen
    ABSTAIN_BASELINE action.

    Integrity failures raise HotspotIntegrityError.
    """

    mri = np.asarray(
        mri_czyx
    )

    coarse_raw = np.asarray(
        coarse_mask_zyx
    )

    variance = np.asarray(
        predictive_variance_zyx,
        dtype=np.float32,
    )

    if mri.ndim != 4:
        raise HotspotIntegrityError(
            "mri_czyx must have shape [C,Z,Y,X]."
        )

    spatial_shape = tuple(
        int(value)
        for value in mri.shape[1:]
    )

    if coarse_raw.shape != spatial_shape:
        raise HotspotIntegrityError(
            "coarse_mask_zyx shape does not match MRI geometry."
        )

    if variance.shape != spatial_shape:
        raise HotspotIntegrityError(
            "predictive_variance_zyx shape does not match MRI geometry."
        )

    if not np.isfinite(
        mri
    ).all():
        raise HotspotIntegrityError(
            "MRI contains non-finite values."
        )

    if (
        np.issubdtype(
            coarse_raw.dtype,
            np.number,
        )
        and not np.isfinite(
            coarse_raw
        ).all()
    ):
        raise HotspotIntegrityError(
            "Coarse mask contains non-finite values."
        )

    if not np.isfinite(
        variance
    ).all():
        raise HotspotIntegrityError(
            "Predictive variance contains non-finite values."
        )

    spacing = _spacing3(
        spacing_zyx_mm
    )

    coarse = coarse_raw.astype(
        bool
    )

    mri_foreground = np.any(
        mri != 0,
        axis=0,
    )

    if not np.any(
        mri_foreground
    ):
        return HotspotResult(
            coordinate_zyx=None,
            predictive_variance=None,
            candidate_count=0,
            abstention_condition=(
                "empty MRI foreground"
            ),
        )

    if not np.any(
        coarse
    ):
        return HotspotResult(
            coordinate_zyx=None,
            predictive_variance=None,
            candidate_count=0,
            abstention_condition=(
                "empty deterministic coarse foreground"
            ),
        )

    shell = (
        physical_boundary_shell_zyx(
            coarse,
            spacing,
            BOUNDARY_SHELL_MM,
        )
    )

    candidates = (
        mri_foreground
        & shell
    )

    candidate_count = int(
        np.count_nonzero(
            candidates
        )
    )

    if candidate_count == 0:
        return HotspotResult(
            coordinate_zyx=None,
            predictive_variance=None,
            candidate_count=0,
            abstention_condition=(
                "no valid uncertainty-boundary hotspot candidate"
            ),
        )

    masked_variance = np.where(
        candidates,
        variance,
        -np.inf,
    )

    flat_index = int(
        np.argmax(
            masked_variance
        )
    )

    coordinate = tuple(
        int(value)
        for value in np.unravel_index(
            flat_index,
            variance.shape,
        )
    )

    value = float(
        variance[
            coordinate
        ]
    )

    return HotspotResult(
        coordinate_zyx=coordinate,
        predictive_variance=value,
        candidate_count=candidate_count,
        abstention_condition=None,
    )
