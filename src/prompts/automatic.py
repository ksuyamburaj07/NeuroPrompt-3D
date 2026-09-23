"""Frozen AutoPrompt-v2 geometry.

Internal NeuroPrompt coordinates are ZYX.

For one automatic uncertainty hotspot, AutoPrompt-v2 defines two
independent candidate hypotheses:

FP correction:
    nearest MRI-foreground voxel inside deterministic coarse foreground
    semantic label 0 (negative)

FN correction:
    nearest MRI-foreground voxel outside deterministic coarse foreground
    semantic label 1 (positive)

Distance is physical Euclidean distance in millimetres.

Candidate coordinates are enumerated with np.argwhere, so exact
distance ties are resolved by the first C-order / lexicographic
coordinate, matching the frozen development implementation.

The final M9C3 v1 policy selects only the FP branch. The FN candidate
may still be generated as part of frozen AutoPrompt-v2 geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import numpy as np


FP_BRANCH = "false_positive_correction"
FN_BRANCH = "false_negative_correction"

FP_LABEL = 0
FN_LABEL = 1

FINAL_V1_SELECTED_BRANCH = FP_BRANCH

NO_VALID_FP_CONDITION = (
    "no valid FP correction hypothesis"
)


class PromptIntegrityError(ValueError):
    """Hard failure for malformed AutoPrompt geometry."""


@dataclass(frozen=True)
class PromptCandidate:
    """One frozen automatic prompt hypothesis."""

    branch: str
    coordinate_zyx: tuple[int, int, int]
    label: int
    distance_mm: float


@dataclass(frozen=True)
class AutoPromptV2Result:
    """Independent frozen FP/FN hypotheses."""

    hotspot_zyx: tuple[int, int, int]
    fp_candidate: Optional[PromptCandidate]
    fn_candidate: Optional[PromptCandidate]

    @property
    def final_v1_fp_available(
        self,
    ) -> bool:
        return (
            self.fp_candidate
            is not None
        )


@dataclass(frozen=True)
class FinalV1PromptSelection:
    """Final M9C3 branch-selection result."""

    candidate: Optional[PromptCandidate]
    abstention_condition: Optional[str]

    @property
    def available(
        self,
    ) -> bool:
        return (
            self.candidate
            is not None
        )


def _spacing3(
    spacing_zyx_mm: Sequence[float],
) -> np.ndarray:
    spacing = np.asarray(
        spacing_zyx_mm,
        dtype=np.float64,
    )

    if spacing.shape != (3,):
        raise PromptIntegrityError(
            "spacing_zyx_mm must contain exactly 3 values."
        )

    if not np.isfinite(
        spacing
    ).all():
        raise PromptIntegrityError(
            "spacing_zyx_mm must be finite."
        )

    if np.any(
        spacing <= 0.0
    ):
        raise PromptIntegrityError(
            "spacing_zyx_mm values must be positive."
        )

    return spacing


def _coordinate3(
    coordinate_zyx: Sequence[int],
) -> np.ndarray:
    coordinate = np.asarray(
        coordinate_zyx,
        dtype=np.int64,
    )

    if coordinate.shape != (3,):
        raise PromptIntegrityError(
            "hotspot_zyx must contain exactly 3 values."
        )

    return coordinate


def _nearest_candidate(
    candidate_mask_zyx: np.ndarray,
    hotspot_zyx: np.ndarray,
    spacing_zyx_mm: np.ndarray,
    *,
    branch: str,
    label: int,
) -> Optional[PromptCandidate]:
    """Return physically nearest candidate with frozen tie behavior."""

    coordinates = np.argwhere(
        candidate_mask_zyx
    )

    if coordinates.shape[0] == 0:
        return None

    displacement_mm = (
        (
            coordinates.astype(
                np.float64
            )
            - hotspot_zyx.astype(
                np.float64
            )
        )
        * spacing_zyx_mm[
            None,
            :
        ]
    )

    squared_distance = np.sum(
        displacement_mm
        ** 2,
        axis=1,
        dtype=np.float64,
    )

    # np.argmin returns the first occurrence. Because coordinates came
    # from np.argwhere, exact ties follow deterministic C-order /
    # lexicographic ZYX ordering.
    selected_index = int(
        np.argmin(
            squared_distance
        )
    )

    selected = coordinates[
        selected_index
    ]

    distance_mm = math.sqrt(
        float(
            squared_distance[
                selected_index
            ]
        )
    )

    return PromptCandidate(
        branch=branch,
        coordinate_zyx=tuple(
            int(value)
            for value in selected
        ),
        label=int(
            label
        ),
        distance_mm=float(
            distance_mm
        ),
    )


def generate_autoprompt_v2(
    mri_czyx: np.ndarray,
    coarse_mask_zyx: np.ndarray,
    hotspot_zyx: Sequence[int],
    spacing_zyx_mm: Sequence[float],
) -> AutoPromptV2Result:
    """Generate independent frozen FP and FN hypotheses.

    Ground truth is neither accepted nor used.
    """

    mri = np.asarray(
        mri_czyx
    )

    coarse_raw = np.asarray(
        coarse_mask_zyx
    )

    if mri.ndim != 4:
        raise PromptIntegrityError(
            "mri_czyx must have shape [C,Z,Y,X]."
        )

    spatial_shape = tuple(
        int(value)
        for value in mri.shape[1:]
    )

    if coarse_raw.shape != spatial_shape:
        raise PromptIntegrityError(
            "coarse_mask_zyx shape does not match MRI geometry."
        )

    if not np.isfinite(
        mri
    ).all():
        raise PromptIntegrityError(
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
        raise PromptIntegrityError(
            "Coarse mask contains non-finite values."
        )

    hotspot = _coordinate3(
        hotspot_zyx
    )

    shape = np.asarray(
        spatial_shape,
        dtype=np.int64,
    )

    if (
        np.any(
            hotspot < 0
        )
        or np.any(
            hotspot >= shape
        )
    ):
        raise PromptIntegrityError(
            "hotspot_zyx lies outside MRI geometry."
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

    hotspot_tuple = tuple(
        int(value)
        for value in hotspot
    )

    if not mri_foreground[
        hotspot_tuple
    ]:
        raise PromptIntegrityError(
            "hotspot_zyx must lie in MRI foreground."
        )

    fp_candidate_mask = (
        mri_foreground
        & coarse
    )

    fn_candidate_mask = (
        mri_foreground
        & np.logical_not(
            coarse
        )
    )

    fp_candidate = _nearest_candidate(
        fp_candidate_mask,
        hotspot,
        spacing,
        branch=FP_BRANCH,
        label=FP_LABEL,
    )

    fn_candidate = _nearest_candidate(
        fn_candidate_mask,
        hotspot,
        spacing,
        branch=FN_BRANCH,
        label=FN_LABEL,
    )

    return AutoPromptV2Result(
        hotspot_zyx=hotspot_tuple,
        fp_candidate=fp_candidate,
        fn_candidate=fn_candidate,
    )


def select_final_v1_fp_prompt(
    result: AutoPromptV2Result,
) -> FinalV1PromptSelection:
    """Apply the frozen M9C3 branch-selection rule.

    Final v1:
        use exactly one FP negative point when available;
        never select the FN branch;
        missing FP hypothesis is semantic abstention.
    """

    candidate = (
        result.fp_candidate
    )

    if candidate is None:
        return FinalV1PromptSelection(
            candidate=None,
            abstention_condition=(
                NO_VALID_FP_CONDITION
            ),
        )

    if (
        candidate.branch
        != FP_BRANCH
        or candidate.label
        != FP_LABEL
    ):
        raise PromptIntegrityError(
            "Final-v1 FP candidate violates the frozen prompt contract."
        )

    return FinalV1PromptSelection(
        candidate=candidate,
        abstention_condition=None,
    )
