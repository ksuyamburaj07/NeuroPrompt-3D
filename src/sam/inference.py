"""Frozen SAM-Med3D inference interface.

This module contains only the tensor and call contract established in
M9A4.

It does not import the external SAM-Med3D repository and does not load
a checkpoint. An already-instantiated SAM-compatible model is supplied
by the caller.

Frozen contract
---------------
Image:
    float32 [1,1,128,128,128] in XYZ spatial order.

Image embedding:
    [1,384,8,8,8].

Point:
    one XYZ coordinate,
    float32 tensor [1,1,3].

Label:
    integer tensor [1,1].
    0 = negative
    1 = positive.

Important:
    The adapter does NOT add +0.5 to prompt coordinates. SAM-Med3D's
    prompt encoder owns its internal half-voxel shift.

Decoder:
    multimask_output=False.

Final mask:
    interpolate logits to 128^3,
    trilinear,
    align_corners=False,
    sigmoid,
    threshold >= 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


SAM_MODEL_SHAPE_XYZ = (
    128,
    128,
    128,
)

SAM_IMAGE_EMBEDDING_SHAPE = (
    1,
    384,
    8,
    8,
    8,
)

SAM_MASK_THRESHOLD = 0.5

SAM_NEGATIVE_LABEL = 0
SAM_POSITIVE_LABEL = 1


class SAMIntegrityError(ValueError):
    """Hard failure for unexpected frozen SAM interface behavior."""


@dataclass(frozen=True)
class SAMPointBranchResult:
    """One independent SAM point-prompt branch result."""

    coordinate_xyz: tuple[int, int, int]
    label: int

    probability_model_xyz: np.ndarray
    mask_model_xyz: np.ndarray


def prepare_sam_image_tensor(
    image_model_xyz: np.ndarray,
    *,
    device: str | torch.device,
) -> torch.Tensor:
    """Convert normalized XYZ model image to frozen SAM tensor format."""

    image = np.asarray(
        image_model_xyz,
        dtype=np.float32,
    )

    if image.shape != SAM_MODEL_SHAPE_XYZ:
        raise SAMIntegrityError(
            "SAM image must have shape [128,128,128]."
        )

    if not np.isfinite(
        image
    ).all():
        raise SAMIntegrityError(
            "SAM image contains non-finite values."
        )

    tensor = (
        torch.from_numpy(
            np.ascontiguousarray(
                image
            )
        )
        .unsqueeze(
            0
        )
        .unsqueeze(
            0
        )
        .to(
            device=torch.device(
                device
            ),
            dtype=torch.float32,
        )
    )

    expected_shape = (
        1,
        1,
        *SAM_MODEL_SHAPE_XYZ,
    )

    if tuple(
        tensor.shape
    ) != expected_shape:
        raise SAMIntegrityError(
            "SAM image tensor has unexpected geometry."
        )

    return tensor


def encode_sam_image(
    sam_model: torch.nn.Module,
    image_tensor: torch.Tensor,
) -> torch.Tensor:
    """Run the frozen SAM image encoder contract."""

    expected_shape = (
        1,
        1,
        *SAM_MODEL_SHAPE_XYZ,
    )

    if tuple(
        image_tensor.shape
    ) != expected_shape:
        raise SAMIntegrityError(
            "SAM image tensor must have shape [1,1,128,128,128]."
        )

    if image_tensor.dtype != torch.float32:
        raise SAMIntegrityError(
            "SAM image tensor must be float32."
        )

    if not torch.isfinite(
        image_tensor
    ).all():
        raise SAMIntegrityError(
            "SAM image tensor contains non-finite values."
        )

    if not hasattr(
        sam_model,
        "image_encoder",
    ):
        raise SAMIntegrityError(
            "SAM model is missing image_encoder."
        )

    with torch.inference_mode():
        image_embedding = (
            sam_model.image_encoder(
                image_tensor
            )
        )

    if not isinstance(
        image_embedding,
        torch.Tensor,
    ):
        raise SAMIntegrityError(
            "SAM image_encoder must return a tensor."
        )

    if tuple(
        image_embedding.shape
    ) != SAM_IMAGE_EMBEDDING_SHAPE:
        raise SAMIntegrityError(
            "SAM image embedding has unexpected geometry."
        )

    if not torch.isfinite(
        image_embedding
    ).all():
        raise SAMIntegrityError(
            "SAM image embedding contains non-finite values."
        )

    return image_embedding


def _validate_point_xyz(
    coordinate_xyz: Sequence[int],
) -> tuple[int, int, int]:
    coordinate = np.asarray(
        coordinate_xyz,
        dtype=np.int64,
    )

    if coordinate.shape != (
        3,
    ):
        raise SAMIntegrityError(
            "SAM point must contain exactly 3 XYZ values."
        )

    if (
        np.any(
            coordinate < 0
        )
        or np.any(
            coordinate
            >= np.asarray(
                SAM_MODEL_SHAPE_XYZ,
                dtype=np.int64,
            )
        )
    ):
        raise SAMIntegrityError(
            "SAM point lies outside the 128^3 model grid."
        )

    return tuple(
        int(value)
        for value in coordinate
    )


def run_sam_point_branch(
    sam_model: torch.nn.Module,
    image_embedding: torch.Tensor,
    coordinate_xyz: Sequence[int],
    label: int,
) -> SAMPointBranchResult:
    """Run exactly one independent point-prompt SAM branch."""

    if tuple(
        image_embedding.shape
    ) != SAM_IMAGE_EMBEDDING_SHAPE:
        raise SAMIntegrityError(
            "SAM image embedding has unexpected geometry."
        )

    if not torch.isfinite(
        image_embedding
    ).all():
        raise SAMIntegrityError(
            "SAM image embedding contains non-finite values."
        )

    coordinate = _validate_point_xyz(
        coordinate_xyz
    )

    label_int = int(
        label
    )

    if label_int not in (
        SAM_NEGATIVE_LABEL,
        SAM_POSITIVE_LABEL,
    ):
        raise SAMIntegrityError(
            "SAM point label must be 0 or 1."
        )

    if not hasattr(
        sam_model,
        "prompt_encoder",
    ):
        raise SAMIntegrityError(
            "SAM model is missing prompt_encoder."
        )

    if not hasattr(
        sam_model,
        "mask_decoder",
    ):
        raise SAMIntegrityError(
            "SAM model is missing mask_decoder."
        )

    device = image_embedding.device

    # Frozen M9A4 behavior:
    # raw integer XYZ becomes float32 [1,1,3].
    # Do NOT add +0.5 here.
    coordinate_tensor = torch.as_tensor(
        coordinate,
        dtype=torch.float32,
        device=device,
    ).reshape(
        1,
        1,
        3,
    )

    label_tensor = torch.as_tensor(
        [
            [
                label_int
            ]
        ],
        dtype=torch.int64,
        device=device,
    )

    with torch.inference_mode():
        sparse_embeddings, dense_embeddings = (
            sam_model.prompt_encoder(
                points=(
                    coordinate_tensor,
                    label_tensor,
                ),
                boxes=None,
                masks=None,
            )
        )

        if not isinstance(
            sparse_embeddings,
            torch.Tensor,
        ):
            raise SAMIntegrityError(
                "SAM sparse prompt embedding must be a tensor."
            )

        if not isinstance(
            dense_embeddings,
            torch.Tensor,
        ):
            raise SAMIntegrityError(
                "SAM dense prompt embedding must be a tensor."
            )

        if not torch.isfinite(
            sparse_embeddings
        ).all():
            raise SAMIntegrityError(
                "SAM sparse prompt embedding contains non-finite values."
            )

        if not torch.isfinite(
            dense_embeddings
        ).all():
            raise SAMIntegrityError(
                "SAM dense prompt embedding contains non-finite values."
            )

        dense_pe = (
            sam_model
            .prompt_encoder
            .get_dense_pe()
        )

        if not isinstance(
            dense_pe,
            torch.Tensor,
        ):
            raise SAMIntegrityError(
                "SAM dense positional encoding must be a tensor."
            )

        if not torch.isfinite(
            dense_pe
        ).all():
            raise SAMIntegrityError(
                "SAM dense positional encoding contains non-finite values."
            )

        decoder_result = (
            sam_model.mask_decoder(
                image_embeddings=(
                    image_embedding
                ),
                image_pe=(
                    dense_pe
                ),
                sparse_prompt_embeddings=(
                    sparse_embeddings
                ),
                dense_prompt_embeddings=(
                    dense_embeddings
                ),
                multimask_output=False,
            )
        )

        if (
            not isinstance(
                decoder_result,
                (
                    tuple,
                    list,
                ),
            )
            or len(
                decoder_result
            ) < 1
        ):
            raise SAMIntegrityError(
                "SAM mask_decoder returned an unexpected result."
            )

        low_res_logits = (
            decoder_result[
                0
            ]
        )

        if not isinstance(
            low_res_logits,
            torch.Tensor,
        ):
            raise SAMIntegrityError(
                "SAM low-resolution logits must be a tensor."
            )

        if (
            low_res_logits.ndim
            != 5
            or int(
                low_res_logits.shape[
                    0
                ]
            )
            != 1
            or int(
                low_res_logits.shape[
                    1
                ]
            )
            != 1
        ):
            raise SAMIntegrityError(
                "SAM low-resolution logits have unexpected geometry."
            )

        if not torch.isfinite(
            low_res_logits
        ).all():
            raise SAMIntegrityError(
                "SAM low-resolution logits contain non-finite values."
            )

        high_res_logits = F.interpolate(
            low_res_logits,
            size=(
                SAM_MODEL_SHAPE_XYZ
            ),
            mode="trilinear",
            align_corners=False,
        )

        if tuple(
            high_res_logits.shape
        ) != (
            1,
            1,
            *SAM_MODEL_SHAPE_XYZ,
        ):
            raise SAMIntegrityError(
                "SAM high-resolution logits have unexpected geometry."
            )

        if not torch.isfinite(
            high_res_logits
        ).all():
            raise SAMIntegrityError(
                "SAM high-resolution logits contain non-finite values."
            )

        probabilities = torch.sigmoid(
            high_res_logits
        )

        if not torch.isfinite(
            probabilities
        ).all():
            raise SAMIntegrityError(
                "SAM probabilities contain non-finite values."
            )

        mask = (
            probabilities
            >= SAM_MASK_THRESHOLD
        )

    probability_xyz = (
        probabilities[
            0,
            0,
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )

    mask_xyz = (
        mask[
            0,
            0,
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            bool,
            copy=False,
        )
    )

    return SAMPointBranchResult(
        coordinate_xyz=(
            coordinate
        ),
        label=(
            label_int
        ),
        probability_model_xyz=(
            np.ascontiguousarray(
                probability_xyz
            )
        ),
        mask_model_xyz=(
            np.ascontiguousarray(
                mask_xyz
            )
        ),
    )
