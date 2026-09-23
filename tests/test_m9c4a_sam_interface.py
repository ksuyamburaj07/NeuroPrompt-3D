import numpy as np
import pytest
import torch
from torch import nn

from src.sam.inference import (
    SAM_IMAGE_EMBEDDING_SHAPE,
    SAM_MASK_THRESHOLD,
    SAM_MODEL_SHAPE_XYZ,
    SAM_NEGATIVE_LABEL,
    SAM_POSITIVE_LABEL,
    SAMIntegrityError,
    encode_sam_image,
    prepare_sam_image_tensor,
    run_sam_point_branch,
)


class FakeImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input_shape = None

    def forward(
        self,
        image,
    ):
        self.last_input_shape = tuple(
            image.shape
        )

        return torch.zeros(
            SAM_IMAGE_EMBEDDING_SHAPE,
            dtype=torch.float32,
            device=image.device,
        )


class FakePromptEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.last_coordinate = None
        self.last_label = None
        self.last_boxes = "UNSET"
        self.last_masks = "UNSET"

    def forward(
        self,
        *,
        points,
        boxes,
        masks,
    ):
        coordinate_tensor, label_tensor = (
            points
        )

        self.last_coordinate = (
            coordinate_tensor
            .detach()
            .cpu()
            .clone()
        )

        self.last_label = (
            label_tensor
            .detach()
            .cpu()
            .clone()
        )

        self.last_boxes = boxes
        self.last_masks = masks

        sparse = torch.zeros(
            (
                1,
                1,
                384,
            ),
            dtype=torch.float32,
            device=coordinate_tensor.device,
        )

        dense = torch.zeros(
            (
                1,
                384,
                8,
                8,
                8,
            ),
            dtype=torch.float32,
            device=coordinate_tensor.device,
        )

        return (
            sparse,
            dense,
        )

    def get_dense_pe(
        self,
    ):
        return torch.zeros(
            (
                1,
                384,
                8,
                8,
                8,
            ),
            dtype=torch.float32,
        )


class FakeMaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        fill_value=0.0,
    ):
        super().__init__()

        self.fill_value = float(
            fill_value
        )

        self.last_multimask_output = None
        self.last_image_embedding_shape = None

    def forward(
        self,
        *,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output,
    ):
        self.last_multimask_output = (
            multimask_output
        )

        self.last_image_embedding_shape = (
            tuple(
                image_embeddings.shape
            )
        )

        device = (
            image_embeddings.device
        )

        logits = torch.full(
            (
                1,
                1,
                8,
                8,
                8,
            ),
            fill_value=self.fill_value,
            dtype=torch.float32,
            device=device,
        )

        iou = torch.ones(
            (
                1,
                1,
            ),
            dtype=torch.float32,
            device=device,
        )

        # Touch the supplied arguments so this fake still verifies that
        # the wrapper passes tensor-like values through the interface.
        assert torch.isfinite(
            image_pe
        ).all()

        assert torch.isfinite(
            sparse_prompt_embeddings
        ).all()

        assert torch.isfinite(
            dense_prompt_embeddings
        ).all()

        return (
            logits,
            iou,
        )


class FakeSAM(nn.Module):
    def __init__(
        self,
        *,
        decoder_fill=0.0,
    ):
        super().__init__()

        self.image_encoder = (
            FakeImageEncoder()
        )

        self.prompt_encoder = (
            FakePromptEncoder()
        )

        self.mask_decoder = (
            FakeMaskDecoder(
                fill_value=decoder_fill
            )
        )


def _image():
    return np.zeros(
        SAM_MODEL_SHAPE_XYZ,
        dtype=np.float32,
    )


def _embedding():
    return torch.zeros(
        SAM_IMAGE_EMBEDDING_SHAPE,
        dtype=torch.float32,
    )


def test_frozen_sam_constants():
    assert SAM_MODEL_SHAPE_XYZ == (
        128,
        128,
        128,
    )

    assert SAM_IMAGE_EMBEDDING_SHAPE == (
        1,
        384,
        8,
        8,
        8,
    )

    assert SAM_MASK_THRESHOLD == 0.5
    assert SAM_NEGATIVE_LABEL == 0
    assert SAM_POSITIVE_LABEL == 1


def test_prepare_image_tensor_exact_contract():
    image = _image()

    tensor = prepare_sam_image_tensor(
        image,
        device="cpu",
    )

    assert tuple(
        tensor.shape
    ) == (
        1,
        1,
        128,
        128,
        128,
    )

    assert (
        tensor.dtype
        == torch.float32
    )


def test_encode_image_exact_frozen_embedding_shape():
    model = FakeSAM()

    image_tensor = (
        prepare_sam_image_tensor(
            _image(),
            device="cpu",
        )
    )

    embedding = encode_sam_image(
        model,
        image_tensor,
    )

    assert tuple(
        embedding.shape
    ) == (
        1,
        384,
        8,
        8,
        8,
    )

    assert (
        model
        .image_encoder
        .last_input_shape
        == (
            1,
            1,
            128,
            128,
            128,
        )
    )


def test_point_is_passed_as_raw_xyz_without_half_voxel_shift():
    model = FakeSAM()

    coordinate = (
        10,
        20,
        30,
    )

    run_sam_point_branch(
        model,
        _embedding(),
        coordinate,
        label=0,
    )

    observed = (
        model
        .prompt_encoder
        .last_coordinate
    )

    expected = torch.tensor(
        [
            [
                [
                    10.0,
                    20.0,
                    30.0,
                ]
            ]
        ],
        dtype=torch.float32,
    )

    assert torch.equal(
        observed,
        expected,
    )

    # Explicitly reject an adapter-side +0.5 interpretation.
    assert not torch.equal(
        observed,
        expected
        + 0.5,
    )


def test_negative_fp_prompt_exact_tensor_contract():
    model = FakeSAM()

    result = run_sam_point_branch(
        model,
        _embedding(),
        (
            12,
            34,
            56,
        ),
        label=0,
    )

    assert (
        model
        .prompt_encoder
        .last_label
        .dtype
        == torch.int64
    )

    assert torch.equal(
        model
        .prompt_encoder
        .last_label,
        torch.tensor(
            [
                [
                    0
                ]
            ],
            dtype=torch.int64,
        ),
    )

    assert (
        model
        .prompt_encoder
        .last_boxes
        is None
    )

    assert (
        model
        .prompt_encoder
        .last_masks
        is None
    )

    assert (
        result.coordinate_xyz
        == (
            12,
            34,
            56,
        )
    )

    assert result.label == 0


def test_decoder_uses_multimask_false():
    model = FakeSAM()

    run_sam_point_branch(
        model,
        _embedding(),
        (
            1,
            2,
            3,
        ),
        label=0,
    )

    assert (
        model
        .mask_decoder
        .last_multimask_output
        is False
    )

    assert (
        model
        .mask_decoder
        .last_image_embedding_shape
        == SAM_IMAGE_EMBEDDING_SHAPE
    )


def test_zero_logits_threshold_at_half_is_foreground():
    model = FakeSAM(
        decoder_fill=0.0
    )

    result = run_sam_point_branch(
        model,
        _embedding(),
        (
            1,
            2,
            3,
        ),
        label=0,
    )

    assert result.probability_model_xyz.shape == (
        128,
        128,
        128,
    )

    assert result.mask_model_xyz.shape == (
        128,
        128,
        128,
    )

    assert result.probability_model_xyz.dtype == np.float32
    assert result.mask_model_xyz.dtype == bool

    assert np.allclose(
        result.probability_model_xyz,
        0.5,
    )

    # Frozen rule is >= 0.5.
    assert np.all(
        result.mask_model_xyz
    )


def test_negative_logits_are_background():
    model = FakeSAM(
        decoder_fill=-2.0
    )

    result = run_sam_point_branch(
        model,
        _embedding(),
        (
            1,
            2,
            3,
        ),
        label=0,
    )

    assert not np.any(
        result.mask_model_xyz
    )


def test_positive_label_supported_for_frozen_v2_geometry():
    model = FakeSAM()

    result = run_sam_point_branch(
        model,
        _embedding(),
        (
            1,
            2,
            3,
        ),
        label=1,
    )

    assert result.label == 1

    assert torch.equal(
        model
        .prompt_encoder
        .last_label,
        torch.tensor(
            [
                [
                    1
                ]
            ],
            dtype=torch.int64,
        ),
    )


@pytest.mark.parametrize(
    "coordinate",
    [
        (
            -1,
            0,
            0,
        ),
        (
            128,
            0,
            0,
        ),
        (
            0,
            128,
            0,
        ),
        (
            0,
            0,
            128,
        ),
    ],
)
def test_out_of_bounds_prompt_hard_fails(
    coordinate,
):
    model = FakeSAM()

    with pytest.raises(
        SAMIntegrityError,
        match="outside",
    ):
        run_sam_point_branch(
            model,
            _embedding(),
            coordinate,
            label=0,
        )


def test_invalid_label_hard_fails():
    model = FakeSAM()

    with pytest.raises(
        SAMIntegrityError,
        match="label must be 0 or 1",
    ):
        run_sam_point_branch(
            model,
            _embedding(),
            (
                1,
                2,
                3,
            ),
            label=7,
        )


def test_wrong_image_shape_hard_fails():
    malformed = np.zeros(
        (
            64,
            128,
            128,
        ),
        dtype=np.float32,
    )

    with pytest.raises(
        SAMIntegrityError,
        match="128",
    ):
        prepare_sam_image_tensor(
            malformed,
            device="cpu",
        )


def test_wrong_embedding_shape_hard_fails():
    model = FakeSAM()

    malformed = torch.zeros(
        (
            1,
            384,
            7,
            8,
            8,
        ),
        dtype=torch.float32,
    )

    with pytest.raises(
        SAMIntegrityError,
        match="embedding",
    ):
        run_sam_point_branch(
            model,
            malformed,
            (
                1,
                2,
                3,
            ),
            label=0,
        )


def test_nonfinite_low_res_logits_hard_fail():
    class NonfiniteDecoder(
        FakeMaskDecoder
    ):
        def forward(
            self,
            **kwargs,
        ):
            logits, iou = super().forward(
                **kwargs
            )

            logits[
                0,
                0,
                0,
                0,
                0,
            ] = torch.nan

            return (
                logits,
                iou,
            )

    model = FakeSAM()

    model.mask_decoder = (
        NonfiniteDecoder()
    )

    with pytest.raises(
        SAMIntegrityError,
        match="non-finite",
    ):
        run_sam_point_branch(
            model,
            _embedding(),
            (
                1,
                2,
                3,
            ),
            label=0,
        )
