"""Dataset utilities for model-ready BraTS cases."""

from pathlib import Path
import torch

from src.data.brats import (
    load_split_case_ids,
    modality_paths_for_case,
)

from src.data.nifti import (
    load_multimodal_case,
    load_segmentation,
)
from src.data.preprocessing import (
    prepare_model_case,
    sample_training_patch,
)
from torch.utils.data import Dataset

def load_prepared_brats_case(
    case_dir: str | Path,
    case_id: str,
):
    """Load and preprocess one BraTS case for model use."""
    case_dir = Path(case_dir)

    modality_paths = modality_paths_for_case(
        case_dir,
        case_id,
    )

    segmentation_path = (
        case_dir
        / f"{case_id}-seg.nii.gz"
    )

    mri = load_multimodal_case(
        modality_paths
    )

    segmentation = load_segmentation(
        segmentation_path,
        reference_path=modality_paths["T1"],
    )

    return prepare_model_case(
        mri,
        segmentation,
    )

class BraTSDataset(Dataset):
    """PyTorch dataset for prepared BraTS cases."""

    def __init__(
        self,
        cases_root: str | Path,
        case_ids: tuple[str, ...],
    ) -> None:
        self.cases_root = Path(cases_root)
        self.case_ids = case_ids

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(
        self,
        index: int,
    ):
        case_id = self.case_ids[index]

        case_dir = (
            self.cases_root
            / case_id
        )

        return load_prepared_brats_case(
            case_dir,
            case_id,
        )

    @classmethod
    def from_manifest(
        cls,
        cases_root: str | Path,
        manifest_path: str | Path,
        split_name: str,
    ):
        case_ids = load_split_case_ids(
            manifest_path,
            split_name,
        )

        return cls(
            cases_root=cases_root,
            case_ids=case_ids,
        )

class TrainingPatchDataset(Dataset):
    """Wrap a full-case dataset and return sampled training patches."""

    def __init__(
        self,
        base_dataset: Dataset,
        spatial_size: tuple[int, int, int],
        positive_probability: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.spatial_size = spatial_size
        self.positive_probability = positive_probability

        self.generator = torch.Generator()

        if seed is not None:
            self.generator.manual_seed(seed)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(
        self,
        index: int,
    ):
        mri, target = self.base_dataset[index]

        return sample_training_patch(
            mri,
            target,
            spatial_size=self.spatial_size,
            positive_probability=self.positive_probability,
            generator=self.generator,
        )
