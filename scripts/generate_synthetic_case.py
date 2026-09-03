"""Generate a four-modality synthetic NeuroPrompt-3D case."""

from pathlib import Path

from src.data.nifti import save_case
from src.data.synthetic import create_synthetic_case
from src.visualization.slices import save_multimodal_preview


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    case_id = "synthetic_001"

    mri, mask = create_synthetic_case(shape=(64, 64, 64), seed=7)
    mri_path, mask_path = save_case(
        mri,
        mask,
        project_root / "data" / "synthetic",
        case_id,
    )
    preview_path = save_multimodal_preview(
        mri,
        mask,
        project_root / "outputs" / f"{case_id}_preview.png",
    )

    print(f"Saved MRI: {mri_path}")
    print(f"Saved mask: {mask_path}")
    print(f"Saved preview: {preview_path}")


if __name__ == "__main__":
    main()
