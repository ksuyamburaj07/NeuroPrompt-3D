import argparse
from pathlib import Path
from typing import Sequence
from src.training.config import BaselineTrainingConfig
from src.training.setup import build_baseline_experiment
from src.training.workflow import run_baseline_training

def parse_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments for baseline training."""

    parser = argparse.ArgumentParser(
        description=(
            "Train the NeuroPrompt-3D baseline "
            "3D U-Net."
        )
    )

    parser.add_argument(
        "--cases-root",
        type=Path,
        required=True,
        help="Root directory containing BraTS cases.",
    )

    parser.add_argument(
        "--manifest-path",
        type=Path,
        required=True,
        help="Path to the frozen split manifest.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for training outputs.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
        help="Training device.",
    )

    return parser.parse_args(argv)

def main(
    argv: Sequence[str] | None = None,
) -> None:
    """Build and run a baseline training experiment."""

    args = parse_arguments(argv)

    if not args.cases_root.exists():
        raise FileNotFoundError(
            f"cases root does not exist: {args.cases_root}"
        )

    if not args.cases_root.is_dir():
        raise NotADirectoryError(
            f"cases root is not a directory: {args.cases_root}"
        )

    if not args.manifest_path.exists():
        raise FileNotFoundError(
            f"manifest path does not exist: {args.manifest_path}"
        )

    if not args.manifest_path.is_file():
        raise FileNotFoundError(
            f"manifest path is not a file: {args.manifest_path}"
        )

    config = BaselineTrainingConfig(
        num_epochs=args.num_epochs,
        device=args.device,
    )

    experiment = build_baseline_experiment(
        cases_root=args.cases_root,
        manifest_path=args.manifest_path,
        config=config,
    )

    run_baseline_training(
        experiment=experiment,
        output_dir=args.output_dir,
    )

if __name__ == "__main__":
    main()
