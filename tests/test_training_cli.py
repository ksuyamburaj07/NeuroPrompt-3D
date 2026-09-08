from pathlib import Path
import pytest
from types import SimpleNamespace
from unittest.mock import Mock

from src.training.cli import main, parse_arguments


def test_parse_arguments_reads_baseline_training_options():
    args = parse_arguments(
        [
            "--cases-root",
            "/data/brats/cases",
            "--manifest-path",
            "splits/frozen.json",
            "--output-dir",
            "outputs/baseline",
            "--num-epochs",
            "5",
            "--device",
            "cpu",
        ]
    )

    assert args.cases_root == Path(
        "/data/brats/cases"
    )

    assert args.manifest_path == Path(
        "splits/frozen.json"
    )

    assert args.output_dir == Path(
        "outputs/baseline"
    )

    assert args.num_epochs == 5
    assert args.device == "cpu"

def test_main_builds_and_runs_baseline_training(
    tmp_path,
    monkeypatch,
):
    experiment = SimpleNamespace()

    build_experiment_mock = Mock(
        return_value=experiment,
    )

    run_training_mock = Mock()

    monkeypatch.setattr(
        "src.training.cli.build_baseline_experiment",
        build_experiment_mock,
    )

    monkeypatch.setattr(
        "src.training.cli.run_baseline_training",
        run_training_mock,
    )

    cases_root = tmp_path / "cases"
    manifest_path = tmp_path / "frozen_split.json"
    output_dir = tmp_path / "baseline_output"


    cases_root.mkdir()

    manifest_path.write_text(
        "{}",
        encoding="utf-8",
    )

    main(
        [
            "--cases-root",
            str(cases_root),
            "--manifest-path",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--num-epochs",
            "5",
            "--device",
            "cpu",
        ]
    )

    build_experiment_mock.assert_called_once()

    build_call = build_experiment_mock.call_args

    assert build_call.kwargs["cases_root"] == cases_root
    assert build_call.kwargs["manifest_path"] == manifest_path

    config = build_call.kwargs["config"]

    assert config.num_epochs == 5
    assert config.device == "cpu"

    run_training_mock.assert_called_once_with(
        experiment=experiment,
        output_dir=output_dir,
    )

def test_main_rejects_missing_cases_root_before_building_experiment(
    tmp_path,
    monkeypatch,
):
    build_experiment_mock = Mock()

    monkeypatch.setattr(
        "src.training.cli.build_baseline_experiment",
        build_experiment_mock,
    )

    missing_cases_root = (
        tmp_path / "missing_cases"
    )

    manifest_path = (
        tmp_path / "frozen_split.json"
    )

    manifest_path.write_text(
        "{}",
        encoding="utf-8",
    )

    output_dir = (
        tmp_path / "baseline_output"
    )

    with pytest.raises(
        FileNotFoundError,
        match="cases root",
    ):
        main(
            [
                "--cases-root",
                str(missing_cases_root),
                "--manifest-path",
                str(manifest_path),
                "--output-dir",
                str(output_dir),
                "--num-epochs",
                "1",
                "--device",
                "cpu",
            ]
        )

    build_experiment_mock.assert_not_called()

def test_main_rejects_missing_manifest_before_building_experiment(
    tmp_path,
    monkeypatch,
):
    build_experiment_mock = Mock()

    monkeypatch.setattr(
        "src.training.cli.build_baseline_experiment",
        build_experiment_mock,
    )

    cases_root = (
        tmp_path / "cases"
    )

    cases_root.mkdir()

    missing_manifest_path = (
        tmp_path / "missing_split.json"
    )

    output_dir = (
        tmp_path / "baseline_output"
    )

    with pytest.raises(
        FileNotFoundError,
        match="manifest path",
    ):
        main(
            [
                "--cases-root",
                str(cases_root),
                "--manifest-path",
                str(missing_manifest_path),
                "--output-dir",
                str(output_dir),
                "--num-epochs",
                "1",
                "--device",
                "cpu",
            ]
        )

    build_experiment_mock.assert_not_called()

def test_main_rejects_cases_root_that_is_not_directory(
    tmp_path,
    monkeypatch,
):
    build_experiment_mock = Mock()

    monkeypatch.setattr(
        "src.training.cli.build_baseline_experiment",
        build_experiment_mock,
    )

    cases_root = (
        tmp_path / "cases"
    )

    cases_root.write_text(
        "not a directory",
        encoding="utf-8",
    )

    manifest_path = (
        tmp_path / "frozen_split.json"
    )

    manifest_path.write_text(
        "{}",
        encoding="utf-8",
    )

    output_dir = (
        tmp_path / "baseline_output"
    )

    with pytest.raises(
        NotADirectoryError,
        match="cases root",
    ):
        main(
            [
                "--cases-root",
                str(cases_root),
                "--manifest-path",
                str(manifest_path),
                "--output-dir",
                str(output_dir),
                "--num-epochs",
                "1",
                "--device",
                "cpu",
            ]
        )

    build_experiment_mock.assert_not_called()

def test_main_rejects_manifest_path_that_is_not_file(
    tmp_path,
    monkeypatch,
):
    build_experiment_mock = Mock()

    monkeypatch.setattr(
        "src.training.cli.build_baseline_experiment",
        build_experiment_mock,
    )

    cases_root = (
        tmp_path / "cases"
    )

    cases_root.mkdir()

    manifest_path = (
        tmp_path / "frozen_split.json"
    )

    manifest_path.mkdir()

    output_dir = (
        tmp_path / "baseline_output"
    )

    with pytest.raises(
        FileNotFoundError,
        match="manifest path",
    ):
        main(
            [
                "--cases-root",
                str(cases_root),
                "--manifest-path",
                str(manifest_path),
                "--output-dir",
                str(output_dir),
                "--num-epochs",
                "1",
                "--device",
                "cpu",
            ]
        )

    build_experiment_mock.assert_not_called()
