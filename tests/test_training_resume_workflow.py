import pytest
from types import SimpleNamespace
from unittest.mock import Mock

import src.training.workflow as workflow_module
from src.training.runner import EpochResult


def test_run_baseline_training_resumes_from_latest_checkpoint(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "baseline_run"
    output_dir.mkdir()

    latest_checkpoint_path = (
        output_dir / "latest_baseline.pt"
    )

    history_path = (
        output_dir / "training_history.json"
    )

    # Only existence matters because loading is mocked here.
    latest_checkpoint_path.touch()
    history_path.write_text(
        "[]",
        encoding="utf-8",
    )

    previous_history = [
        EpochResult(
            epoch=1,
            train_loss=1.4,
            validation_loss=1.5,
        ),
        EpochResult(
            epoch=2,
            train_loss=1.2,
            validation_loss=1.3,
        ),
    ]

    resumed_epoch = EpochResult(
        epoch=3,
        train_loss=1.0,
        validation_loss=1.1,
    )

    checkpoint = {
        "epoch": 2,
        "best_validation_loss": 1.3,
    }

    restore_mock = Mock(
        return_value=checkpoint,
    )

    load_history_mock = Mock(
        return_value=previous_history,
    )

    save_checkpoint_mock = Mock()
    save_history_mock = Mock()

    random_state = {
        "example": "state",
    }

    capture_random_state_mock = Mock(
        return_value=random_state,
    )

    def fake_run_baseline_epochs(
        experiment,
        num_epochs,
        checkpoint_path,
        on_epoch_complete,
        start_epoch,
        initial_best_validation_loss,
    ):
        assert num_epochs == 4
        assert start_epoch == 3
        assert initial_best_validation_loss == 1.3

        current_history = [
            resumed_epoch,
        ]

        on_epoch_complete(
            current_history,
            1.1,
        )

        return current_history

    monkeypatch.setattr(
        workflow_module,
        "restore_baseline_training_state",
        restore_mock,
        raising=False,
    )

    monkeypatch.setattr(
        workflow_module,
        "load_training_history",
        load_history_mock,
        raising=False,
    )

    monkeypatch.setattr(
        workflow_module,
        "run_baseline_epochs",
        fake_run_baseline_epochs,
    )

    monkeypatch.setattr(
        workflow_module,
        "save_training_checkpoint",
        save_checkpoint_mock,
    )

    monkeypatch.setattr(
        workflow_module,
        "save_training_history",
        save_history_mock,
    )

    monkeypatch.setattr(
        workflow_module,
        "capture_training_random_state",
        capture_random_state_mock,
    )

    experiment = SimpleNamespace(
        model=object(),
        optimizer=object(),
        train_loader=object(),
        config=SimpleNamespace(
            num_epochs=4,
        ),
    )

    returned_history = (
        workflow_module.run_baseline_training(
            experiment=experiment,
            output_dir=output_dir,
            resume=True,
        )
    )

    assert returned_history == [
        *previous_history,
        resumed_epoch,
    ]

    restore_mock.assert_called_once_with(
        experiment=experiment,
        checkpoint_path=latest_checkpoint_path,
    )

    load_history_mock.assert_called_once_with(
        path=history_path,
    )

    save_history_mock.assert_called_once_with(
        path=history_path,
        history=[
            *previous_history,
            resumed_epoch,
        ],
    )

    latest_save = save_checkpoint_mock.call_args

    assert latest_save.kwargs["path"] == (
        latest_checkpoint_path
    )

    assert latest_save.kwargs["epoch"] == 3
    assert latest_save.kwargs[
        "best_validation_loss"
    ] == 1.1

    assert latest_save.kwargs[
        "random_state"
    ] is random_state

def test_resume_requires_latest_checkpoint(
    tmp_path,
):
    experiment = SimpleNamespace(
        config=SimpleNamespace(
            num_epochs=4,
        ),
    )

    with pytest.raises(
        FileNotFoundError,
        match="latest checkpoint does not exist",
    ):
        workflow_module.run_baseline_training(
            experiment=experiment,
            output_dir=tmp_path,
            resume=True,
        )


def test_resume_requires_training_history(
    tmp_path,
):
    latest_checkpoint_path = (
        tmp_path / "latest_baseline.pt"
    )

    latest_checkpoint_path.touch()

    experiment = SimpleNamespace(
        config=SimpleNamespace(
            num_epochs=4,
        ),
    )

    with pytest.raises(
        FileNotFoundError,
        match="history file does not exist",
    ):
        workflow_module.run_baseline_training(
            experiment=experiment,
            output_dir=tmp_path,
            resume=True,
        )


def test_resume_rejects_checkpoint_history_epoch_mismatch(
    tmp_path,
    monkeypatch,
):
    latest_checkpoint_path = (
        tmp_path / "latest_baseline.pt"
    )

    history_path = (
        tmp_path / "training_history.json"
    )

    latest_checkpoint_path.touch()
    history_path.write_text(
        "[]",
        encoding="utf-8",
    )

    checkpoint = {
        "epoch": 3,
        "best_validation_loss": 1.2,
    }

    previous_history = [
        EpochResult(
            epoch=1,
            train_loss=1.4,
            validation_loss=1.5,
        ),
        EpochResult(
            epoch=2,
            train_loss=1.2,
            validation_loss=1.3,
        ),
    ]

    monkeypatch.setattr(
        workflow_module,
        "restore_baseline_training_state",
        Mock(
            return_value=checkpoint,
        ),
        raising=False,
    )

    monkeypatch.setattr(
        workflow_module,
        "load_training_history",
        Mock(
            return_value=previous_history,
        ),
        raising=False,
    )

    experiment = SimpleNamespace(
        model=object(),
        optimizer=object(),
        train_loader=object(),
        config=SimpleNamespace(
            num_epochs=4,
        ),
    )

    with pytest.raises(
        ValueError,
        match="checkpoint epoch does not match",
    ):
        workflow_module.run_baseline_training(
            experiment=experiment,
            output_dir=tmp_path,
            resume=True,
        )

def test_resume_recovers_when_history_is_one_epoch_ahead(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "baseline_run"
    output_dir.mkdir()

    latest_checkpoint_path = (
        output_dir / "latest_baseline.pt"
    )

    history_path = (
        output_dir / "training_history.json"
    )

    latest_checkpoint_path.touch()
    history_path.write_text(
        "[]",
        encoding="utf-8",
    )

    checkpoint = {
        "epoch": 2,
        "best_validation_loss": 1.3,
    }

    previous_history = [
        EpochResult(
            epoch=1,
            train_loss=1.4,
            validation_loss=1.5,
        ),
        EpochResult(
            epoch=2,
            train_loss=1.2,
            validation_loss=1.3,
        ),
        EpochResult(
            epoch=3,
            train_loss=1.0,
            validation_loss=1.1,
        ),
    ]

    rerun_epoch = EpochResult(
        epoch=3,
        train_loss=0.9,
        validation_loss=1.0,
    )

    monkeypatch.setattr(
        workflow_module,
        "restore_baseline_training_state",
        Mock(
            return_value=checkpoint,
        ),
    )

    monkeypatch.setattr(
        workflow_module,
        "load_training_history",
        Mock(
            return_value=previous_history,
        ),
    )

    save_history_mock = Mock()

    monkeypatch.setattr(
        workflow_module,
        "save_training_history",
        save_history_mock,
    )

    monkeypatch.setattr(
        workflow_module,
        "save_training_checkpoint",
        Mock(),
    )

    monkeypatch.setattr(
        workflow_module,
        "capture_training_random_state",
        Mock(
            return_value={},
        ),
    )

    def fake_run_baseline_epochs(
        experiment,
        num_epochs,
        checkpoint_path,
        start_epoch,
        initial_best_validation_loss,
        on_epoch_complete,
    ):
        assert start_epoch == 3
        assert initial_best_validation_loss == 1.3

        on_epoch_complete(
            [rerun_epoch],
            1.0,
        )

        return [rerun_epoch]

    monkeypatch.setattr(
        workflow_module,
        "run_baseline_epochs",
        fake_run_baseline_epochs,
    )

    experiment = SimpleNamespace(
        model=object(),
        optimizer=object(),
        train_loader=object(),
        config=SimpleNamespace(
            num_epochs=4,
        ),
    )

    returned_history = (
        workflow_module.run_baseline_training(
            experiment=experiment,
            output_dir=output_dir,
            resume=True,
        )
    )

    assert returned_history == [
        previous_history[0],
        previous_history[1],
        rerun_epoch,
    ]

    assert save_history_mock.call_args.kwargs[
        "history"
    ] == [
        previous_history[0],
        previous_history[1],
        rerun_epoch,
    ]


def test_resume_rejects_history_more_than_one_epoch_ahead(
    tmp_path,
    monkeypatch,
):
    latest_checkpoint_path = (
        tmp_path / "latest_baseline.pt"
    )

    history_path = (
        tmp_path / "training_history.json"
    )

    latest_checkpoint_path.touch()
    history_path.write_text(
        "[]",
        encoding="utf-8",
    )

    checkpoint = {
        "epoch": 2,
        "best_validation_loss": 1.3,
    }

    previous_history = [
        EpochResult(
            epoch=1,
            train_loss=1.4,
            validation_loss=1.5,
        ),
        EpochResult(
            epoch=2,
            train_loss=1.2,
            validation_loss=1.3,
        ),
        EpochResult(
            epoch=3,
            train_loss=1.0,
            validation_loss=1.1,
        ),
        EpochResult(
            epoch=4,
            train_loss=0.9,
            validation_loss=1.0,
        ),
    ]

    monkeypatch.setattr(
        workflow_module,
        "restore_baseline_training_state",
        Mock(
            return_value=checkpoint,
        ),
    )

    monkeypatch.setattr(
        workflow_module,
        "load_training_history",
        Mock(
            return_value=previous_history,
        ),
    )

    run_epochs_mock = Mock()

    monkeypatch.setattr(
        workflow_module,
        "run_baseline_epochs",
        run_epochs_mock,
    )

    experiment = SimpleNamespace(
        model=object(),
        optimizer=object(),
        train_loader=object(),
        config=SimpleNamespace(
            num_epochs=5,
        ),
    )

    with pytest.raises(
        ValueError,
        match="checkpoint epoch does not match",
    ):
        workflow_module.run_baseline_training(
            experiment=experiment,
            output_dir=tmp_path,
            resume=True,
        )

    run_epochs_mock.assert_not_called()
