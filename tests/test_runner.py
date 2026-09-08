from types import SimpleNamespace
from unittest.mock import Mock
from src.training.runner import run_baseline_epochs


def test_run_baseline_epochs_records_train_and_validation_losses(
    monkeypatch,
):
    train_losses = iter([
        1.4,
        1.2,
        1.0,
    ])

    validation_losses = iter([
        1.5,
        1.3,
        1.1,
    ])

    def fake_train_one_epoch(
        model,
        optimizer,
        dataloader,
    ):
        return next(train_losses)

    def fake_evaluate_sliding_window_epoch(
        model,
        dataloader,
        roi_size,
        overlap,
    ):
        return next(validation_losses)

    monkeypatch.setattr(
        "src.training.runner.train_one_epoch",
        fake_train_one_epoch,
    )

    monkeypatch.setattr(
        "src.training.runner.evaluate_sliding_window_epoch",
        fake_evaluate_sliding_window_epoch,
    )

    experiment = SimpleNamespace(
        model=object(),
        optimizer=object(),
        train_loader=object(),
        validation_loader=object(),
        config=SimpleNamespace(
            patch_size=(96, 96, 96),
            validation_overlap=0.25,
        ),
    )

    history = run_baseline_epochs(
        experiment=experiment,
        num_epochs=3,
    )

    assert len(history) == 3

    assert history[0].epoch == 1
    assert history[0].train_loss == 1.4
    assert history[0].validation_loss == 1.5

    assert history[1].epoch == 2
    assert history[1].train_loss == 1.2
    assert history[1].validation_loss == 1.3

    assert history[2].epoch == 3
    assert history[2].train_loss == 1.0
    assert history[2].validation_loss == 1.1

def test_run_baseline_epochs_saves_only_when_validation_improves(
    tmp_path,
    monkeypatch,
):
    train_losses = iter([
        1.4,
        1.3,
        1.1,
        1.0,
    ])

    validation_losses = iter([
        1.5,
        1.6,
        1.2,
        1.3,
    ])

    def fake_train_one_epoch(
        model,
        optimizer,
        dataloader,
    ):
        return next(train_losses)

    def fake_evaluate_sliding_window_epoch(
        model,
        dataloader,
        roi_size,
        overlap,
    ):
        return next(validation_losses)

    checkpoint_mock = Mock()

    monkeypatch.setattr(
        "src.training.runner.train_one_epoch",
        fake_train_one_epoch,
    )

    monkeypatch.setattr(
        "src.training.runner.evaluate_sliding_window_epoch",
        fake_evaluate_sliding_window_epoch,
    )

    monkeypatch.setattr(
        "src.training.runner.save_training_checkpoint",
        checkpoint_mock,
    )

    config = SimpleNamespace(
        patch_size=(96, 96, 96),
        validation_overlap=0.25,
    )

    experiment = SimpleNamespace(
        model=object(),
        optimizer=object(),
        train_loader=object(),
        validation_loader=object(),
        config=config,
    )

    checkpoint_path = (
        tmp_path / "best_baseline.pt"
    )

    run_baseline_epochs(
        experiment=experiment,
        num_epochs=4,
        checkpoint_path=checkpoint_path,
    )

    assert checkpoint_mock.call_count == 2

    first_save = checkpoint_mock.call_args_list[0]
    second_save = checkpoint_mock.call_args_list[1]

    assert first_save.kwargs["epoch"] == 1
    assert first_save.kwargs["train_loss"] == 1.4
    assert first_save.kwargs["validation_loss"] == 1.5

    assert second_save.kwargs["epoch"] == 3
    assert second_save.kwargs["train_loss"] == 1.1
    assert second_save.kwargs["validation_loss"] == 1.2

    assert first_save.kwargs["path"] == checkpoint_path
    assert second_save.kwargs["path"] == checkpoint_path

def test_run_baseline_epochs_does_not_overwrite_best_on_equal_loss(
    tmp_path,
    monkeypatch,
):
    train_losses = iter([
        1.4,
        1.3,
        1.2,
    ])

    validation_losses = iter([
        1.2,
        1.2,
        1.3,
    ])

    def fake_train_one_epoch(
        model,
        optimizer,
        dataloader,
    ):
        return next(train_losses)

    def fake_evaluate_sliding_window_epoch(
        model,
        dataloader,
        roi_size,
        overlap,
    ):
        return next(validation_losses)

    checkpoint_mock = Mock()

    monkeypatch.setattr(
        "src.training.runner.train_one_epoch",
        fake_train_one_epoch,
    )

    monkeypatch.setattr(
        "src.training.runner.evaluate_sliding_window_epoch",
        fake_evaluate_sliding_window_epoch,
    )

    monkeypatch.setattr(
        "src.training.runner.save_training_checkpoint",
        checkpoint_mock,
    )

    experiment = SimpleNamespace(
        model=object(),
        optimizer=object(),
        train_loader=object(),
        validation_loader=object(),
        config=SimpleNamespace(
            patch_size=(96, 96, 96),
            validation_overlap=0.25,
        ),
    )

    run_baseline_epochs(
        experiment=experiment,
        num_epochs=3,
        checkpoint_path=tmp_path / "best_baseline.pt",
    )

    assert checkpoint_mock.call_count == 1

    saved_checkpoint = checkpoint_mock.call_args

    assert saved_checkpoint.kwargs["epoch"] == 1
    assert saved_checkpoint.kwargs["validation_loss"] == 1.2
