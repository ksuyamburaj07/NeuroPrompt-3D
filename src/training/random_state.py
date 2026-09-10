import torch


def capture_training_random_state(
    train_loader,
) -> dict:
    """Capture RNG state needed to continue baseline training."""
    cuda_rng_state = None

    if torch.cuda.is_available():
        cuda_rng_state = [
            state.clone()
            for state in torch.cuda.get_rng_state_all()
        ]

    return {
        "torch_cpu_rng_state": (
            torch.get_rng_state().clone()
        ),
        "torch_cuda_rng_state_all": cuda_rng_state,
        "shuffle_generator_state": (
            train_loader.generator.get_state().clone()
        ),
        "patch_generator_state": (
            train_loader.dataset.generator.get_state().clone()
        ),
    }


def restore_training_random_state(
    state: dict,
    train_loader,
) -> None:
    """Restore RNG state captured during baseline training."""
    torch.set_rng_state(
        state["torch_cpu_rng_state"]
    )

    cuda_rng_state = state[
        "torch_cuda_rng_state_all"
    ]

    if cuda_rng_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Checkpoint contains CUDA RNG state, "
                "but CUDA is not available."
            )

        torch.cuda.set_rng_state_all(
            cuda_rng_state
        )

    train_loader.generator.set_state(
        state["shuffle_generator_state"]
    )

    train_loader.dataset.generator.set_state(
        state["patch_generator_state"]
    )
