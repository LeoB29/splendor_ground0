from splendor_ai.training import resolve_training_backend


def test_backend_defaults_to_cuda_first_policy() -> None:
    backend = resolve_training_backend()

    assert backend.name == "cuda"


def test_backend_can_fall_back_to_directml() -> None:
    backend = resolve_training_backend(prefer_cuda=False, allow_directml=True)

    assert backend.name == "directml"


def test_backend_can_fall_back_to_cpu() -> None:
    backend = resolve_training_backend(prefer_cuda=False, allow_directml=False)

    assert backend.name == "cpu"
