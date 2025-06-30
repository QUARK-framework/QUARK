from dataclasses import dataclass


@dataclass
class BackendResult:
    """
    Result returned from a quantum backend.
    """
    counts: list[dict[str, int]]
    n_shots: int
