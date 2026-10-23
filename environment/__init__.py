"""Environment package: hazard grid and disaster targets."""

from environment.grid import (
    EnvironmentGenerationError,
    GridEnvironment,
    generate_environment,
)
from environment.targets import Target, total_priority

__all__ = [
    "EnvironmentGenerationError",
    "GridEnvironment",
    "Target",
    "generate_environment",
    "total_priority",
]
