"""Experiment entrypoints.

Do not eagerly import experiment modules here.

This package is used with `python -m provetok.experiments.<module> ...`. Python
initializes the package (`__init__.py`) before importing the submodule, so any
heavy imports here can create surprising import-time side effects and circular
imports.
"""

__all__ = [
    "run_scaling_experiment",
    "ScalingExperimentConfig",
    "ScalingExperimentResult",
    "run_allocation_experiment",
    "AllocationExperimentConfig",
    "AllocationExperimentResult",
]


def __getattr__(name: str):
    if name in ("run_scaling_experiment", "ScalingExperimentConfig", "ScalingExperimentResult"):
        from .fig2_scaling_law import (
            run_scaling_experiment,
            ScalingExperimentConfig,
            ScalingExperimentResult,
        )

        return {
            "run_scaling_experiment": run_scaling_experiment,
            "ScalingExperimentConfig": ScalingExperimentConfig,
            "ScalingExperimentResult": ScalingExperimentResult,
        }[name]

    if name in ("run_allocation_experiment", "AllocationExperimentConfig", "AllocationExperimentResult"):
        from .fig3_allocation import (
            run_allocation_experiment,
            AllocationExperimentConfig,
            AllocationExperimentResult,
        )

        return {
            "run_allocation_experiment": run_allocation_experiment,
            "AllocationExperimentConfig": AllocationExperimentConfig,
            "AllocationExperimentResult": AllocationExperimentResult,
        }[name]

    raise AttributeError(name)
