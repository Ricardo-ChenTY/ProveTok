"""ProveTok 实验脚本

包含论文 Fig 2 和 Fig 3 的复现代码。
"""
from .fig2_scaling_law import (
    run_scaling_experiment,
    ScalingExperimentConfig,
    ScalingExperimentResult,
)
from .fig3_allocation import (
    run_allocation_experiment,
    AllocationExperimentConfig,
    AllocationExperimentResult,
)
