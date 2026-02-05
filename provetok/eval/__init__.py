from .metrics_frames import (
    frame_f1,
    compute_frame_f1,
    FrameMetrics,
    FrameMatchResult,
    hungarian_match_frames,
    aggregate_frame_metrics,
    format_frame_metrics,
)
from .metrics_grounding import (
    GroundingMetrics,
    compute_iou,
    compute_dice,
    compute_citation_grounding,
    compute_generation_grounding,
    compute_grounding_metrics,
    compute_mask_sanity,
    format_grounding_metrics,
    union_citations,
    union_lesion_masks,
    omega_permutation_test,
    citation_swap_test,
)
from .scaling import (
    ScalingFit,
    AllocationConfig,
    AllocationResult,
    AllocationModel,
    fit_scaling_law,
    fit_power_law,
    fit_log_saturation,
    compute_diminishing_returns_point,
    format_scaling_report,
    format_allocation_report,
)

from .stats import (
    PairedBootstrapResult,
    paired_bootstrap_mean_diff,
    holm_bonferroni,
)

from .counterfactual import (
    reindex_tokens,
    permute_cell_ids,
    permute_embeddings,
    swap_citations,
    drop_cited_tokens,
    remove_all_citations,
    issue_counts,
    issue_rate,
)

from .metrics_text import (
    TextMetricConfig,
    MissingTextMetricDependency,
    compute_text_metrics,
    compute_text_metrics_batch,
    compute_bertscore_f1_batch,
)
