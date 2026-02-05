"""Refusal Calibration: 可校准的拒答机制

根据 proposal §4.3.5:
- 每个 frame 输出支持概率 q_k ∈ [0,1]
- 若 q_k < τ_refuse 则输出 uncertain/refuse
- τ_refuse 在开发集按约束 critical miss-rate ≤ δ 选择一次
- 在所有预算 B 上固定，防止"按预算调阈值刷表"

必须同时报告（反封嘴主表约束）：
- unsupported rate（越低越好）
- critical miss-rate（不得上升）
- refusal rate（拒答比例）
- refusal ECE / reliability（校准好坏）
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

from ..types import Frame, Generation, Issue, Token
from ..verifier.taxonomy import is_critical_finding


@dataclass
class RefusalDecision:
    """单个 frame 的拒答决策"""
    frame_idx: int
    q_k: float              # 支持概率
    should_refuse: bool     # 是否拒答
    is_critical: bool       # 是否为 critical finding
    reason: str             # 拒答原因


@dataclass
class CalibrationMetrics:
    """校准指标集合"""
    # 核心指标
    unsupported_rate: float      # verifier U1 issue rate
    critical_miss_rate: float    # critical findings 被错误拒答的比例
    refusal_rate: float          # 总拒答比例
    refusal_ece: float           # Expected Calibration Error

    # 分桶统计
    reliability_bins: Dict[str, float]  # bin -> (avg_q, avg_correct)
    critical_refusal_count: int
    total_critical_count: int

    def is_valid(self, max_miss_rate: float = 0.05) -> bool:
        """检查是否满足安全约束"""
        return self.critical_miss_rate <= max_miss_rate


class RefusalCalibrator:
    """拒答校准器

    核心功能:
    1. 根据 q_k 和 τ_refuse 决定是否拒答
    2. 在开发集上选择最优 τ_refuse
    3. 计算校准指标（ECE, reliability diagram）
    """

    def __init__(
        self,
        tau_refuse: float = 0.5,
        max_critical_miss_rate: float = 0.05,
        num_bins: int = 10,
    ):
        """
        Args:
            tau_refuse: 拒答阈值
            max_critical_miss_rate: critical miss-rate 上限 (δ)
            num_bins: ECE 计算的分桶数
        """
        self.tau_refuse = tau_refuse
        self.max_critical_miss_rate = max_critical_miss_rate
        self.num_bins = num_bins

    def should_refuse(self, q_k: float, is_critical: bool = False) -> bool:
        """决定是否拒答

        对于 critical findings，使用更低的阈值以减少漏报
        """
        if is_critical:
            # Paper-grade safety: do not refuse critical findings (avoid "封嘴").
            return False

        return q_k < self.tau_refuse

    def decide_refusals(
        self,
        generation: Generation,
        *,
        max_refusal_rate: Optional[float] = None,
    ) -> List[RefusalDecision]:
        """对一个 Generation 的所有 frames 做拒答决策"""
        decisions = []

        for idx, frame in enumerate(generation.frames):
            q_k = generation.q.get(idx, 0.5)
            is_critical = self._is_critical_finding(frame)
            # Refusal is only meaningful for *positive* (present) claims.
            # For absent claims, keep refusal=False to avoid inflating refusal_rate.
            if str(frame.polarity) not in ("present", "positive"):
                refuse = False
            else:
                refuse = self.should_refuse(q_k, is_critical)

            reason = ""
            if refuse:
                reason = f"q_k={q_k:.3f} < tau={self.tau_refuse:.3f}"
                if is_critical:
                    reason += " (critical, conservative threshold)"

            decisions.append(RefusalDecision(
                frame_idx=idx,
                q_k=q_k,
                should_refuse=refuse,
                is_critical=is_critical,
                reason=reason,
            ))

        # Hard cap: enforce a maximum refusal rate per generation to avoid
        # degenerate solutions (e.g., refusing almost everything).
        if max_refusal_rate is not None:
            rr = float(max_refusal_rate)
            rr = max(0.0, min(1.0, rr))
            allowed = int(np.floor(rr * float(len(decisions)) + 1e-12))
            refused = [d for d in decisions if bool(d.should_refuse)]
            if len(refused) > allowed:
                # Keep the lowest-q refusals (most uncertain), deterministic tie-break by frame_idx.
                keep = {d.frame_idx for d in sorted(refused, key=lambda d: (float(d.q_k), int(d.frame_idx)))[:allowed]}
                capped: List[RefusalDecision] = []
                for d in decisions:
                    if d.should_refuse and d.frame_idx not in keep:
                        capped.append(
                            RefusalDecision(
                                frame_idx=int(d.frame_idx),
                                q_k=float(d.q_k),
                                should_refuse=False,
                                is_critical=bool(d.is_critical),
                                reason=str(d.reason) + " (capped)",
                            )
                        )
                    else:
                        capped.append(d)
                decisions = capped

        return decisions

    def _is_critical_finding(self, frame: Frame) -> bool:
        """判断是否为 critical finding"""
        return bool(is_critical_finding(frame.finding))

    def compute_calibration_metrics(
        self,
        generations: List[Generation],
        ground_truths: List[List[Frame]],
        issues_list: List[List[Issue]],
    ) -> CalibrationMetrics:
        """计算完整的校准指标

        Args:
            generations: 生成结果列表
            ground_truths: ground truth frames 列表
            issues_list: 每个 generation 的 verifier issues

        Returns:
            CalibrationMetrics
        """
        # 收集所有 (q_k, correct, is_critical) 三元组
        samples = []
        unsupported_count = 0
        total_frames = 0
        total_refused = 0
        critical_refused = 0
        total_critical = 0

        # Treat only severity>=2 unsupported issues as "material" for calibration.
        # Minor coverage heuristics (severity=1) are too noisy and can make the
        # calibration label degenerate on real manifests.
        min_unsupported_severity = 2

        for gen, gt_frames, issues in zip(generations, ground_truths, issues_list):
            # Unsupported bookkeeping
            unsupported_frames = set()
            for issue in issues:
                if "unsupported" in str(issue.issue_type).lower() and int(getattr(issue, "severity", 0)) >= int(min_unsupported_severity):
                    unsupported_count += 1
                    try:
                        unsupported_frames.add(int(issue.frame_idx))
                    except Exception:
                        pass

            for idx, frame in enumerate(gen.frames):
                q_k = gen.q.get(idx, 0.5)
                is_critical = self._is_critical_finding(frame)
                refused = gen.refusal.get(idx, False)
                total_refused += int(bool(refused))

                # NOTE: q_k is intended to represent *support probability* (evidence-backed),
                # not necessarily clinical correctness. Therefore, for ECE/reliability we
                # treat "correct" as "supported by verifier (no unsupported issues)".
                supported = (int(idx) not in unsupported_frames)
                # ECE only applies to *asserted* (non-refused) positive claims.
                if (not bool(refused)) and (str(frame.polarity) in ("present", "positive")):
                    samples.append(
                        {
                            "q_k": float(q_k),
                            "correct": bool(supported),
                            "is_critical": bool(is_critical),
                            "refused": bool(refused),
                        }
                    )

                total_frames += 1

                if is_critical:
                    total_critical += 1
                    # Critical miss-rate is still defined against GT (avoid "封嘴" on true findings).
                    gt_correct = bool(self._frame_in_gt(frame, gt_frames))
                    if refused and gt_correct:
                        critical_refused += 1

        # 计算 ECE
        ece, bins = self._compute_ece(samples)

        # 计算各指标
        unsupported_rate = unsupported_count / max(total_frames, 1)
        critical_miss_rate = critical_refused / max(total_critical, 1)
        refusal_rate = total_refused / max(total_frames, 1)

        return CalibrationMetrics(
            unsupported_rate=unsupported_rate,
            critical_miss_rate=critical_miss_rate,
            refusal_rate=refusal_rate,
            refusal_ece=ece,
            reliability_bins=bins,
            critical_refusal_count=critical_refused,
            total_critical_count=total_critical,
        )

    def _frame_in_gt(self, frame: Frame, gt_frames: List[Frame]) -> bool:
        """检查 frame 是否在 ground truth 中（简化匹配）"""
        for gt in gt_frames:
            if (frame.finding == gt.finding and
                frame.polarity == gt.polarity):
                return True
        return False

    def _compute_ece(
        self,
        samples: List[Dict],
    ) -> Tuple[float, Dict[str, float]]:
        """计算 Expected Calibration Error

        ECE = Σ_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

        Args:
            samples: list of {q_k, correct, ...}

        Returns:
            (ece, bin_stats)
        """
        if not samples:
            return 0.0, {}

        # 分桶
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        bin_stats = {}
        total_ece = 0.0
        n = len(samples)

        for i in range(self.num_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            bin_name = f"{lo:.1f}-{hi:.1f}"

            # 筛选落入该 bin 的样本
            bin_samples = [
                s for s in samples
                if lo <= s["q_k"] < hi or (i == self.num_bins - 1 and s["q_k"] == hi)
            ]

            if not bin_samples:
                bin_stats[bin_name] = {"avg_conf": 0, "avg_acc": 0, "count": 0}
                continue

            avg_conf = np.mean([s["q_k"] for s in bin_samples])
            avg_acc = np.mean([1 if s["correct"] else 0 for s in bin_samples])
            count = len(bin_samples)

            bin_stats[bin_name] = {
                "avg_conf": float(avg_conf),
                "avg_acc": float(avg_acc),
                "count": count,
            }

            # ECE 累加
            total_ece += (count / n) * abs(avg_acc - avg_conf)

        return total_ece, bin_stats

    def fit_q_calibration_map(
        self,
        generations: List[Generation],
        ground_truths: List[List[Frame]],
        issues_list: Optional[List[List[Issue]]] = None,
    ) -> Dict[str, Any]:
        """Fit a simple histogram-binning calibration map for q_k -> P(correct).

        This is a lightweight way to make `q_k` interpretable as a probability and
        reduce ECE. We intentionally keep it simple and auditable (bin boundaries +
        per-bin empirical accuracy).
        """
        if not generations:
            return {"bin_boundaries": np.linspace(0, 1, self.num_bins + 1).tolist(), "bin_avg_acc": [0.0] * int(self.num_bins)}

        # Keep the calibration label non-degenerate by focusing on material issues (severity>=2).
        min_unsupported_severity = 2

        samples: List[Dict[str, Any]] = []
        if issues_list is not None:
            for gen, issues in zip(generations, issues_list):
                unsupported_frames = set()
                for iss in issues or []:
                    if (
                        "unsupported" in str(getattr(iss, "issue_type", "")).lower()
                        and int(getattr(iss, "severity", 0)) >= int(min_unsupported_severity)
                    ):
                        try:
                            unsupported_frames.add(int(getattr(iss, "frame_idx", -1)))
                        except Exception:
                            pass
                for idx, frame in enumerate(gen.frames):
                    if str(getattr(frame, "polarity", "")) not in ("present", "positive"):
                        continue
                    q_k = float(gen.q.get(idx, 0.5))
                    supported = (int(idx) not in unsupported_frames)
                    samples.append({"q_k": float(max(0.0, min(1.0, q_k))), "correct": bool(supported)})
        else:
            for gen, gt_frames in zip(generations, ground_truths):
                for idx, frame in enumerate(gen.frames):
                    q_k = float(gen.q.get(idx, 0.5))
                    correct = bool(self._frame_in_gt(frame, gt_frames))
                    samples.append({"q_k": float(max(0.0, min(1.0, q_k))), "correct": bool(correct)})

        boundaries = np.linspace(0, 1, self.num_bins + 1)
        bin_avg_acc: List[float] = []
        for i in range(self.num_bins):
            lo, hi = float(boundaries[i]), float(boundaries[i + 1])
            bin_samples = [s for s in samples if (lo <= float(s["q_k"]) < hi) or (i == self.num_bins - 1 and float(s["q_k"]) == hi)]
            if not bin_samples:
                # Empty bin: fall back to mid-point (keeps map in [0,1] and monotone-ish in expectation).
                bin_avg_acc.append(float(0.5 * (lo + hi)))
                continue
            avg_acc = float(np.mean([1.0 if bool(s["correct"]) else 0.0 for s in bin_samples]))
            bin_avg_acc.append(avg_acc)

        return {"bin_boundaries": boundaries.tolist(), "bin_avg_acc": bin_avg_acc}

    def find_optimal_threshold(
        self,
        val_generations: List[Generation],
        val_ground_truths: List[List[Frame]],
        val_issues: List[List[Issue]],
        candidate_taus: Optional[List[float]] = None,
        max_refusal_rate: Optional[float] = None,
        max_refusal_ece: Optional[float] = None,
        *,
        val_tokens: Optional[List[List[Token]]] = None,
        verifier_fn: Optional[Any] = None,
    ) -> Tuple[float, CalibrationMetrics]:
        """在验证集上找最优 τ_refuse

        约束: critical_miss_rate ≤ max_critical_miss_rate
        目标: 最小化 unsupported_rate

        Args:
            val_generations: 验证集生成结果
            val_ground_truths: 验证集 ground truth
            val_issues: 验证集 issues
            candidate_taus: 候选阈值列表

        Returns:
            (best_tau, best_metrics)
        """
        if candidate_taus is None:
            # Include fine-grained small taus to avoid the degenerate "either refuse none
            # or always saturate the max_refusal_rate cap" regime.
            candidate_taus = sorted(
                {
                    0.0,
                    *[i / 1000.0 for i in range(1, 21)],  # 0.001 .. 0.020
                    *[i / 100.0 for i in range(1, 21)],  # 0.01 .. 0.20
                    0.25,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                }
            )

        best_tau = self.tau_refuse
        best_metrics = None
        best_unsupported = float("inf")
        best_refusal = float("inf")
        # Heuristic tie-break: when unsupported is within a small tolerance of the
        # best value, prefer lower refusal_rate to avoid living right at the cap.
        #
        # This does not change the *proof* constraints, but makes the selected tau
        # more robust (margin against "anti-gaming" refusal-rate limits).
        rel_unsupported_tol = 0.01  # allow +1% unsupported to gain lower refusal

        for tau in candidate_taus:
            # 临时设置阈值
            old_tau = self.tau_refuse
            self.tau_refuse = tau

            # 重新计算 refusals
            updated_gens = []
            updated_issues = []
            for gen in val_generations:
                decisions = self.decide_refusals(gen, max_refusal_rate=max_refusal_rate)
                new_refusal = {d.frame_idx: d.should_refuse for d in decisions}
                tmp = Generation(
                    frames=gen.frames,
                    citations=gen.citations,
                    q=gen.q,
                    refusal=new_refusal,
                    text="",
                )
                from .narrative import render_generation_text

                updated_gens.append(
                    Generation(
                        frames=tmp.frames,
                        citations=tmp.citations,
                        q=tmp.q,
                        refusal=tmp.refusal,
                        text=render_generation_text(tmp),
                    )
                )
            if val_tokens is not None and verifier_fn is not None:
                if len(val_tokens) != len(updated_gens):
                    raise ValueError(f"val_tokens length mismatch: {len(val_tokens)} vs {len(updated_gens)} generations")
                for gen_u, toks in zip(updated_gens, val_tokens):
                    updated_issues.append(verifier_fn(gen_u, toks))
            else:
                updated_issues = val_issues

            # 计算指标
            metrics = self.compute_calibration_metrics(updated_gens, val_ground_truths, updated_issues)

            # 检查约束
            ok = metrics.is_valid(self.max_critical_miss_rate)
            if max_refusal_rate is not None:
                ok = ok and (float(metrics.refusal_rate) <= float(max_refusal_rate) + 1e-12)
            if max_refusal_ece is not None:
                ok = ok and (float(metrics.refusal_ece) <= float(max_refusal_ece) + 1e-12)
            if ok:
                u = float(metrics.unsupported_rate)
                rr = float(metrics.refusal_rate)
                is_strict_better = u < best_unsupported - 1e-12
                is_close = u <= (best_unsupported * (1.0 + rel_unsupported_tol) + 1e-12)
                is_better_refusal = rr < best_refusal - 1e-12
                if is_strict_better or (is_close and is_better_refusal):
                    best_unsupported = float(u)
                    best_refusal = float(rr)
                    best_tau = float(tau)
                    best_metrics = metrics

            self.tau_refuse = old_tau

        # 如果没有满足约束的，返回最保守的
        if best_metrics is None:
            self.tau_refuse = min(candidate_taus)
            best_tau = self.tau_refuse
            best_metrics = self.compute_calibration_metrics(val_generations, val_ground_truths, val_issues)

        self.tau_refuse = best_tau
        return best_tau, best_metrics


def apply_q_calibration_to_generation(generation: Generation, q_calibration: Dict[str, Any]) -> Generation:
    """Return a new Generation with q_k calibrated via a histogram-binning map."""
    boundaries = q_calibration.get("bin_boundaries") or []
    bin_avg_acc = q_calibration.get("bin_avg_acc") or []
    if not isinstance(boundaries, list) or not isinstance(bin_avg_acc, list) or not boundaries or not bin_avg_acc:
        return generation

    # boundaries length should be num_bins+1; if malformed, be conservative.
    num_bins = max(1, len(boundaries) - 1)
    if len(bin_avg_acc) < num_bins:
        return generation

    def _apply(q: float) -> float:
        q = float(max(0.0, min(1.0, q)))
        # Find bin index (linear scan is fine for small num_bins).
        for i in range(num_bins):
            lo = float(boundaries[i])
            hi = float(boundaries[i + 1])
            if (lo <= q < hi) or (i == num_bins - 1 and q == hi):
                return float(max(0.0, min(1.0, float(bin_avg_acc[i]))))
        return float(max(0.0, min(1.0, q)))

    new_q = {int(k): _apply(float(v)) for k, v in (generation.q or {}).items()}
    tmp = Generation(frames=generation.frames, citations=generation.citations, q=new_q, refusal=generation.refusal, text="")
    from .narrative import render_generation_text

    return Generation(
        frames=tmp.frames,
        citations=tmp.citations,
        q=tmp.q,
        refusal=tmp.refusal,
        text=render_generation_text(tmp),
    )


@dataclass(frozen=True)
class RefusalPolicy:
    """A reusable refusal policy calibrated on a dev set.

    Contract:
    - Calibrate once on dev/val under a critical miss-rate constraint.
    - Reuse the same `tau_refuse` across budgets B (do not tune per-budget).
    """

    tau_refuse: float
    max_critical_miss_rate: float = 0.05
    max_refusal_rate: Optional[float] = None
    num_bins: int = 10
    q_calibration: Optional[Dict[str, Any]] = None

    def apply(self, generation: Generation) -> Generation:
        gen = generation
        if self.q_calibration:
            gen = apply_q_calibration_to_generation(gen, self.q_calibration)
        calibrator = RefusalCalibrator(
            tau_refuse=float(self.tau_refuse),
            max_critical_miss_rate=float(self.max_critical_miss_rate),
            num_bins=int(self.num_bins),
        )
        return apply_refusal_to_generation(gen, calibrator, max_refusal_rate=self.max_refusal_rate)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tau_refuse": float(self.tau_refuse),
            "max_critical_miss_rate": float(self.max_critical_miss_rate),
            "max_refusal_rate": (float(self.max_refusal_rate) if self.max_refusal_rate is not None else None),
            "num_bins": int(self.num_bins),
            "q_calibration": self.q_calibration,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RefusalPolicy":
        return cls(
            tau_refuse=float(d["tau_refuse"]),
            max_critical_miss_rate=float(d.get("max_critical_miss_rate", 0.05)),
            max_refusal_rate=(float(d["max_refusal_rate"]) if d.get("max_refusal_rate", None) is not None else None),
            num_bins=int(d.get("num_bins", 10)),
            q_calibration=d.get("q_calibration"),
        )


def apply_refusal_to_generation(
    generation: Generation,
    calibrator: RefusalCalibrator,
    *,
    max_refusal_rate: Optional[float] = None,
) -> Generation:
    """应用拒答决策到 Generation

    返回更新了 refusal 字段的新 Generation
    """
    decisions = calibrator.decide_refusals(generation, max_refusal_rate=max_refusal_rate)
    new_refusal = {d.frame_idx: d.should_refuse for d in decisions}

    tmp = Generation(
        frames=generation.frames,
        citations=generation.citations,
        q=generation.q,
        refusal=new_refusal,
        text="",
    )
    from .narrative import render_generation_text
    return Generation(
        frames=tmp.frames,
        citations=tmp.citations,
        q=tmp.q,
        refusal=tmp.refusal,
        text=render_generation_text(tmp),
    )


def format_calibration_report(metrics: CalibrationMetrics) -> str:
    """格式化校准报告（用于日志/可视化）"""
    lines = [
        "=" * 50,
        "Refusal Calibration Report",
        "=" * 50,
        f"Unsupported Rate:     {metrics.unsupported_rate:.4f}",
        f"Critical Miss Rate:   {metrics.critical_miss_rate:.4f}",
        f"Refusal Rate:         {metrics.refusal_rate:.4f}",
        f"Refusal ECE:          {metrics.refusal_ece:.4f}",
        "",
        f"Critical Findings:    {metrics.critical_refusal_count}/{metrics.total_critical_count} refused",
        "",
        "Reliability Bins:",
    ]

    for bin_name, stats in sorted(metrics.reliability_bins.items()):
        if isinstance(stats, dict):
            lines.append(
                f"  {bin_name}: conf={stats.get('avg_conf', 0):.3f}, "
                f"acc={stats.get('avg_acc', 0):.3f}, n={stats.get('count', 0)}"
            )

    lines.append("=" * 50)
    return "\n".join(lines)
