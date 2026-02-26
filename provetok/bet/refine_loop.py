"""BET Refine Loop: 预算约束下的证据 Tokenization 迭代优化

根据 proposal Algorithm 1: Deterministic Greedy BET-Refine
1. 构建/更新 evidence graph
2. 对每个可 split cell 计算 Δ(c)
3. 选择 c* = argmax Δ(c)，tie-break: 最小 cell_id
4. 若 Δ(c*) < ε 或 |S| >= B_enc: 停止
5. split c* 为 8 children
6. 每 M 步运行一次 PCG+verifier 刷新 issues
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional, Sequence, Set, Tuple
import torch

from ..grid.cells import root_cell, split, Cell
from ..types import Token, Generation, Issue, Frame
from .tokenize import TokenEncoder
from .allocator import PickPrefer, pick_cell_to_split
from .evidence_head import (
    EvidenceHead,
    EvidenceScore,
    compute_delta,
    rank_cells_by_delta,
    select_cell_to_split_by_delta,
)


@dataclass
class RefineTrace:
    """单步 refine 的详细跟踪信息"""
    step: int
    num_cells: int
    num_tokens: int
    num_frames: int
    num_issues: int
    verifier_refreshed: bool
    issues: List[Dict[str, Any]]
    split_cell: Optional[str] = None
    delta: Optional[float] = None
    delta_breakdown: Optional[Dict[str, float]] = None


@dataclass
class RefineResult:
    """Refine loop 的完整结果"""
    tokens: List[Token]
    gen: Generation
    issues: List[Issue]
    trace: List[RefineTrace]
    # 额外的可审计信息
    final_cells: List[Cell] = field(default_factory=list)
    total_steps: int = 0
    stopped_reason: str = ""


def _clone_generation_with_updates(
    gen: Generation,
    *,
    frames: Optional[List[Frame]] = None,
    citations: Optional[Dict[int, List[int]]] = None,
    citations_ref: Optional[Dict[int, List[str]]] = None,
) -> Generation:
    return Generation(
        frames=list(frames if frames is not None else list(gen.frames)),
        citations={int(k): list(v) for k, v in (citations if citations is not None else dict(gen.citations)).items()},
        q={int(k): float(v) for k, v in dict(gen.q).items()},
        refusal={int(k): bool(v) for k, v in dict(gen.refusal).items()},
        citations_ref={int(k): list(v) for k, v in (citations_ref if citations_ref is not None else dict(gen.citations_ref or {})).items()},
        text=str(getattr(gen, "text", "") or ""),
        impression=str(getattr(gen, "impression", "") or ""),
        report_text=str(getattr(gen, "report_text", "") or ""),
    )


def _despecify_frame(frame: Frame, *, confidence_cap: float = 0.6) -> Frame:
    return Frame(
        finding=str(getattr(frame, "finding", "normal")),
        polarity=str(getattr(frame, "polarity", "present")),
        laterality="unspecified",
        confidence=min(float(confidence_cap), float(getattr(frame, "confidence", 0.5))),
        location=str(getattr(frame, "location", "unspecified")),
        size_bin=str(getattr(frame, "size_bin", "unspecified")),
        severity=str(getattr(frame, "severity", "unspecified")),
        uncertain=True,
    )


def _apply_semantic_citation_rerank(
    gen: Generation,
    *,
    tokens: List[Token],
    issues: List[Issue],
    top_k: int,
    rule_ids: Set[str],
) -> Tuple[Generation, bool]:
    if not tokens or not gen.frames or not issues or not rule_ids:
        return gen, False

    token_ref_map = {int(t.token_id): str(getattr(t, "ref", t.cell_id)) for t in tokens}
    issues_by_frame: Dict[int, Set[str]] = {}
    for iss in issues:
        rid = str(getattr(iss, "rule_id", "") or "").strip()
        if rid not in rule_ids:
            continue
        idx = int(getattr(iss, "frame_idx", -1))
        if idx < 0:
            continue
        issues_by_frame.setdefault(idx, set()).add(rid)

    if not issues_by_frame:
        return gen, False

    try:
        from ..verifier.rules import FINDING_ALLOWED_ANATOMY
    except Exception:
        FINDING_ALLOWED_ANATOMY = {}

    new_citations = {int(k): list(v) for k, v in dict(gen.citations).items()}
    new_citations_ref = {int(k): list(v) for k, v in dict(gen.citations_ref or {}).items()}
    changed = False
    keep = max(1, int(top_k))

    for frame_idx, frame in enumerate(gen.frames):
        frame_rules = issues_by_frame.get(int(frame_idx))
        if not frame_rules:
            continue
        if bool(gen.refusal.get(int(frame_idx), False)):
            continue
        if str(getattr(frame, "polarity", "")).lower() not in ("present", "positive"):
            continue

        allowed: Set[str] = set()
        if "R5" in frame_rules:
            finding = str(getattr(frame, "finding", "")).strip().lower()
            allowed = {str(x).strip().lower() for x in FINDING_ALLOWED_ANATOMY.get(finding, set())}

        scored: List[Tuple[float, int]] = []
        for tok in tokens:
            anatomy_label = str(getattr(tok, "anatomy_label", "") or "").strip().lower()
            sem = float(tok.ctclip_similarity) if getattr(tok, "ctclip_similarity", None) is not None else float(tok.score)
            rank_score = float(sem) + 0.2 * float(tok.score) + 0.02 * float(tok.level)
            if allowed:
                if anatomy_label and anatomy_label in allowed:
                    rank_score += 1.0
                else:
                    rank_score -= 1.0
            scored.append((rank_score, int(tok.token_id)))

        scored.sort(key=lambda x: (-float(x[0]), int(x[1])))
        selected: List[int] = []
        seen: Set[int] = set()
        for _, tid in scored:
            if tid in seen:
                continue
            seen.add(int(tid))
            selected.append(int(tid))
            if len(selected) >= keep:
                break

        if not selected:
            continue

        old = list(new_citations.get(int(frame_idx), []))
        if old == selected:
            continue
        new_citations[int(frame_idx)] = selected
        new_citations_ref[int(frame_idx)] = [token_ref_map.get(int(tid), str(tid)) for tid in selected]
        changed = True

    if not changed:
        return gen, False
    return _clone_generation_with_updates(gen, citations=new_citations, citations_ref=new_citations_ref), True


def _apply_despecify_fallback(
    gen: Generation,
    *,
    issues: List[Issue],
    confidence_cap: float,
) -> Tuple[Generation, bool]:
    if not gen.frames or not issues:
        return gen, False

    issue_frames = {int(getattr(iss, "frame_idx", -1)) for iss in issues if int(getattr(iss, "frame_idx", -1)) >= 0}
    if not issue_frames:
        return gen, False

    frames = list(gen.frames)
    changed = False
    for idx, fr in enumerate(frames):
        if int(idx) not in issue_frames:
            continue
        if bool(gen.refusal.get(int(idx), False)):
            continue
        if str(getattr(fr, "polarity", "")).lower() not in ("present", "positive"):
            continue
        newer = _despecify_frame(fr, confidence_cap=float(confidence_cap))
        if newer != fr:
            frames[idx] = newer
            changed = True

    if not changed:
        return gen, False
    return _clone_generation_with_updates(gen, frames=frames), True


def run_refine_loop(
    volume: torch.Tensor,
    budget_tokens: int,
    steps: int,
    generator_fn: Callable[[List[Token]], Generation],
    verifier_fn: Callable[[Generation, List[Token]], List[Issue]],
    emb_dim: int = 32,
    seed: int = 0,
    encoder: Optional[Any] = None,
    affine_zyx: Optional[Any] = None,
    anatomy_labels: Optional[Any] = None,
    anatomy_label_map: Optional[Dict[int, str]] = None,
    require_full_budget: bool = False,
    # Evidence Head 相关参数
    evidence_head: Optional[EvidenceHead] = None,
    use_evidence_head: bool = True,
    epsilon: float = 0.01,
    init_level: int = 0,
    max_depth: int = 4,
    lambda_uncertainty: float = 0.3,
    verifier_refresh_period: int = 1,  # M: 每 M 步刷新 verifier
    pcg_refresh_period: int = 1,       # 每 M 步刷新 PCG（LLM 模式下用于降频）
    allocator_prefer: PickPrefer = "uncertainty",
    token_score_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    token_score_scale: float = 1.0,
    token_score_fuse: str = "override",  # override|max|blend
    token_score_blend_alpha: float = 1.0,
    token_score_max_beta: float = 1.0,
    score_to_uncertainty: bool = False,
    score_level_power: float = 0.0,
    # Optional Stage-4 semantic close-loop hooks (default off; non-breaking).
    semantic_rerank_on_violation: bool = False,
    semantic_rerank_topk: int = 3,
    semantic_rerank_rule_ids: Optional[Sequence[str]] = None,
    despecify_on_remaining_issues: bool = False,
    despecify_confidence_cap: float = 0.6,
) -> RefineResult:
    """运行 BET refine loop

    Args:
        volume: 3D CT volume tensor
        budget_tokens: 最大 token 数量 (B_enc)
        steps: 最大迭代步数
        generator_fn: PCG 生成函数
        verifier_fn: Verifier 验证函数
        emb_dim: Token embedding 维度
        seed: 随机种子
        evidence_head: Evidence Head 模型（可选，如果 None 则使用原始 allocator）
        use_evidence_head: 是否使用 Evidence Head 选择 cell
        epsilon: 停机阈值（Δ < ε 时停止）
        max_depth: Cell 最大深度
        lambda_uncertainty: 不确定性项的权重
        verifier_refresh_period: Verifier 刷新周期

    Returns:
        RefineResult 包含 tokens, generation, issues 和详细 trace
    """
    init_level = int(init_level)
    max_depth = int(max_depth)
    if init_level > max_depth:
        init_level = max_depth
    verifier_refresh_period = max(1, int(verifier_refresh_period))
    pcg_refresh_period = max(1, int(pcg_refresh_period))

    # Guardrail: verifier evaluates (gen, tokens) pairs. If we refresh verifier
    # more frequently than PCG, we end up checking a stale generation against a
    # new token set (citations no longer align). For expensive PCGs (LLMs), users
    # often set `pcg_refresh_period>1` but forget to change verifier period.
    # Default to a consistent schedule in that case.
    if pcg_refresh_period > 1 and verifier_refresh_period == 1:
        verifier_refresh_period = pcg_refresh_period
    semantic_rules = {
        str(x).strip()
        for x in (semantic_rerank_rule_ids if semantic_rerank_rule_ids is not None else ("R5", "R6"))
        if str(x).strip()
    }
    cells: List[Cell] = [root_cell()]
    if init_level > 0:
        n = 2 ** init_level
        cells = [Cell(level=init_level, ix=ix, iy=iy, iz=iz) for ix in range(n) for iy in range(n) for iz in range(n)]
    trace: List[RefineTrace] = []
    issues: List[Issue] = []
    gen: Generation = None  # type: ignore
    stopped_reason = "max_steps"

    # If the caller requests spending the full budget, make sure the step cap is
    # large enough to actually reach `budget_tokens` when possible.
    if require_full_budget and budget_tokens > 0:
        missing = int(budget_tokens) - int(len(cells))
        if missing > 0:
            min_steps_to_fill = int((missing + 6) // 7)  # each split increases token count by +7
            if int(steps) < min_steps_to_fill:
                steps = int(min_steps_to_fill)

    # 如果使用 Evidence Head 但未提供，则创建默认的
    if use_evidence_head and evidence_head is None:
        evidence_head = EvidenceHead(
            emb_dim=emb_dim,
            lambda_uncertainty=lambda_uncertainty,
        )

    token_encoder = TokenEncoder(
        volume=volume,
        emb_dim=emb_dim,
        seed=seed,
        encoder=encoder,
        affine_zyx=affine_zyx,
        anatomy_labels=anatomy_labels,
        anatomy_label_map=anatomy_label_map,
    )

    def _apply_token_scores(tokens_in: List[Token]) -> List[Token]:
        if token_score_fn is None or not tokens_in:
            return tokens_in
        fuse = str(token_score_fuse)
        alpha = float(token_score_blend_alpha)
        beta = float(token_score_max_beta)
        scale = float(token_score_scale)
        if fuse not in ("override", "max", "blend"):
            raise ValueError(f"token_score_fuse must be one of override|max|blend (got {fuse!r})")
        if fuse == "blend" and not (0.0 <= alpha <= 1.0):
            raise ValueError(f"token_score_blend_alpha must be in [0,1] when fuse=blend (got {alpha})")
        if fuse == "max" and beta < 0.0:
            raise ValueError(f"token_score_max_beta must be >=0 when fuse=max (got {beta})")
        if scale < 0.0:
            raise ValueError(f"token_score_scale must be >=0 (got {scale})")
        emb = torch.stack([t.embedding for t in tokens_in], dim=0)
        scores_t = token_score_fn(emb)
        if not isinstance(scores_t, torch.Tensor):
            scores_t = torch.tensor(scores_t)  # type: ignore[arg-type]
        scores = scores_t.detach().cpu().flatten().tolist()
        if len(scores) != len(tokens_in):
            raise ValueError(f"token_score_fn must return N scores, got {len(scores)} for N={len(tokens_in)}")
        out: List[Token] = []
        for t, s in zip(tokens_in, scores):
            ss = float(s)
            ss = max(0.0, min(1.0, scale * ss))
            base = float(t.score)
            if fuse == "max":
                ss = max(ss, beta * base)
            elif fuse == "blend":
                ss = alpha * ss + (1.0 - alpha) * base
            if float(score_level_power) != 0.0:
                ss = ss * float((1 + int(t.level)) ** float(score_level_power))
            out.append(
                Token(
                    token_id=int(t.token_id),
                    cell_id=str(t.cell_id),
                    level=int(t.level),
                    embedding=t.embedding,
                    score=ss,
                    uncertainty=(1.0 - ss) if score_to_uncertainty else float(t.uncertainty),
                    ref=str(getattr(t, "ref", t.cell_id)),
                    bounds_voxel=tuple(getattr(t, "bounds_voxel", (0, 0, 0, 0, 0, 0))),
                    center_voxel=tuple(getattr(t, "center_voxel", (0.0, 0.0, 0.0))),
                    bounds_mm=getattr(t, "bounds_mm", None),
                    center_mm=getattr(t, "center_mm", None),
                    anatomy_label=getattr(t, "anatomy_label", None),
                    ctclip_similarity=getattr(t, "ctclip_similarity", None),
                )
            )
        return out

    last_gen: Optional[Generation] = None
    for step in range(steps):
        # 1. 编码当前 cells 为 tokens（cache-aware）
        tokens = token_encoder.encode(cells)
        tokens = _apply_token_scores(tokens)

        # 2. 运行 PCG 生成
        should_refresh_pcg = (step % pcg_refresh_period == 0) or (step == 0) or (last_gen is None)
        if should_refresh_pcg:
            gen = generator_fn(tokens)
            last_gen = gen
        else:
            gen = last_gen

        # 3. 运行 Verifier（根据 refresh period）
        should_refresh_verifier = (step % verifier_refresh_period == 0) or (step == 0)
        if should_refresh_verifier:
            issues = verifier_fn(gen, tokens)

        # 4. 记录 trace
        trace_entry = RefineTrace(
            step=step,
            num_cells=len(cells),
            num_tokens=len(tokens),
            num_frames=len(gen.frames),
            num_issues=len(issues),
            verifier_refreshed=should_refresh_verifier,
            issues=[
                dict(
                    frame_idx=i.frame_idx,
                    type=i.issue_type,  # IssueType is already a string Literal
                    rule=i.rule_id,
                    severity=i.severity,
                )
                for i in issues
            ],
        )

        # 5. 检查停止条件
        if len(tokens) >= budget_tokens:
            stopped_reason = "budget_reached"
            trace.append(trace_entry)
            break

        if not issues and not require_full_budget:
            stopped_reason = "no_issues"
            trace.append(trace_entry)
            break

        # 6. 选择要 split 的 cell
        if use_evidence_head and evidence_head is not None:
            # 使用 Evidence Head 计算 Δ(c)
            cell_embeddings = {t.cell_id: t.embedding for t in tokens}

            ranked_scores = rank_cells_by_delta(
                cells=cells,
                cell_embeddings=cell_embeddings,
                evidence_head=evidence_head,
                current_issues=issues,
                max_depth=max_depth,
                epsilon=(0.0 if require_full_budget else epsilon),
            )

            if not ranked_scores:
                c_star = pick_cell_to_split(cells, tokens, issues)
                if c_star is None:
                    stopped_reason = "no_splittable_cells"
                    trace.append(trace_entry)
                    break
                trace_entry.split_cell = c_star.id()
                trace.append(trace_entry)
                cells = [c for c in cells if c.id() != c_star.id()] + split(c_star)
                continue

            best_score = ranked_scores[0]
            c_star = best_score.cell

            # 检查 Δ < ε
            if best_score.delta < epsilon and not require_full_budget:
                stopped_reason = f"delta_below_epsilon (Δ={best_score.delta:.4f} < ε={epsilon})"
                trace.append(trace_entry)
                break

            # 记录 Δ 信息
            trace_entry.split_cell = c_star.id()
            trace_entry.delta = best_score.delta
            trace_entry.delta_breakdown = {
                "issue_reduction": best_score.issue_reduction,
                "uncertainty": best_score.uncertainty,
            }
        else:
            # 使用原始 allocator
            splittable = [c for c in cells if c.level < max_depth]
            if not splittable:
                stopped_reason = "max_depth_reached"
                trace.append(trace_entry)
                break

            c_star = pick_cell_to_split(splittable, tokens, issues, prefer=allocator_prefer)
            if c_star is not None:
                trace_entry.split_cell = c_star.id()

        trace.append(trace_entry)

        if c_star is None:
            stopped_reason = "no_cell_to_split"
            break

        # 7. Split 选中的 cell
        cells = [c for c in cells if c.id() != c_star.id()] + split(c_star)

    # 最终编码和验证
    tokens = token_encoder.encode(cells)
    tokens = _apply_token_scores(tokens)
    gen = generator_fn(tokens)
    issues = verifier_fn(gen, tokens)

    if bool(semantic_rerank_on_violation) and semantic_rules:
        gen_reranked, reranked = _apply_semantic_citation_rerank(
            gen,
            tokens=tokens,
            issues=issues,
            top_k=int(semantic_rerank_topk),
            rule_ids=semantic_rules,
        )
        if reranked:
            gen = gen_reranked
            issues = verifier_fn(gen, tokens)

    if bool(despecify_on_remaining_issues) and issues:
        gen_despecified, changed = _apply_despecify_fallback(
            gen,
            issues=issues,
            confidence_cap=float(despecify_confidence_cap),
        )
        if changed:
            gen = gen_despecified
            issues = verifier_fn(gen, tokens)

    return RefineResult(
        tokens=tokens,
        gen=gen,
        issues=issues,
        trace=trace,
        final_cells=cells,
        total_steps=len(trace),
        stopped_reason=stopped_reason,
    )


def run_refine_loop_with_grounding(
    volume: torch.Tensor,
    budget_tokens: int,
    steps: int,
    generator_fn: Callable[[List[Token]], Generation],
    verifier_fn: Callable[[Generation, List[Token]], List[Issue]],
    lesion_masks: Optional[Dict[int, Any]] = None,
    emb_dim: int = 32,
    seed: int = 0,
    evidence_head: Optional[EvidenceHead] = None,
    epsilon: float = 0.01,
    max_depth: int = 4,
) -> RefineResult:
    """带 grounding 监督的 refine loop（M2 阶段）

    Args:
        lesion_masks: frame_idx -> 3D lesion mask，用于 grounding consistency
        其他参数同 run_refine_loop

    Returns:
        RefineResult，trace 中包含 grounding 相关信息
    """
    # 基础 refine loop
    result = run_refine_loop(
        volume=volume,
        budget_tokens=budget_tokens,
        steps=steps,
        generator_fn=generator_fn,
        verifier_fn=verifier_fn,
        emb_dim=emb_dim,
        seed=seed,
        evidence_head=evidence_head,
        use_evidence_head=True,
        epsilon=epsilon,
        max_depth=max_depth,
    )

    # 如果有 lesion masks，计算 grounding 信息
    if lesion_masks is not None:
        from ..eval.metrics_grounding import compute_citation_grounding

        volume_shape = tuple(volume.shape)
        grounding_scores = []

        for frame_idx, cites in result.gen.citations.items():
            if frame_idx in lesion_masks:
                score = compute_citation_grounding(
                    citations=cites,
                    tokens=result.tokens,
                    lesion_mask=lesion_masks[frame_idx],
                    volume_shape=volume_shape,
                )
                grounding_scores.append(score)

        # 添加到 result（可以扩展 RefineResult 或使用 trace）
        if result.trace:
            result.trace[-1].delta_breakdown = result.trace[-1].delta_breakdown or {}
            result.trace[-1].delta_breakdown["grounding_iou"] = (
                sum(s["iou_union"] for s in grounding_scores) / len(grounding_scores)
                if grounding_scores else 0.0
            )

    return result


# 向后兼容的简化接口
def refine_loop_simple(
    volume: torch.Tensor,
    budget_tokens: int,
    steps: int,
    generator_fn: Callable[[List[Token]], Generation],
    verifier_fn: Callable[[Generation, List[Token]], List[Issue]],
    emb_dim: int = 32,
    seed: int = 0,
) -> RefineResult:
    """简化接口，不使用 Evidence Head"""
    return run_refine_loop(
        volume=volume,
        budget_tokens=budget_tokens,
        steps=steps,
        generator_fn=generator_fn,
        verifier_fn=verifier_fn,
        emb_dim=emb_dim,
        seed=seed,
        use_evidence_head=False,
    )
