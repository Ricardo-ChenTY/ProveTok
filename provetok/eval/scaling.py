"""Scaling Law 拟合与 Compute Allocation Model

根据 proposal §5:
1. Performance-Budget scaling（可解释拟合，不绑死形状）
   - 饱和幂律: P(B) = P_∞ - a(B+b_0)^{-α}
   - 对数饱和: P(B) = c_0 + c_1·log(B+b_0)

2. Allocation model（必须能"预测最优分配"，不是只画曲线）
   - 统一 compute 单位为 FLOPs
   - 学习回归预测器 P̂(θ)
   - 解约束优化: max_θ P̂(θ) s.t. FLOPs_total(θ) ≤ B
   - 报告 regret（预测最优 vs 实际最优差距）
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.stats import pearsonr
import warnings


@dataclass
class ScalingFit:
    """Scaling law 拟合结果"""
    model_type: str           # "power_law" or "log_saturation"
    params: Dict[str, float]  # 拟合参数
    aic: float                # Akaike Information Criterion
    bic: float                # Bayesian Information Criterion
    r_squared: float          # R² 拟合度
    residual_std: float       # 残差标准差

    def predict(self, budget: float) -> float:
        """预测给定预算下的性能"""
        if self.model_type == "power_law":
            return _power_law(budget, **self.params)
        elif self.model_type == "log_saturation":
            return _log_saturation(budget, **self.params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


@dataclass
class AllocationConfig:
    """Compute allocation 配置"""
    b_enc: int          # encoder token 预算
    n_refine: int       # refine 迭代次数
    b_gen: int          # generation token 预算
    n_verify: int       # verifier 调用次数

    def total_flops(self, flops_per_enc: float, flops_per_dec: float, flops_per_verify: float) -> float:
        """计算总 FLOPs"""
        return (
            self.b_enc * flops_per_enc +
            self.b_gen * flops_per_dec +
            self.n_verify * flops_per_verify
        )


@dataclass
class AllocationResult:
    """Allocation model 预测结果"""
    optimal_config: AllocationConfig
    predicted_performance: float
    actual_performance: Optional[float]
    regret: Optional[float]  # predicted optimal - actual optimal


# ============================================================
# Scaling Law 函数
# ============================================================

def _power_law(B: float, P_inf: float, a: float, b_0: float, alpha: float) -> float:
    """饱和幂律: P(B) = P_∞ - a(B+b_0)^{-α}"""
    return P_inf - a * np.power(B + b_0, -alpha)


def _log_saturation(B: float, c_0: float, c_1: float, b_0: float) -> float:
    """对数饱和: P(B) = c_0 + c_1·log(B+b_0)，截断到 [0,1]"""
    val = c_0 + c_1 * np.log(B + b_0)
    return np.clip(val, 0.0, 1.0)


def fit_power_law(
    budgets: np.ndarray,
    performances: np.ndarray,
    max_iter: int = 10000,
) -> Optional[ScalingFit]:
    """拟合饱和幂律模型"""
    try:
        # 初始猜测
        p0 = [
            np.max(performances),  # P_inf
            0.1,                   # a
            1.0,                   # b_0
            0.5,                   # alpha
        ]

        # 边界
        bounds = (
            [0.0, 0.0, 0.1, 0.01],    # lower
            [1.0, 10.0, 100.0, 5.0],  # upper
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                lambda B, P_inf, a, b_0, alpha: _power_law(B, P_inf, a, b_0, alpha),
                budgets, performances,
                p0=p0, bounds=bounds, maxfev=max_iter
            )

        params = {
            "P_inf": popt[0],
            "a": popt[1],
            "b_0": popt[2],
            "alpha": popt[3],
        }

        # 计算拟合度
        predictions = np.array([_power_law(b, **params) for b in budgets])
        return _compute_fit_stats("power_law", params, performances, predictions)

    except Exception:
        return None


def fit_log_saturation(
    budgets: np.ndarray,
    performances: np.ndarray,
    max_iter: int = 10000,
) -> Optional[ScalingFit]:
    """拟合对数饱和模型"""
    try:
        # 初始猜测
        p0 = [
            np.min(performances),  # c_0
            0.1,                   # c_1
            1.0,                   # b_0
        ]

        # 边界
        bounds = (
            [-1.0, 0.0, 0.1],     # lower
            [1.0, 1.0, 100.0],    # upper
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                lambda B, c_0, c_1, b_0: _log_saturation(B, c_0, c_1, b_0),
                budgets, performances,
                p0=p0, bounds=bounds, maxfev=max_iter
            )

        params = {
            "c_0": popt[0],
            "c_1": popt[1],
            "b_0": popt[2],
        }

        # 计算拟合度
        predictions = np.array([_log_saturation(b, **params) for b in budgets])
        return _compute_fit_stats("log_saturation", params, performances, predictions)

    except Exception:
        return None


def _compute_fit_stats(
    model_type: str,
    params: Dict[str, float],
    actual: np.ndarray,
    predicted: np.ndarray,
) -> ScalingFit:
    """计算拟合统计量"""
    n = len(actual)
    k = len(params)

    # 残差
    residuals = actual - predicted
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)

    # R²
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # 残差标准差
    residual_std = np.std(residuals)

    # AIC / BIC
    # 假设正态误差
    sigma2 = ss_res / n
    log_likelihood = -n/2 * (np.log(2*np.pi) + np.log(sigma2) + 1)
    aic = 2*k - 2*log_likelihood
    bic = k*np.log(n) - 2*log_likelihood

    return ScalingFit(
        model_type=model_type,
        params=params,
        aic=aic,
        bic=bic,
        r_squared=r_squared,
        residual_std=residual_std,
    )


def fit_scaling_law(
    budgets: List[float],
    performances: List[float],
    criterion: str = "bic",
) -> Tuple[ScalingFit, str]:
    """拟合 scaling law，自动选择最佳模型

    Args:
        budgets: 预算列表
        performances: 对应的性能指标
        criterion: 模型选择标准 ("aic" or "bic")

    Returns:
        (best_fit, reason) - 最佳拟合结果和选择原因
    """
    budgets = np.array(budgets)
    performances = np.array(performances)

    fits = []

    # 尝试两种模型
    power_fit = fit_power_law(budgets, performances)
    if power_fit:
        fits.append(power_fit)

    log_fit = fit_log_saturation(budgets, performances)
    if log_fit:
        fits.append(log_fit)

    if not fits:
        raise ValueError("无法拟合任何 scaling law 模型")

    # 按 criterion 选择最佳
    if criterion == "aic":
        best_fit = min(fits, key=lambda f: f.aic)
        reason = f"Selected by AIC ({best_fit.aic:.2f})"
    else:  # bic
        best_fit = min(fits, key=lambda f: f.bic)
        reason = f"Selected by BIC ({best_fit.bic:.2f})"

    return best_fit, reason


def compute_diminishing_returns_point(
    fit: ScalingFit,
    threshold: float = 0.01,
    max_budget: float = 1000,
) -> float:
    """计算边际收益递减点

    找到 dP/dB < threshold 的最小 budget

    Args:
        fit: Scaling law 拟合结果
        threshold: 边际收益阈值
        max_budget: 搜索上限

    Returns:
        边际收益递减点的 budget
    """
    for b in np.linspace(1, max_budget, 1000):
        # 数值微分
        eps = 0.1
        grad = (fit.predict(b + eps) - fit.predict(b)) / eps

        if grad < threshold:
            return b

    return max_budget


# ============================================================
# Allocation Model
# ============================================================

class AllocationModel:
    """Compute Allocation 预测模型

    学习从配置 θ 到性能 P 的映射，并在预算约束下优化
    """

    def __init__(
        self,
        flops_per_enc: float = 1.0,
        flops_per_dec: float = 2.0,
        flops_per_verify: float = 0.5,
    ):
        """
        Args:
            flops_per_enc: 每个 encoder token 的 FLOPs（归一化）
            flops_per_dec: 每个 decoder token 的 FLOPs
            flops_per_verify: 每次 verifier 调用的 FLOPs
        """
        self.flops_per_enc = flops_per_enc
        self.flops_per_dec = flops_per_dec
        self.flops_per_verify = flops_per_verify

        # 训练数据
        self.configs: List[AllocationConfig] = []
        self.performances: List[float] = []

        # 拟合的预测器（简单线性回归）
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0

    def add_sample(self, config: AllocationConfig, performance: float):
        """添加一个训练样本"""
        self.configs.append(config)
        self.performances.append(performance)

    def _config_to_features(self, config: AllocationConfig) -> np.ndarray:
        """将配置转换为特征向量"""
        return np.array([
            config.b_enc,
            config.n_refine,
            config.b_gen,
            config.n_verify,
            np.log1p(config.b_enc),  # log 特征
            np.log1p(config.n_refine),
            config.b_enc * config.n_refine,  # 交互项
        ])

    def fit(self):
        """拟合预测模型"""
        if len(self.configs) < 5:
            raise ValueError("需要至少 5 个训练样本")

        # 构建特征矩阵
        X = np.array([self._config_to_features(c) for c in self.configs])
        y = np.array(self.performances)

        # 添加偏置
        X = np.hstack([X, np.ones((len(X), 1))])

        # 最小二乘拟合
        self.weights, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        self.bias = self.weights[-1]
        self.weights = self.weights[:-1]

    def predict(self, config: AllocationConfig) -> float:
        """预测给定配置的性能"""
        if self.weights is None:
            raise ValueError("模型尚未拟合")

        features = self._config_to_features(config)
        return float(np.dot(features, self.weights) + self.bias)

    def find_optimal_config(
        self,
        budget_flops: float,
        b_enc_range: Tuple[int, int] = (8, 128),
        n_refine_range: Tuple[int, int] = (1, 10),
        b_gen_range: Tuple[int, int] = (64, 512),
        n_verify_range: Tuple[int, int] = (1, 5),
    ) -> AllocationResult:
        """在预算约束下找最优配置

        max_θ P̂(θ) s.t. FLOPs_total(θ) ≤ budget_flops

        使用网格搜索（配置空间较小）
        """
        if self.weights is None:
            raise ValueError("模型尚未拟合")

        best_config = None
        best_predicted = -float('inf')

        # 网格搜索
        for b_enc in range(b_enc_range[0], b_enc_range[1] + 1, 8):
            for n_refine in range(n_refine_range[0], n_refine_range[1] + 1):
                for b_gen in range(b_gen_range[0], b_gen_range[1] + 1, 32):
                    for n_verify in range(n_verify_range[0], n_verify_range[1] + 1):
                        config = AllocationConfig(
                            b_enc=b_enc,
                            n_refine=n_refine,
                            b_gen=b_gen,
                            n_verify=n_verify,
                        )

                        # 检查预算约束
                        total_flops = config.total_flops(
                            self.flops_per_enc,
                            self.flops_per_dec,
                            self.flops_per_verify,
                        )

                        if total_flops > budget_flops:
                            continue

                        # 预测性能
                        predicted = self.predict(config)

                        if predicted > best_predicted:
                            best_predicted = predicted
                            best_config = config

        if best_config is None:
            raise ValueError("在预算约束下找不到有效配置")

        return AllocationResult(
            optimal_config=best_config,
            predicted_performance=best_predicted,
            actual_performance=None,
            regret=None,
        )

    def compute_regret(
        self,
        predicted_result: AllocationResult,
        actual_optimal_performance: float,
    ) -> float:
        """计算 regret

        regret = actual_optimal - predicted_optimal 的实际性能
        """
        return actual_optimal_performance - (predicted_result.actual_performance or 0.0)


def format_scaling_report(
    fit: ScalingFit,
    budgets: List[float],
    performances: List[float],
) -> str:
    """格式化 scaling law 报告"""
    lines = [
        "=" * 60,
        "Scaling Law Analysis",
        "=" * 60,
        f"Model Type: {fit.model_type}",
        "",
        "Parameters:",
    ]

    for key, val in fit.params.items():
        lines.append(f"  {key}: {val:.4f}")

    lines.extend([
        "",
        "Fit Statistics:",
        f"  R²: {fit.r_squared:.4f}",
        f"  Residual Std: {fit.residual_std:.4f}",
        f"  AIC: {fit.aic:.2f}",
        f"  BIC: {fit.bic:.2f}",
        "",
        "Predictions vs Actual:",
    ])

    for b, p in zip(budgets, performances):
        pred = fit.predict(b)
        lines.append(f"  B={b}: actual={p:.4f}, pred={pred:.4f}, err={abs(p-pred):.4f}")

    # 边际收益递减点
    dr_point = compute_diminishing_returns_point(fit)
    lines.append(f"\nDiminishing Returns Point: B={dr_point:.1f}")

    lines.append("=" * 60)
    return "\n".join(lines)


def format_allocation_report(
    model: AllocationModel,
    result: AllocationResult,
    budget_flops: float,
) -> str:
    """格式化 allocation model 报告"""
    config = result.optimal_config
    lines = [
        "=" * 60,
        "Compute Allocation Analysis",
        "=" * 60,
        f"Budget: {budget_flops:.1f} FLOPs",
        "",
        "Optimal Configuration:",
        f"  B_enc (tokens): {config.b_enc}",
        f"  n_refine: {config.n_refine}",
        f"  B_gen (tokens): {config.b_gen}",
        f"  n_verify: {config.n_verify}",
        "",
        f"Predicted Performance: {result.predicted_performance:.4f}",
    ]

    if result.actual_performance is not None:
        lines.append(f"Actual Performance: {result.actual_performance:.4f}")

    if result.regret is not None:
        lines.append(f"Regret: {result.regret:.4f}")

    actual_flops = config.total_flops(
        model.flops_per_enc,
        model.flops_per_dec,
        model.flops_per_verify,
    )
    lines.append(f"\nActual FLOPs Used: {actual_flops:.1f} ({100*actual_flops/budget_flops:.1f}%)")

    lines.append("=" * 60)
    return "\n".join(lines)
