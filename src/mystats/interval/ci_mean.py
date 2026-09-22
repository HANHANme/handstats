"""区间估计：均值的置信区间（σ 已知用 z，未知用 t）。

这个模块同时演示了“区间和检验共享同一套枢轴量”：
ci_mean 与 ttest_1samp 用的是同一个 (样本均值 - μ)/(s/√n) 结构，
所以阶段 2（区间估计扩充）的实际工作量很小——这正是骨架分层的好处。
"""
from __future__ import annotations

import numpy as np

from mystats import distributions
from mystats._registry import register
from mystats.base import IntervalResult
from mystats.validate import as_sample, check_confidence, check_positive


@register("ci_mean")
def ci_mean(x, confidence=0.95, sigma=None):
    """总体均值的置信区间。

    sigma=None → t 区间：均值 ± t(α/2, n-1)·s/√n（σ 未知，最常用）
    sigma=数值 → z 区间：均值 ± z(α/2)·σ/√n（σ 已知）
    """
    confidence = check_confidence(confidence)
    x = as_sample(x)
    n = x.size
    xbar = float(np.mean(x))
    q = 1 - (1 - confidence) / 2 # 上侧分位点，如 95% → 0.975

    if sigma is not None:
        sigma = check_positive(sigma, "sigma")
        crit = distributions.norm_ppf(q)
        se = sigma / np.sqrt(n)
        method = "z 区间（σ 已知）"
        steps = [
            f"样本量 n = {n}，样本均值 = {xbar:.6g}",
            f"已知 σ = {sigma:g}，标准误 SE = σ/√n = {se:.6g}",
            f"临界值 z(α/2) = 标准正态的 {q:g} 分位数 = {crit:.6g}",
            f"{confidence:.0%} 置信区间 = 均值 ± z·SE = {xbar:.6g} ± {crit * se:.6g}",
        ]
    else:
        if n < 2:
            raise ValueError("σ 未知时求 t 区间需要至少 2 个观测值")
        df = n - 1
        s = float(np.std(x, ddof=1))
        crit = distributions.t_ppf(q, df)
        se = s / np.sqrt(n)
        method = "t 区间（σ 未知）"
        steps = [
            f"样本量 n = {n}，自由度 df = n - 1 = {df}",
            f"样本均值 = {xbar:.6g}，样本标准差 s = {s:.6g}",
            f"标准误 SE = s/√n = {se:.6g}",
            f"临界值 t(α/2, df) = {crit:.6g}",
            f"{confidence:.0%} 置信区间 = 均值 ± t·SE = {xbar:.6g} ± {crit * se:.6g}",
        ]

    half = crit * se
    return IntervalResult(
        estimate=xbar,
        lower=xbar - half,
        upper=xbar + half,
        confidence=confidence,
        method=method,
        params={"n": n, "se": float(se), "critical_value": float(crit)},
        steps=steps,
    )
