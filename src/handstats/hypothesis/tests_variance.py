"""方差类假设检验：两总体方差的 F 检验（方差齐性）。"""
from __future__ import annotations

import numpy as np

from handstats import distributions
from handstats._registry import register
from handstats.base import TestResult
from handstats.hypothesis._common import _h0_h1
from handstats.validate import as_sample, check_alpha, check_alternative


@register("ftest_2samp_var")
def ftest_2samp_var(x1, x2, alternative="two-sided", alpha=0.05):
    """双样本方差 F 检验：H0: σ1² = σ2²（两总体均近似正态时适用）。

    统计量 F = s1²/s2²，自由度 (n1-1, n2-1)；双侧 p = 2·min[P(F ≤ F0),
    P(F ≥ F0)]，与 R 的 var.test 同式——不需要手工把大方差挪到分子。
    """
    alternative = check_alternative(alternative)
    alpha = check_alpha(alpha)
    x1 = as_sample(x1, "x1")
    x2 = as_sample(x2, "x2")
    if x1.size < 2 or x2.size < 2:
        raise ValueError("F 检验需要每个样本至少 2 个观测值")

    n1, n2 = x1.size, x2.size
    v1 = float(np.var(x1, ddof=1))
    v2 = float(np.var(x2, ddof=1))
    if v1 == 0 or v2 == 0:
        raise ValueError("样本方差为 0（该组数据完全相同），F 统计量无定义")
    dfn, dfd = n1 - 1, n2 - 1
    F = v1 / v2
    p = distributions.p_value(F, alternative, dist="f", dfn=dfn, dfd=dfd)

    if alternative == "two-sided":
        p_line = (
            f"p = 2·min[P(F ≤ {F:.6g}), P(F ≥ {F:.6g})] = {p:.6g}"
            f"（F 分布，dfn = {dfn}，dfd = {dfd}）"
        )
    else:
        sign = "≤" if alternative == "less" else "≥"
        p_line = f"p = P(F {sign} {F:.6g}) = {p:.6g}（F 分布，dfn = {dfn}，dfd = {dfd}）"

    steps = [
        f"样本 1：n1 = {n1}，样本方差 s1² = {v1:.6g}（自由度 dfn = n1 - 1 = {dfn}）",
        f"样本 2：n2 = {n2}，样本方差 s2² = {v2:.6g}（自由度 dfd = n2 - 1 = {dfd}）",
        f"统计量 F = s1²/s2² = {v1:.6g}/{v2:.6g} = {F:.6g}",
        p_line,
    ]
    h0, h1 = _h0_h1("σ1²", "σ2²", alternative)
    return TestResult(
        statistic=F,
        pvalue=p,
        method="双样本方差 F 检验",
        alternative=alternative,
        alpha=alpha,
        h0=h0,
        h1=h1,
        params={"n1": n1, "n2": n2, "dfn": dfn, "dfd": dfd, "var1": v1, "var2": v2},
        steps=steps,
    )
