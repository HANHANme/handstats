"""比例类假设检验（大样本正态近似）：单比例、双比例 z 检验。

比例数据没有“样本标准差”，标准误完全由原假设（或合并样本）的比例算出，
这是它与均值检验最大的差别——steps 里专门写清楚这一点。
"""
from __future__ import annotations

import math

from mystats import distributions
from mystats._registry import register
from mystats.base import TestResult
from mystats.hypothesis._common import _h0_h1, _p_step
from mystats.validate import (
    check_alpha,
    check_alternative,
    check_positive,
    check_proportion,
    check_successes,
)

@register("ztest_1prop")
def ztest_1prop(x, n, p0, alternative="two-sided", alpha=0.05):
    """单比例 z 检验（大样本正态近似）：H0: p = p0。

    参数
    ----
    x : “成功”次数（如 100 次投掷中正面朝上的次数，须为整数）
    n : 总试验次数
    p0 : 原假设中的总体比例
    """
    alternative = check_alternative(alternative)
    alpha = check_alpha(alpha)
    n = check_positive(n, "n")
    x = check_successes(x, n, "x")
    p0 = check_proportion(p0, "p0")

    phat = x / n # 样本比例
    se = math.sqrt(p0 * (1 - p0) / n)
    z = (phat - p0) / se
    p = distributions.p_value(z, alternative, dist="norm")

    steps = [
        f"试验次数 n = {n:g}，成功次数 x = {x:g}，样本比例 = x/n = {phat:.6g}",
        f"H0 下的标准误 SE = √(p0(1-p0)/n) = √({p0:g}×{1 - p0:g}/{n:g}) = {se:.6g}",
        f"统计量 z = (样本比例 - p0)/SE = ({phat:.6g} - {p0:g})/{se:.6g} = {z:.6g}",
        _p_step(alternative, p, "z", "标准正态分布"),
    ]
    h0, h1 = _h0_h1("p", f"{p0:g}", alternative)
    return TestResult(
        statistic=z,
        pvalue=p,
        method="单比例 z 检验（大样本）",
        alternative=alternative,
        alpha=alpha,
        h0=h0,
        h1=h1,
        params={"n": n, "x": x, "phat": phat, "se": se},
        steps=steps,
    )


@register("ztest_2prop")
def ztest_2prop(x1, n1, x2, n2, alternative="two-sided", alpha=0.05):
    """双比例 z 检验（大样本正态近似）：H0: p1 = p2。

    标准误用 H0 下的合并比例 (x1+x2)/(n1+n2) 计算——教材标准做法：
    两比例在 H0 下相等，合并后才能得到更稳的公共比例估计。
    """
    alternative = check_alternative(alternative)
    alpha = check_alpha(alpha)
    n1 = check_positive(n1, "n1")
    n2 = check_positive(n2, "n2")
    x1 = check_successes(x1, n1, "x1")
    x2 = check_successes(x2, n2, "x2")

    phat1, phat2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (phat1 - phat2) / se
    p = distributions.p_value(z, alternative, dist="norm")

    steps = [
        f"样本 1：n1 = {n1:g}，成功 x1 = {x1:g}，样本比例1 = {phat1:.6g}",
        f"样本 2：n2 = {n2:g}，成功 x2 = {x2:g}，样本比例2 = {phat2:.6g}",
        f"合并比例 = (x1+x2)/(n1+n2) = {pooled:.6g}",
        f"标准误 SE = √[合并比例×(1-合并比例)×(1/n1+1/n2)] = {se:.6g}",
        f"统计量 z = (样本比例1 - 样本比例2)/SE = ({phat1:.6g} - {phat2:.6g})/{se:.6g} = {z:.6g}",
        _p_step(alternative, p, "z", "标准正态分布"),
    ]
    h0, h1 = _h0_h1("p1", "p2", alternative)
    return TestResult(
        statistic=z,
        pvalue=p,
        method="双比例 z 检验（大样本）",
        alternative=alternative,
        alpha=alpha,
        h0=h0,
        h1=h1,
        params={"n1": n1, "n2": n2, "x1": x1, "x2": x2,
                "phat1": phat1, "phat2": phat2, "pooled": pooled, "se": se},
        steps=steps,
    )
