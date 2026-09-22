"""区间估计：总体比例 p 的置信区间（Wald 大样本近似）。"""
from __future__ import annotations

import math

from mystats import distributions
from mystats._registry import register
from mystats.base import IntervalResult
from mystats.validate import check_confidence, check_positive, check_successes


@register("ci_proportion")
def ci_proportion(x, n, confidence=0.95):
    """总体比例 p 的置信区间（Wald 近似，大样本）。

    公式：样本比例 ± z(α/2)·√[样本比例(1-样本比例)/n]。
    成功次数或失败次数 < 5 时正态近似不可靠，steps 会自动提示
    （数据驱动的精确区间将来在 resampling 子包用 Bootstrap 补充）。
    """
    confidence = check_confidence(confidence)
    n = check_positive(n, "n")
    x = check_successes(x, n, "x")

    phat = x / n
    se = math.sqrt(phat * (1 - phat) / n)
    q = 1 - (1 - confidence) / 2
    crit = distributions.norm_ppf(q)
    half = crit * se

    steps = [
        f"试验次数 n = {n:g}，成功次数 x = {x:g}，样本比例 = x/n = {phat:.6g}",
        f"标准误 SE = √[样本比例×(1-样本比例)/n] = {se:.6g}",
        f"临界值 z(α/2) = 标准正态的 {q:g} 分位数 = {crit:.6g}",
        f"{confidence:.0%} 置信区间 = 样本比例 ± z·SE = {phat:.6g} ± {half:.6g}",
    ]
    if min(x, n - x) < 5:
        steps.append(
            f"注意：成功/失败次数最小为 {min(x, n - x):g} < 5，正态近似可能不可靠"
        )
    return IntervalResult(
        estimate=phat,
        lower=phat - half,
        upper=phat + half,
        confidence=confidence,
        method="比例 Wald 区间（大样本）",
        params={"n": n, "x": x, "phat": phat, "se": se, "critical_value": crit},
        steps=steps,
    )
