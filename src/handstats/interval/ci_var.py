"""区间估计：总体方差 σ² 的置信区间（卡方区间）。"""
from __future__ import annotations

import numpy as np

from handstats import distributions
from handstats._registry import register
from handstats.base import IntervalResult
from handstats.validate import as_sample, check_confidence


@register("ci_var")
def ci_var(x, confidence=0.95):
    """总体方差 σ² 的置信区间（要求总体近似正态）。

    公式：(n-1)s² / χ²(α/2) ≤ σ² ≤ (n-1)s² / χ²(1-α/2)，
    其中 χ²(α/2)、χ²(1-α/2) 是右尾面积为 α/2、1-α/2 的卡方分位数。
    注意区间不对称：卡方分布偏态，点估计 s² 不在区间正中间——
    这与 z/t 区间不同，是初学时最容易犯迷糊的地方。
    params 附带标准差 σ 的区间（对 σ² 区间开方）。
    """
    confidence = check_confidence(confidence)
    x = as_sample(x)
    if x.size < 2:
        raise ValueError("求方差区间至少需要 2 个观测值")
    n = x.size
    df = n - 1
    s2 = float(np.var(x, ddof=1))
    alpha = 1 - confidence
    crit_hi = distributions.chi2_ppf(1 - alpha / 2, df) # 右尾 α/2（较大的值）
    crit_lo = distributions.chi2_ppf(alpha / 2, df)     # 右尾 1-α/2（较小的值）
    lower = (n - 1) * s2 / crit_hi
    upper = (n - 1) * s2 / crit_lo

    steps = [
        f"样本量 n = {n}，自由度 df = n - 1 = {df}，样本方差 s² = {s2:.6g}",
        f"临界值 χ²(α/2, df)（右尾 α/2）= {crit_hi:.6g}，"
        f"χ²(1-α/2, df)（右尾 1-α/2）= {crit_lo:.6g}",
        f"下界 = (n-1)s² / χ²(α/2) = ({n - 1})×{s2:.6g}/{crit_hi:.6g} = {lower:.6g}",
        f"上界 = (n-1)s² / χ²(1-α/2) = ({n - 1})×{s2:.6g}/{crit_lo:.6g} = {upper:.6g}",
    ]
    return IntervalResult(
        estimate=s2,
        lower=lower,
        upper=upper,
        confidence=confidence,
        method="方差卡方区间",
        params={"n": n, "df": df, "var": s2,
                "sd_lower": float(np.sqrt(lower)), "sd_upper": float(np.sqrt(upper))},
        steps=steps,
    )
