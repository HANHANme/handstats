"""方差齐性检验：Levene / Brown-Forsythe（做 ANOVA 前的前提检查）。

原理很巧：把每个观测换成“到组中心的绝对离差”z_ij = |y_ij - 中心_i|，
再对 z 做一遍单因素 ANOVA——方差大的组，离差的平均幅度自然也大。
中心用均值 = 经典 Levene；用中位数 = Brown-Forsythe（对偏态/离群更稳健）。
"""
from __future__ import annotations

import numpy as np

from mystats import distributions
from mystats._registry import register
from mystats.base import TestResult
from mystats.anova.oneway import _as_groups, _oneway_core
from mystats.validate import check_alpha


@register("levene_test")
def levene_test(groups, alpha=0.05, center="mean"):
    """Levene 方差齐性检验：H0: k 个总体方差全相等。

    参数
    ----
    groups : 若干组样本的序列
    center : "mean" 经典 Levene（正态数据下功效高）；
             "median" Brown-Forsythe（数据偏态或有离群点时更稳健）
    alpha : 显著性水平

    注：两总体场合可用 hypothesis.ftest_2samp_var（精确 F 检验，
    但要求正态）；多总体场合用本检验。
    """
    alpha = check_alpha(alpha)
    center = str(center).strip().lower()
    if center not in ("mean", "median"):
        raise ValueError(f'center 只能是 "mean" 或 "median"，收到 {center!r}')
    arrays = _as_groups(groups)

    if center == "mean":
        centers = [float(np.mean(a)) for a in arrays]
        desc = "均值"
    else:
        centers = [float(np.median(a)) for a in arrays]
        desc = "中位数（Brown-Forsythe）"
    z = [np.abs(a - c) for a, c in zip(arrays, centers)]
    core = _oneway_core(z)
    W = core["f_stat"]
    p = distributions.f_sf(W, core["df_a"], core["df_e"])

    centers_str = ", ".join(f"{c:.6g}" for c in centers)
    steps = [
        f"组数 k = {core['k']}，各组中心（{desc}） = ({centers_str})",
        "离差 z_ij = |y_ij - 组中心|（若方差齐，各组 z 的平均水平应接近）",
        f"对 z 做单因素 ANOVA：W = MS_A/MS_E = {W:.6g}，"
        f"df = ({core['df_a']}, {core['df_e']})",
        f"p = P(F ≥ W) = {p:.6g}（F 分布）",
    ]
    return TestResult(
        statistic=W,
        pvalue=p,
        method=f"Levene 方差齐性检验（{desc}）",
        alternative="greater",
        alpha=alpha,
        h0=f"{core['k']} 个总体方差全相等",
        h1="至少有两个总体方差不相等",
        params={"k": core["k"], "df_a": core["df_a"], "df_e": core["df_e"],
                "centers": centers, "center": center},
        steps=steps,
    )
