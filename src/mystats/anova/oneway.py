"""单因素方差分析（ANOVA）与它的两个配套：ANOVA 表、事后比较的公共底座。

方差分析的骨架是一张表（教材核心）：
    SS_A（组间/因素） + SS_E（组内/误差） = SS_T（总）
    F = MS_A/MS_E = (SS_A/(k-1))/(SS_E/(n-k))
H0: k 个总体均值全相等；H1: 至少两个不等。F 越大越异常（右尾）。
本文件提供 _as_groups / _oneway_core / _anova_table 三个内部工具，
tukey_hsd（posthoc.py）与 levene_test（levene.py）都复用它们。
"""
from __future__ import annotations

import numpy as np

from mystats import distributions
from mystats._registry import register
from mystats.base import TestResult
from mystats.validate import as_sample, check_alpha


def _as_groups(groups):
    """把“若干组样本”统一为若干一维数组并检查（单因素家族共用）。"""
    if isinstance(groups, (str, bytes)):
        raise ValueError("groups 应是若干组样本的序列（列表/元组），不是字符串")
    try:
        items = list(groups)
    except TypeError:
        raise ValueError("groups 应是若干组样本的序列（列表/元组）")
    if len(items) < 2:
        raise ValueError(f"方差分析至少需要 2 组，收到 {len(items)} 组")
    arrays = [as_sample(g, f"groups[{i}]") for i, g in enumerate(items)]
    n = int(sum(a.size for a in arrays))
    if n - len(arrays) < 1:
        raise ValueError("总观测数必须大于组数（否则误差自由度 n - k = 0）")
    return arrays


def _oneway_core(arrays):
    """单因素 ANOVA 的 SS 分解与 F（Levene 内部对 |离差| 复用同一函数）。"""
    k = len(arrays)
    n = int(sum(a.size for a in arrays))
    means = np.array([float(a.mean()) for a in arrays])
    grand = float(np.concatenate(arrays).mean())
    ss_a = float(sum(arrays[i].size * (means[i] - grand) ** 2 for i in range(k)))
    ss_e = float(sum(float(np.sum((a - a.mean()) ** 2)) for a in arrays))
    df_a, df_e = k - 1, n - k
    ms_a, ms_e = float(ss_a / df_a), float(ss_e / df_e)
    return {
        "k": k, "n": n, "means": means, "sizes": [int(a.size) for a in arrays],
        "grand": grand, "ss_a": ss_a, "ss_e": ss_e, "df_a": df_a, "df_e": df_e,
        "ms_a": ms_a, "ms_e": ms_e,
        # ms_e = 0（组内无变异）时 F 视为无穷；调用方负责给出友好的 ValueError
        "f_stat": float("inf") if ms_e == 0 else float(ms_a / ms_e),
    }


def _anova_table(rows):
    """渲染方差分析表：rows = [(来源, ss, df, ms, f, p), ...]，None 表示空格。

    教材里的方差分析表（来源 / SS / df / MS / F / p）逐行对应。
    """
    lines = [f"{'来源':<6}{'SS':>14}{'df':>6}{'MS':>14}{'F':>11}{'p':>11}"]
    for src, ss, df, ms, f_, p in rows:
        ms_s = f"{ms:>14.6g}" if ms is not None else f"{'':>14}"
        f_s = f"{f_:>11.6g}" if f_ is not None else f"{'':>11}"
        p_s = f"{p:>11.6g}" if p is not None else f"{'':>11}"
        lines.append(f"{src:<6}{ss:>14.6g}{df:>6g}{ms_s}{f_s}{p_s}")
    return lines


@register("anova_oneway")
def anova_oneway(groups, alpha=0.05):
    """单因素方差分析：H0: k 个总体均值全相等。

    参数
    ----
    groups : 若干组样本的序列（各组样本量可以不同）
    alpha : 显著性水平

    返回 TestResult；params 里带完整 SS 分解、各组均值和 η²（效应量：
    因素解释的方差占比，SS_A/SS_T）以及 anova_table（表格文本行）。
    前提：各组近似正态、方差齐（用 levene_test 检查）。
    """
    alpha = check_alpha(alpha)
    arrays = _as_groups(groups)
    core = _oneway_core(arrays)
    k = core["k"]
    if core["ms_e"] <= 0:
        raise ValueError("组内均方为 0（每组内部数据完全相同），F 无定义")
    F = core["f_stat"]
    p = distributions.f_sf(F, core["df_a"], core["df_e"])

    means_str = ", ".join(f"{m:.6g}" for m in core["means"])
    ss_t = core["ss_a"] + core["ss_e"]
    table = _anova_table([
        ("因素A", core["ss_a"], core["df_a"], core["ms_a"], F, p),
        ("误差E", core["ss_e"], core["df_e"], core["ms_e"], None, None),
        ("总和T", ss_t, core["n"] - 1, None, None, None),
    ])
    steps = [
        f"组数 k = {k}，各组样本量 = {core['sizes']}，总 n = {core['n']}",
        f"各组均值 = ({means_str})，总均值 = {core['grand']:.6g}",
        f"SS_A = Σn_i·(组均值_i - 总均值)² = {core['ss_a']:.6g}，"
        f"SS_E = ΣΣ(y_ij - 组均值_i)² = {core['ss_e']:.6g}",
        "方差分析表：", *table,
        f"p = P(F ≥ {F:.6g}) = {p:.6g}"
        f"（F 分布，dfn = {core['df_a']}，dfd = {core['df_e']}）",
    ]
    return TestResult(
        statistic=F,
        pvalue=p,
        method="单因素方差分析",
        alternative="greater",
        alpha=alpha,
        h0=f"{k} 个总体均值全相等",
        h1="至少有两个总体均值不相等",
        params={**core, "ss_t": ss_t, "eta_sq": core["ss_a"] / ss_t,
                "anova_table": table},
        steps=steps,
    )
