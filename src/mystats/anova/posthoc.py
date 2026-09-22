"""事后多重比较（post-hoc）：Tukey HSD。

为什么 ANOVA 拒绝 H0 之后不直接逐对做 t 检验？
--------
k 组有 k(k-1)/2 个均值对；逐对 t 检验会让“至少一个假阳性”的族错误率
随对数膨胀（3 组时 3 对、6 组时 15 对）。Tukey HSD 把族错误率精确控制
在 α：用学生化极差分布的临界值 q(α; k, df)，所有对共用同一个
“诚实显著差”HSD = q·SE，一次比较全部对。
"""
from __future__ import annotations

import numpy as np

from mystats import distributions
from mystats._registry import register
from mystats.base import TestResult
from mystats.anova.oneway import _as_groups, _oneway_core
from mystats.validate import check_alpha


@register("tukey_hsd")
def tukey_hsd(groups, alpha=0.05):
    """Tukey HSD 事后多重比较（族错误率 = α）：到底哪些组的均值不同？

    参数
    ----
    groups : 若干组样本的序列（与 anova_oneway 相同的输入）
    alpha : 族错误率（整体犯第一类错误的概率上限）

    返回 list[TestResult]，按 (组i, 组j) 字典序排列（i < j）；
    每个的 params 里有均值差 diff、区间 [lower, upper]、HSD 半宽。
    组样本量不同时自动用 Tukey-Kramer 公式。
    """
    alpha = check_alpha(alpha)
    arrays = _as_groups(groups)
    core = _oneway_core(arrays)
    if core["ms_e"] <= 0:
        raise ValueError("组内均方为 0（每组内部数据完全相同），无法做多重比较")
    k, df_e, mse = core["k"], core["df_e"], core["ms_e"]
    q_crit = distributions.studentized_range_ppf(1 - alpha, k, df_e)

    results = []
    for i in range(k):
        for j in range(i + 1, k):
            ni, nj = arrays[i].size, arrays[j].size
            diff = float(core["means"][i] - core["means"][j])
            se = float(np.sqrt(mse / 2.0 * (1.0 / ni + 1.0 / nj)))
            q = abs(diff) / se
            p = distributions.studentized_range_sf(q, k, df_e)
            half = q_crit * se
            steps = [
                f"前提：ANOVA 的 MSE = {mse:.6g}，误差自由度 df = {df_e}，组数 k = {k}",
                f"均值{i + 1} = {core['means'][i]:.6g}（n = {ni}），"
                f"均值{j + 1} = {core['means'][j]:.6g}（n = {nj}），"
                f"均值差 = {diff:.6g}",
                f"SE = √(MSE/2·(1/n_i + 1/n_j)) = {se:.6g}",
                f"统计量 q = |均值差|/SE = {q:.6g}",
                f"临界值 q(α; k, df) = {q_crit:.6g}，"
                f"诚实显著差 HSD = 临界值×SE = {half:.6g}",
                f"区间 = 均值差 ± HSD = [{diff - half:.6g}, {diff + half:.6g}]，"
                f"p = P(Q ≥ q) = {p:.6g}（学生化极差分布，k = {k}，df = {df_e}）",
            ]
            results.append(TestResult(
                statistic=float(q),
                pvalue=float(p),
                method=f"Tukey HSD：组{i + 1} vs 组{j + 1}",
                alternative="greater",
                alpha=alpha,
                h0=f"μ{i + 1} = μ{j + 1}",
                h1=f"μ{i + 1} ≠ μ{j + 1}",
                params={"diff": diff, "se": se, "lower": diff - half,
                        "upper": diff + half, "hsd": half, "q_crit": q_crit,
                        "n_i": ni, "n_j": nj, "k": k, "df": df_e},
                steps=steps,
            ))
    return results
