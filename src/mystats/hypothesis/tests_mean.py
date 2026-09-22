"""均值类假设检验：z 检验（σ 已知）与 t 检验（σ 未知）。

★ 全包的“黄金模板” ★
以后每新增一个检验（卡方、F、秩和、似然比……），照本文件的固定四步走：

    ① 校验   validate.as_sample / check_alternative / check_alpha
    ② 计算   纯 numpy 算出统计量——数学只发生在这一步
    ③ p 值   distributions.p_value（统计量 → p 值）
    ④ 打包   base.TestResult，并把每一步算式写进 steps（手算核对表）

四步彼此独立：换分布只动 ③，改输出格式只动 ④，数学错误只会出在 ②。
"""
from __future__ import annotations

import numpy as np

from mystats import distributions
from mystats._registry import register
from mystats.base import TestResult
from mystats.hypothesis._common import _p_step
from mystats.validate import as_sample, check_alpha, check_alternative, check_positive

# 单样本 / 双样本场合 H1 的文字模板
_H1_ONE = {"two-sided": "μ ≠ {mu0}", "less": "μ < {mu0}", "greater": "μ > {mu0}"}
_H1_TWO = {"two-sided": "μ1 ≠ μ2", "less": "μ1 < μ2", "greater": "μ1 > μ2"}


@register("ztest_1samp")
def ztest_1samp(x, mu0, sigma, alternative="two-sided", alpha=0.05):
    """单样本 z 检验：总体方差 σ² 已知时检验 H0: μ = μ0。

    参数
    ----
    x : 一维样本
    mu0 : 原假设下的总体均值
    sigma : 已知的总体标准差（正数）
    alternative : "two-sided" / "less" / "greater"（也接受 双侧/左侧/右侧）
    alpha : 显著性水平
    """
    alternative = check_alternative(alternative)
    alpha = check_alpha(alpha)
    sigma = check_positive(sigma, "sigma")
    x = as_sample(x)

    n = x.size
    xbar = float(np.mean(x))
    se = float(sigma / np.sqrt(n)) # 标准误 σ/√n（σ 已知，不需要估）
    z = (xbar - mu0) / se
    p = distributions.p_value(z, alternative, dist="norm")

    h1 = _H1_ONE[alternative].format(mu0=f"{mu0:g}")
    steps = [
        f"样本量 n = {n}",
        f"样本均值 = {xbar:.6g}",
        f"已知 σ = {sigma:g}，标准误 SE = σ/√n = {se:.6g}",
        f"统计量 z = (样本均值 - μ0)/SE = ({xbar:.6g} - {mu0:g})/{se:.6g} = {z:.6g}",
        _p_step(alternative, p, "z", "标准正态分布"),
    ]
    return TestResult(
        statistic=z,
        pvalue=p,
        method="单样本 z 检验（σ 已知）",
        alternative=alternative,
        alpha=alpha,
        h0=f"μ = {mu0:g}",
        h1=h1,
        params={"n": n, "mean": xbar, "se": se},
        steps=steps,
    )


@register("ttest_1samp")
def ttest_1samp(x, mu0, alternative="two-sided", alpha=0.05):
    """单样本 t 检验：σ 未知时检验 H0: μ = μ0（用样本标准差 s 代替 σ）。"""
    alternative = check_alternative(alternative)
    alpha = check_alpha(alpha)
    x = as_sample(x)
    if x.size < 2:
        raise ValueError("t 检验需要至少 2 个观测值（要算样本方差）")

    n = x.size
    df = n - 1
    xbar = float(np.mean(x))
    s = float(np.std(x, ddof=1)) # ddof=1 即除以 n-1，和教材手算公式一致
    se = float(s / np.sqrt(n))
    t = (xbar - mu0) / se
    p = distributions.p_value(t, alternative, dist="t", df=df)

    h1 = _H1_ONE[alternative].format(mu0=f"{mu0:g}")
    steps = [
        f"样本量 n = {n}，自由度 df = n - 1 = {df}",
        f"样本均值 = {xbar:.6g}",
        f"样本标准差 s = √[Σ(xi - 均值)²/(n-1)] = {s:.6g}",
        f"标准误 SE = s/√n = {se:.6g}",
        f"统计量 t = (样本均值 - μ0)/SE = ({xbar:.6g} - {mu0:g})/{se:.6g} = {t:.6g}",
        _p_step(alternative, p, "t", f"t 分布，df = {df}"),
    ]
    return TestResult(
        statistic=t,
        pvalue=p,
        method="单样本 t 检验（σ 未知）",
        alternative=alternative,
        alpha=alpha,
        h0=f"μ = {mu0:g}",
        h1=h1,
        params={"n": n, "df": df, "mean": xbar, "sd": s, "se": se},
        steps=steps,
    )


@register("ttest_2samp_ind")
def ttest_2samp_ind(x1, x2, equal_var=True, alternative="two-sided", alpha=0.05):
    """独立双样本 t 检验：H0: μ1 = μ2。

    equal_var=True  用合并方差（教材经典公式，df = n1 + n2 - 2）；
    equal_var=False 用 Welch 近似（两总体方差不等时更稳妥）。
    """
    alternative = check_alternative(alternative)
    alpha = check_alpha(alpha)
    x1 = as_sample(x1, "x1")
    x2 = as_sample(x2, "x2")
    if x1.size < 2 or x2.size < 2:
        raise ValueError("t 检验需要每个样本至少 2 个观测值")

    n1, n2 = x1.size, x2.size
    m1, m2 = float(np.mean(x1)), float(np.mean(x2))
    v1, v2 = float(np.var(x1, ddof=1)), float(np.var(x2, ddof=1))

    if equal_var:
        df = n1 + n2 - 2
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df # 合并方差 sp²
        se = np.sqrt(sp2 * (1 / n1 + 1 / n2))
        method = "独立双样本 t 检验（合并方差）"
        var_steps = [
            f"合并方差 sp² = [(n1-1)·s1² + (n2-1)·s2²]/(n1+n2-2) = {sp2:.6g}",
            f"标准误 SE = sp·√(1/n1 + 1/n2) = {se:.6g}",
            f"自由度 df = n1 + n2 - 2 = {df}",
        ]
    else:
        se = np.sqrt(v1 / n1 + v2 / n2)
        # Welch–Satterthwaite 自由度近似
        df = float(se**4 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)))
        method = "独立双样本 t 检验（Welch，方差不等）"
        var_steps = [
            f"标准误 SE = √(s1²/n1 + s2²/n2) = {se:.6g}",
            f"Welch 自由度 df = SE^4 / [ (s1²/n1)²/(n1-1) + (s2²/n2)²/(n2-1) ] = {df:.6g}",
        ]

    t = (m1 - m2) / se
    p = distributions.p_value(t, alternative, dist="t", df=df)

    steps = [
        f"样本 1：n1 = {n1}，均值 = {m1:.6g}，方差 s1² = {v1:.6g}",
        f"样本 2：n2 = {n2}，均值 = {m2:.6g}，方差 s2² = {v2:.6g}",
        *var_steps,
        f"统计量 t = (均值1 - 均值2)/SE = ({m1:.6g} - {m2:.6g})/{se:.6g} = {t:.6g}",
        _p_step(alternative, p, "t", f"t 分布，df = {df:.6g}"),
    ]
    return TestResult(
        statistic=t,
        pvalue=p,
        method=method,
        alternative=alternative,
        alpha=alpha,
        h0="μ1 = μ2",
        h1=_H1_TWO[alternative],
        params={"n1": n1, "n2": n2, "df": df, "mean1": m1, "mean2": m2, "se": float(se)},
        steps=steps,
    )
