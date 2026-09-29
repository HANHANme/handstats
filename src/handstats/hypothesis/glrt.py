"""广义似然比检验（GLRT）：通用引擎 + 两个可解析对照的实例。

思想（教材标准框架）
--------
H0: θ ∈ Θ0 vs H1: θ ∈ Θ1 = Θ \\ Θ0。定义似然比
    λ(x) = max_{θ∈Θ0} L(θ; x) / max_{θ∈Θ} L(θ; x) ∈ (0, 1]
λ 越小，说明数据在“无约束”下比在“H0 约束”下出现得多，越应拒绝 H0。
精确分布一般难求，用 Wilks 定理：H0 成立且 n 大时，
    -2 ln λ ≈ χ²( dim(Θ) - dim(Θ0) )，自由度 = 参数空间维数之差。

本模块三层：
1. glrt_test             通用引擎：任何模型，给出两个最大对数似然即可检验；
2. glrt_normal_mean      正态均值（可解析）：验证 GLRT 与 z / t 检验的精确关系；
3. glrt_exponential_mean 指数均值（可解析）：对照精确枢轴与 Wilks 渐近。
"""
from __future__ import annotations

import math

import numpy as np

from handstats import distributions
from handstats._registry import register
from handstats.base import TestResult
from handstats.validate import as_sample, check_alpha, check_positive


@register("glrt_test")
def glrt_test(loglik_full, loglik_null, df, alpha=0.05):
    """GLRT 通用引擎（Wilks 渐近版）。

    参数
    ----
    loglik_full : 对数似然在无约束 MLE 处的最大值（手算或数值优化均可；
                  与参数无关的常数项在两个对数似然相减时抵消，可省略）
    loglik_null : 对数似然在 Θ0 约束下的最大值
    df : Wilks 自由度 = dim(Θ) - dim(Θ0)
    alpha : 显著性水平

    适用前提：正则条件 + 大样本（经验上 df 小、n ≥ 30 时近似良好；
    想看“渐近 vs 精确”差多少，用本模块两个可解析实例对照）。
    """
    alpha = check_alpha(alpha)
    ll_full, ll_null = float(loglik_full), float(loglik_null)
    diff = ll_full - ll_null
    if diff < -1e-8:
        raise ValueError(
            "约束下的最大对数似然不可能大于无约束的（lnL_H0 > lnL_full），请检查输入"
        )
    df = int(df)
    if df < 1:
        raise ValueError(f"Wilks 自由度必须是正整数，收到 df = {df!r}")
    stat = 2.0 * max(0.0, diff) # 容差内的浮点毛刺截为 0
    lam = math.exp(-stat / 2)
    p = distributions.p_value(stat, "greater", dist="chi2", df=df)

    steps = [
        f"无约束最大对数似然 lnL_full = {ll_full:.6g}（θ 在全空间取最大）",
        f"H0 约束下最大对数似然 lnL_H0 = {ll_null:.6g}（θ 限制在 Θ0 内取最大）",
        f"似然比 λ = exp(lnL_H0 - lnL_full) = {lam:.6g}（0 < λ ≤ 1，越小越拒绝）",
        f"统计量 -2 ln λ = 2(lnL_full - lnL_H0) = {stat:.6g}",
        f"Wilks 定理（大样本）：H0 下 -2 ln λ ≈ χ²({df})，"
        f"p = P(χ² ≥ {stat:.6g}) = {p:.6g}",
    ]
    return TestResult(
        statistic=stat,
        pvalue=p,
        method="广义似然比检验 GLRT（Wilks 渐近）",
        alternative="greater",
        alpha=alpha,
        h0="θ ∈ Θ0（参数受约束）",
        h1="θ 不在 Θ0 内（参数不受约束）",
        params={"lambda": lam, "loglik_full": ll_full,
                "loglik_null": ll_null, "df": df},
        steps=steps,
    )


@register("glrt_normal_mean")
def glrt_normal_mean(x, mu0, sigma=None, alpha=0.05):
    """正态总体均值的 GLRT：H0: μ = μ0（双侧）。

    σ 已知：-2 ln λ = z²，且 z² 精确服从 χ²(1)——Wilks 渐近在这里是精确的；
    σ 未知：-2 ln λ = n·ln(1 + t²/(n-1))，它是 |t| 的单调增函数，
            因此 GLRT 的精确实现就是双侧 t 检验（教材经典结论）。
    结果里同时给出 Wilks 渐近 p 与精确 p，供对比学习。

    注：GLRT 天然是双侧的；单侧检验需要序约束最大化，超出本模块范围。
    """
    alpha = check_alpha(alpha)
    x = as_sample(x)
    n = x.size
    xbar = float(np.mean(x))

    if sigma is not None:
        sigma = check_positive(sigma, "sigma")
        z = (xbar - mu0) / (sigma / np.sqrt(n))
        stat = z * z
        lam = math.exp(-stat / 2)
        p = distributions.p_value(stat, "greater", dist="chi2", df=1)
        p_exact = distributions.p_value(z, "two-sided", dist="norm")
        steps = [
            f"样本量 n = {n}，样本均值 = {xbar:.6g}，已知 σ = {sigma:g}",
            "对数似然（省略与 μ 无关的常数）lnL(μ) = -Σ(xi - μ)²/(2σ²)，"
            "在 μ = 样本均值 处最大（无约束），在 μ = mu0 处取值（H0 约束）",
            f"-2 ln λ = [Σ(xi - mu0)² - Σ(xi - 均值)²]/σ² "
            f"= n(均值 - mu0)²/σ² = z² = {stat:.6g}",
            f"Wilks：-2 ln λ ≈ χ²(1)，p = P(χ² ≥ {stat:.6g}) = {p:.6g}",
            f"本例 z² 严格服从 χ²(1)：渐近 p = 精确 p = {p_exact:.6g}（双侧 z 检验）",
        ]
        return TestResult(
            statistic=stat,
            pvalue=p,
            method="正态均值 GLRT（σ 已知）",
            alternative="greater",
            alpha=alpha,
            h0=f"μ = {mu0:g}",
            h1=f"μ ≠ {mu0:g}",
            params={"n": n, "mean": xbar, "z": z, "z2": stat, "lambda": lam,
                    "df": 1, "exact_p": p_exact},
            steps=steps,
        )

    if n < 2:
        raise ValueError("σ 未知时至少需要 2 个观测值")
    sse = float(np.sum((x - xbar) ** 2))
    if sse == 0:
        raise ValueError("数据全部相同（σ² 的 MLE = 0），GLRT 无定义")
    sse0 = float(np.sum((x - mu0) ** 2))
    df = n - 1
    t_stat = float((xbar - mu0) / np.sqrt(sse / df / n)) # = (均值-μ0)/(s/√n)
    stat = n * math.log(sse0 / sse)
    lam = math.exp(-stat / 2)
    p_wilks = distributions.p_value(stat, "greater", dist="chi2", df=1)
    p_exact = distributions.p_value(t_stat, "two-sided", dist="t", df=df)

    steps = [
        f"样本量 n = {n}，样本均值 = {xbar:.6g}",
        f"无约束最大似然：μ = 样本均值，σ² = (1/n)Σ(xi - 均值)² = {sse / n:.6g}"
        "（MLE 除以 n，不是样本方差的 n-1）",
        f"H0 下最大似然：μ 固定为 {mu0:g}，σ² = (1/n)Σ(xi - mu0)² = {sse0 / n:.6g}",
        f"似然比 λ = (σ²_无约束 / σ²_H0)^(n/2) = {lam:.6g}",
        f"统计量 -2 ln λ = n·ln(σ²_H0 / σ²_无约束) = n·ln(1 + t²/(n-1)) = {stat:.6g}，"
        f"其中 t = (均值 - mu0)/(s/√n) = {t_stat:.6g}",
        f"Wilks 渐近：-2 ln λ ≈ χ²(1)，p = P(χ² ≥ {stat:.6g}) = {p_wilks:.6g}",
        f"精确结论：-2 ln λ 随 |t| 单调增 → GLRT 精确等价于双侧 t 检验，"
        f"精确 p = {p_exact:.6g}",
    ]
    return TestResult(
        statistic=stat,
        pvalue=p_wilks,
        method="正态均值 GLRT（σ 未知）",
        alternative="greater",
        alpha=alpha,
        h0=f"μ = {mu0:g}",
        h1=f"μ ≠ {mu0:g}",
        params={"n": n, "mean": xbar, "t": t_stat, "df": df,
                "sigma2_mle": sse / n, "sigma2_h0": sse0 / n, "lambda": lam,
                "wilks_p": p_wilks, "exact_p": p_exact},
        steps=steps,
    )


@register("glrt_exponential_mean")
def glrt_exponential_mean(x, theta0, alpha=0.05):
    """指数总体均值的 GLRT：H0: E(X) = θ0（密度 f(x) = (1/θ)·e^(-x/θ)）。

    lnL(θ) = -n·ln θ - Σxi/θ，无约束 MLE 是 θ = 样本均值；
    记 u = 均值/θ0：λ = (u·e^(1-u))^n，-2 ln λ = 2n·[u - 1 - ln u] ≥ 0。
    精确对照：H0 下 2Σxi/θ0 ~ χ²(2n)，params 附带等尾精确 p 供对比——
    它与 Wilks 渐近 p 略有差别（卡方偏态所致），正是“渐近 vs 精确”的学习点。
    """
    alpha = check_alpha(alpha)
    x = as_sample(x)
    if np.any(x <= 0):
        raise ValueError("指数分布要求观测值全部为正")
    n = x.size
    xbar = float(np.mean(x))
    theta0 = check_positive(theta0, "theta0")

    u = xbar / theta0
    stat = 2.0 * n * (u - 1 - math.log(u))
    lam = math.exp(-stat / 2)
    p = distributions.p_value(stat, "greater", dist="chi2", df=1)
    pivot = 2.0 * float(x.sum()) / theta0
    p_exact = distributions.p_value(pivot, "two-sided", dist="chi2", df=2 * n)

    steps = [
        f"样本量 n = {n}，样本均值 = {xbar:.6g}，H0 均值 θ0 = {theta0:g}",
        "对数似然 lnL(θ) = -n·ln θ - Σxi/θ → 无约束 MLE：θ = 样本均值",
        f"记 u = 均值/θ0 = {u:.6g}，似然比 λ = (u·e^(1-u))^n = {lam:.6g}",
        f"统计量 -2 ln λ = 2n·[u - 1 - ln u] = {stat:.6g}（u = 1 时取 0，恒 ≥ 0）",
        f"Wilks 渐近：-2 ln λ ≈ χ²(1)，p = P(χ² ≥ {stat:.6g}) = {p:.6g}",
        f"精确对照：H0 下 2Σxi/θ0 = {pivot:.6g} ~ χ²({2 * n})，"
        f"等尾精确 p = {p_exact:.6g}",
    ]
    return TestResult(
        statistic=stat,
        pvalue=p,
        method="指数均值 GLRT（Wilks 渐近）",
        alternative="greater",
        alpha=alpha,
        h0=f"E(X) = θ0 = {theta0:g}",
        h1=f"E(X) ≠ {theta0:g}",
        params={"n": n, "mean": xbar, "u": u, "lambda": lam,
                "exact_p_equal_tail": p_exact},
        steps=steps,
    )
