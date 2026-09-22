"""区间估计：两总体均值差 μ1 - μ2 的置信区间（独立样本 / 配对样本）。

与 hypothesis/tests_mean.py 里的双样本 t 检验共享同一套枢轴量：
区间 = 点估计 ± 临界值 × 标准误，标准误的三种给法（σ 已知 / 合并 / Welch）
与检验完全一致——这正是“检验与区间是一枚硬币两面”的代码体现。
"""
from __future__ import annotations

import numpy as np

from mystats import distributions
from mystats._registry import register
from mystats.base import IntervalResult
from mystats.validate import as_sample, check_confidence, check_positive


@register("ci_mean_2samp")
def ci_mean_2samp(x1, x2, confidence=0.95, sigma1=None, sigma2=None, equal_var=True):
    """两独立样本均值差 μ1 - μ2 的置信区间。

    参数
    ----
    sigma1, sigma2 : 两总体标准差都已知时给出 → z 区间（SE 含 σ）
    都缺省（最常用） → t 区间；equal_var=True 用合并方差
    （df = n1 + n2 - 2，教材经典公式），False 用 Welch 近似（方差不等时）
    """
    confidence = check_confidence(confidence)
    x1 = as_sample(x1, "x1")
    x2 = as_sample(x2, "x2")
    if x1.size < 2 or x2.size < 2:
        raise ValueError("每个样本至少需要 2 个观测值")
    if (sigma1 is None) != (sigma2 is None):
        raise ValueError("sigma1 与 sigma2 必须同时给出或同时缺省")

    n1, n2 = x1.size, x2.size
    m1, m2 = float(np.mean(x1)), float(np.mean(x2))
    diff = m1 - m2
    q = 1 - (1 - confidence) / 2 # 上侧分位点，如 95% → 0.975

    if sigma1 is not None:
        sigma1 = check_positive(sigma1, "sigma1")
        sigma2 = check_positive(sigma2, "sigma2")
        se = float(np.sqrt(sigma1**2 / n1 + sigma2**2 / n2))
        crit = distributions.norm_ppf(q)
        method = "z 区间（两 σ 已知）"
        params = {"n1": n1, "n2": n2, "mean1": m1, "mean2": m2,
                  "se": se, "critical_value": crit}
        head_steps = [
            f"样本 1：n1 = {n1}，均值 = {m1:.6g}（σ1 = {sigma1:g} 已知）",
            f"样本 2：n2 = {n2}，均值 = {m2:.6g}（σ2 = {sigma2:g} 已知）",
        ]
        se_steps = [
            f"标准误 SE = √(σ1²/n1 + σ2²/n2) = {se:.6g}",
            f"临界值 z(α/2) = 标准正态的 {q:g} 分位数 = {crit:.6g}",
        ]
    else:
        v1 = float(np.var(x1, ddof=1))
        v2 = float(np.var(x2, ddof=1))
        head_steps = [
            f"样本 1：n1 = {n1}，均值 = {m1:.6g}，方差 s1² = {v1:.6g}",
            f"样本 2：n2 = {n2}，均值 = {m2:.6g}，方差 s2² = {v2:.6g}",
        ]
        if equal_var:
            df = n1 + n2 - 2
            sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
            se = float(np.sqrt(sp2 * (1 / n1 + 1 / n2)))
            crit = distributions.t_ppf(q, df)
            method = "t 区间（合并方差）"
            se_steps = [
                f"合并方差 sp² = [(n1-1)·s1² + (n2-1)·s2²]/(n1+n2-2) = {sp2:.6g}",
                f"标准误 SE = sp·√(1/n1 + 1/n2) = {se:.6g}",
                f"自由度 df = n1 + n2 - 2 = {df}",
                f"临界值 t(α/2, df) = {crit:.6g}",
            ]
            params = {"n1": n1, "n2": n2, "mean1": m1, "mean2": m2,
                      "df": df, "se": se, "critical_value": crit}
        else:
            se = float(np.sqrt(v1 / n1 + v2 / n2))
            df = float(se**4 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)))
            crit = distributions.t_ppf(q, df)
            method = "t 区间（Welch）"
            se_steps = [
                f"标准误 SE = √(s1²/n1 + s2²/n2) = {se:.6g}",
                f"Welch 自由度 df = SE^4 / [(s1²/n1)²/(n1-1) + (s2²/n2)²/(n2-1)] = {df:.6g}",
                f"临界值 t(α/2, df) = {crit:.6g}",
            ]
            params = {"n1": n1, "n2": n2, "mean1": m1, "mean2": m2,
                      "df": df, "se": se, "critical_value": crit}

    half = crit * se
    steps = [
        *head_steps,
        f"均值差 = 均值1 - 均值2 = {diff:.6g}",
        *se_steps,
        f"{confidence:.0%} 置信区间 = 均值差 ± 临界值·SE = {diff:.6g} ± {half:.6g}",
    ]
    return IntervalResult(
        estimate=diff,
        lower=diff - half,
        upper=diff + half,
        confidence=confidence,
        method=method,
        params=params,
        steps=steps,
    )


@register("ci_paired_diff")
def ci_paired_diff(x1, x2, confidence=0.95):
    """配对样本均值差 μd 的置信区间（t 区间）。

    先逐对求差 d_i = x1_i - x2_i，再对差值做单样本 t 区间。
    配对设计消除了个体间差异（如同一批人培训前后对照），
    要求两样本一一对应、长度相同。
    """
    confidence = check_confidence(confidence)
    x1 = as_sample(x1, "x1")
    x2 = as_sample(x2, "x2")
    if x1.size != x2.size:
        raise ValueError(f"配对样本长度必须相同：x1 有 {x1.size} 个，x2 有 {x2.size} 个")
    if x1.size < 2:
        raise ValueError("配对样本至少需要 2 对")

    d = x1 - x2
    n = d.size
    df = n - 1
    dbar = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    se = float(sd / np.sqrt(n))
    q = 1 - (1 - confidence) / 2
    crit = distributions.t_ppf(q, df)
    half = crit * se

    steps = [
        f"逐对求差 d_i = x1_i - x2_i，对数 n = {n}，差值均值 = {dbar:.6g}",
        f"差值标准差 s_d = {sd:.6g}，标准误 SE = s_d/√n = {se:.6g}",
        f"自由度 df = n - 1 = {df}，临界值 t(α/2, df) = {crit:.6g}",
        f"{confidence:.0%} 置信区间 = 差值均值 ± t·SE = {dbar:.6g} ± {half:.6g}",
    ]
    return IntervalResult(
        estimate=dbar,
        lower=dbar - half,
        upper=dbar + half,
        confidence=confidence,
        method="配对差值 t 区间",
        params={"n": n, "df": df, "mean_diff": dbar, "sd": sd, "se": se},
        steps=steps,
    )
