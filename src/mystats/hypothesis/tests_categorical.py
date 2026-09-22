"""分类数据的假设检验：卡方拟合优度、卡方独立性（列联表）。

沿用黄金模板四步：① 校验 → ② 计算 → ③ p 值 → ④ 打包 + steps。
卡方类检验的统计量天然“越大越异常”，p 值一律取右尾 P(χ² ≥ 统计量)。
"""
from __future__ import annotations

import numpy as np

from mystats import distributions
from mystats._registry import register
from mystats.base import TestResult
from mystats.validate import as_counts, as_sample, as_table, check_alpha


@register("chisquare_gof")
def chisquare_gof(obs, expected=None, n_params=0, alpha=0.05):
    """卡方拟合优度检验：H0: 总体服从给定的分布。

    参数
    ----
    obs : 各类别的观测频数（一维非负）
    expected : 各类别的理论概率（和为 1）或理论频数（和为 n），
               缺省 = 均匀分布（每类概率 1/k）
    n_params : 分布中由样本估计的参数个数（如用样本均值代替 λ 则为 1），
               自由度按 df = k - 1 - n_params 扣减
    alpha : 显著性水平
    """
    alpha = check_alpha(alpha)
    obs = as_counts(obs, "obs")
    k = obs.size
    n = int(obs.sum())

    if expected is None:
        probs = np.full(k, 1.0 / k)
        prob_desc = "缺省 = 均匀分布（每类概率 1/k）"
    else:
        probs = as_sample(expected, "expected")
        if probs.size != k:
            raise ValueError(
                f"expected 长度必须与 obs 一致：obs 有 {k} 类，expected 有 {probs.size} 类"
            )
        total = float(probs.sum())
        if total <= 0:
            raise ValueError("expected 的总和必须为正")
        if not np.isclose(total, 1.0, rtol=0.0, atol=1e-8):
            probs = probs / total # 概率、频数两种给法都自动归一化
        prob_desc = "给定分布"

    n_params = float(n_params)
    if n_params != int(n_params) or n_params < 0:
        raise ValueError("n_params 必须是非负整数")
    n_params = int(n_params)
    df = k - 1 - n_params
    if df < 1:
        raise ValueError(f"自由度 df = k - 1 - n_params = {df} < 1：类别太少或估计参数太多")

    exp_counts = n * probs
    chi2 = float(np.sum((obs - exp_counts) ** 2 / exp_counts))
    p = distributions.p_value(chi2, "greater", dist="chi2", df=df)

    steps = [
        f"类别数 k = {k}，总频数 n = {n}（{prob_desc}）",
        f"理论频数 E_i = n × p_i：{[round(float(v), 4) for v in exp_counts]}",
        f"统计量 χ² = Σ(O_i - E_i)²/E_i = {chi2:.6g}",
        f"自由度 df = k - 1 - {n_params} = {df}",
        f"p = P(χ² ≥ {chi2:.6g}) = {p:.6g}（卡方分布，df = {df}）",
    ]
    min_exp = float(exp_counts.min())
    if min_exp < 5:
        steps.append(
            f"注意：最小理论频数 = {min_exp:.4g} < 5，卡方近似可能不可靠（教材经验规则）"
        )

    return TestResult(
        statistic=chi2,
        pvalue=p,
        method="卡方拟合优度检验",
        alternative="greater", # χ² 越大越异常，天然右尾
        alpha=alpha,
        h0="总体服从给定分布",
        h1="总体不服从给定分布",
        params={"k": k, "n": n, "df": df, "min_expected": min_exp},
        steps=steps,
    )


@register("chisquare_ind")
def chisquare_ind(table, alpha=0.05):
    """卡方独立性检验：H0: 行、列两个分类变量相互独立。

    参数
    ----
    table : r×c 列联表（二维频数，r、c ≥ 2）
    alpha : 显著性水平

    注：采用教材公式，不做 Yates 连续性校正——2×2 表与 scipy
    chi2_contingency 的默认结果会不同（对拍时需传 correction=False）。
    """
    alpha = check_alpha(alpha)
    tab = as_table(table, "table")
    r, c = tab.shape
    n = float(tab.sum())
    row_totals = tab.sum(axis=1, keepdims=True)
    col_totals = tab.sum(axis=0, keepdims=True)
    expected = row_totals @ col_totals / n # 期望频数矩阵 E_ij
    chi2 = float(np.sum((tab - expected) ** 2 / expected))
    df = (r - 1) * (c - 1)
    p = distributions.p_value(chi2, "greater", dist="chi2", df=df)

    exp_str = "; ".join(" ".join(f"{v:.4g}" for v in row) for row in expected)
    steps = [
        f"列联表 {r}×{c}，总频数 n = {n:g}",
        f"期望频数 E_ij = 行合计 × 列合计 / n，矩阵 = [{exp_str}]",
        f"统计量 χ² = Σ(O_ij - E_ij)²/E_ij = {chi2:.6g}",
        f"自由度 df = (r-1)(c-1) = ({r}-1)×({c}-1) = {df}",
        f"p = P(χ² ≥ {chi2:.6g}) = {p:.6g}（卡方分布，df = {df}）",
    ]
    min_exp = float(expected.min())
    if min_exp < 5:
        steps.append(
            f"注意：最小期望频数 = {min_exp:.4g} < 5，卡方近似可能不可靠（教材经验规则）"
        )

    return TestResult(
        statistic=chi2,
        pvalue=p,
        method="卡方独立性检验",
        alternative="greater",
        alpha=alpha,
        h0="两个分类变量相互独立",
        h1="两个分类变量不独立（有关联）",
        params={
            "r": r,
            "c": c,
            "n": int(n),
            "df": df,
            "cramers_v": float(np.sqrt(chi2 / (n * (min(r, c) - 1)))), # 效应量
            "expected": expected,
            "min_expected": min_exp,
        },
        steps=steps,
    )
