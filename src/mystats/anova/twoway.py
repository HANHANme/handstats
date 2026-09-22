"""双因素方差分析：无重复试验（r×c 表）与等重复试验（r×c×m）。

两种设计（教材标准划分）：
--------
无重复（每格 1 个观测）：A、B 可加（无交互）假设下，误差 = 交互的混入，
                          df_E = (r-1)(c-1)，无法单独检验交互；
等重复（每格 m ≥ 2 个观测）：交互作用 A×B 可从误差中分离出来单独检验，
                          df_E = rc(m-1)。

返回 TwoWayResult（复合结果）：因素 A / B /（有重复时）交互 A×B 各带一个
TestResult，可独立 print；summary() 汇总各因素结论，show_steps() 打印
完整的方差分析表。这是包里第一个“复合结果对象”——因为双因素天然有
两个（或三个）结论，塞进单个 TestResult 会丢信息。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mystats import distributions
from mystats._registry import register
from mystats.base import TestResult, _print_handcheck
from mystats.anova.oneway import _anova_table
from mystats.validate import check_alpha


@dataclass
class TwoWayResult:
    """双因素方差分析的复合结果：每个因素一个 TestResult。

    factor_a / factor_b / factor_ab（交互；无重复设计时为 None）
    各自是完整的 TestResult（可 print / show_steps）；
    summary() 汇总各因素结论，show_steps() 打印共享的方差分析表。
    """

    method: str = ""
    factor_a: TestResult | None = None
    factor_b: TestResult | None = None
    factor_ab: TestResult | None = None
    params: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"方法：{self.method}"] if self.method else []
        for label, fac in (
            ("因素 A", self.factor_a),
            ("因素 B", self.factor_b),
            ("交互 A×B", self.factor_ab),
        ):
            if fac is None:
                continue
            lines.append(f"— {label} —")
            lines.append(fac.conclusion())
        return "\n".join(lines)

    def show_steps(self) -> None:
        _print_handcheck(self.method, self.steps)

    def __str__(self) -> str:
        return self.summary()


def _factor_test(F, p, dfn, dfd, method, h0, h1, alpha, ms, ms_e, ss, df):
    """构造单个因素的 TestResult（步骤含 F 的完整算式）。"""
    steps = [
        f"MS = SS/df = {ss:.6g}/{df} = {ms:.6g}，MS_E = {ms_e:.6g}",
        f"F = MS/MS_E = {ms:.6g}/{ms_e:.6g} = {F:.6g}，"
        f"p = P(F ≥ {F:.6g}) = {p:.6g}（F 分布，dfn = {dfn}，dfd = {dfd}）",
    ]
    return TestResult(
        statistic=F,
        pvalue=p,
        method=method,
        alternative="greater",
        alpha=alpha,
        h0=h0,
        h1=h1,
        params={"dfn": dfn, "dfd": dfd, "ms": ms, "ss": ss},
        steps=steps,
    )


@register("anova_twoway")
def anova_twoway(data, alpha=0.05, names=None):
    """双因素方差分析（完全设计，等重复）。

    参数
    ----
    data : 二维 (r, c) —— 无重复试验（每格 1 个观测，不能检验交互）；
           三维 (r, c, m) —— 等重复试验（每格 m ≥ 2 个观测，可检验交互）
           行 = 因素 A 的 r 个水平，列 = 因素 B 的 c 个水平
    names : 因素名称对，如 ("温度", "催化剂")；缺省 ("因素A", "因素B")
    alpha : 显著性水平

    前提：数据近似正态、方差齐、每格重复数相同（非平衡设计超出本函数范围）。
    """
    alpha = check_alpha(alpha)
    arr = np.asarray(data, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("data 中含有 NaN 或 Inf")
    if names is None:
        names = ("因素A", "因素B")
    else:
        if len(names) != 2:
            raise ValueError("names 必须是 (因素A名, 因素B名) 两个元素的序列")
        names = (str(names[0]), str(names[1]))

    if arr.ndim == 2:
        if min(arr.shape) < 2:
            raise ValueError(f"无重复双因素至少需要 2×2 的表，收到形状 {arr.shape}")
        return _no_replication(arr, alpha, names)
    if arr.ndim == 3:
        r, c, m = arr.shape
        if min(r, c) < 2 or m < 2:
            raise ValueError(
                "等重复双因素需要 r≥2、c≥2、每格重复 m≥2；"
                "每格只有 1 个观测请传二维表（此时无法检验交互）"
            )
        return _with_replication(arr, alpha, names)
    raise ValueError(f"data 必须是二维或三维数组，实际 ndim = {arr.ndim}")


def _no_replication(tab, alpha, names):
    """无重复双因素：行 = 因素 A，列 = 因素 B，每格 1 个观测。"""
    na, nb = names
    r, c = tab.shape
    n = r * c
    row = tab.mean(axis=1) # A 各水平（行）均值
    col = tab.mean(axis=0) # B 各水平（列）均值
    grand = float(tab.mean())
    ss_a = float(c * np.sum((row - grand) ** 2))
    ss_b = float(r * np.sum((col - grand) ** 2))
    ss_e = float(np.sum((tab - row[:, None] - col[None, :] + grand) ** 2))
    if ss_e <= 0:
        raise ValueError("误差平方和为 0（数据完全可加），F 无定义")
    ss_t = ss_a + ss_b + ss_e
    df_a, df_b, df_e = r - 1, c - 1, (r - 1) * (c - 1)
    ms_a, ms_b, ms_e = ss_a / df_a, ss_b / df_b, ss_e / df_e
    F_a, F_b = ms_a / ms_e, ms_b / ms_e
    p_a = distributions.f_sf(F_a, df_a, df_e)
    p_b = distributions.f_sf(F_b, df_b, df_e)

    row_str = ", ".join(f"{v:.6g}" for v in row)
    col_str = ", ".join(f"{v:.6g}" for v in col)
    table = _anova_table([
        ("因素A", ss_a, df_a, ms_a, F_a, p_a),
        ("因素B", ss_b, df_b, ms_b, F_b, p_b),
        ("误差E", ss_e, df_e, ms_e, None, None),
        ("总和T", ss_t, n - 1, None, None, None),
    ])
    steps = [
        f"{na} {r} 个水平 × {nb} {c} 个水平，每格 1 个观测，总 n = {n}",
        f"行均值（{na}）= ({row_str})，列均值（{nb}）= ({col_str})，"
        f"总均值 = {grand:.6g}",
        f"SS_A = c·Σ(行均值 - 总均值)² = {ss_a:.6g}，"
        f"SS_B = r·Σ(列均值 - 总均值)² = {ss_b:.6g}",
        f"SS_E = ΣΣ(y_ij - 行均值_i - 列均值_j + 总均值)² = {ss_e:.6g}，"
        f"df_E = (r-1)(c-1) = {df_e}（无重复时误差 = 交互的混入，无法单独检验交互）",
        "方差分析表：", *table,
    ]
    factor_a = _factor_test(
        F_a, p_a, df_a, df_e, f"双因素方差分析 · {na}",
        f"{na} 的 {r} 个水平均值全相等（水平间无差异）",
        f"{na} 的水平均值不全相等", alpha, ms_a, ms_e, ss_a, df_a)
    factor_b = _factor_test(
        F_b, p_b, df_b, df_e, f"双因素方差分析 · {nb}",
        f"{nb} 的 {c} 个水平均值全相等（水平间无差异）",
        f"{nb} 的水平均值不全相等", alpha, ms_b, ms_e, ss_b, df_b)
    return TwoWayResult(
        method=f"双因素方差分析（无重复，{na}×{nb}）",
        factor_a=factor_a,
        factor_b=factor_b,
        params={"r": r, "c": c, "m": 1, "n": n, "row_means": row,
                "col_means": col, "grand": grand, "ss_a": ss_a, "ss_b": ss_b,
                "ss_e": ss_e, "ss_t": ss_t, "df_a": df_a, "df_b": df_b,
                "df_e": df_e, "ms_a": ms_a, "ms_b": ms_b, "ms_e": ms_e,
                "anova_table": table},
        steps=steps,
    )


def _with_replication(arr, alpha, names):
    """等重复双因素：形状 (r, c, m)，可检验交互作用 A×B。"""
    na, nb = names
    r, c, m = arr.shape
    n = r * c * m
    cell = arr.mean(axis=2) # 各格均值（每格 m 个观测）
    row = cell.mean(axis=1) # A 各水平均值（对格均值再平均）
    col = cell.mean(axis=0) # B 各水平均值
    grand = float(arr.mean())
    ss_a = float(c * m * np.sum((row - grand) ** 2))
    ss_b = float(r * m * np.sum((col - grand) ** 2))
    ss_ab = float(m * np.sum((cell - row[:, None] - col[None, :] + grand) ** 2))
    ss_e = float(np.sum((arr - cell[:, :, None]) ** 2))
    if ss_e <= 0:
        raise ValueError("误差平方和为 0（每格内部数据完全相同），F 无定义")
    ss_t = ss_a + ss_b + ss_ab + ss_e
    df_a, df_b, df_ab = r - 1, c - 1, (r - 1) * (c - 1)
    df_e = r * c * (m - 1)
    ms_a, ms_b, ms_ab, ms_e = (
        ss_a / df_a, ss_b / df_b, ss_ab / df_ab, ss_e / df_e)
    F_a, F_b, F_ab = ms_a / ms_e, ms_b / ms_e, ms_ab / ms_e
    p_a = distributions.f_sf(F_a, df_a, df_e)
    p_b = distributions.f_sf(F_b, df_b, df_e)
    p_ab = distributions.f_sf(F_ab, df_ab, df_e)

    row_str = ", ".join(f"{v:.6g}" for v in row)
    col_str = ", ".join(f"{v:.6g}" for v in col)
    table = _anova_table([
        ("因素A", ss_a, df_a, ms_a, F_a, p_a),
        ("因素B", ss_b, df_b, ms_b, F_b, p_b),
        ("A×B", ss_ab, df_ab, ms_ab, F_ab, p_ab),
        ("误差E", ss_e, df_e, ms_e, None, None),
        ("总和T", ss_t, n - 1, None, None, None),
    ])
    steps = [
        f"{na} {r} 个水平 × {nb} {c} 个水平，每格重复 m = {m}，总 n = {n}",
        f"行均值（{na}）= ({row_str})，列均值（{nb}）= ({col_str})，"
        f"总均值 = {grand:.6g}",
        f"SS_A = cm·Σ(行均值 - 总均值)² = {ss_a:.6g}，"
        f"SS_B = rm·Σ(列均值 - 总均值)² = {ss_b:.6g}",
        f"SS_A×B = m·ΣΣ(格均值 - 行均值 - 列均值 + 总均值)² = {ss_ab:.6g}"
        "（交互 = 偏离可加性的部分）",
        f"SS_E = ΣΣΣ(y_ijk - 格均值)² = {ss_e:.6g}，"
        f"df_E = rc(m-1) = {df_e}",
        "方差分析表：", *table,
    ]
    factor_a = _factor_test(
        F_a, p_a, df_a, df_e, f"双因素方差分析 · {na}",
        f"{na} 的 {r} 个水平均值全相等（水平间无差异）",
        f"{na} 的水平均值不全相等", alpha, ms_a, ms_e, ss_a, df_a)
    factor_b = _factor_test(
        F_b, p_b, df_b, df_e, f"双因素方差分析 · {nb}",
        f"{nb} 的 {c} 个水平均值全相等（水平间无差异）",
        f"{nb} 的水平均值不全相等", alpha, ms_b, ms_e, ss_b, df_b)
    factor_ab = _factor_test(
        F_ab, p_ab, df_ab, df_e, f"双因素方差分析 · {na}×{nb} 交互",
        f"{na} 与 {nb} 无交互作用（可加模型成立）",
        f"{na} 与 {nb} 存在交互作用", alpha, ms_ab, ms_e, ss_ab, df_ab)
    return TwoWayResult(
        method=f"双因素方差分析（等重复，{na}×{nb}，每格 {m} 次）",
        factor_a=factor_a,
        factor_b=factor_b,
        factor_ab=factor_ab,
        params={"r": r, "c": c, "m": m, "n": n, "cell_means": cell,
                "row_means": row, "col_means": col, "grand": grand,
                "ss_a": ss_a, "ss_b": ss_b, "ss_ab": ss_ab, "ss_e": ss_e,
                "ss_t": ss_t, "df_a": df_a, "df_b": df_b, "df_ab": df_ab,
                "df_e": df_e, "ms_a": ms_a, "ms_b": ms_b, "ms_ab": ms_ab,
                "ms_e": ms_e, "anova_table": table},
        steps=steps,
    )
