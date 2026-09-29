"""假设检验子包内部共用的小工具（下划线开头 = 仅包内使用）。"""
from __future__ import annotations


def _p_step(alternative: str, p: float, letter: str, dist_desc: str) -> str:
    """steps 里“p 值怎么来的”那一行——三种备择假设的算法不同。

    letter 传统计量的字母记号（如 "z"、"t"）；卡方/F 等需要自定义
    表述的检验可以不经过本函数，直接手写 steps 那一行。
    """
    L = letter.upper()
    if alternative == "two-sided":
        return f"p = 2·P({L} ≥ |{letter}|) = {p:.6g}（{dist_desc}）"
    if alternative == "less":
        return f"p = P({L} ≤ {letter}) = {p:.6g}（{dist_desc}）"
    return f"p = P({L} ≥ {letter}) = {p:.6g}（{dist_desc}）"


# H0/H1 的关系符：单侧检验的原假设是复合假设，教材写法为 ≤/≥
_H0_OP = {"two-sided": "=", "less": "≥", "greater": "≤"}
_H1_OP = {"two-sided": "≠", "less": "<", "greater": ">"}


def _h0_h1(lhs: str, rhs: str, alternative: str) -> tuple[str, str]:
    """按备择假设方向生成教材写法的 H0/H1 文本对。

    双侧：H0: μ = μ0    H1: μ ≠ μ0
    左侧：H0: μ ≥ μ0    H1: μ < μ0   （原假设是复合假设，不是简单的等号）
    右侧：H0: μ ≤ μ0    H1: μ > μ0
    """
    return f"{lhs} {_H0_OP[alternative]} {rhs}", f"{lhs} {_H1_OP[alternative]} {rhs}"
