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
