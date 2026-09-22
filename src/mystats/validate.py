"""输入校验：所有模块共用的“第一道关卡”。

集中在一个文件的好处：以后几十个检验函数不用各自重复写
“是不是一维、有没有 NaN”这类检查；报错信息也全局统一，
使用者（未来的你）看到报错就知道是数据问题而不是算法问题。
"""
from __future__ import annotations

import numpy as np

# 备择假设的别名归一化表：英文习惯写法 + 中文写法都收
_ALTERNATIVES = {
    "two-sided": "two-sided",
    "two_sided": "two-sided",
    "2-sided": "two-sided",
    "双侧": "two-sided",
    "双边": "two-sided",
    "less": "less",
    "左侧": "less",
    "左边": "less",
    "greater": "greater",
    "右侧": "greater",
    "右边": "greater",
}


def as_sample(x, name: str = "x") -> np.ndarray:
    """把输入转成一维 float 数组并做基本检查；返回新数组，不改调用方数据。"""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} 必须是一维样本数组，实际维度 ndim = {arr.ndim}")
    if arr.size == 0:
        raise ValueError(f"{name} 不能为空")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 中含有 NaN 或 Inf，请先清洗数据")
    return arr


def check_alpha(alpha: float) -> float:
    """显著性水平 α 必须严格落在 (0, 1) 内。"""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha 必须在 (0, 1) 内，收到 {alpha!r}")
    return float(alpha)


def check_confidence(level: float) -> float:
    """置信水平必须严格落在 (0, 1) 内。"""
    if not 0.0 < level < 1.0:
        raise ValueError(f"confidence 必须在 (0, 1) 内，收到 {level!r}")
    return float(level)


def check_positive(value: float, name: str) -> float:
    """必须是有限的正数（用于 σ、样本量等天然非负的量）。"""
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} 必须是正数，收到 {value!r}")
    return value


def check_alternative(alternative: str) -> str:
    """把各种写法（two-sided / 双侧 / Less ...）归一化成规范三选一。"""
    key = _ALTERNATIVES.get(str(alternative).strip().lower())
    if key is None:
        raise ValueError(
            f"alternative 只能是 two-sided / less / greater（或 双侧/左侧/右侧），"
            f"收到 {alternative!r}"
        )
    return key
