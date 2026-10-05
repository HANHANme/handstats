"""外壳的数据输入解析：把粘贴的文本变成包函数要的参数。

设计原则：
- 分隔符宽容——空格、逗号（中英文）、分号、顿号、换行都认，
  这样从 Excel 复制过来的数据能直接粘贴；
- 报错信息面向小白（指明哪一行、哪个词不是数字）；
- 不依赖 streamlit，可以被 pytest 直接测试（specs/parsing 与 UI 分离
  是外壳的可测试性设计）。
"""
from __future__ import annotations

import re

import numpy as np

# 空白 / 中英文逗号 / 中英文分号 / 顿号 都当分隔符
_SPLIT = re.compile(r"[\s,，;；、]+")


def _to_floats(line: str, where: str = "") -> list[float]:
    tokens = [t for t in _SPLIT.split(line.strip()) if t]
    out = []
    for t in tokens:
        try:
            out.append(float(t))
        except ValueError:
            raise ValueError(f"{where}无法识别为数字：{t!r}")
    return out


def parse_sample(text, label: str = "样本数据") -> list[float]:
    """一维样本：整段文本全部当数字解析。"""
    if text is None or not str(text).strip():
        raise ValueError(f"请先粘贴{label}（可从 Excel 复制一列）")
    return _to_floats(str(text), f"{label}中")


def parse_counts(text, label: str = "频数") -> list[float]:
    """一维频数（非负性由包内 as_counts 校验并给出中文报错）。"""
    return parse_sample(text, label)


def parse_groups(text, label: str = "分组数据") -> list[list[float]]:
    """每组一行：ANOVA / Tukey / Levene 的输入格式。"""
    groups = []
    for line in str(text or "").splitlines():
        vals = _to_floats(line)
        if vals:
            groups.append(vals)
    if len(groups) < 2:
        raise ValueError(f"{label}至少需要 2 行（每行一组的观测值）")
    return groups


def parse_table2d(text, label: str = "表格数据", min_rows: int = 2) -> np.ndarray:
    """二维表：每行一个表格行（列联表、双因素无重复表）。"""
    rows = []
    for i, line in enumerate(str(text or "").splitlines(), 1):
        vals = _to_floats(line, f"{label}第 {i} 行")
        if vals:
            rows.append(vals)
    if len(rows) < min_rows:
        raise ValueError(f"{label}至少需要 {min_rows} 行")
    width = len(rows[0])
    for i, r in enumerate(rows, 1):
        if len(r) != width:
            raise ValueError(
                f"{label}每行的数值个数必须相同：第 1 行有 {width} 个，"
                f"第 {i} 行有 {len(r)} 个"
            )
    return np.array(rows, dtype=float)


def parse_xycols(text, label: str = "观测数据"):
    """每行一个观测、两列（自变量 因变量）：一元回归的输入格式。"""
    tab = parse_table2d(text, label, min_rows=3)
    if tab.shape[1] != 2:
        raise ValueError(f"{label}必须是两列（自变量x 因变量y），当前有 {tab.shape[1]} 列")
    return tab[:, 0], tab[:, 1]


def parse_xytable(text, label: str = "观测数据"):
    """每行一个观测：最后一列是因变量 y，其余列全是自变量。返回 (X, y)。

    与 parse_xycols 的差别：允许任意列数（≥2）——一元回归的两列数据
    天然兼容，多元回归把各自变量列依次排在前即可。
    """
    tab = parse_table2d(text, label, min_rows=3)
    if tab.shape[1] < 2:
        raise ValueError(f"{label}至少需要两列：前面的列是自变量，最后一列是因变量 y")
    return tab[:, :-1], tab[:, -1]


# kind → 解析器 + 文本框的提示文案（app.py 渲染用）
PARSERS = {
    "sample": {
        "fn": parse_sample,
        "placeholder": "5.1  4.9  5.3, 5.0\n4.8  5.2  5.05 ...",
        "help": "空格、逗号或换行分隔；可直接从 Excel 复制一列粘贴",
    },
    "counts": {
        "fn": parse_counts,
        "placeholder": "8  13  9  12  7  11",
        "help": "各类别的个数，如掷骰子 60 次各面出现的次数",
    },
    "groups": {
        "fn": parse_groups,
        "placeholder": "75 80 68 72 85\n70 78 74 69 81\n82 79 88 76 90",
        "help": "每组一行（示例为 3 组，每组 5 个观测）",
    },
    "table2d": {
        "fn": parse_table2d,
        "placeholder": "45 55\n30 70",
        "help": "每行一个表格行（示例为 2×2 列联表）",
    },
    "xycols": {
        "fn": parse_xycols,
        "placeholder": "1.2  5.3\n2.0  6.1\n2.8  7.0 ...",
        "help": "每行一个观测，两列：自变量x 因变量y；可从 Excel 复制两列粘贴",
    },
    "xytable": {
        "fn": parse_xytable,
        "placeholder": "x1   x2   y\n1.2  3.5  5.3\n2.0  4.1  6.1 ...",
        "help": "每行一个观测，最后一列是因变量 y，其余列是自变量；可从 Excel 复制多列粘贴",
    },
}
