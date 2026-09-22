"""mystats —— 面向学习的数理统计工具包。

设计理念
--------
1. 每个统计过程都返回“富结果对象”（TestResult / IntervalResult / FitResult），
   而不是一个裸数字：统计量、p 值、结论、中间步骤一次拿全；
2. steps 机制输出【手算核对表】：把教材里的每一步计算（标准误、统计量、
   自由度、p 值来源）逐行打印，方便和考试手算互相核对；
3. 底层数值统一走 distributions.py 一个接口（目前由 scipy 提供支持），
   以后想换成自己实现的数值算法，只改那一个文件。

目录结构（可延展性）
--------
hypothesis/     假设检验（已有 z/t；规划：卡方、F、似然比 GLRT……）
interval/       区间估计（已有均值 z/t 区间）
regression/     回归分析（预留，阶段 3）
anova/          方差分析（预留，阶段 4）
nonparametric/  非参数检验（预留）
resampling/     Bootstrap / 置换检验（预留）
multivariate/   多元统计分析（预留）
"""
from __future__ import annotations

from mystats._registry import PROCEDURES, register
from mystats.base import FitResult, IntervalResult, TestResult
from mystats.hypothesis import ttest_1samp, ttest_2samp_ind, ztest_1samp
from mystats.interval import ci_mean

__version__ = "0.1.0"

__all__ = [
    "TestResult",
    "IntervalResult",
    "FitResult",
    "ztest_1samp",
    "ttest_1samp",
    "ttest_2samp_ind",
    "ci_mean",
    "register",
    "list_procedures",
]


def list_procedures() -> dict:
    """返回 {过程名: 函数}——当前已注册的所有统计过程一览。

    将来做命令行界面或图形界面时，直接遍历这个注册表就能自动生成
    “可用方法”菜单，不需要改动任何已有代码。
    """
    return dict(PROCEDURES)
