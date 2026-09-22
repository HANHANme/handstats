"""假设检验子包。

已有：单样本 z / t 检验、独立双样本 t 检验。
规划（阶段 1）：卡方拟合优度 / 独立性检验、F 方差齐性检验、
比例检验（单比例 / 双比例）、以及结合课程进度加入似然比检验（GLRT）。
"""
from mystats.hypothesis.tests_mean import ttest_1samp, ttest_2samp_ind, ztest_1samp

__all__ = ["ztest_1samp", "ttest_1samp", "ttest_2samp_ind"]
