"""假设检验子包。

已有：
- 均值：单样本 z / t、独立双样本 t（合并方差 / Welch）
- 分类数据：卡方拟合优度、卡方独立性
- 方差：双样本 F 检验（方差齐性）
- 比例：单比例 / 双比例 z 检验（大样本正态近似）

规划：似然比检验（GLRT，配合课程进度）；精确二项检验、秩检验等
归入 nonparametric 子包。
"""
from mystats.hypothesis.tests_categorical import chisquare_gof, chisquare_ind
from mystats.hypothesis.tests_mean import ttest_1samp, ttest_2samp_ind, ztest_1samp
from mystats.hypothesis.tests_proportion import ztest_1prop, ztest_2prop
from mystats.hypothesis.tests_variance import ftest_2samp_var

__all__ = [
    "ztest_1samp",
    "ttest_1samp",
    "ttest_2samp_ind",
    "chisquare_gof",
    "chisquare_ind",
    "ftest_2samp_var",
    "ztest_1prop",
    "ztest_2prop",
]
