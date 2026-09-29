"""假设检验子包。

已有：
- 均值：单样本 z / t、独立双样本 t（合并方差 / Welch）
- 分类数据：卡方拟合优度、卡方独立性
- 方差：双样本 F 检验（方差齐性）
- 比例：单比例 / 双比例 z 检验（大样本正态近似）
- GLRT：广义似然比检验（通用引擎 + 正态/指数均值可解析实例）

规划：精确二项检验、秩检验等归入 nonparametric 子包。
"""
from handstats.hypothesis.glrt import glrt_exponential_mean, glrt_normal_mean, glrt_test
from handstats.hypothesis.tests_categorical import chisquare_gof, chisquare_ind
from handstats.hypothesis.tests_mean import ttest_1samp, ttest_2samp_ind, ztest_1samp
from handstats.hypothesis.tests_proportion import ztest_1prop, ztest_2prop
from handstats.hypothesis.tests_variance import ftest_2samp_var

__all__ = [
    "ztest_1samp",
    "ttest_1samp",
    "ttest_2samp_ind",
    "chisquare_gof",
    "chisquare_ind",
    "ftest_2samp_var",
    "ztest_1prop",
    "ztest_2prop",
    "glrt_test",
    "glrt_normal_mean",
    "glrt_exponential_mean",
]
