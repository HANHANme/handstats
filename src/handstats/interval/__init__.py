"""区间估计子包。

已有：
- 均值：单样本 z/t 区间（ci_mean）
- 均值差：独立双样本 z/t 区间（合并/Welch）、配对差值 t 区间
- 方差：σ² 卡方区间（params 附带 σ 区间）
- 比例：Wald 大样本区间

规划（阶段 3+）：回归系数与预测区间；Bootstrap 区间见 resampling 子包。
"""
from handstats.interval.ci_mean import ci_mean
from handstats.interval.ci_mean_2 import ci_mean_2samp, ci_paired_diff
from handstats.interval.ci_prop import ci_proportion
from handstats.interval.ci_var import ci_var

__all__ = ["ci_mean", "ci_mean_2samp", "ci_paired_diff", "ci_var", "ci_proportion"]
