"""区间估计子包。

已有：均值的置信区间（σ 已知 z 区间 / σ 未知 t 区间）。
规划（阶段 2）：两样本均值差、比例、方差、回归系数的置信区间。
"""
from mystats.interval.ci_mean import ci_mean

__all__ = ["ci_mean"]
