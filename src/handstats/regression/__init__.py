"""回归分析子包（阶段 3）。

已有：
- lin_reg：一元/多元 OLS（t / F 检验、系数置信区间、R²、
  残差诊断、predict 预测区间）
- nonlin_reg：非线性最小二乘（scipy curve_fit + 参数推断）

规划：加权最小二乘（WLS）、广义线性模型（逻辑/泊松回归）、逐步回归。
动工模板：见 ols.py 与 base.FitResult。
"""
from handstats.regression.nonlinear import nonlin_reg
from handstats.regression.ols import lin_reg

__all__ = ["lin_reg", "nonlin_reg"]
