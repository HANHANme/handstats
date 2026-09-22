"""方差分析子包（阶段 4）。

已有：
- anova_oneway：单因素方差分析（ANOVA 表、η² 效应量）
- tukey_hsd：事后多重比较（族错误率控制在 α，Tukey-Kramer 支持不等样本量）
- levene_test：方差齐性检验（Levene / Brown-Forsythe，ANOVA 的前提检查）
- anova_twoway：双因素（无重复 r×c / 等重复 r×c×m 含交互检验），
  返回复合结果 TwoWayResult

规划：随机区组设计、非平衡双因素、多元方差分析（见 multivariate）。
"""
from mystats.anova.levene import levene_test
from mystats.anova.oneway import anova_oneway
from mystats.anova.posthoc import tukey_hsd
from mystats.anova.twoway import TwoWayResult, anova_twoway

__all__ = ["anova_oneway", "anova_twoway", "tukey_hsd", "levene_test", "TwoWayResult"]
