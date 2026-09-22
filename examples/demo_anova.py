"""阶段 4 演示：单因素/双因素方差分析 + Tukey 事后比较 + Levene 前提检查。

运行：python examples/demo_anova.py
"""
import sys

import numpy as np

from mystats import anova_oneway, anova_twoway, levene_test, tukey_hsd

# 中文 Windows 控制台若是 GBK 编码，重定向输出时防止个别字符报错
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(errors="replace")

rng = np.random.default_rng(2026)

print("=" * 60)
print("1) 单因素：三种教学法的期末成绩")
print("=" * 60)
groups = [
    rng.normal(75, 5, 15), # 讲授法
    rng.normal(80, 5, 15), # 讨论法
    rng.normal(70, 5, 15), # 实验法
]
res = anova_oneway(groups)
print(res)
res.show_steps()

print()
print("=" * 60)
print("2) Levene：先检查 ANOVA 的方差齐性前提")
print("=" * 60)
print(levene_test(groups))

print()
print("=" * 60)
print("3) Tukey HSD：ANOVA 拒绝了 H0，到底哪两种教学法有差别？")
print("=" * 60)
for t in tukey_hsd(groups):
    print(t)

print()
print("=" * 60)
print("4) 双因素无重复：4 台机器 × 3 名操作员的日产量")
print("=" * 60)
tab = (80 + np.array([0.0, 2.0, -1.0, 1.5])[:, None] # 机器效应
       + np.array([0.0, 1.0, -0.5])[None, :] # 操作员效应
       + rng.normal(0, 0.6, (4, 3)))
res2 = anova_twoway(tab, names=("机器", "操作员"))
print(res2)
res2.show_steps()

print()
print("=" * 60)
print("5) 双因素等重复：温度 × 催化剂（每格 2 次），检验交互作用")
print("=" * 60)
data = (62.0
        + np.array([0.0, 3.0, 6.0])[:, None, None] # 温度主效应
        + np.array([0.0, 2.0])[None, :, None] # 催化剂主效应
        + np.array([[0.0, 0.0], [0.0, 1.5], [0.0, 2.5]])[:, :, None] # 交互
        + rng.normal(0, 0.8, (3, 2, 2)))
res3 = anova_twoway(data, names=("温度", "催化剂"))
print(res3)
res3.show_steps()
