"""阶段 3 演示：一元/多元线性回归 + 预测区间 + 非线性回归。

运行：python examples/demo_regression.py
"""
import sys

import numpy as np

from mystats import lin_reg, nonlin_reg

# 中文 Windows 控制台若是 GBK 编码，重定向输出时防止个别字符报错
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(errors="replace")

rng = np.random.default_rng(2026)

print("=" * 60)
print("1) 一元线性回归：复习时长 x（小时）与考试得分 y")
print("=" * 60)
x = rng.uniform(1, 10, size=20)
y = 35 + 5.5 * x + rng.normal(0, 6, size=20)
res = lin_reg(x, y)
print(res) # 摘要：R²、F 检验、系数表
print()
res.show_steps() # 手算核对表（Lxx / Lxy / Lyy 教材记号）
print()
pred = res.predict(x0=6.0, interval="prediction") # 复习 6 小时能考多少分
print(f"复习 6 小时的成绩预测：{pred}")
pred.show_steps()
print()
print(f"诊断：|标准化残差| 最大 = {np.max(np.abs(res.std_residuals)):.4g}，"
      f"Cook 距离最大 = {np.max(res.cooks_d):.4g}")

print()
print("=" * 60)
print("2) 多元线性回归：得分 ~ 复习时长 + 模拟考次数")
print("=" * 60)
x1 = rng.uniform(1, 10, size=25)
x2 = rng.integers(1, 6, size=25).astype(float)
y2 = 20 + 4.0 * x1 + 3.5 * x2 + rng.normal(0, 5, size=25)
res2 = lin_reg(np.column_stack([x1, x2]), y2,
               coef_names=["截距", "复习时长", "模拟考次数"])
print(res2)

print()
print("=" * 60)
print("3) 非线性回归：药物浓度衰减 c(t) = c0·exp(-k·t)")
print("=" * 60)
t = np.linspace(0.5, 8, 20)
c = 8.0 * np.exp(-0.35 * t) + rng.normal(0, 0.15, size=20)
res3 = nonlin_reg(lambda tt, c0, k: c0 * np.exp(-k * tt), t, c,
                  p0=[1.0, 0.1], coef_names=["c0", "k"])
print(res3)
res3.show_steps()
