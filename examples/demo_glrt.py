"""GLRT（广义似然比检验）演示：通用引擎 + 两个可解析实例 + 泊松引擎用法。

运行：python examples/demo_glrt.py
"""
import sys

import numpy as np

from handstats import glrt_exponential_mean, glrt_normal_mean, glrt_test

# 中文 Windows 控制台若是 GBK 编码，重定向输出时防止个别字符报错
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(errors="replace")

rng = np.random.default_rng(2026)

print("=" * 60)
print("1) 正态均值 GLRT（σ 未知）：H0: μ = 5")
print("   看核对表最后一行：Wilks 渐近 p 与精确 t p 的对比")
print("=" * 60)
x = rng.normal(5.4, 1.2, size=20)
res = glrt_normal_mean(x, mu0=5.0)
print(res)
res.show_steps()

print()
print("=" * 60)
print("2) 正态均值 GLRT（σ 已知，同一批数据）：H0: μ = 5")
print("   z² 恰好严格服从 χ²(1) → 渐近 p = 精确 p")
print("=" * 60)
res = glrt_normal_mean(x, mu0=5.0, sigma=1.2)
print(res)
res.show_steps()

print()
print("=" * 60)
print("3) 指数均值 GLRT：20 个灯泡寿命（小时），H0: 平均寿命 = 1000")
print("   Wilks 渐近 p vs 等尾精确 p（卡方偏态导致细微差别）")
print("=" * 60)
life = np.array([896, 1053, 1204, 763, 987, 1120, 655, 1341, 902, 1108,
                 774, 1015, 1230, 868, 955, 1402, 1090, 815, 970, 1148],
                dtype=float)
res = glrt_exponential_mean(life, theta0=1000.0)
print(res)
res.show_steps()

print()
print("=" * 60)
print("4) 通用引擎：泊松分布 H0: λ = 4（手算两个对数似然，交给引擎）")
print("=" * 60)
x = rng.poisson(3.2, size=30)
n = x.size
s = float(x.sum())
xbar = s / n
lam0 = 4.0
# 泊松对数似然 lnL(λ) = -nλ + (Σxi)·ln λ - Σln(xi!)
# 末项与 λ 无关，两个对数似然相减时抵消 → 喂给引擎时可直接省略
ll_full = -n * xbar + s * np.log(xbar)
ll_h0 = -n * lam0 + s * np.log(lam0)
res = glrt_test(ll_full, ll_h0, df=1) # 维数差 dim(Θ) - dim(Θ0) = 1
print(res)
res.show_steps()
