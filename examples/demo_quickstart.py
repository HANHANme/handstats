"""快速体验：跑一个 t 检验 + 一个置信区间，看看“手算核对表”长什么样。

运行：python examples/demo_quickstart.py
"""
import sys

import numpy as np

from mystats import ci_mean, ttest_1samp

# 中文 Windows 控制台若是 GBK 编码，重定向输出时防止个别字符报错
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(errors="replace")

rng = np.random.default_rng(2026)
x = rng.normal(5.2, 1.3, size=16) # 模拟一批数据：真值 μ = 5.2

print("=" * 60)
print("1) 单样本 t 检验：H0: μ = 5.0")
print("=" * 60)
res = ttest_1samp(x, mu0=5.0)
print(res) # 结论（方法 / H0 / H1 / 统计量 / p / 判定）
print()
res.show_steps() # 手算核对表：每一步算式
print()
print(f"res.reject = {res.reject}    res.params = {res.params}")

print()
print("=" * 60)
print("2) 均值的 95% 置信区间（σ 未知 → t 区间）")
print("=" * 60)
ci = ci_mean(x, confidence=0.95)
print(ci)
print()
ci.show_steps()
print()
print(f"真值 μ = 5.2 是否落在区间里：{ci.contains(5.2)}")
