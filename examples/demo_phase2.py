"""阶段 2 新增过程演示：均值差区间、配对区间、方差区间、比例区间。

运行：python examples/demo_phase2.py
"""
import sys

import numpy as np

from mystats import ci_mean_2samp, ci_paired_diff, ci_proportion, ci_var

# 中文 Windows 控制台若是 GBK 编码，重定向输出时防止个别字符报错
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(errors="replace")

rng = np.random.default_rng(2026)

print("=" * 60)
print("1) 两种工艺的均值差 95% 置信区间（独立双样本 t 区间，合并方差）")
print("=" * 60)
a = rng.normal(52.0, 3.0, size=12) # 新工艺强度
b = rng.normal(49.5, 3.5, size=14) # 旧工艺强度
ci = ci_mean_2samp(a, b)
print(ci)
ci.show_steps()

print()
print("=" * 60)
print("2) 配对设计：10 名学生培训前后的成绩差 95% 置信区间")
print("=" * 60)
before = rng.normal(70.0, 8.0, size=10)
after = before + rng.normal(5.0, 3.0, size=10)
ci = ci_paired_diff(after, before)
print(ci)
ci.show_steps()

print()
print("=" * 60)
print("3) 总体方差 σ² 的 95% 置信区间（卡方区间，注意它不对称）")
print("=" * 60)
x = rng.normal(0.0, 2.0, size=15)
ci = ci_var(x)
print(ci)
ci.show_steps()
print(f"params 里的 σ 区间：[{ci.params['sd_lower']:.4g}, {ci.params['sd_upper']:.4g}]")

print()
print("=" * 60)
print("4) 总体比例的 95% 置信区间（Wald）：100 件中 3 件次品")
print("=" * 60)
ci = ci_proportion(x=3, n=100) # 成功次数少 → 看核对表末尾的提示
print(ci)
ci.show_steps()
