"""阶段 1 新增过程演示：卡方拟合优度、卡方独立性、单比例 z 检验。

运行：python examples/demo_phase1.py
"""
import sys

from handstats import chisquare_gof, chisquare_ind, ztest_1prop

# 中文 Windows 控制台若是 GBK 编码，重定向输出时防止个别字符报错
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(errors="replace")

print("=" * 60)
print("1) 卡方拟合优度：掷骰子 60 次，检验骰子是否均匀")
print("=" * 60)
res = chisquare_gof(obs=[8, 13, 9, 12, 7, 11]) # 各面出现次数，期望每面 10 次
print(res)
res.show_steps()

print()
print("=" * 60)
print("2) 卡方独立性：吸烟与患呼吸道疾病是否有关（2×2 列联表）")
print("=" * 60)
table = [[45, 55], [30, 70]] # 行 = 是否吸烟，列 = 是否患病
res = chisquare_ind(table)
print(res)
res.show_steps()

print()
print("=" * 60)
print("3) 单比例 z 检验：抛硬币 100 次正面 65 次，硬币是否不均匀")
print("=" * 60)
res = ztest_1prop(x=65, n=100, p0=0.5)
print(res)
res.show_steps()
