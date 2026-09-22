# mystats

> 面向学习的数理统计工具包：假设检验 · 区间估计 · 回归 · 方差分析（建设中）

`mystats` 与 scipy.stats 的差别只有一句话：**每个过程都输出手算核对表**——
样本量、均值、标准差、标准误、统计量、自由度、p 值的每一步算式都逐行
打印出来，方便和课本 / 考试手算互相核对。数值底层由 numpy / scipy 提供，
正确性由与 scipy 的对拍测试保证。

## 安装（开发模式，改代码立即生效）

```bash
cd mystats
python -m pip install -e ".[dev]"
python -m pytest          # 应当全部通过
```

## 快速上手

```python
import numpy as np
from mystats import ttest_1samp, ci_mean

rng = np.random.default_rng(2026)
x = rng.normal(5.2, 1.3, size=16)

res = ttest_1samp(x, mu0=5.0)
print(res)          # 方法 / H0 / H1 / 统计量 / p 值 / 结论
res.show_steps()    # 手算核对表：n、均值、s、SE、t、p 一步步列出
res.reject          # True / False
res.params          # {"n": 16, "df": 15, ...} 中间量都在

ci = ci_mean(x)     # 均值 95% t 置信区间
ci.show_steps()
ci.contains(5.2)    # 检查区间是否覆盖真值
```

完整演示：`python examples/demo_quickstart.py`（入门）、
`python examples/demo_phase1.py`（卡方 / F / 比例）

全部可用过程一览：`python -m mystats` 或 `mystats.list_procedures()`

## 目录结构与路线图

```
src/mystats/
├── base.py           结果对象：TestResult / IntervalResult / FitResult（全包统一出口）
├── validate.py       输入校验（全包共用的第一道关卡）
├── distributions.py  分布层：scipy 的薄封装，未来自研数值算法的换芯点
├── _registry.py      过程注册表：新过程挂上即被 list_procedures() 发现
├── hypothesis/       假设检验   —— 已有 z/t/卡方/F/比例；规划：GLRT
├── interval/         区间估计   —— 已有均值 z/t 区间
├── regression/       回归分析   —— 阶段 3：OLS、非线性、诊断
├── anova/            方差分析   —— 阶段 4：单/双因素、事后检验
├── nonparametric/    非参数检验 —— 预留：符号、秩和、KS
├── resampling/       重抽样     —— 预留：Bootstrap、置换检验
└── multivariate/     多元统计   —— 预留：Hotelling T2、PCA
```

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | 包骨架 + 注册表 + 结果对象 + 黄金模板（z/t 检验、均值区间）+ 对拍测试 | ✅ 当前 |
| 1 | 假设检验扩充：卡方拟合优度 / 独立性、F 方差齐性、单/双比例 z 检验 | ✅ |
| 2 | 区间估计扩充：两样本均值差、比例、方差 | ⬜ 下一站 |
| 3 | 回归：一元 / 多元 OLS、显著性、诊断、非线性 | ⬜ |
| 4 | 方差分析：单 / 双因素、ANOVA 表、事后检验 | ⬜ |
| 5 | 打磨文档与示例、发布 PyPI | ⬜ |

## 如何新增一个统计过程（黄金模板四步）

所有模块统一走 `hypothesis/tests_mean.py` 里的四步流水线：

1. **校验** `validate`：`as_sample` / `check_alternative` / `check_alpha`；
2. **计算**：纯 numpy 算统计量——数学只发生在这一步；
3. **p 值** `distributions.p_value`：统计量 → p 值；
4. **打包** `TestResult`，把每一步算式写进 `steps`。

最后用 `@register("名字")` 挂上注册表，并在 `tests/` 里补一条与 scipy
的对拍测试。

## 未来：无代码外壳（规划）

目标：不写代码也能用——网页表单选方法、填数据、看结论和手算核对表。
骨架已为此预留两个钩子：注册表（`python -m mystats` 一览全部过程，
外壳遍历它即可自动生成方法菜单）和统一结果对象（外壳只需展示
`conclusion()` / `show_steps()` 的文本）。计划在核心过程齐备后用
Streamlit 实现网页版，并可免费部署成公开链接。

## 开发约定

- 测试与源码 1:1：新检验不对拍不合入（对拍就是正确性的锚）；
- 输出文本只用 GBK 安全字符（√ ² μ σ ≠ ≤ ≥ 等），保证中文 Windows
  控制台 / 重定向不会因编码崩掉；
- 包名 `mystats` 是占位：发布前想改名，改 `pyproject.toml` 的 `name`
  和 `src/mystats/` 目录名即可（宜早不宜迟，PyPI 上 mystats 可能已被占用）。

## 推送到 GitHub（第一次）

先在 GitHub 网页上新建一个**空**仓库（不要勾选初始化 README），然后：

```bash
git remote add origin git@github.com:<你的用户名>/mystats.git
git push -u origin main
```

`.github/workflows/tests.yml` 已配好：push 之后每次提交会自动在
Ubuntu + 三个 Python 版本上跑测试，在仓库的 Actions 标签页查看。

## 许可证

MIT（见 LICENSE；记得把版权行改成你自己的名字）。
