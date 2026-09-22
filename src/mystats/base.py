"""结果对象：全包所有统计过程统一的“出口”。

为什么所有过程都返回统一结构的对象，而不是返回一个数字？
--------
1. 调用方（你自己、将来的命令行界面、图形界面）拿到的字段永远一致：
   .statistic / .pvalue / .reject / .conclusion() ...，写批处理不用特判；
2. steps 字段承载【手算核对表】——本包面向教学的核心功能；
3. 将来新增回归、方差分析等模块，只需扩充 FitResult，老代码不受影响。

编码约定：输出文本只用 GBK/UTF-8 都安全的字符（√ ² μ σ ≠ ≤ ≥ 等），
不用 x̄（组合上横线）、Unicode 上下标等 GBK 缺失的字符，避免中文
Windows 控制台在重定向输出时抛 UnicodeEncodeError。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# 备择假设方向的中文说法（conclusion() 输出用）
_ALT_CN = {"two-sided": "双侧", "less": "左侧", "greater": "右侧"}


def _print_handcheck(method: str, steps: list) -> None:
    """按统一版式打印手算核对表。"""
    bar = "─" * 56
    print(bar)
    print(f"  手算核对表 · {method}")
    print(bar)
    for i, line in enumerate(steps, start=1):
        print(f"  {i:>2}. {line}")
    print(bar)


@dataclass
class TestResult:
    """假设检验的结果对象。

    常用属性 / 方法
    --------
    statistic : 检验统计量的值（z / t / 卡方 / F ...）
    pvalue    : p 值
    reject    : 属性（自动计算），p ≤ alpha 时为 True
    conclusion() : 返回中文结论文本；print(结果) 等价于打印它
    show_steps() : 打印手算核对表
    params    : 各检验自由存放的中间量（df、n、标准误、效应量……）
    """

    statistic: float
    pvalue: float
    method: str = ""
    alternative: str = "two-sided" # less / greater / two-sided
    alpha: float = 0.05
    h0: str = "" # 原假设的文字描述，如 "μ = 5"
    h1: str = "" # 备择假设的文字描述
    params: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)

    @property
    def reject(self) -> bool:
        """是否拒绝 H0：p 值法（p ≤ α），连续分布下与临界值法等价。"""
        return self.pvalue <= self.alpha

    def conclusion(self) -> str:
        cmp_ = "≤" if self.reject else ">"
        verdict = "拒绝原假设 H0" if self.reject else "不能拒绝原假设 H0"
        lines = []
        if self.method:
            alt_cn = _ALT_CN.get(self.alternative, self.alternative)
            lines.append(f"方法：{self.method}（{alt_cn}）")
        if self.h0:
            lines.append(f"H0：{self.h0}    H1：{self.h1}")
        lines.append(f"统计量 = {self.statistic:.6g}，p = {self.pvalue:.6g}")
        lines.append(f"判定：p {cmp_} α = {self.alpha:g}，{verdict}")
        return "\n".join(lines)

    def show_steps(self) -> None:
        _print_handcheck(self.method, self.steps)

    def __str__(self) -> str:
        return self.conclusion()


@dataclass
class IntervalResult:
    """区间估计的结果对象。

    estimate  : 点估计
    lower/upper : 置信区间下/上界
    contains(v) : 判断某值是否落在区间内（教学中常用来检查是否覆盖真值）
    """

    estimate: float
    lower: float
    upper: float
    confidence: float = 0.95
    method: str = ""
    params: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    def show_steps(self) -> None:
        _print_handcheck(self.method, self.steps)

    def __str__(self) -> str:
        return (
            f"{self.confidence:.0%} 置信区间：[{self.lower:.6g}, {self.upper:.6g}]"
            f"（点估计 = {self.estimate:.6g}，方法：{self.method}）"
        )


@dataclass
class FitResult:
    """“拟合类”结果（回归、方差分析模型）的占位骨架，阶段 3 扩充。

    规划字段（回归模块动工时定稿）：params（系数表）、params_se、
    residuals、df_model / df_resid、r_squared、conf_int 等。
    现在就让它存在，是为了让外部代码从第一天起就面向统一的
    三大结果类型（检验 / 区间 / 拟合）编程。
    """

    params: dict = field(default_factory=dict)
    method: str = ""
    steps: list = field(default_factory=list)
