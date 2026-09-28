"""mystats 包的“公共出口”：三大结果对象 + 手算核对表打印器。

为什么所有过程都返回统一结构的对象，而不是返回一个数字？
--------
1. 调用方（你自己、将来的命令行界面、图形界面）拿到的字段永远一致，
   写批处理不用特判；
2. steps 字段承载【手算核对表】——本包面向教学的核心功能；
3. 三大类型覆盖全部统计过程：检验 TestResult、区间 IntervalResult、
   拟合 FitResult（回归 / 方差分析）。

编码约定：输出文本只用 GBK/UTF-8 都安全的字符（√ ² μ σ ≠ ≤ ≥ 等），
不用 x̄、β̂、ŷ、组合上标等 GBK 缺失的字符，避免中文 Windows 控制台
在重定向输出时抛 UnicodeEncodeError。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mystats import distributions

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
    statistic : 检验统计量的值（z / t / 卡方 / F / -2lnλ ...）
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
        if self.reject and self.h1 and self.alternative in ("less", "greater"):
            # 单侧检验的 H0 是复合假设，拒绝时应指明方向性结论落在 H1 一侧
            verdict += f"，证据支持 H1：{self.h1}"
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
    """“拟合类”结果（回归 / 方差分析），阶段 3 定稿。

    常用属性 / 方法
    --------
    coefficients / coef_se / coef_t / coef_p / coef_ci : 系数及其推断
    r_squared / adj_r_squared / sigma / sigma2 : 拟合优度与残差方差
    f_stat / f_pvalue : 整体显著性 F 检验（H0: 所有斜率 = 0）
    fitted / residuals / std_residuals / leverage / cooks_d : 拟合值与诊断
    summary() / coef_table() : 文本摘要与系数表；print(结果) 等价于 summary()
    show_steps() : 打印手算核对表
    predict(x0, interval=...) : 新点预测（仅线性回归，返回 IntervalResult）
    """

    method: str = ""
    coef_names: list = field(default_factory=list)
    coefficients: np.ndarray | None = None
    coef_se: np.ndarray | None = None
    coef_t: np.ndarray | None = None
    coef_p: np.ndarray | None = None
    coef_ci: np.ndarray | None = None # 形状 (p, 2)：每行 [下界, 上界]
    r_squared: float = 0.0
    adj_r_squared: float = 0.0
    sigma: float = 0.0
    sigma2: float = 0.0
    n: int = 0
    df_model: int = 0 # 斜率（自变量）个数 k
    df_resid: int = 0 # n - p，p 为含截距的参数个数
    f_stat: float | None = None
    f_pvalue: float | None = None
    fitted: np.ndarray | None = None
    residuals: np.ndarray | None = None
    std_residuals: np.ndarray | None = None
    leverage: np.ndarray | None = None # 帽子矩阵对角元 h_ii（非线性回归为 None）
    cooks_d: np.ndarray | None = None # Cook 距离（非线性回归为 None）
    alpha: float = 0.05
    params: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)

    def coef_table(self) -> str:
        """系数表：估计值、标准误、t 值、p 值、置信区间。"""
        if self.coefficients is None:
            return ""
        conf = 1 - self.alpha
        header = (
            f"{'系数':<6}{'估计':>13}{'标准误':>13}{'t 值':>10}{'p 值':>10}"
            f"   {conf:.0%} 区间"
        )
        rows = []
        for i, name in enumerate(self.coef_names):
            lo, hi = self.coef_ci[i]
            rows.append(
                f"{name:<6}{self.coefficients[i]:>13.6g}{self.coef_se[i]:>13.6g}"
                f"{self.coef_t[i]:>10.4g}{self.coef_p[i]:>10.4g}   [{lo:.6g}, {hi:.6g}]"
            )
        return "\n".join([header, "─" * 76, *rows])

    def summary(self) -> str:
        lines = []
        if self.method:
            lines.append(f"方法：{self.method}")
        lines.append(
            f"n = {self.n}，R² = {self.r_squared:.6g}，"
            f"调整 R² = {self.adj_r_squared:.6g}，残差标准差 = {self.sigma:.6g}"
        )
        if self.f_stat is not None:
            if self.f_pvalue is not None and self.f_pvalue <= self.alpha:
                verdict = "拒绝 H0（模型整体显著）"
            else:
                verdict = "不能拒绝 H0（模型整体不显著）"
            lines.append(
                f"整体 F 检验（H0: 所有斜率 = 0）：F = {self.f_stat:.6g}，"
                f"p = {self.f_pvalue:.6g} → {verdict}"
            )
        table = self.coef_table()
        if table:
            lines.append(table)
        return "\n".join(lines)

    def show_steps(self) -> None:
        _print_handcheck(self.method, self.steps)

    def predict(self, x0, interval="prediction") -> IntervalResult:
        """用拟合的线性模型在新点 x0 预测，返回 IntervalResult。

        x0 : 预测点的自变量值（不含截距）。一元回归给标量，
             多元回归给长度 k 的序列。
        interval : "prediction" 个别值预测区间（含 y 的随机波动，考试重点）；
                   "confidence" 均值的置信区间（只含回归线本身的不确定性，
                   总比预测区间窄）。
        """
        if "xtx_inv" not in self.params:
            raise ValueError("predict 仅支持 lin_reg 的结果（非线性模型无此信息）")
        if interval not in ("prediction", "confidence"):
            raise ValueError('interval 只能是 "prediction" 或 "confidence"')
        if np.ndim(x0) == 0:
            row = np.array([1.0, float(x0)])
        else:
            row = np.concatenate([[1.0], np.asarray(x0, dtype=float)])
        if row.size != self.coefficients.size:
            raise ValueError(
                f"预测点需要 {self.coefficients.size - 1} 个自变量值，收到 {row.size - 1} 个"
            )
        yhat = float(row @ self.coefficients)
        lev = float(row @ self.params["xtx_inv"] @ row) # 预测点的杠杆 h0
        crit = self.params["critical_t"]
        extra = "1 + " if interval == "prediction" else ""
        se = float(self.sigma * np.sqrt((1.0 if interval == "prediction" else 0.0) + lev))
        half = float(crit * se)
        label = "预测区间（个别值）" if interval == "prediction" else "均值置信区间"
        steps = [
            f"预测点（含截距 1）= {[round(float(v), 6) for v in row]}",
            f"点预测 = 预测点 · 系数向量 = {yhat:.6g}",
            f"杠杆 h0 = 预测点^T (X^T X)^(-1) 预测点 = {lev:.6g}",
            f"SE = σ·√({extra}h0) = {se:.6g}，"
            f"临界值 t(α/2, df = {self.df_resid}) = {crit:.6g}",
            f"{label} = 点预测 ± t·SE = {yhat:.6g} ± {half:.6g}",
        ]
        return IntervalResult(
            estimate=yhat,
            lower=yhat - half,
            upper=yhat + half,
            confidence=1 - self.alpha,
            method=label,
            params={"leverage": lev, "se": se, "critical_value": float(crit)},
            steps=steps,
        )

    def __str__(self) -> str:
        return self.summary()
