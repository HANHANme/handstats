"""非线性回归：scipy.optimize.curve_fit 的教学封装 + 参数推断。"""
from __future__ import annotations

import numpy as np

from handstats import distributions
from handstats._registry import register
from handstats.base import FitResult
from handstats.validate import as_sample, check_alpha


@register("nonlin_reg")
def nonlin_reg(model, x, y, p0, alpha=0.05, coef_names=None):
    """非线性最小二乘回归（课程：非线性回归）。

    参数
    ----
    model : 形如 f(x, a, b, ...) 的模型函数（需支持 numpy 向量化）
    x : 自变量（一维；多维模型可给形状 (n, d) 的数组）
    y : 因变量（一维）
    p0 : 参数初值列表——初值不合适可能不收敛，换个初值再试
    coef_names : 参数名列表（缺省 b1, b2, ...）
    alpha : 显著性水平

    示例
    ----
    nonlin_reg(lambda t, c0, k: c0 * np.exp(-k * t), t, c, p0=[1.0, 0.1])

    推断说明：标准误来自协方差矩阵 pcov 的对角元（线性化近似），
    t / p 值用 t 分布（df = n - m）近似。非线性模型没有精确的小样本
    理论，这是业界通行做法；R² 在非线性场合仅作参考。
    """
    from scipy.optimize import curve_fit # 全包唯一用到优化器的地方

    alpha = check_alpha(alpha)
    y = as_sample(y, "y")
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 0:
        raise ValueError("x 不能是标量")
    if x_arr.shape[0] != y.size:
        raise ValueError(f"x 与 y 的观测数不一致：x 有 {x_arr.shape[0]}，y 有 {y.size}")

    beta, pcov = curve_fit(model, x_arr, y, p0=p0)
    m = int(beta.size)
    n = y.size
    df_resid = n - m
    if df_resid < 1:
        raise ValueError(f"参数个数 m = {m} 必须小于样本量 n = {n}")

    fitted = np.asarray(model(x_arr, *beta), dtype=float)
    resid = y - fitted
    sse = float(resid @ resid)
    ym = float(np.mean(y))
    sst = float(np.sum((y - ym) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df_resid
    sigma2 = sse / df_resid
    sigma = float(np.sqrt(sigma2))
    se = np.sqrt(np.diag(pcov))
    tvals = beta / se
    pvals = np.array(
        [distributions.p_value(t, "two-sided", dist="t", df=df_resid) for t in tvals]
    )
    crit = distributions.t_ppf(1 - alpha / 2, df_resid)
    ci = np.column_stack([beta - crit * se, beta + crit * se])

    if coef_names is None:
        names = [f"b{i + 1}" for i in range(m)]
    else:
        names = list(coef_names)
        if len(names) != m:
            raise ValueError(f"coef_names 需给出全部 {m} 个参数名，收到 {len(names)} 个")

    steps = [
        f"模型由用户给定，参数个数 m = {m}，n = {n}，残差自由度 df = n - m = {df_resid}",
        f"curve_fit 数值优化（无 bounds 时为 Levenberg-Marquardt 法）得参数 = "
        f"{[round(float(v), 6) for v in beta]}",
        f"残差平方和 SSE = {sse:.6g}，σ² = SSE/(n-m) = {sigma2:.6g}",
        f"标准误 = √(协方差矩阵 pcov 对角元) = {[round(float(v), 6) for v in se]}",
        f"t = 参数/标准误 = {[round(float(v), 4) for v in tvals]}，"
        f"p ≈ {[round(float(v), 4) for v in pvals]}（t 分布，df = {df_resid}，近似）",
        f"R² = 1 - SSE/SST = {r2:.6g}（非线性模型中 R² 仅作参考）",
    ]
    return FitResult(
        method="非线性最小二乘回归",
        coef_names=names,
        coefficients=beta,
        coef_se=se,
        coef_t=tvals,
        coef_p=pvals,
        coef_ci=ci,
        r_squared=r2,
        adj_r_squared=adj_r2,
        sigma=sigma,
        sigma2=sigma2,
        n=n,
        df_model=m - 1,
        df_resid=df_resid,
        f_stat=None, # 非线性模型的整体 F 检验无标准定义，不提供
        f_pvalue=None,
        fitted=fitted,
        residuals=resid,
        std_residuals=resid / sigma, # 简单标准化（非线性无杠杆理论）
        leverage=None,
        cooks_d=None,
        alpha=alpha,
        params={"p0": list(p0), "pcov": pcov, "critical_t": crit},
        steps=steps,
    )
