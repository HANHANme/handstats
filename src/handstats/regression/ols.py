"""线性回归（OLS）：一元 / 多元，含推断、区间预测与残差诊断。

黄金模板的“拟合版”流程：
    ① 校验 → ② 矩阵求解 β 与残差方差 → ③ 推断量（t / F / 区间 / R²）
    → ④ 打包 FitResult + steps
一元场合额外用教材记号（Lxx、Lxy、Lyy）写核对表，可与手算逐步对照。
"""
from __future__ import annotations

import numpy as np

from handstats import distributions
from handstats._registry import register
from handstats.base import FitResult
from handstats.validate import as_sample, check_alpha


def _design(x, y, coef_names):
    """校验自变量并组装含截距列的设计矩阵；返回 X, 名称列表, n, k。（y 已校验）"""
    X_raw = np.asarray(x, dtype=float)
    if X_raw.ndim == 1:
        X_raw = X_raw.reshape(-1, 1)
    if X_raw.ndim != 2:
        raise ValueError("x 必须是一维（一元回归）或二维（多元回归，每列一个自变量）")
    n, k = X_raw.shape
    if y.size != n:
        raise ValueError(f"x 与 y 长度不一致：x 有 {n} 行，y 有 {y.size} 个")
    p = k + 1
    if n <= p:
        raise ValueError(
            f"样本量 n = {n} 必须大于参数个数 p = {p}（含截距），否则没有残差自由度"
        )
    X = np.column_stack([np.ones(n), X_raw])
    if np.linalg.matrix_rank(X) < p:
        raise ValueError("设计矩阵不满秩：自变量之间存在完全共线性（重复列或常数列）")
    if coef_names is None:
        names = ["截距"] + [f"x{i + 1}" for i in range(k)]
    else:
        names = list(coef_names)
        if len(names) != p:
            raise ValueError(
                f"coef_names 需给出全部 {p} 个名称（含截距），收到 {len(names)} 个"
            )
    return X, names, n, k


@register("lin_reg")
def lin_reg(x, y, alpha=0.05, coef_names=None):
    """一元 / 多元线性回归（最小二乘，含截距）。

    参数
    ----
    x : 一维（一元回归）或二维 (n, k)（多元回归，每列一个自变量）
    y : 一维因变量
    alpha : 显著性水平（用于系数置信区间与 F 检验判定）
    coef_names : 系数名列表（缺省：截距, x1, x2, ...）

    返回的 FitResult 常用：summary() / coef_table() / show_steps() /
    predict(x0, interval="prediction")；诊断字段 std_residuals、
    leverage、cooks_d（|标准化残差| > 2 或 Cook 距离大 → 值得检查的点）。
    """
    alpha = check_alpha(alpha)
    y = as_sample(y, "y")
    X, names, n, k = _design(x, y, coef_names)
    p = k + 1

    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    fitted = X @ beta
    resid = y - fitted
    sse = float(resid @ resid)
    ym = float(np.mean(y))
    sst = float(np.sum((y - ym) ** 2))
    ssr = sst - sse
    df_resid = n - p
    df_model = k
    sigma2 = sse / df_resid
    sigma = float(np.sqrt(sigma2))
    r2 = 1.0 - sse / sst
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df_resid
    se = np.sqrt(sigma2 * np.diag(XtX_inv))
    tvals = beta / se
    pvals = np.array(
        [distributions.p_value(t, "two-sided", dist="t", df=df_resid) for t in tvals]
    )
    crit = distributions.t_ppf(1 - alpha / 2, df_resid)
    ci = np.column_stack([beta - crit * se, beta + crit * se])
    f_stat = float((ssr / df_model) / sigma2)
    f_p = distributions.p_value(f_stat, "greater", dist="f", dfn=df_model, dfd=df_resid)

    # 诊断：杠杆 h_ii（帽子矩阵对角元）、标准化残差、Cook 距离
    hii = np.einsum("ij,jk,ik->i", X, XtX_inv, X) # 等价 diag(X(X'X)^(-1)X')，不生成 n×n 矩阵
    std_resid = resid / (sigma * np.sqrt(1.0 - hii)) # 内学生化残差 r_i = e_i/(s·√(1-h_ii))
    # Cook 距离：定义 D_i = e_i²·h_ii/(p·MSE·(1-h_ii)²)，代入 r_i 恒等变形为
    # D_i = r_i²·h_ii/(p·(1-h_ii))（注意分母只有一个 (1-h_ii)，勿再平方）
    cooks = (std_resid**2 / p) * (hii / (1.0 - hii))

    if k == 1:
        xr = X[:, 1]
        xm = float(np.mean(xr))
        Lxx = float(np.sum((xr - xm) ** 2))
        Lxy = float(np.sum((xr - xm) * (y - ym)))
        b, a = float(beta[1]), float(beta[0])
        steps = [
            f"n = {n}，均值(x) = {xm:.6g}，均值(y) = {ym:.6g}",
            f"Lxx = Σ(xi - 均值x)² = {Lxx:.6g}，"
            f"Lxy = Σ(xi - 均值x)(yi - 均值y) = {Lxy:.6g}，Lyy = {sst:.6g}",
            f"斜率 b = Lxy/Lxx = {b:.6g}，截距 a = 均值y - b·均值x = {a:.6g}",
            f"回归平方和 SSR = b·Lxy = {ssr:.6g}，残差平方和 SSE = Lyy - SSR = {sse:.6g}，"
            f"σ² = SSE/(n-2) = {sigma2:.6g}",
            f"SE(b) = σ/√Lxx = {se[1]:.6g}，SE(a) = σ·√(1/n + 均值x²/Lxx) = {se[0]:.6g}",
            f"t(b) = b/SE(b) = {tvals[1]:.6g}（p = {pvals[1]:.6g}），"
            f"t(a) = a/SE(a) = {tvals[0]:.6g}（p = {pvals[0]:.6g}），df = n - 2 = {df_resid}",
            f"R² = SSR/Lyy = {r2:.6g}，F = (SSR/1)/(SSE/(n-2)) = t(b)² = {f_stat:.6g}，"
            f"p = {f_p:.6g}",
        ]
    else:
        steps = [
            f"n = {n}，自变量个数 k = {k}，参数个数 p = k + 1 = {p}，残差自由度 n - p = {df_resid}",
            f"系数 = (X^T X)^(-1) X^T y = {[round(float(v), 6) for v in beta]}"
            "（X 为含截距列的设计矩阵）",
            f"残差平方和 SSE = Σ(yi - 拟合值)² = {sse:.6g}，σ² = SSE/(n-p) = {sigma2:.6g}",
            f"各系数标准误 = √(σ²·(X^T X)^(-1) 对角元) = {[round(float(v), 6) for v in se]}",
            f"各系数 t = 系数/标准误 = {[round(float(v), 4) for v in tvals]}，"
            f"p 值 = {[round(float(v), 4) for v in pvals]}（t 分布，df = {df_resid}）",
            f"R² = 1 - SSE/SST = {r2:.6g}，调整 R² = {adj_r2:.6g}",
            f"整体 F = (SSR/{k})/(SSE/(n-p)) = {f_stat:.6g}，p = {f_p:.6g}"
            f"（H0: 所有斜率 = 0）",
        ]

    kind = "一元" if k == 1 else "多元"
    return FitResult(
        method=f"线性回归 OLS（{kind}，含截距）",
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
        df_model=df_model,
        df_resid=df_resid,
        f_stat=f_stat,
        f_pvalue=f_p,
        fitted=fitted,
        residuals=resid,
        std_residuals=std_resid,
        leverage=hii,
        cooks_d=cooks,
        alpha=alpha,
        params={"xtx_inv": XtX_inv, "critical_t": crit},
        steps=steps,
    )
