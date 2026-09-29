"""回归模块的对拍测试：scipy.linregress / 手工矩阵公式 / curve_fit / 删除法定义 / OLS 交叉验证。"""
import numpy as np
import pytest
from scipy import stats
from scipy.optimize import curve_fit

from handstats.regression import lin_reg, nonlin_reg

rng = np.random.default_rng(20260922)


def _assert_close(a, b):
    assert np.isclose(a, b, rtol=1e-9, atol=1e-12), f"{a} != {b}"


def test_simple_reg_matches_linregress():
    x = rng.uniform(0, 10, size=30)
    y = 2.0 + 1.5 * x + rng.normal(0, 1.0, size=30)
    ref = stats.linregress(x, y)
    res = lin_reg(x, y)
    _assert_close(res.coefficients[0], ref.intercept)
    _assert_close(res.coefficients[1], ref.slope)
    _assert_close(res.coef_se[0], ref.intercept_stderr)
    _assert_close(res.coef_se[1], ref.stderr)
    _assert_close(res.r_squared, ref.rvalue**2)
    _assert_close(res.coef_p[1], ref.pvalue)
    assert res.df_resid == 28 and res.df_model == 1


def test_multi_reg_matches_manual_matrix():
    X = np.column_stack([rng.uniform(0, 5, 40), rng.normal(0, 1, 40)])
    y = 1.0 + 2.0 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(0, 0.8, 40)
    res = lin_reg(X, y)
    # 手工矩阵公式（含截距列）
    Xd = np.column_stack([np.ones(40), X])
    beta_ref = np.linalg.solve(Xd.T @ Xd, Xd.T @ y)
    resid = y - Xd @ beta_ref
    sse = float(resid @ resid)
    mse = sse / (40 - 3)
    se_ref = np.sqrt(mse * np.diag(np.linalg.inv(Xd.T @ Xd)))
    np.testing.assert_allclose(res.coefficients, beta_ref, rtol=1e-9)
    np.testing.assert_allclose(res.coef_se, se_ref, rtol=1e-9)
    _assert_close(res.r_squared, 1 - sse / np.sum((y - y.mean()) ** 2))
    assert res.df_resid == 37 and res.df_model == 2


def test_simple_exact_fit():
    """完全落在直线上的数据：系数精确还原，R² = 1。"""
    x = np.arange(1.0, 6.0)
    y = 1.0 + 2.0 * x
    with np.errstate(divide="ignore", invalid="ignore"): # σ² = 0 → t/F 无穷大
        res = lin_reg(x, y)
    _assert_close(res.coefficients[0], 1.0)
    _assert_close(res.coefficients[1], 2.0)
    _assert_close(res.r_squared, 1.0)
    _assert_close(res.sigma2, 0.0)


def test_predict_intervals_match_manual():
    x = rng.uniform(0, 10, size=25)
    y = 3.0 + 0.8 * x + rng.normal(0, 1.2, size=25)
    res = lin_reg(x, y)
    x0 = 6.0
    Xd = np.column_stack([np.ones(25), x])
    XtXi = np.linalg.inv(Xd.T @ Xd)
    row = np.array([1.0, x0])
    yhat = float(row @ res.coefficients)
    lev = float(row @ XtXi @ row)
    crit = res.params["critical_t"]
    cases = [
        ("confidence", res.sigma * np.sqrt(lev)),
        ("prediction", res.sigma * np.sqrt(1 + lev)),
    ]
    for kind, scale in cases:
        pred = res.predict(x0, interval=kind)
        ref = stats.t.interval(0.95, res.df_resid, loc=yhat, scale=scale)
        _assert_close(pred.estimate, yhat)
        _assert_close(pred.lower, ref[0])
        _assert_close(pred.upper, ref[1])
    # 预测区间必须比均值置信区间宽（考试重点结论）
    ci = res.predict(x0, interval="confidence")
    pi = res.predict(x0, interval="prediction")
    assert pi.upper - pi.lower > ci.upper - ci.lower


def test_diagnostics_properties():
    x = rng.normal(0, 1, 30)
    y = 1.0 + x + rng.normal(0, 1, 30)
    res = lin_reg(x, y)
    _assert_close(float(np.sum(res.leverage)), 2.0) # 帽子矩阵的迹 = 参数个数 p
    assert np.all(res.cooks_d >= 0)
    assert res.std_residuals.shape == (30,)
    assert "手算核对表" not in res.summary() # summary 与核对表分离


def test_cooks_distance_matches_deletion_definition():
    """Cook 距离金标准对拍：删除法定义 D_i = (β̂-β̂_(i))'X'X(β̂-β̂_(i))/(p·MSE)。

    数据里故意放一个高杠杆点（x=6）：若公式错成多除一个 (1-h_ii)，
    该点会被放大 1/(1-h_ii) 倍，本测试必然失败（低杠杆点上两种写法
    差异在小数位以下，捕捉不到）。
    """
    x = rng.normal(0, 1, 20)
    x[0] = 6.0 # 高杠杆点
    y = 1.0 + 0.8 * x + rng.normal(0, 0.6, 20)
    res = lin_reg(x, y)

    n = 20
    Xd = np.column_stack([np.ones(n), x])
    p = Xd.shape[1] # 含截距的参数个数
    XtX = Xd.T @ Xd
    beta_all = np.linalg.solve(XtX, Xd.T @ y)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        Xm, ym = Xd[mask], y[mask]
        beta_i = np.linalg.solve(Xm.T @ Xm, Xm.T @ ym)
        diff = beta_all - beta_i
        d_ref = float(diff @ XtX @ diff) / (p * res.sigma2) # MSE 取全模型的
        _assert_close(res.cooks_d[i], d_ref)
    # 高杠杆点上 1/(1-h_ii) 因子的放大效应显著，防止将来回归到错误公式
    assert res.leverage[0] > 0.5 # 确保该测试数据确实造出了高杠杆点
    wrong = res.cooks_d[0] / (1.0 - res.leverage[0]) # 错误公式的值
    assert abs(wrong - res.cooks_d[0]) / res.cooks_d[0] > 0.2


def test_input_errors():
    y = rng.normal(0, 1, 20)
    dup = np.column_stack([rng.normal(0, 1, 20), np.zeros(20)]) # 常数列共线
    with pytest.raises(ValueError):
        lin_reg(dup, y)
    with pytest.raises(ValueError):
        lin_reg(rng.normal(0, 1, 20), y[:5]) # 长度不一致
    with pytest.raises(ValueError):
        lin_reg([1.0, 2.0], [1.0, 2.0, 3.0]) # n ≤ p
    with pytest.raises(ValueError):
        lin_reg(rng.normal(0, 1, 10), rng.normal(0, 1, 10), coef_names=["只有一个"]) # 名称数不符


def test_nonlinear_matches_curve_fit():
    x = np.linspace(0, 5, 30)
    y = 3.0 * np.exp(-0.6 * x) + rng.normal(0, 0.05, size=30)
    model = lambda t, a, b: a * np.exp(-b * t) # noqa: E731
    ref_beta, ref_pcov = curve_fit(model, x, y, p0=[1.0, 0.5])
    res = nonlin_reg(model, x, y, p0=[1.0, 0.5])
    np.testing.assert_allclose(res.coefficients, ref_beta, rtol=1e-8)
    np.testing.assert_allclose(res.coef_se, np.sqrt(np.diag(ref_pcov)), rtol=1e-8)
    assert res.r_squared > 0.99
    assert res.coef_names == ["b1", "b2"]
    assert res.f_stat is None # 非线性不提供整体 F 检验


def test_nonlinear_linear_agrees_with_ols():
    """线性模型走非线性拟合器，应与 OLS 结果一致（交叉验证）。"""
    x = rng.uniform(0, 5, 30)
    y = 2.0 - 0.7 * x + rng.normal(0, 0.5, size=30)
    res_nl = nonlin_reg(lambda t, a, b: a + b * t, x, y, p0=[0.0, 0.0])
    res_ols = lin_reg(x, y)
    np.testing.assert_allclose(res_nl.coefficients, res_ols.coefficients, rtol=1e-5)
    np.testing.assert_allclose(res_nl.coef_se, res_ols.coef_se, rtol=1e-4)
