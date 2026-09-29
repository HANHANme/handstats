"""GLRT 模块的测试：与 t / z 检验、scipy、手算公式三方对照。"""
import numpy as np
import pytest
from scipy import stats

from handstats.hypothesis import glrt_exponential_mean, glrt_normal_mean, glrt_test

rng = np.random.default_rng(20260922)


def _assert_close(a, b):
    assert np.isclose(a, b, rtol=1e-9, atol=1e-12), f"{a} != {b}"


def test_glrt_normal_unknown_sigma_matches_ttest():
    """核心恒等式：-2 ln λ = n·ln(1 + t²/(n-1))，精确 p = 双侧 t 检验 p。"""
    x = rng.normal(2.0, 1.5, size=25)
    mu0 = 1.2
    res = glrt_normal_mean(x, mu0)
    ref = stats.ttest_1samp(x, mu0)
    _assert_close(res.params["t"], ref.statistic)
    _assert_close(res.params["exact_p"], ref.pvalue)
    n = len(x)
    _assert_close(res.statistic, n * np.log(1 + ref.statistic**2 / (n - 1)))
    _assert_close(res.pvalue, stats.chi2.sf(res.statistic, 1)) # Wilks 渐近
    assert 0.0 < res.params["lambda"] <= 1.0


def test_glrt_normal_known_sigma_wilks_is_exact():
    """σ 已知时 z² 精确服从 χ²(1)：Wilks 渐近 p 必须等于精确 p。"""
    x = rng.normal(5.0, 2.0, size=30)
    res = glrt_normal_mean(x, 4.0, sigma=2.0)
    z = (x.mean() - 4.0) / (2.0 / np.sqrt(len(x)))
    _assert_close(res.statistic, z**2)
    _assert_close(res.pvalue, 2 * stats.norm.sf(abs(z)))
    _assert_close(res.params["exact_p"], res.pvalue)


def test_glrt_normal_mu0_equals_mean():
    """数据均值恰好等于 mu0：λ = 1，-2lnλ = 0，p = 1。"""
    res = glrt_normal_mean([1.0, 2.0, 3.0, 4.0, 5.0], 3.0)
    _assert_close(res.statistic, 0.0)
    _assert_close(res.params["lambda"], 1.0)
    _assert_close(res.pvalue, 1.0)
    assert not res.reject


def test_glrt_normal_requires_variation():
    with pytest.raises(ValueError):
        glrt_normal_mean([2.0, 2.0, 2.0], 5.0) # σ² 的 MLE = 0


def test_glrt_exponential_matches_manual_formula():
    x = rng.exponential(scale=800.0, size=24)
    theta0 = 1000.0
    n = len(x)
    u = x.mean() / theta0
    res = glrt_exponential_mean(x, theta0)
    _assert_close(res.statistic, 2 * n * (u - 1 - np.log(u)))
    _assert_close(res.params["lambda"], (u * np.exp(1 - u)) ** n)
    pivot = 2 * x.sum() / theta0
    _assert_close(
        res.params["exact_p_equal_tail"],
        min(1.0, 2 * min(stats.chi2.cdf(pivot, 2 * n), stats.chi2.sf(pivot, 2 * n))),
    )


def test_glrt_exponential_rejects_nonpositive():
    with pytest.raises(ValueError):
        glrt_exponential_mean([100.0, -5.0, 300.0], 200.0)


def test_glrt_engine_reproduces_ztest():
    """引擎 + 手算对数似然（省略公共常数）应重现 z 检验的结论。"""
    x = rng.normal(0.0, 1.0, size=40)
    sigma, mu0 = 1.0, 0.3
    sse = ((x - x.mean()) ** 2).sum()
    sse0 = ((x - mu0) ** 2).sum()
    # lnL(μ) = -Σ(xi-μ)²/(2σ²)，省略与 μ 无关的常数项
    res = glrt_test(-sse / (2 * sigma**2), -sse0 / (2 * sigma**2), df=1)
    z = (x.mean() - mu0) / (sigma / np.sqrt(len(x)))
    _assert_close(res.statistic, z**2)
    _assert_close(res.pvalue, 2 * stats.norm.sf(abs(z))) # 此例 Wilks 精确


def test_glrt_engine_input_errors():
    with pytest.raises(ValueError):
        glrt_test(-10.0, -5.0, df=1) # lnL_H0 > lnL_full，矛盾
    with pytest.raises(ValueError):
        glrt_test(-10.0, -12.0, df=0) # df < 1
