"""阶段 2 新增区间的对拍测试：与 scipy 公式核对 + 手算性质验收。"""
import numpy as np
import pytest
from scipy import stats

from handstats.interval import ci_mean_2samp, ci_paired_diff, ci_proportion, ci_var

rng = np.random.default_rng(20260922)


def _assert_close(a, b):
    assert np.isclose(a, b, rtol=1e-9, atol=1e-12), f"{a} != {b}"


def test_ci_mean_2samp_pooled_matches_scipy():
    a = rng.normal(2.0, 1.0, size=20)
    b = rng.normal(3.5, 1.4, size=26)
    sp2 = (19 * a.var(ddof=1) + 25 * b.var(ddof=1)) / 44
    se = np.sqrt(sp2 * (1 / 20 + 1 / 26))
    ref = stats.t.interval(0.95, 44, loc=a.mean() - b.mean(), scale=se)
    res = ci_mean_2samp(a, b)
    _assert_close(res.lower, ref[0])
    _assert_close(res.upper, ref[1])
    _assert_close(res.params["df"], 44)


def test_ci_mean_2samp_welch_matches_scipy():
    a = rng.normal(0.0, 1.0, size=18)
    b = rng.normal(1.0, 2.5, size=12)
    v1n, v2n = a.var(ddof=1) / 18, b.var(ddof=1) / 12
    se = np.sqrt(v1n + v2n)
    df = se**4 / (v1n**2 / 17 + v2n**2 / 11)
    ref = stats.t.interval(0.95, df, loc=a.mean() - b.mean(), scale=se)
    res = ci_mean_2samp(a, b, equal_var=False)
    _assert_close(res.lower, ref[0])
    _assert_close(res.upper, ref[1])


def test_ci_mean_2samp_z_known_sigma():
    a = rng.normal(5.0, 2.0, size=15)
    b = rng.normal(4.0, 3.0, size=22)
    se = np.sqrt(4.0 / 15 + 9.0 / 22)
    ref = stats.norm.interval(0.90, loc=a.mean() - b.mean(), scale=se)
    res = ci_mean_2samp(a, b, confidence=0.90, sigma1=2.0, sigma2=3.0)
    _assert_close(res.lower, ref[0])
    _assert_close(res.upper, ref[1])


def test_ci_mean_2samp_sigma_pair_error():
    with pytest.raises(ValueError):
        ci_mean_2samp([1, 2, 3], [4, 5, 6], sigma1=2.0) # 只给一个 σ


def test_ci_paired_diff_matches_scipy():
    a = rng.normal(5.0, 1.0, size=16)
    b = a - rng.normal(0.8, 0.5, size=16) # 配对（同源数据）
    d = a - b
    ref = stats.t.interval(0.95, 15, loc=d.mean(), scale=d.std(ddof=1) / 4)
    res = ci_paired_diff(a, b)
    _assert_close(res.lower, ref[0])
    _assert_close(res.upper, ref[1])
    _assert_close(res.estimate, d.mean())


def test_ci_paired_diff_length_error():
    with pytest.raises(ValueError):
        ci_paired_diff([1, 2, 3], [1, 2]) # 长度不同


def test_ci_var_matches_manual_formula():
    x = rng.normal(1.0, 2.0, size=14)
    s2 = x.var(ddof=1)
    lo = 13 * s2 / stats.chi2.ppf(0.975, 13)
    hi = 13 * s2 / stats.chi2.ppf(0.025, 13)
    res = ci_var(x)
    _assert_close(res.lower, lo)
    _assert_close(res.upper, hi)
    assert res.lower < s2 < res.upper # 点估计必在区间内
    _assert_close(res.params["sd_lower"], np.sqrt(lo)) # σ 区间 = σ² 区间开方


def test_ci_proportion_matches_scipy():
    x, n = 65, 100
    phat = 0.65
    se = np.sqrt(0.65 * 0.35 / 100)
    ref = stats.norm.interval(0.95, loc=phat, scale=se)
    res = ci_proportion(x, n)
    _assert_close(res.lower, ref[0])
    _assert_close(res.upper, ref[1])


def test_ci_proportion_small_sample_warning():
    res = ci_proportion(x=3, n=100) # 成功次数 < 5 → steps 出提示
    assert any("正态近似" in line for line in res.steps)


def test_ci_proportion_input_errors():
    with pytest.raises(ValueError):
        ci_proportion(x=101, n=100) # x > n
    with pytest.raises(ValueError):
        ci_proportion(x=3.5, n=100) # 非整数
