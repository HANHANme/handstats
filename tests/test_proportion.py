"""比例 z 检验的对拍测试（用公式核对 + 经典数字验收）。"""
import numpy as np
import pytest
from scipy import stats

from mystats.hypothesis import ztest_1prop, ztest_2prop


def _assert_close(a, b):
    assert np.isclose(a, b, rtol=1e-9, atol=1e-12), f"{a} != {b}"


def test_1prop_classic_coin():
    """抛硬币 100 次正面 65 次，H0: p = 0.5：z = 3，p ≈ 0.0027（能手算）。"""
    res = ztest_1prop(x=65, n=100, p0=0.5)
    _assert_close(res.statistic, 3.0)
    _assert_close(res.pvalue, 2 * stats.norm.sf(3.0))
    assert res.reject


def test_1prop_one_sided_matches_formula():
    res_g = ztest_1prop(65, 100, 0.5, alternative="greater")
    res_l = ztest_1prop(65, 100, 0.5, alternative="less")
    _assert_close(res_g.pvalue, stats.norm.sf(3.0))
    _assert_close(res_l.pvalue, stats.norm.cdf(3.0))


def test_1prop_fair_coin():
    """50/100 → z = 0，p = 1。"""
    res = ztest_1prop(x=50, n=100, p0=0.5)
    _assert_close(res.statistic, 0.0)
    _assert_close(res.pvalue, 1.0)
    assert not res.reject


def test_2prop_matches_formula():
    x1, n1, x2, n2 = 45, 80, 56, 95
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z_ref = (p1 - p2) / se
    p_ref = 2 * stats.norm.sf(abs(z_ref))
    res = ztest_2prop(x1, n1, x2, n2)
    _assert_close(res.statistic, z_ref)
    _assert_close(res.pvalue, p_ref)
    _assert_close(res.params["pooled"], pooled)


def test_prop_input_errors():
    with pytest.raises(ValueError):
        ztest_1prop(x=101, n=100, p0=0.5) # x > n
    with pytest.raises(ValueError):
        ztest_1prop(x=50.5, n=100, p0=0.5) # 非整数
    with pytest.raises(ValueError):
        ztest_1prop(x=10, n=100, p0=1.5) # p0 出界
    with pytest.raises(ValueError):
        ztest_2prop(x1=10, n1=0, x2=10, n2=20) # n1 = 0
