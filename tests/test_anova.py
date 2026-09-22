"""方差分析的对拍测试：scipy（f_oneway/levene/tukey_hsd）+ 手算数据 +
哑变量回归交叉验证（用已对拍过的 lin_reg 反推 SS 分解）。"""
import numpy as np
import pytest
from scipy import stats

from mystats.anova import anova_oneway, anova_twoway, levene_test, tukey_hsd
from mystats.regression import lin_reg

rng = np.random.default_rng(20260922)


def _assert_close(a, b):
    assert np.isclose(a, b, rtol=1e-9, atol=1e-12), f"{a} != {b}"


def _sse(X, y):
    """含截距设计矩阵 X 的 OLS 残差平方和（借用已对拍过的 lin_reg）。"""
    fit = lin_reg(X, y)
    return float(fit.residuals @ fit.residuals)


def _dummy(idx, levels):
    """哑变量矩阵：levels 个水平 → levels-1 列（第一水平为基准）。"""
    D = np.zeros((len(idx), levels - 1))
    for lv in range(1, levels):
        D[idx == lv, lv - 1] = 1.0
    return D


def test_oneway_matches_scipy():
    groups = [rng.normal(5.0, 2.0, 12), rng.normal(6.0, 2.0, 15),
              rng.normal(5.5, 2.0, 10)]
    ref = stats.f_oneway(*groups)
    res = anova_oneway(groups)
    _assert_close(res.statistic, ref.statistic)
    _assert_close(res.pvalue, ref.pvalue)


def test_oneway_hand_verifiable():
    """均值 2/4/6、各 3 个观测：SS_A = 24，SS_E = 6，F = 12（能手算）。"""
    groups = [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]]
    res = anova_oneway(groups)
    _assert_close(res.params["ss_a"], 24.0)
    _assert_close(res.params["ss_e"], 6.0)
    _assert_close(res.params["ss_t"], 30.0)
    _assert_close(res.statistic, 12.0)
    _assert_close(res.pvalue, stats.f.sf(12.0, 2, 6))
    _assert_close(res.params["eta_sq"], 0.8)


def test_oneway_input_errors():
    with pytest.raises(ValueError):
        anova_oneway([[1.0, 2.0]]) # 只有一组
    with pytest.raises(ValueError):
        anova_oneway([[1.0], [2.0]]) # n - k = 0
    with pytest.raises(ValueError):
        anova_oneway([[1.0, 2.0], [3.0, np.nan]])
    with pytest.raises(ValueError):
        anova_oneway([[1.0, 1.0], [2.0, 2.0]]) # 组内均方 = 0


def test_levene_matches_scipy():
    groups = [rng.normal(0, 1.0, 15), rng.normal(0, 2.0, 12), rng.normal(0, 0.5, 18)]
    ref = stats.levene(*groups, center="mean")
    res = levene_test(groups)
    _assert_close(res.statistic, ref.statistic)
    _assert_close(res.pvalue, ref.pvalue)
    ref_med = stats.levene(*groups, center="median")
    res_med = levene_test(groups, center="median")
    _assert_close(res_med.statistic, ref_med.statistic)
    _assert_close(res_med.pvalue, ref_med.pvalue)


def test_tukey_matches_scipy():
    if not hasattr(stats, "tukey_hsd"): # scipy < 1.11
        pytest.skip("当前 scipy 无 tukey_hsd，跳过对拍")
    groups = [rng.normal(10, 2, 12), rng.normal(12, 2, 12), rng.normal(13, 2, 12),
              rng.normal(10.5, 2, 9)] # 不等样本量 → Tukey-Kramer
    ref = stats.tukey_hsd(*groups)
    res_list = tukey_hsd(groups)
    idx = 0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            _assert_close(res_list[idx].pvalue, ref.pvalue[i, j])
            _assert_close(res_list[idx].params["diff"],
                          groups[i].mean() - groups[j].mean())
            idx += 1
    # 拒绝判定与“置信区间不含 0”必须一致
    for res in res_list:
        lo, hi = res.params["lower"], res.params["upper"]
        assert res.reject == (lo > 0 or hi < 0)


def test_twoway_no_rep_hand():
    """[[1,2],[3,6]]：SS_A = 9，SS_B = 4，SS_E = 1（能手算）。"""
    tab = np.array([[1.0, 2.0], [3.0, 6.0]])
    res = anova_twoway(tab)
    _assert_close(res.params["ss_a"], 9.0)
    _assert_close(res.params["ss_b"], 4.0)
    _assert_close(res.params["ss_e"], 1.0)
    _assert_close(res.params["ss_t"], 14.0)
    _assert_close(res.factor_a.statistic, 9.0)
    _assert_close(res.factor_b.statistic, 4.0)
    assert res.factor_ab is None # 无重复设计没有交互检验


def test_twoway_no_rep_matches_dummy_regression():
    r, c = 3, 4
    tab = (rng.normal(5.0, 1.0, (r, c)) + np.arange(r)[:, None] * 0.8
           - np.arange(c)[None, :] * 0.3)
    res = anova_twoway(tab)
    y = tab.ravel()
    row_idx = np.repeat(np.arange(r), c)
    col_idx = np.tile(np.arange(c), r)
    A, B = _dummy(row_idx, r), _dummy(col_idx, c)
    # 注意：不自己加截距列——lin_reg 会自动加（重复加会共线）
    sse_full = _sse(np.hstack([A, B]), y) # 可加模型
    sse_woA = _sse(B, y) # 去掉 A 的模型
    sse_woB = _sse(A, y) # 去掉 B 的模型
    df_e = (r - 1) * (c - 1)
    _assert_close(res.params["ss_e"], sse_full)
    _assert_close(res.params["ss_a"], sse_woA - sse_full)
    _assert_close(res.params["ss_b"], sse_woB - sse_full)
    _assert_close(res.factor_a.statistic,
                  ((sse_woA - sse_full) / (r - 1)) / (sse_full / df_e))


def test_twoway_with_rep_matches_dummy_regression():
    r, c, m = 3, 2, 2
    data = (5.0 + np.arange(r)[:, None, None] * 0.6 - np.arange(c)[None, :, None] * 0.5
            + rng.normal(0, 0.4, (r, c, m))[:, :, :] + rng.normal(0, 0.3, (r, c, m)))
    res = anova_twoway(data)
    y = data.ravel()
    row_idx = np.repeat(np.arange(r), c * m)
    col_idx = np.tile(np.repeat(np.arange(c), m), r)
    A, B = _dummy(row_idx, r), _dummy(col_idx, c)
    AB = np.hstack([A[:, a:a + 1] * B[:, b:b + 1]
                    for a in range(r - 1) for b in range(c - 1)])
    # 注意：不自己加截距列——lin_reg 会自动加（重复加会共线）
    sse_full = _sse(np.hstack([A, B, AB]), y) # 饱和模型（拟合 = 格均值）
    sse_main = _sse(np.hstack([A, B]), y) # 可加模型
    sse_woA = _sse(B, y)
    sse_woB = _sse(A, y)
    df_e = r * c * (m - 1)
    ss_a = sse_woA - sse_main
    ss_b = sse_woB - sse_main
    ss_ab = sse_main - sse_full
    _assert_close(res.params["ss_e"], sse_full)
    _assert_close(res.params["ss_a"], ss_a)
    _assert_close(res.params["ss_b"], ss_b)
    _assert_close(res.params["ss_ab"], ss_ab)
    _assert_close(res.factor_a.statistic, (ss_a / (r - 1)) / (sse_full / df_e))
    _assert_close(res.factor_b.statistic, (ss_b / (c - 1)) / (sse_full / df_e))
    _assert_close(res.factor_ab.statistic,
                  (ss_ab / ((r - 1) * (c - 1))) / (sse_full / df_e))
    # 平衡设计的分解恒等式：SS_T = SS_A + SS_B + SS_AB + SS_E
    _assert_close(res.params["ss_t"], float(np.sum((data - data.mean()) ** 2)))
    _assert_close(res.params["ss_t"], ss_a + ss_b + ss_ab + sse_full)


def test_twoway_input_errors():
    with pytest.raises(ValueError):
        anova_twoway([1, 2, 3]) # 一维
    with pytest.raises(ValueError):
        anova_twoway(np.ones((1, 4))) # r = 1
    with pytest.raises(ValueError):
        anova_twoway(np.ones((3, 2, 1))) # m = 1 应传二维
    with pytest.raises(ValueError):
        anova_twoway(np.ones((3, 3)), names=("只有一个名字",))


def test_twoway_custom_names_in_output():
    res = anova_twoway(rng.normal(0, 1, (3, 3)), names=("温度", "压力"))
    assert "温度" in res.summary()
    assert "温度" in res.method
