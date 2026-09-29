"""外壳（Streamlit）的测试：数据解析器 + spec 与注册表/签名的一致性 +
全支持过程冒烟。刻意不依赖 streamlit——parsing/specs 与 UI 分离的目的之一。
"""
import numpy as np
import pytest
from inspect import signature

from handstats import list_procedures
from handstats.shell.parsing import (
    parse_groups,
    parse_sample,
    parse_table2d,
    parse_xycols,
)
from handstats.shell.specs import SPECS


def test_parse_sample_mixed_separators():
    # 空格、中英文逗号、分号、顿号混用都能解析
    assert parse_sample("5.1, 4.9  5.3；5.0，4.8、5.05") == [
        5.1, 4.9, 5.3, 5.0, 4.8, 5.05,
    ]


def test_parse_sample_rejects_bad_input():
    with pytest.raises(ValueError, match="无法识别"):
        parse_sample("1 2 abc 3")
    with pytest.raises(ValueError, match="请先粘贴"):
        parse_sample("   \n  ")


def test_parse_groups_each_line_one_group():
    groups = parse_groups("75 80 68\n70 78 74\n\n82 79 88") # 空行应被忽略
    assert groups == [[75, 80, 68], [70, 78, 74], [82, 79, 88]]
    with pytest.raises(ValueError, match="至少需要 2 行"):
        parse_groups("1 2 3")


def test_parse_table2d_shape_and_ragged_rows():
    tab = parse_table2d("45 55\n30 70")
    assert tab.shape == (2, 2)
    assert np.allclose(tab, [[45, 55], [30, 70]])
    with pytest.raises(ValueError, match="每行的数值个数必须相同"):
        parse_table2d("45 55 60\n30 70") # 第二行少一个


def test_parse_xycols_splits_columns():
    x, y = parse_xycols("1.2 5.3\n2.0 6.1\n2.8 7.0")
    assert np.allclose(x, [1.2, 2.0, 2.8])
    assert np.allclose(y, [5.3, 6.1, 7.0])
    with pytest.raises(ValueError, match="必须是两列"):
        parse_xycols("1 2 3\n4 5 6\n7 8 9") # 三列


def test_specs_are_registered_procedures():
    procs = set(list_procedures())
    assert set(SPECS) <= procs, "SPEC 里有未注册的过程名"
    # 唯一不支持表单化的是 nonlin_reg（需要自定义模型函数）
    assert procs - set(SPECS) == {"nonlin_reg"}


def test_spec_data_params_exist_in_signatures():
    """spec 声明的数据参数名必须真实存在于函数签名（防改名后悄悄失配）。"""
    procs = list_procedures()
    for name, spec in SPECS.items():
        sig_params = set(signature(procs[name]).parameters)
        for d in spec["data"]:
            params = d["params"]
            for p in (params if isinstance(params, tuple) else (params,)):
                assert p in sig_params, f"{name} 的 spec 绑定了不存在的参数 {p}"


# 每个支持过程的最小示例——外壳绑定的参数组合必须真的能跑通
EXAMPLES = {
    "ztest_1samp": {"x": [5.1, 4.9, 5.3], "mu0": 5.0, "sigma": 1.0},
    "ttest_1samp": {"x": [1, 2, 3, 4, 5], "mu0": 3.0},
    "ttest_2samp_ind": {"x1": [1, 2, 3], "x2": [3, 4, 5]},
    "ztest_1prop": {"x": 65, "n": 100, "p0": 0.5},
    "ztest_2prop": {"x1": 45, "n1": 80, "x2": 56, "n2": 95},
    "chisquare_gof": {"obs": [8, 13, 9, 12, 7, 11]},
    "chisquare_ind": {"table": [[45, 55], [30, 70]]},
    "ftest_2samp_var": {"x1": [1, 2, 3, 4], "x2": [2, 4, 6, 8]},
    "glrt_test": {"loglik_full": -10.0, "loglik_null": -12.0, "df": 1},
    "glrt_normal_mean": {"x": [5.1, 4.9, 5.3, 5.2], "mu0": 5.0},
    "glrt_exponential_mean": {"x": [900, 1000, 1100, 1050], "theta0": 1000.0},
    "ci_mean": {"x": [1, 2, 3, 4, 5]},
    "ci_mean_2samp": {"x1": [1, 2, 3], "x2": [3, 4, 5]},
    "ci_paired_diff": {"x1": [1, 2, 3, 4], "x2": [2, 3, 4, 5]},
    "ci_var": {"x": [1, 2, 3, 4, 5]},
    "ci_proportion": {"x": 3, "n": 100},
    "lin_reg": {"x": [1, 2, 3, 4, 5], "y": [2.1, 2.9, 4.2, 5.1, 5.9]},
    "anova_oneway": {"groups": [[1, 2, 3], [3, 4, 5], [5, 6, 7]]},
    "levene_test": {"groups": [[1, 2, 3], [3, 4, 5]]},
    "tukey_hsd": {"groups": [[75, 80, 68, 72, 85], [70, 78, 74, 69, 81],
                             [82, 79, 88, 76, 90]]},
    "anova_twoway": {"data": [[1, 2], [3, 6]]},
}


def test_all_supported_procedures_run_with_examples():
    procs = list_procedures()
    for name, kwargs in EXAMPLES.items():
        res = procs[name](**kwargs)
        assert res is not None, f"{name} 未返回结果"
