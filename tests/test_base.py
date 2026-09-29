"""结果对象与注册表的行为测试。"""
import pytest

from handstats import list_procedures
from handstats._registry import register
from handstats.hypothesis import ttest_1samp


def test_show_steps_prints_handcheck(capsys):
    res = ttest_1samp([1.0, 2.0, 3.0, 4.0, 5.0], mu0=3.0)
    res.show_steps()
    out = capsys.readouterr().out
    assert "手算核对表" in out
    assert "统计量 t" in out
    assert "p =" in out


def test_str_and_reject():
    res = ttest_1samp([1, 2, 3, 4, 5], mu0=100) # 数据远离 μ0 → 拒绝
    assert res.reject is True
    res2 = ttest_1samp([1, 2, 3, 4, 5], mu0=3) # p = 1 → 不能拒绝
    assert res2.reject is False
    assert "不能拒绝" in str(res2)


def test_list_procedures_contains_builtins():
    procs = list_procedures()
    assert {"ztest_1samp", "ttest_1samp", "ttest_2samp_ind", "ci_mean"} <= set(procs)


def test_register_rejects_duplicate_names():
    with pytest.raises(ValueError):
        @register("ttest_1samp") # 已被占用的名字
        def _dummy(x):
            return x
