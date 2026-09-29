"""分布层：全包唯一的“数值引擎”接口（目前由 scipy 提供支持）。

为什么要把 scipy 包一层，而不是在各检验里直接 import scipy？
--------
1. 可替换性：以后若想练习数值计算、自己实现 t 分布的 CDF（数值积分、
   连分式展开等），只需改这一个文件，包里其他几十个模块一行不动；
2. 统一命名：全包只出现 t_sf / t_ppf 这类名字，查分布相关代码只看这里。

目前封装：正态 norm、t、卡方 chi2、F（阶段 1 的检验全都够用）。
"""
from __future__ import annotations

from scipy import stats as _sp

# ---------------- 正态分布 ----------------

def norm_cdf(x: float) -> float:
    return float(_sp.norm.cdf(x))


def norm_sf(x: float) -> float:
    return float(_sp.norm.sf(x))


def norm_ppf(q: float) -> float:
    return float(_sp.norm.ppf(q))


# ---------------- t 分布（自由度 df）----------------

def t_cdf(x: float, df: float) -> float:
    return float(_sp.t.cdf(x, df))


def t_sf(x: float, df: float) -> float:
    return float(_sp.t.sf(x, df))


def t_ppf(q: float, df: float) -> float:
    return float(_sp.t.ppf(q, df))


# ---------------- 卡方分布（自由度 df）----------------

def chi2_cdf(x: float, df: float) -> float:
    return float(_sp.chi2.cdf(x, df))


def chi2_sf(x: float, df: float) -> float:
    return float(_sp.chi2.sf(x, df))


def chi2_ppf(q: float, df: float) -> float:
    return float(_sp.chi2.ppf(q, df))


# ---------------- F 分布（分子自由度 dfn，分母自由度 dfd）----------------

def f_cdf(x: float, dfn: float, dfd: float) -> float:
    return float(_sp.f.cdf(x, dfn, dfd))


def f_sf(x: float, dfn: float, dfd: float) -> float:
    return float(_sp.f.sf(x, dfn, dfd))


def f_ppf(q: float, dfn: float, dfd: float) -> float:
    return float(_sp.f.ppf(q, dfn, dfd))


# ---------------- 学生化极差分布（组数 k，误差自由度 df；Tukey HSD 用）----------------
# scipy 1.11+ 提供；handstats 依赖 scipy>=1.8，运行环境若过旧会在调用时报错
def studentized_range_ppf(q: float, k: float, df: float) -> float:
    return float(_sp.studentized_range.ppf(q, k, df))


def studentized_range_sf(x: float, k: float, df: float) -> float:
    return float(_sp.studentized_range.sf(x, k, df))


# ---------------- 统计量 → p 值的共用出口 ----------------

_ALLOWED_DISTS = ("norm", "t", "chi2", "f")


def p_value(stat: float, alternative: str, dist: str = "t", **shape) -> float:
    """由检验统计量算 p 值——全包所有检验共用的“最后一跳”。

    参数
    ----
    stat : 检验统计量的实现值
    alternative : "less" / "greater" / "two-sided"（须已通过 validate 归一化）
    dist : 原假设下统计量服从的分布："norm" / "t" / "chi2" / "f"
    **shape : 分布的形状参数，如 df=19，或 dfn=3, dfd=16

    公式
    ----
    less:      p = P(T ≤ stat) = CDF(stat)
    greater:   p = P(T ≥ stat) = SF(stat)
    two-sided: p = 2·min(CDF(stat), SF(stat))，封顶为 1
      （对正态/t 这类对称分布即教材里的 2·P(|T| ≥ |stat|)；
       卡方/F 一般只用 greater；偏态分布的双侧公式将来需要时再细化。）
    """
    if dist not in _ALLOWED_DISTS:
        raise ValueError(f"dist 只能取 {_ALLOWED_DISTS}，收到 {dist!r}")
    d = getattr(_sp, dist)
    if alternative == "less":
        return float(d.cdf(stat, **shape))
    if alternative == "greater":
        return float(d.sf(stat, **shape))
    return float(min(1.0, 2.0 * min(d.cdf(stat, **shape), d.sf(stat, **shape))))
