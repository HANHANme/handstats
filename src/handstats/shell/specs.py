"""过程表单声明：外壳为每个统计过程渲染输入表单所需的元数据。

与注册表的分工（可延展性的另一半）：
- 这里每个条目的 key 必须是已注册的过程名；未列出的过程在外壳菜单的
  “（更多过程）”里列出并说明原因，不会静默消失；
- 数据参数（样本/频数/分组/二维表/x-y 两列）在这里声明绑定；
  其余标量参数（α、置信水平、μ0、下拉框……）由函数签名自动生成控件，
  个别特殊参数（可选 σ、center）用 overrides 覆盖；
- 以后新增统计过程：包本体照常 @register，外壳在这里加一个条目即可。

数据 kind 的含义见 parsing.PARSERS：
sample 一维样本 / counts 一维频数 / groups 每行一组 /
table2d 二维表 / xycols 每行一个观测（x y）
"""
from __future__ import annotations

CATEGORY_ORDER = [
    "均值检验",
    "比例检验",
    "分类数据检验",
    "方差检验",
    "GLRT 似然比检验",
    "区间估计",
    "回归分析",
    "方差分析",
]

# 标量参数的中文标签（自动控件的缺省文案，spec["labels"] 可覆盖）
LABELS = {
    "mu0": "原假设值 μ0",
    "sigma": "已知总体标准差 σ",
    "sigma1": "已知总体标准差 σ1",
    "sigma2": "已知总体标准差 σ2",
    "p0": "原假设比例 p0",
    "theta0": "原假设均值 θ0",
    "alpha": "显著性水平 α",
    "confidence": "置信水平",
    "equal_var": "两总体方差相等？（勾选 = 合并方差经典 t，不勾 = Welch）",
    "alternative": "备择假设方向",
    "x": "成功次数 x",
    "n": "试验总次数 n",
    "x1": "成功次数 x1",
    "n1": "试验次数 n1",
    "x2": "成功次数 x2",
    "n2": "试验次数 n2",
    "df": "Wilks 自由度（参数维数之差）",
    "loglik_full": "无约束最大对数似然 lnL_full",
    "loglik_null": "H0 下最大对数似然 lnL_H0",
    "n_params": "由数据估计的参数个数（扣自由度用）",
}

SPECS = {
    # ---------------- 均值检验 ----------------
    "ztest_1samp": {
        "category": "均值检验",
        "desc": "单样本 z 检验（σ 已知）",
        "data": [{"params": "x", "kind": "sample", "label": "样本数据"}],
        "defaults": {"mu0": 5.0, "sigma": 1.0},
    },
    "ttest_1samp": {
        "category": "均值检验",
        "desc": "单样本 t 检验（σ 未知，最常用）",
        "data": [{"params": "x", "kind": "sample", "label": "样本数据"}],
        "defaults": {"mu0": 5.0},
    },
    "ttest_2samp_ind": {
        "category": "均值检验",
        "desc": "独立双样本 t 检验（合并方差 / Welch）",
        "data": [
            {"params": "x1", "kind": "sample", "label": "样本 1 数据"},
            {"params": "x2", "kind": "sample", "label": "样本 2 数据"},
        ],
    },
    # ---------------- 比例检验 ----------------
    "ztest_1prop": {
        "category": "比例检验",
        "desc": "单比例 z 检验（如：硬币偏吗？次品率达标吗？）",
        "data": [],
        "defaults": {"x": 65, "n": 100, "p0": 0.5},
    },
    "ztest_2prop": {
        "category": "比例检验",
        "desc": "双比例 z 检验（两总体比例是否相同）",
        "data": [],
        "defaults": {"x1": 45, "n1": 80, "x2": 56, "n2": 95},
    },
    # ---------------- 分类数据检验 ----------------
    "chisquare_gof": {
        "category": "分类数据检验",
        "desc": "卡方拟合优度（数据是否服从给定分布）",
        "data": [
            {"params": "obs", "kind": "counts", "label": "各类别观测频数"},
            {"params": "expected", "kind": "counts",
             "label": "各类别理论概率/频数（缺省 = 均匀分布）", "optional": True},
        ],
        "defaults": {"n_params": 0},
    },
    "chisquare_ind": {
        "category": "分类数据检验",
        "desc": "卡方独立性检验（2×2 或 r×c 列联表）",
        "data": [{"params": "table", "kind": "table2d", "label": "列联表"}],
    },
    # ---------------- 方差检验 ----------------
    "ftest_2samp_var": {
        "category": "方差检验",
        "desc": "双样本方差 F 检验（方差齐性，两总体）",
        "data": [
            {"params": "x1", "kind": "sample", "label": "样本 1 数据"},
            {"params": "x2", "kind": "sample", "label": "样本 2 数据"},
        ],
    },
    # ---------------- GLRT 似然比检验 ----------------
    "glrt_test": {
        "category": "GLRT 似然比检验",
        "desc": "GLRT 通用引擎（手算两个最大对数似然，Wilks 渐近）",
        "data": [],
        "defaults": {"loglik_full": -10.0, "loglik_null": -12.0, "df": 1},
    },
    "glrt_normal_mean": {
        "category": "GLRT 似然比检验",
        "desc": "正态均值 GLRT（附 Wilks 渐近 p 与精确 t p 对比）",
        "data": [{"params": "x", "kind": "sample", "label": "样本数据"}],
        "defaults": {"mu0": 5.0},
        "overrides": {
            "sigma": {"kind": "optional_number", "label": "σ 已知？",
                      "input_label": "已知总体标准差 σ", "value": 1.0, "min": 0.001},
        },
    },
    "glrt_exponential_mean": {
        "category": "GLRT 似然比检验",
        "desc": "指数均值 GLRT（如灯泡平均寿命，渐近 vs 精确对照）",
        "data": [{"params": "x", "kind": "sample", "label": "寿命数据（全部为正）"}],
        "defaults": {"theta0": 1000.0},
    },
    # ---------------- 区间估计 ----------------
    "ci_mean": {
        "category": "区间估计",
        "desc": "均值的置信区间（σ 已知 z / 未知 t）",
        "data": [{"params": "x", "kind": "sample", "label": "样本数据"}],
        "overrides": {
            "sigma": {"kind": "optional_number", "label": "σ 已知？",
                      "input_label": "已知总体标准差 σ", "value": 1.0, "min": 0.001},
        },
    },
    "ci_mean_2samp": {
        "category": "区间估计",
        "desc": "两独立样本均值差的置信区间",
        "data": [
            {"params": "x1", "kind": "sample", "label": "样本 1 数据"},
            {"params": "x2", "kind": "sample", "label": "样本 2 数据"},
        ],
        "overrides": {
            "sigma1": {"kind": "optional_number", "label": "两个 σ 都已知？",
                       "input_label": "已知 σ1", "value": 1.0, "min": 0.001},
            "sigma2": {"kind": "optional_number", "label": "（σ1 与 σ2 需同时给出）",
                       "input_label": "已知 σ2", "value": 1.0, "min": 0.001},
        },
    },
    "ci_paired_diff": {
        "category": "区间估计",
        "desc": "配对差值的置信区间（前后对照等）",
        "data": [
            {"params": "x1", "kind": "sample", "label": "处理后 / 后测数据"},
            {"params": "x2", "kind": "sample", "label": "处理前 / 前测数据（与左侧等长）"},
        ],
    },
    "ci_var": {
        "category": "区间估计",
        "desc": "方差的置信区间（卡方区间，不对称）",
        "data": [{"params": "x", "kind": "sample", "label": "样本数据"}],
    },
    "ci_proportion": {
        "category": "区间估计",
        "desc": "比例的置信区间（Wald 大样本）",
        "data": [],
        "defaults": {"x": 3, "n": 100},
    },
    # ---------------- 回归分析 ----------------
    "lin_reg": {
        "category": "回归分析",
        "desc": "一元线性回归（最小二乘 + t/F 检验 + 预测）",
        "data": [{"params": ("x", "y"), "kind": "xycols",
                  "label": "观测数据（每行：自变量x 因变量y）"}],
    },
    # ---------------- 方差分析 ----------------
    "anova_oneway": {
        "category": "方差分析",
        "desc": "单因素方差分析（k 组均值全相等吗？）",
        "data": [{"params": "groups", "kind": "groups", "label": "分组数据（每组一行）"}],
    },
    "levene_test": {
        "category": "方差分析",
        "desc": "Levene 方差齐性检验（ANOVA 前提检查，k 组）",
        "data": [{"params": "groups", "kind": "groups", "label": "分组数据（每组一行）"}],
        "overrides": {
            "center": {"kind": "selectbox", "options": {
                "mean": "均值中心（经典 Levene）",
                "median": "中位数中心（Brown-Forsythe，更稳健）",
            }},
        },
    },
    "tukey_hsd": {
        "category": "方差分析",
        "desc": "Tukey HSD 事后多重比较（ANOVA 拒绝后：哪两组不同？）",
        "data": [{"params": "groups", "kind": "groups", "label": "分组数据（每组一行）"}],
    },
    "anova_twoway": {
        "category": "方差分析",
        "desc": "双因素方差分析（无重复设计：行 = 因素A，列 = 因素B）",
        "data": [{"params": "data", "kind": "table2d",
                  "label": "双因素表（等重复的三维设计暂请用代码调用）"}],
    },
}
