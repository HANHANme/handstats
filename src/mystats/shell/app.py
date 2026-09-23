"""mystats 网页外壳（Streamlit）：不写代码也能用全部统计过程。

启动：python -m mystats.shell（浏览器自动打开 http://localhost:8501）

外壳只依赖两个稳定接口——这正是包骨架预留的钩子，新增过程近零改动：
1. 注册表 list_procedures()：菜单自动发现全部过程；
2. 统一结果对象：str(res) 给结论/摘要全文，res.show_steps() 给手算
   核对表（用 redirect_stdout 捕获打印，外壳不解析 steps 内部结构）。

参数表单：数据参数按 specs.py 声明渲染；标量参数由函数签名自动生成
控件（bool → 复选框，(0,1) 内的小数 → α/置信水平，alternative → 中文
下拉，值传中文由校验层归一化）。
"""
from __future__ import annotations

import contextlib
import inspect
import io

import streamlit as st

from mystats import list_procedures
from mystats.shell.parsing import PARSERS
from mystats.shell.specs import CATEGORY_ORDER, LABELS, SPECS

st.set_page_config(page_title="mystats 统计工作室", page_icon="📊", layout="wide")

PROCS = list_procedures()

st.title("mystats 统计工作室")
st.caption(
    "三步上手：① 左侧选类别和方法 → ② 粘贴数据、填参数 → ③ 点【开始分析】。"
    "结果附带【手算核对表】，可与课本 / 考试手算逐步核对。"
)

# ---------------- 侧边栏：类别 → 方法（注册表 + spec 驱动） ----------------

CATEGORIES: dict[str, list[str]] = {}
for _name, _spec in SPECS.items():
    CATEGORIES.setdefault(_spec["category"], []).append(_name)
UNSUPPORTED = sorted(set(PROCS) - set(SPECS))

with st.sidebar:
    st.header("选择统计方法")
    cat_options = [c for c in CATEGORY_ORDER if c in CATEGORIES]
    cat = st.selectbox("① 类别", cat_options + ["（更多过程）"])
    if cat == "（更多过程）":
        st.info(
            "以下过程需要自定义模型函数，请用 Python 调用（见包的 README）：\n\n"
            + "\n".join(f"- `{n}`" for n in UNSUPPORTED)
        )
        st.stop()
    proc = st.selectbox(
        "② 方法", CATEGORIES[cat], format_func=lambda n: SPECS[n]["desc"]
    )

# ---------------- 主区：数据输入 ----------------

spec = SPECS[proc]
func = PROCS[proc]

st.header(spec["desc"])
st.caption(f"过程名 `{proc}`（在 Python 中调用同名函数）")

st.subheader("③ 输入数据")
if not spec["data"]:
    st.caption("本方法无需粘贴数据，直接在下方填写数字参数即可。")

data_kwargs: dict = {}
data_ready = True
for d in spec["data"]:
    params = d["params"]
    plist = params if isinstance(params, tuple) else (params,)
    label = d["label"]
    meta = PARSERS[d["kind"]]
    if d.get("optional", False):
        show = st.checkbox(f"提供「{label}」", key=f"{proc}.{plist[0]}.on")
    else:
        show = True
    if not show:
        continue # 可选数据未勾选：不传该参数，走函数默认值
    text = st.text_area(
        label,
        height=140,
        placeholder=meta["placeholder"],
        help=meta["help"],
        key=f"{proc}.{plist[0]}",
    )
    try:
        parsed = meta["fn"](text, label)
        if isinstance(params, tuple):
            for p, v in zip(plist, parsed):
                data_kwargs[p] = v
        else:
            data_kwargs[params] = parsed
    except ValueError as e:
        st.error(str(e))
        data_ready = False

# ---------------- 主区：方法参数（签名自动生成 + overrides） ----------------

st.subheader("④ 方法参数")
scalar_kwargs: dict = {}
sig = inspect.signature(func)
data_params_bound = {
    p
    for d in spec["data"]
    for p in (d["params"] if isinstance(d["params"], tuple) else (d["params"],))
}

for pname, pinfo in sig.parameters.items():
    if pname in data_params_bound:
        continue
    key = f"{proc}.{pname}"
    label = spec.get("labels", {}).get(pname) or LABELS.get(pname, pname)
    dv = pinfo.default

    override = spec.get("overrides", {}).get(pname)
    if override is not None:
        if override["kind"] == "optional_number":
            if st.checkbox(override.get("label", label), key=key + ".on"):
                scalar_kwargs[pname] = st.number_input(
                    override.get("input_label", label),
                    value=float(override.get("value", 1.0)),
                    min_value=float(override.get("min", 0.0)),
                    key=key + ".val",
                )
        elif override["kind"] == "selectbox":
            options = override["options"] # {值: 中文标签}
            scalar_kwargs[pname] = st.selectbox(
                label,
                list(options),
                format_func=lambda v, o=options: o[v],
                key=key,
            )
        continue

    if dv is inspect.Parameter.empty:
        # 必填数字参数（mu0、成功次数、对数似然……）
        default = spec.get("defaults", {}).get(pname, 0)
        scalar_kwargs[pname] = st.number_input(label, value=default, key=key)
    elif isinstance(dv, bool):
        scalar_kwargs[pname] = st.checkbox(label, value=dv, key=key)
    elif isinstance(dv, float) and 0.0 < dv < 1.0:
        # α / 置信水平这一类
        scalar_kwargs[pname] = st.number_input(
            label, min_value=0.0, max_value=1.0, value=dv, step=0.01, key=key
        )
    elif isinstance(dv, (int, float)):
        scalar_kwargs[pname] = st.number_input(label, value=dv, key=key)
    elif isinstance(dv, str) and pname == "alternative":
        idx = {"two-sided": 0, "less": 1, "greater": 2}.get(dv, 0)
        scalar_kwargs[pname] = st.selectbox(
            label, ["双侧", "左侧", "右侧"], index=idx, key=key,
            help="备择假设 H1 的方向；中文选项会被校验层自动归一化",
        )
    else:
        pass # 默认为 None 的参数（coef_names、names 等）：不传，用函数默认

# ---------------- 运行与结果渲染 ----------------


def _capture(fn) -> str:
    """捕获 print 型输出的统一办法（show_steps / summary 都走 stdout）。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue()


def _metrics(res) -> None:
    cols = st.columns(3)
    if hasattr(res, "pvalue"): # 检验类
        cols[0].metric("统计量", f"{res.statistic:.6g}")
        cols[1].metric("p 值", f"{res.pvalue:.6g}")
        cols[2].metric(
            f"判定（α = {getattr(res, 'alpha', 0.05):g}）",
            "拒绝 H0" if res.reject else "不能拒绝 H0",
        )
    elif hasattr(res, "lower"): # 区间类
        cols[0].metric("点估计", f"{res.estimate:.6g}")
        cols[1].metric("区间下界", f"{res.lower:.6g}")
        cols[2].metric("区间上界", f"{res.upper:.6g}")
    elif getattr(res, "coefficients", None) is not None: # 回归类
        cols[0].metric("R²", f"{res.r_squared:.6g}")
        cols[1].metric("调整 R²", f"{res.adj_r_squared:.6g}")
        cols[2].metric("残差标准差", f"{res.sigma:.6g}")


def _render_single(res) -> None:
    _metrics(res)
    st.code(str(res)) # 结论/摘要全文（统一出口：__str__）
    with st.expander("手算核对表（点开逐步核对）"):
        st.code(_capture(res.show_steps))


def _render_result(res) -> None:
    st.success("分析完成")
    if isinstance(res, list): # tukey_hsd：两两比较列表
        for r in res:
            with st.expander(f"{r.method}（p = {r.pvalue:.4g}）"):
                _metrics(r)
                st.code(str(r))
                st.code(_capture(r.show_steps))
    else:
        _render_single(res)


st.divider()
if st.button("开始分析", type="primary"):
    if not data_ready:
        st.error("请先修正上方数据输入的红色提示，再点【开始分析】")
        st.stop()
    try:
        res = func(**data_kwargs, **scalar_kwargs)
    except Exception as e: # 表单填错会得到包的中文校验报错（也是教学的一部分）
        st.error(f"计算失败：{e}")
        st.stop()
    _render_result(res)
