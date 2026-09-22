"""过程注册表：包里所有统计过程的“花名册”。

设计目的（可延展性的关键一环）
--------
- 每写一个新过程（检验/区间/回归……），只要在函数头上加
  @register("名字")，它就自动出现在 mystats.list_procedures() 里；
- 将来做 CLI / GUI / 自动生成文档时，遍历 PROCEDURES 即可，零改动；
- 重复登记同名过程会直接抛错，防止笔误悄悄覆盖别人的函数。
"""
from __future__ import annotations

from typing import Callable

# {过程名: 函数}。包内模块通过 @register 写入，外部通过 list_procedures() 读取。
PROCEDURES: dict[str, Callable] = {}


def register(name: str) -> Callable:
    """装饰器：把一个统计过程登记进注册表。

    用法::

        @register("ttest_1samp")
        def ttest_1samp(x, mu0, ...):
            ...
    """

    def decorator(func: Callable) -> Callable:
        if name in PROCEDURES:
            raise ValueError(f"过程名 {name!r} 已被注册，请换一个名字")
        PROCEDURES[name] = func
        func._procedure_name = name # 反查用：从函数找到它的注册名
        return func

    return decorator
