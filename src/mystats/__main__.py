"""`python -m mystats`：命令行一览当前包里所有可用的统计过程。

这是未来“无代码外壳”的雏形：外壳程序同样只需要遍历注册表，
就能自动生成方法菜单——新增过程时不需要为它写任何界面代码。
"""
from mystats import __version__, list_procedures


def main() -> None:
    procs = sorted(list_procedures())
    print(f"mystats v{__version__} —— 可用统计过程（{len(procs)} 个）：")
    for i, name in enumerate(procs, start=1):
        print(f"  {i:>2}. {name}")


if __name__ == "__main__":
    main()
