"""`python -m mystats.shell`：启动无代码网页外壳（Streamlit）。"""
import os
import sys


def main() -> None:
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        sys.exit(
            "未安装 streamlit。请先运行：python -m pip install mystats[shell]"
        )
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    # 命令行参数原样透传给 streamlit（如 --server.port 8501）
    sys.argv = [sys.argv[0], "run", app_path, *sys.argv[1:]]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
