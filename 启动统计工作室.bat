@echo off
chcp 65001 >nul
title handstats 统计工作室

rem ===== 1. 找一个可用的 Python =====
set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY if exist "C:\Python314\python.exe" set "PY=C:\Python314\python.exe"
if not defined PY if exist "D:\Anaconda\envs\dbn-tcn\python.exe" set "PY=D:\Anaconda\envs\dbn-tcn\python.exe"
if not defined PY (
    echo [X] 没有找到 Python。请先安装：https://www.python.org/downloads/
    pause
    exit /b 1
)

rem ===== 2. 首次运行自动安装依赖（已装过则秒过）=====
%PY% -c "import handstats, streamlit" >nul 2>nul
if errorlevel 1 (
    echo 首次运行：正在安装依赖（约 1-2 分钟，走国内镜像）...
    %PY% -m pip install "handstats[shell]" -i https://mirrors.aliyun.com/pypi/simple/
    if errorlevel 1 (
        echo [X] 安装失败，请检查网络后重试。
        pause
        exit /b 1
    )
)

rem ===== 3. 启动（浏览器会自动打开，关闭本窗口即退出）=====
echo 正在启动统计工作室，浏览器将自动打开 http://localhost:8501
echo 关闭本窗口或按 Ctrl+C 即可退出。
%PY% -m handstats.shell
pause
