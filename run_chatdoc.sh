#!/bin/bash

# ChatDoc 快速启动脚本

echo "=========================================="
echo "ChatDoc 智能体 - 快速启动"
echo "=========================================="
echo ""

# 激活虚拟环境
source .venv/bin/activate

# 检查 Python
echo "Python 版本:"
python --version
echo ""

# 运行依赖检查
echo "检查依赖..."
python check_dependencies.py
echo ""

# 询问用户选择
echo "请选择运行模式:"
echo "1. 完整版 ChatDoc (需要安装依赖)"
echo "2. 简化版 ChatDoc (无需依赖，立即可用)"
echo "3. 仅检查依赖"
echo ""
read -p "请输入选项 (1/2/3): " choice

case $choice in
    1)
        echo ""
        echo "启动完整版 ChatDoc..."
        python chatdoc_real_agent.py
        ;;
    2)
        echo ""
        echo "启动简化版 ChatDoc..."
        python chatdoc_simple.py
        ;;
    3)
        echo ""
        echo "依赖检查已完成"
        ;;
    *)
        echo "无效选项"
        ;;
esac

echo ""
echo "=========================================="
