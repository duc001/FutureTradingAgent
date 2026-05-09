#!/bin/bash

# ChatDoc 智能体启动脚本
# 功能：自动检查环境并运行 ChatDoc 问答系统

echo "========================================"
echo "  ChatDoc 智能体问答系统"
echo "========================================"
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 检查 Python 是否安装
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 Python3"
    echo "   请先安装 Python 3.13+"
    exit 1
fi

echo "✓ Python 版本: $(python3 --version)"
echo ""

# 检查虚拟环境
if [ -d ".venv" ]; then
    echo "✓ 检测到虚拟环境 .venv"
    source .venv/bin/activate
else
    echo "⚠ 未检测到虚拟环境，使用系统 Python"
fi

# 检查必要的依赖包
echo "正在检查依赖..."
python3 -c "import sentence_transformers" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 错误: sentence-transformers 未安装"
    echo "   请运行: pip install sentence-transformers numpy chromadb openai"
    exit 1
fi
echo "✓ sentence-transformers 已安装"

python3 -c "import chromadb" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠ ChromaDB 未安装，将使用 NumPy"
else
    echo "✓ ChromaDB 已安装"
fi

python3 -c "import openai" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 错误: openai SDK 未安装"
    echo "   请运行: pip install openai"
    exit 1
fi
echo "✓ OpenAI SDK 已安装"

echo ""
echo "========================================"
echo "  启动 ChatDoc..."
echo "========================================"
echo ""

# 运行主程序
python3 chatdoc_real_agent.py

# 退出码
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ 程序执行完成"
else
    echo ""
    echo "❌ 程序执行失败 (退出码: $EXIT_CODE)"
fi

exit $EXIT_CODE
