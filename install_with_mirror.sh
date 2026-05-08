#!/bin/bash

# 使用清华镜像源安装 ChatDoc 依赖

echo "=========================================="
echo "使用清华镜像源安装依赖"
echo "=========================================="
echo ""

# 激活虚拟环境
source .venv/bin/activate

echo "1. 配置 pip 使用清华镜像..."
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn
echo ""

echo "2. 升级 pip..."
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
echo ""

echo "3. 安装核心依赖（必需）..."
pip install sentence-transformers numpy chromadb -i https://pypi.tuna.tsinghua.edu.cn/simple
echo ""

echo "4. 安装文档解析（可选）..."
pip install PyPDF2 python-docx -i https://pypi.tuna.tsinghua.edu.cn/simple
echo ""

echo "5. 安装 LLM 支持（可选）..."
pip install openai requests -i https://pypi.tuna.tsinghua.edu.cn/simple
echo ""

echo "6. 验证安装..."
python check_dependencies.py

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
