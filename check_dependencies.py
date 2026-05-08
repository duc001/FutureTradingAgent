"""
检查 ChatDoc 依赖安装情况
"""

import sys
import subprocess

def check_package(package_name, import_name=None):
    """检查包是否安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False

def get_installed_packages():
    """获取已安装的包列表"""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'list'],
            capture_output=True,
            text=True
        )
        return result.stdout
    except Exception as e:
        return f"Error: {e}"

def main():
    print("="*70)
    print("ChatDoc 依赖检查")
    print("="*70)
    print()
    
    # 显示 Python 信息
    print(f"Python 版本: {sys.version}")
    print(f"Python 路径: {sys.executable}")
    print()
    
    # 显示已安装的包
    print("-"*70)
    print("当前已安装的包:")
    print("-"*70)
    print(get_installed_packages())
    print()
    
    # 检查关键依赖
    print("-"*70)
    print("ChatDoc 依赖检查:")
    print("-"*70)
    
    dependencies = [
        ("sentence-transformers", "sentence_transformers", "核心 - 文本向量化"),
        ("faiss-cpu", "faiss", "核心 - 向量检索"),
        ("numpy", "numpy", "核心 - 数值计算"),
        ("PyPDF2", "PyPDF2", "可选 - PDF解析"),
        ("python-docx", "docx", "可选 - Word解析"),
        ("openai", "openai", "可选 - OpenAI LLM"),
        ("requests", "requests", "可选 - HTTP请求"),
    ]
    
    required = []
    optional = []
    missing_required = []
    missing_optional = []
    
    for pkg_name, import_name, description in dependencies:
        installed = check_package(pkg_name, import_name)
        status = "✓ 已安装" if installed else "✗ 未安装"
        
        if "可选" in description:
            optional.append((pkg_name, installed, description))
            if not installed:
                missing_optional.append(pkg_name)
        else:
            required.append((pkg_name, installed, description))
            if not installed:
                missing_required.append(pkg_name)
        
        print(f"{status:10} {pkg_name:30} ({description})")
    
    print()
    
    # 总结
    print("-"*70)
    print("安装建议:")
    print("-"*70)
    
    if missing_required:
        print("\n❌ 缺少核心依赖（必须安装）:")
        print(f"   pip install {' '.join(missing_required)}")
    
    if missing_optional:
        print(f"\n⚠ 缺少可选依赖（按需安装）:")
        print(f"   pip install {' '.join(missing_optional)}")
    
    if not missing_required and not missing_optional:
        print("\n✓ 所有依赖已安装！")
    
    print()
    print("-"*70)
    print("快速安装命令:")
    print("-"*70)
    print("\n# 安装核心依赖（必需）")
    print("pip install sentence-transformers faiss-cpu numpy")
    print("\n# 安装文档解析（可选）")
    print("pip install PyPDF2 python-docx")
    print("\n# 安装 LLM 支持（可选）")
    print("pip install openai requests")
    print()
    
    print("="*70)
    print("运行建议:")
    print("="*70)
    print("\n1. 简化版（无需依赖，立即可用）:")
    print("   python chatdoc_simple.py")
    print("\n2. 完整版（需要安装依赖）:")
    print("   python chatdoc_real_agent.py")
    print()


if __name__ == '__main__':
    main()
