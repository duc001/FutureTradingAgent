"""
手动下载 tiktoken 编码文件
解决网络超时问题
"""
import os
import requests

# 创建缓存目录
cache_dir = './tiktoken_cache'
os.makedirs(cache_dir, exist_ok=True)

print("正在下载 tiktoken 编码文件...")
print("如果下载失败，请尝试使用代理或手动下载")

urls = [
    "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
]

for url in urls:
    filename = os.path.basename(url)
    filepath = os.path.join(cache_dir, filename)
    
    if os.path.exists(filepath):
        print(f"✅ {filename} 已存在")
        continue
    
    try:
        print(f"⬇️  下载 {filename}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ {filename} 下载成功")
    except Exception as e:
        print(f"❌ {filename} 下载失败: {e}")
        print(f"\n💡 解决方案:")
        print(f"   1. 使用科学上网工具")
        print(f"   2. 手动下载文件并放到 {cache_dir} 目录")
        print(f"   3. 运行简化版示例: python example_selector_simple.py")

print("\n下载完成！现在可以运行主程序了。")
