"""快速测试分块功能"""
import time

def split_text_simple(text: str, chunk_size: int = 800, overlap: int = 100):
    """最简单的分块实现"""
    if not text:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # 简单分割：直接按位置切分，不找分隔符
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # 移动到下一个位置
        start = end - overlap
        
        # 安全检查
        if start >= text_len or start <= 0:
            break
    
    return chunks

# 测试
if __name__ == '__main__':
    # 读取文件
    file_path = "/Users/anjuke/duchuan/工作区/2026/2602.txt"
    
    print("正在读取文件...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"文件大小: {len(content)} 字符")
    
    # 测试分块
    print("开始分块...")
    start_time = time.time()
    
    chunks = split_text_simple(content)
    
    elapsed = time.time() - start_time
    print(f"✓ 分块完成！")
    print(f"  耗时: {elapsed:.4f} 秒")
    print(f"  分块数: {len(chunks)}")
    print(f"  平均每块: {len(content) / len(chunks):.0f} 字符")
    
    # 显示前3个块的预览
    print("\n前3个块预览:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- 块 {i+1} ({len(chunk)} 字符) ---")
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
