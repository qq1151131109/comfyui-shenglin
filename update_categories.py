#!/usr/bin/env python3
"""
统一ComfyUI分组工具
将所有Shenglin节点统一到一个主分组下
"""

import os
import re
from pathlib import Path

def update_category_in_file(file_path, new_category_mapping):
    """更新单个文件中的分组"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        modified = False

        # 查找所有CATEGORY定义
        category_pattern = r'CATEGORY\s*=\s*["\']([^"\']+)["\']'

        def replace_category(match):
            nonlocal modified
            old_category = match.group(1)

            # 根据映射表替换
            for old_pattern, new_category in new_category_mapping.items():
                if old_pattern in old_category:
                    modified = True
                    print(f"  {old_category} → {new_category}")
                    return f'CATEGORY = "{new_category}"'

            return match.group(0)  # 如果没有匹配，保持原样

        content = re.sub(category_pattern, replace_category, content)

        # 如果有修改，写回文件
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True

        return False

    except Exception as e:
        print(f"❌ 更新文件失败 {file_path}: {e}")
        return False

def main():
    """主函数"""
    print("🔧 统一ComfyUI Shenglin分组")
    print("=" * 50)

    # 新的统一分组方案
    category_mapping = {
        # RunningHub相关 → 主要功能
        "🎨 Shenglin/RunningHub": "🔥 Shenglin/图像生成",
        "⚙️ Shenglin/RunningHub/Tools": "🔥 Shenglin/工具",

        # 音频相关 → 音频处理
        "🎵 Shenglin/Audio": "🔥 Shenglin/音频处理",

        # 视频相关 → 视频处理
        "🎬 Shenglin/Video": "🔥 Shenglin/视频处理"
    }

    # 获取所有Python文件
    base_dir = Path(__file__).parent
    python_files = list(base_dir.rglob("*.py"))

    # 过滤掉当前脚本和__pycache__
    python_files = [f for f in python_files if f.name != "update_categories.py" and "__pycache__" not in str(f)]

    print(f"📁 找到 {len(python_files)} 个Python文件")

    updated_files = 0
    total_changes = 0

    for file_path in python_files:
        print(f"\\n📝 检查文件: {file_path.relative_to(base_dir)}")

        if update_category_in_file(file_path, category_mapping):
            updated_files += 1
            total_changes += 1

    print("\\n" + "=" * 50)
    print(f"✅ 统一分组完成!")
    print(f"📊 更新了 {updated_files} 个文件")

    print("\\n🎯 新的统一分组结构:")
    print("🔥 Shenglin/")
    print("├── 图像生成 (原RunningHub主要功能)")
    print("├── 工具 (原RunningHub Tools)")
    print("├── 音频处理 (原Audio)")
    print("└── 视频处理 (原Video)")

    print("\\n💡 现在所有节点都在统一的'🔥 Shenglin'主分组下！")

    return updated_files > 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)