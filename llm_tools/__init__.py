"""
LLM大语言模型工具集
LLM Tools - 基于comfyui_LLM_party的轻量级集成

注意：此模块提供LLM Party的引用，实际功能由独立的comfyui_LLM_party提供
"""

# 尝试导入LLM Party节点
try:
    import sys
    import os

    # 查找comfyui_LLM_party路径
    llm_party_path = None
    for path in sys.path:
        potential_path = os.path.join(path, 'custom_nodes', 'comfyui_LLM_party')
        if os.path.exists(potential_path):
            llm_party_path = potential_path
            break

    if llm_party_path and os.path.exists(os.path.join(llm_party_path, '__init__.py')):
        sys.path.insert(0, llm_party_path)
        from comfyui_LLM_party.llm import NODE_CLASS_MAPPINGS as LLM_NODE_CLASS_MAPPINGS
        from comfyui_LLM_party.llm import NODE_DISPLAY_NAME_MAPPINGS as LLM_NODE_DISPLAY_NAME_MAPPINGS
        LLM_PARTY_AVAILABLE = True
        print("✅ LLM Party 集成成功")
    else:
        LLM_PARTY_AVAILABLE = False
        LLM_NODE_CLASS_MAPPINGS = {}
        LLM_NODE_DISPLAY_NAME_MAPPINGS = {}
        print("⚠️ LLM Party 未找到，请确保已安装 comfyui_LLM_party")

except Exception as e:
    LLM_PARTY_AVAILABLE = False
    LLM_NODE_CLASS_MAPPINGS = {}
    LLM_NODE_DISPLAY_NAME_MAPPINGS = {}
    print(f"⚠️ LLM Party 集成失败: {e}")

__all__ = ["LLM_NODE_CLASS_MAPPINGS", "LLM_NODE_DISPLAY_NAME_MAPPINGS", "LLM_PARTY_AVAILABLE"]