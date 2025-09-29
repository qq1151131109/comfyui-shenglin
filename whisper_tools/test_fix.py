#!/usr/bin/env python3
"""
测试修复后的WhisperX对齐节点
"""

import sys
sys.path.append('/shenglin/ComfyUI')

import torch
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入我们的节点
from apply_whisperx_alignment import ApplyWhisperXAlignmentNode

def test_fixed_node():
    """测试修复后的节点"""
    print("🚀 测试修复后的WhisperX对齐节点")
    print("=" * 50)
    
    # 创建节点实例
    node = ApplyWhisperXAlignmentNode()
    
    # 准备测试数据
    audio_file = "/shenglin/ComfyUI/input/H_V2VinfiniteTalk_00004-audio.mp4"
    
    # 使用torchaudio加载真实音频文件
    import torchaudio
    waveform, sample_rate = torchaudio.load(audio_file)
    
    # 创建AUDIO输入（从Load Audio节点来的格式）
    audio_input = {
        "waveform": waveform,
        "sample_rate": sample_rate
    }
    
    reference_text = ""  # 空文本，让它自动识别
    model = "base"
    language = "auto"
    
    try:
        print("📝 调用WhisperX对齐节点...")
        result = node.apply_whisperx_alignment(
            audio=audio_input,
            reference_text=reference_text,
            model=model,
            language=language,
            return_char_alignments=False
        )
        
        aligned_text, segments_alignment, words_alignment, log_output = result
        
        print("✅ 节点执行成功!")
        print(f"📊 结果统计:")
        print(f"  - 最终文本长度: {len(aligned_text)} 字符")
        print(f"  - 句子对齐数: {len(segments_alignment)}")
        print(f"  - 词语对齐数: {len(words_alignment)}")
        
        if len(words_alignment) > 0:
            print(f"  - 第一个词: {words_alignment[0]}")
            print("🎉 词级对齐数据成功生成!")
        else:
            print("❌ 词级对齐数据仍为空")
            
        print("\n📋 详细日志:")
        print(log_output[-1000:])  # 显示最后1000个字符
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    # 使用真实的音频文件进行测试
    import os
    audio_file = "/shenglin/ComfyUI/input/H_V2VinfiniteTalk_00004-audio.mp4"
    if os.path.exists(audio_file):
        test_fixed_node()
    else:
        print(f"❌ 测试音频文件不存在: {audio_file}")
        print("请确保音频文件存在于正确位置")