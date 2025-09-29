#!/usr/bin/env python3
"""
WhisperX调试脚本
系统性测试每个环节，找出问题所在
"""

import os
import sys
import json
import logging
import traceback

# 添加ComfyUI路径
sys.path.append('/shenglin/ComfyUI')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """测试依赖导入"""
    print("=" * 50)
    print("🔍 测试1: 依赖导入检查")
    print("=" * 50)
    
    try:
        import whisperx
        print(f"✅ WhisperX版本: {getattr(whisperx, '__version__', 'Unknown')}")
    except ImportError as e:
        print(f"❌ WhisperX导入失败: {e}")
        return False
    
    try:
        import pyannote.audio
        print(f"✅ Pyannote版本: {getattr(pyannote.audio, '__version__', 'Unknown')}")
    except ImportError as e:
        print(f"❌ Pyannote导入失败: {e}")
    
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA版本: {torch.version.cuda}")
    except ImportError as e:
        print(f"❌ PyTorch导入失败: {e}")
    
    return True

def test_whisperx_basic():
    """测试WhisperX基础功能"""
    print("\n" + "=" * 50)
    print("🔍 测试2: WhisperX基础功能")
    print("=" * 50)
    
    try:
        import whisperx
        import torch
        
        # 测试模型加载
        print("📥 加载WhisperX模型...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        model = whisperx.load_model("base", device=device)
        print("✅ WhisperX模型加载成功")
        
        # 测试对齐模型加载
        print("📥 加载对齐模型...")
        try:
            align_model, metadata = whisperx.load_align_model(language_code="en", device=device)
            print("✅ 对齐模型加载成功")
            print(f"📋 对齐模型metadata keys: {list(metadata.keys()) if metadata else 'None'}")
            return model, align_model, metadata
        except Exception as e:
            print(f"❌ 对齐模型加载失败: {e}")
            print(f"📋 错误详情: {traceback.format_exc()}")
            return model, None, None
            
    except Exception as e:
        print(f"❌ WhisperX基础功能测试失败: {e}")
        print(f"📋 错误详情: {traceback.format_exc()}")
        return None, None, None

def create_test_audio():
    """创建测试音频文件"""
    print("\n" + "=" * 50)
    print("🔍 测试3: 创建测试音频")
    print("=" * 50)
    
    # 查找现有的音频文件
    test_paths = [
        "/shenglin/ComfyUI/input/H_V2VinfiniteTalk_00004-audio.mp4",
        "/shenglin/ComfyUI/temp/whisperx_*.wav"
    ]
    
    for path_pattern in test_paths:
        if '*' in path_pattern:
            import glob
            files = glob.glob(path_pattern)
            if files:
                audio_file = files[0]
                print(f"✅ 找到测试音频: {audio_file}")
                return audio_file
        else:
            if os.path.exists(path_pattern):
                print(f"✅ 找到测试音频: {path_pattern}")
                return path_pattern
    
    print("❌ 未找到测试音频文件")
    return None

def test_whisperx_transcription(model, audio_file):
    """测试WhisperX转录"""
    print("\n" + "=" * 50)
    print("🔍 测试4: WhisperX转录")
    print("=" * 50)
    
    if not model or not audio_file:
        print("❌ 缺少必要参数")
        return None
    
    try:
        print(f"🎵 转录音频: {os.path.basename(audio_file)}")
        result = model.transcribe(audio_file, batch_size=16)
        
        print("✅ 转录成功!")
        print(f"📋 结果keys: {list(result.keys())}")
        
        segments = result.get('segments', [])
        print(f"📋 segments数量: {len(segments)}")
        
        if segments:
            first_seg = segments[0]
            print(f"📋 第一个segment keys: {list(first_seg.keys())}")
            print(f"📋 第一个segment内容: {first_seg.get('text', 'N/A')[:100]}")
            
            # 检查是否有words字段
            if 'words' in first_seg:
                words = first_seg['words']
                print(f"📋 第一个segment的words数量: {len(words) if words else 0}")
                if words:
                    print(f"📋 第一个word: {words[0]}")
        
        return result
        
    except Exception as e:
        print(f"❌ 转录失败: {e}")
        print(f"📋 错误详情: {traceback.format_exc()}")
        return None

def test_whisperx_alignment(transcribe_result, align_model, metadata, audio_file):
    """测试WhisperX对齐"""
    print("\n" + "=" * 50)
    print("🔍 测试5: WhisperX对齐")
    print("=" * 50)
    
    if not all([transcribe_result, align_model, metadata, audio_file]):
        print("❌ 缺少必要参数")
        return None
    
    try:
        import torch
        import whisperx
        
        segments = transcribe_result.get('segments', [])
        if not segments:
            print("❌ 没有segments可以对齐")
            return None
        
        print(f"🎯 对齐 {len(segments)} 个segments")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 执行对齐
        aligned_result = whisperx.align(
            segments,
            align_model,
            metadata,
            audio_file,
            device,
            return_char_alignments=False
        )
        
        print("✅ 对齐成功!")
        print(f"📋 对齐结果keys: {list(aligned_result.keys())}")
        
        # 详细分析对齐结果
        aligned_segments = aligned_result.get("segments", [])
        word_segments = aligned_result.get("word_segments", [])
        
        print(f"📋 对齐后segments数量: {len(aligned_segments)}")
        print(f"📋 word_segments数量: {len(word_segments)}")
        
        # 检查第一个对齐后的segment
        if aligned_segments:
            first_aligned = aligned_segments[0]
            print(f"📋 第一个对齐segment keys: {list(first_aligned.keys())}")
            
            if 'words' in first_aligned and first_aligned['words']:
                words_in_segment = first_aligned['words']
                print(f"📋 第一个segment中的words数量: {len(words_in_segment)}")
                print(f"📋 第一个word详情: {words_in_segment[0] if words_in_segment else 'N/A'}")
        
        return aligned_result
        
    except Exception as e:
        print(f"❌ 对齐失败: {e}")
        print(f"📋 错误详情: {traceback.format_exc()}")
        return None

def test_data_extraction(aligned_result):
    """测试数据提取"""
    print("\n" + "=" * 50)
    print("🔍 测试6: 数据提取分析")
    print("=" * 50)
    
    if not aligned_result:
        print("❌ 没有对齐结果")
        return
    
    # 方法1: 从word_segments提取
    word_segments = aligned_result.get("word_segments", [])
    print(f"📋 方法1 - word_segments: {len(word_segments)} 个词")
    
    # 方法2: 从segments中的words提取
    segments = aligned_result.get("segments", [])
    total_words_from_segments = 0
    
    for i, seg in enumerate(segments):
        words_in_seg = seg.get('words', [])
        if words_in_seg:
            total_words_from_segments += len(words_in_seg)
            if i == 0:  # 显示第一个的详情
                print(f"📋 第一个segment的words示例: {words_in_seg[0]}")
    
    print(f"📋 方法2 - 从segments提取words: {total_words_from_segments} 个词")
    
    # 分析哪种方法有数据
    if word_segments:
        print("✅ 建议使用: word_segments字段")
        return word_segments
    elif total_words_from_segments > 0:
        print("✅ 建议使用: segments中的words字段")
        words = []
        for seg in segments:
            if 'words' in seg and seg['words']:
                words.extend(seg['words'])
        return words
    else:
        print("❌ 两种方法都没有找到词级数据")
        return []

def main():
    """主测试函数"""
    print("🚀 开始WhisperX全面测试")
    print("目标：找出words为0的根本原因")
    
    # 测试1: 依赖导入
    if not test_imports():
        return
    
    # 测试2: WhisperX基础功能
    model, align_model, metadata = test_whisperx_basic()
    if not model:
        return
    
    # 测试3: 找到测试音频
    audio_file = create_test_audio()
    if not audio_file:
        print("💡 请将测试音频文件放在 /shenglin/ComfyUI/input/ 目录下")
        return
    
    # 测试4: 转录
    transcribe_result = test_whisperx_transcription(model, audio_file)
    if not transcribe_result:
        return
    
    # 测试5: 对齐(如果有对齐模型)
    if align_model and metadata:
        aligned_result = test_whisperx_alignment(transcribe_result, align_model, metadata, audio_file)
        if aligned_result:
            # 测试6: 数据提取
            words = test_data_extraction(aligned_result)
            
            print("\n" + "=" * 50)
            print("🎉 测试完成 - 结果总结")
            print("=" * 50)
            print(f"✅ 最终提取到的words数量: {len(words)}")
            if words:
                print(f"✅ 第一个word示例: {words[0]}")
                print("✅ 词级对齐数据可用!")
            else:
                print("❌ 未能提取到词级对齐数据")
                print("💡 这就是为什么字幕渲染没有数据的原因")
        else:
            print("❌ 对齐步骤失败，这可能是根本原因")
    else:
        print("❌ 对齐模型加载失败，无法进行对齐测试")
        print("💡 这可能是words为0的根本原因")

if __name__ == "__main__":
    main()