"""
批量音频预览节点
专门处理多个独立音频的播放预览
"""

import os
import tempfile
import torch
import torchaudio
import folder_paths
import json
import random
from typing import List, Dict, Any

class BatchAudioPreview:
    """
    批量音频预览节点

    接收音频列表，为每个音频生成独立的预览文件
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_list": ("*", {"tooltip": "批量音频列表"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "preview_batch_audio"
    OUTPUT_NODE = True
    CATEGORY = "🎵 Shenglin/Audio"
    DESCRIPTION = "预览批量音频，每个音频独立播放"

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

    def preview_batch_audio(self, audio_list, prompt=None, extra_pnginfo=None):
        """
        批量音频预览处理
        """
        if not isinstance(audio_list, list):
            # 如果输入不是列表，尝试转换
            if hasattr(audio_list, '__iter__'):
                audio_list = list(audio_list)
            else:
                audio_list = [audio_list]

        results = []

        for i, audio_item in enumerate(audio_list):
            try:
                # 处理每个音频项
                if isinstance(audio_item, dict) and "waveform" in audio_item:
                    # 标准的ComfyUI音频格式
                    waveform = audio_item["waveform"]
                    sample_rate = audio_item["sample_rate"]

                    # 如果是batch格式，取第一个
                    if len(waveform.shape) == 3:
                        waveform = waveform[0]

                    # 生成临时文件名
                    filename = f"batch_audio_{self.prefix_append}_{i:03d}.wav"
                    filepath = os.path.join(self.output_dir, filename)

                    # 保存音频文件
                    torchaudio.save(filepath, waveform.cpu(), sample_rate)

                    results.append({
                        "filename": filename,
                        "subfolder": "",
                        "type": self.type,
                        "format": "audio/wav"
                    })

                    print(f"🎵 保存音频 {i+1}: {filename}")

            except Exception as e:
                print(f"❌ 音频 {i+1} 处理失败: {str(e)}")
                continue

        if not results:
            # 如果没有成功的音频，创建一个静音占位
            filename = f"batch_audio_{self.prefix_append}_empty.wav"
            filepath = os.path.join(self.output_dir, filename)

            # 创建1秒静音
            sample_rate = 32000
            silent_waveform = torch.zeros(1, sample_rate)
            torchaudio.save(filepath, silent_waveform, sample_rate)

            results.append({
                "filename": filename,
                "subfolder": "",
                "type": self.type,
                "format": "audio/wav"
            })

        print(f"🎵 批量音频预览完成: 生成 {len(results)} 个音频文件")

        # 返回UI结果
        return {"ui": {"audio": results}}


# 注册节点
NODE_CLASS_MAPPINGS = {
    "BatchAudioPreview": BatchAudioPreview
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchAudioPreview": "🎵 批量音频预览"
}