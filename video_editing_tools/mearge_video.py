import os
import glob
import tempfile
from pathlib import Path
import ffmpeg
import folder_paths
import shutil

class VideoMergeNode:
    """
    视频合并节点
    将素材视频和游戏视频按照指定位置关系合并
    支持上下位置合并，自动处理尺寸和时间轴
    """
    
    @classmethod
    def get_input_folders(cls):
        """获取输入目录下的所有子文件夹"""
        try:
            input_dir = folder_paths.get_input_directory()
            if not os.path.exists(input_dir):
                return ["input"]
            
            folders = ["input"]  # 默认包含根目录
            for item in os.listdir(input_dir):
                item_path = os.path.join(input_dir, item)
                if os.path.isdir(item_path):
                    folders.append(item)
            
            return sorted(folders)
        except Exception:
            return ["input"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "material_folder": (cls.get_input_folders(), {"default": "input", "tooltip": "选择素材视频所在的文件夹"}),
                "game_folder": (cls.get_input_folders(), {"default": "input", "tooltip": "选择游戏视频所在的文件夹"}),
                "position": (["up", "down"], {"default": "up", "tooltip": "up: 素材视频在上方, down: 素材视频在下方"}),
                "audio_mode": (["game_only", "mix"], {"default": "game_only", "tooltip": "game_only: 只使用游戏音频, mix: 素材和游戏音频混音"}),
                "material_audio_volume": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "素材音频音量占比 (0.0-2.0)"}),
                "game_audio_volume": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "游戏音频音量占比 (0.0-2.0)"}),
                "output_folder_name": ("STRING", {"default": "merged_videos", "multiline": False, "tooltip": "输出文件夹名称，合并后的视频将保存到此文件夹"}),
            },
            "optional": {
                "material_path": ("STRING", {"default": "", "multiline": False, "tooltip": "直接输入素材文件夹的完整路径，优先级高于下拉框选择"}),
                "game_path": ("STRING", {"default": "", "multiline": False, "tooltip": "直接输入游戏文件夹的完整路径，优先级高于下拉框选择"}),
                "gif_path": ("STRING", {"default": "", "multiline": False, "tooltip": "GIF动态图路径，如果存在则在素材和游戏视频结合处叠加显示"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_paths",)
    FUNCTION = "merge_videos"
    CATEGORY = "video_editing"
    
    def get_video_info(self, video_path, threshold_db=-60.0):
        """获取视频信息"""
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
            
            if not video_stream:
                return None
            
            print(f"音频检测 - 文件: {os.path.basename(video_path)}")
            print(f"  静音阈值: {threshold_db} dB")
            
            # 第一步：检查是否有音轨
            has_audio_track = audio_stream is not None
            print(f"  第一步 - 音轨检测: {'有音轨' if has_audio_track else '无音轨'}")
            
            if audio_stream:
                print(f"    音频编码: {audio_stream.get('codec_name', '未知')}")
                print(f"    音频时长: {audio_stream.get('duration', '未知')}")
                print(f"    采样率: {audio_stream.get('sample_rate', '未知')}")
            
            # 第二步：如果有音轨，检测音量
            has_audio = False
            if has_audio_track:
                print(f"  第二步 - 音量检测:")
                try:
                    # 使用ffmpeg-python分析音量
                    input_stream = ffmpeg.input(video_path)
                    audio_stream_test = input_stream.audio
                    
                    # 创建音量检测流
                    volume_stream = audio_stream_test.filter('volumedetect')
                    
                    # 输出到null设备进行分析
                    output_stream = ffmpeg.output(volume_stream, 'pipe:', format='null')
                    
                    # 运行分析
                    process = ffmpeg.run(output_stream, capture_stdout=True, capture_stderr=True, quiet=True)
                    
                    # 解析stderr中的音量信息
                    stderr_output = process[1].decode('utf-8') if process[1] else ''
                    
                    if 'mean_volume:' in stderr_output:
                        # 提取音量信息
                        lines = stderr_output.split('\n')
                        for line in lines:
                            if 'mean_volume:' in line:
                                volume_str = line.split('mean_volume:')[1].strip()
                                try:
                                    volume_db = float(volume_str.split()[0])
                                    print(f"    平均音量: {volume_db} dB")
                                    
                                    # 如果音量大于阈值，认为有声音
                                    has_audio = volume_db > threshold_db
                                    print(f"    音量判断: {'有声音' if has_audio else '静音'} (阈值: {threshold_db} dB)")
                                    break
                                except Exception as parse_e:
                                    print(f"    音量解析失败: {parse_e}")
                                    has_audio = True  # 解析失败时默认认为有声音
                                    print(f"    音量判断: 有声音（解析失败）")
                                    break
                        else:
                            has_audio = True
                            print(f"    音量判断: 有声音（未找到音量信息）")
                    else:
                        has_audio = True
                        print(f"    音量判断: 有声音（无音量信息）")
                        
                except Exception as volume_e:
                    print(f"    音量检测异常: {volume_e}")
                    has_audio = True  # 检测失败时默认认为有声音
                    print(f"    音量判断: 有声音（检测失败）")
            else:
                print(f"  第二步 - 跳过音量检测（无音轨）")
                has_audio = False
            
            print(f"  最终判断: {'有音频' if has_audio else '无音频'}")
            
            return {
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'duration': float(probe['format']['duration']),
                'fps': eval(video_stream['r_frame_rate']),
                'has_audio': has_audio
            }
        except Exception as e:
            print(f"获取视频信息失败 {video_path}: {str(e)}")
            return None
    
    def resize_video_to_width(self, input_path, target_width, output_path):
        """将视频缩放到指定宽度，保持宽高比"""
        try:
            video_info = self.get_video_info(input_path)
            if not video_info:
                return False
                
            original_width = video_info['width']
            original_height = video_info['height']
            
            # 计算新的高度
            new_height = int((target_width * original_height) / original_width)
            
            # 确保高度是偶数（视频编码要求）
            if new_height % 2 != 0:
                new_height += 1
            
            # 使用ffmpeg进行缩放，兼容有声音和没有声音的情况
            input_stream = ffmpeg.input(input_path)
            
            print(f"缩放视频: {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
            print(f"  原始尺寸: {original_width}x{original_height}")
            print(f"  目标尺寸: {target_width}x{new_height}")
            print(f"  有音频: {video_info['has_audio']}")
            
            if video_info['has_audio']:
                # 有音频的情况
                print("  使用音频输出模式")
                (
                    ffmpeg
                    .output(
                        input_stream.video.filter('scale', target_width, new_height),
                        input_stream.audio,  # 保留音频
                        output_path, 
                        vcodec='libx264', 
                        acodec='aac', 
                        preset='medium'
                    )
                    .overwrite_output()
                    .run(quiet=False)  # 显示详细信息
                )
            else:
                # 没有音频的情况
                print("  使用无音频输出模式")
                (
                    ffmpeg
                    .output(
                        input_stream.video.filter('scale', target_width, new_height),
                        output_path, 
                        vcodec='libx264', 
                        preset='medium'
                    )
                    .overwrite_output()
                    .run(quiet=False)  # 显示详细信息
                )
            
            return True
        except Exception as e:
            print(f"视频缩放失败 {input_path}: {str(e)}")
            return False
    
    def merge_videos_vertically(self, material_path, game_path, output_path, position="up", audio_mode="game_only", material_audio_volume=0.5, game_audio_volume=0.5, gif_path=""):
        """垂直合并视频"""
        try:
            # 获取视频信息用于调试
            material_info = self.get_video_info(material_path)
            game_info = self.get_video_info(game_path)
            
            print(f"垂直合并视频:")
            print(f"  素材: {os.path.basename(material_path)} (有音频: {material_info['has_audio'] if material_info else '未知'})")
            print(f"  游戏: {os.path.basename(game_path)} (有音频: {game_info['has_audio'] if game_info else '未知'})")
            print(f"  位置: {position}, 音频模式: {audio_mode}")
            
            # 根据位置确定视频顺序并合并
            if position == "up":
                # 素材在上，游戏在下
                material_input = ffmpeg.input(material_path)
                game_input = ffmpeg.input(game_path)
                
                # 使用vstack filter合并视频
                video_output = ffmpeg.filter([material_input.video, game_input.video], 'vstack', inputs=2)
                
                # 检查是否需要叠加GIF
                if gif_path and gif_path.strip() and os.path.exists(gif_path.strip()):
                    print(f"  检测到GIF文件: {os.path.basename(gif_path)}")
                    
                    # 获取GIF信息
                    gif_info = self.get_video_info(gif_path.strip())
                    if not gif_info:
                        print(f"  警告: 无法获取GIF信息，跳过GIF叠加")
                    else:
                        # 获取合并后视频的尺寸和时长
                        material_height = material_info['height']
                        game_height = game_info['height']
                        video_width = material_info['width']  # 两个视频宽度相同
                        total_height = material_height + game_height
                        
                        # 计算合并后视频的总时长（使用游戏视频时长）
                        material_duration = material_info['duration']
                        game_duration = game_info['duration']
                        total_duration = game_duration  # 使用游戏视频时长作为目标时长
                        
                        print(f"  游戏视频时长: {game_duration:.2f}秒")
                        print(f"  素材视频时长: {material_duration:.2f}秒")
                        print(f"  目标合并时长: {total_duration:.2f}秒")
                        print(f"  GIF原始时长: {gif_info['duration']:.2f}秒")
                        
                        # 计算GIF缩放后的高度（等比缩放）
                        gif_original_width = gif_info['width']
                        gif_original_height = gif_info['height']
                        gif_new_height = int((video_width * gif_original_height) / gif_original_width)
                        
                        # 确保高度是偶数（视频编码要求）
                        if gif_new_height % 2 != 0:
                            gif_new_height += 1
                        
                        print(f"  GIF原始尺寸: {gif_original_width}x{gif_original_height}")
                        print(f"  GIF缩放尺寸: {video_width}x{gif_new_height}")
                        
                        # 创建GIF输入
                        gif_input = ffmpeg.input(gif_path.strip())
                        
                        # 使用loop filter实现真正的循环播放
                        gif_looped = gif_input.video.filter('loop', loop=-1, size=32767, start=0)
                        
                        # 缩放GIF到与游戏视频相同的宽度
                        gif_scaled = gif_looped.filter('scale', video_width, gif_new_height)
                        
                        # 计算GIF位置：在结合处居中显示
                        # GIF的中心位置应该在结合处（material_height位置）
                        gif_center_y = material_height
                        
                        # 叠加缩放后的GIF到视频上，使用shortest=1确保输出时长由游戏视频决定
                        video_output = ffmpeg.filter([video_output, gif_scaled], 'overlay', 
                                                   x='(W-w)/2',  # 水平居中
                                                   y=f'{gif_center_y}-h/2',  # 垂直居中在结合处
                                                   shortest=1)  # 输出时长由最短的输入决定（游戏视频）
                        print(f"  GIF循环播放设置: 使用loop filter实现无限循环，输出时长由游戏视频决定")
                        print(f"  GIF叠加位置: 水平居中，垂直位置在结合处 (y={gif_center_y})")
                
                # 根据音频模式处理音频
                if audio_mode == "mix":
                    # 混音模式：先检查音频状态
                    print(f"  混音模式 - 素材音量: {material_audio_volume}, 游戏音量: {game_audio_volume}")
                    
                    # 检查素材和游戏视频的音频状态
                    material_info = self.get_video_info(material_path)
                    game_info = self.get_video_info(game_path)
                    
                    if not material_info or not game_info:
                        print("  无法获取视频信息，使用游戏音频")
                        audio_output = game_input.audio
                    elif not material_info['has_audio'] and not game_info['has_audio']:
                        # 两个视频都没有音频
                        print("  错误：素材视频和游戏视频都没有音频，无法进行混音处理")
                        raise ValueError("素材视频和游戏视频都没有音频，无法进行混音处理")
                    elif not material_info['has_audio']:
                        # 只有游戏视频有音频
                        print("  素材视频没有音频，使用游戏音频")
                        audio_output = game_input.audio
                    elif not game_info['has_audio']:
                        # 只有素材视频有音频
                        print("  游戏视频没有音频，使用素材音频")
                        audio_output = material_input.audio
                    else:
                        # 两个视频都有音频，进行混音
                        print("  两个视频都有音频，进行混音处理")
                        # 对素材音频应用音量调整
                        material_audio_adjusted = material_input.audio.filter('volume', material_audio_volume)
                        # 对游戏音频应用音量调整
                        game_audio_adjusted = game_input.audio.filter('volume', game_audio_volume)
                        
                        # 使用amix filter混合音频
                        audio_output = ffmpeg.filter([material_audio_adjusted, game_audio_adjusted], 'amix', inputs=2, duration='longest')
                        print("  混音模式：成功创建混音")
                else:
                    # 只使用游戏音频
                    print("  使用游戏音频")
                    audio_output = game_input.audio
                
                # 输出合并后的视频
                (
                    ffmpeg
                    .output(
                        video_output,
                        audio_output,
                        output_path,
                        vcodec='libx264',
                        acodec='aac',
                        audio_bitrate='128k',
                        preset='medium'
                    )
                    .overwrite_output()
                    .run(quiet=False)  # 显示详细错误信息
                )
            else:
                # 游戏在上，素材在下
                game_input = ffmpeg.input(game_path)
                material_input = ffmpeg.input(material_path)
                
                # 使用vstack filter合并视频
                video_output = ffmpeg.filter([game_input.video, material_input.video], 'vstack', inputs=2)
                
                # 检查是否需要叠加GIF
                if gif_path and gif_path.strip() and os.path.exists(gif_path.strip()):
                    print(f"  检测到GIF文件: {os.path.basename(gif_path)}")
                    
                    # 获取GIF信息
                    gif_info = self.get_video_info(gif_path.strip())
                    if not gif_info:
                        print(f"  警告: 无法获取GIF信息，跳过GIF叠加")
                    else:
                        # 获取合并后视频的尺寸和时长
                        material_height = material_info['height']
                        game_height = game_info['height']
                        video_width = game_info['width']  # 两个视频宽度相同
                        
                        # 计算合并后视频的总时长（使用游戏视频时长）
                        material_duration = material_info['duration']
                        game_duration = game_info['duration']
                        total_duration = game_duration  # 使用游戏视频时长作为目标时长
                        
                        print(f"  游戏视频时长: {game_duration:.2f}秒")
                        print(f"  素材视频时长: {material_duration:.2f}秒")
                        print(f"  目标合并时长: {total_duration:.2f}秒")
                        print(f"  GIF原始时长: {gif_info['duration']:.2f}秒")
                        
                        # 计算GIF缩放后的高度（等比缩放）
                        gif_original_width = gif_info['width']
                        gif_original_height = gif_info['height']
                        gif_new_height = int((video_width * gif_original_height) / gif_original_width)
                        
                        # 确保高度是偶数（视频编码要求）
                        if gif_new_height % 2 != 0:
                            gif_new_height += 1
                        
                        print(f"  GIF原始尺寸: {gif_original_width}x{gif_original_height}")
                        print(f"  GIF缩放尺寸: {video_width}x{gif_new_height}")
                        
                        # 创建GIF输入
                        gif_input = ffmpeg.input(gif_path.strip())
                        
                        # 使用loop filter实现真正的循环播放
                        gif_looped = gif_input.video.filter('loop', loop=-1, size=32767, start=0)
                        
                        # 缩放GIF到与游戏视频相同的宽度
                        gif_scaled = gif_looped.filter('scale', video_width, gif_new_height)
                        
                        # 计算GIF位置：在结合处居中显示
                        # GIF的中心位置应该在结合处（game_height位置）
                        gif_center_y = game_height
                        
                        # 叠加缩放后的GIF到视频上，使用shortest=1确保输出时长由游戏视频决定
                        video_output = ffmpeg.filter([video_output, gif_scaled], 'overlay', 
                                                   x='(W-w)/2',  # 水平居中
                                                   y=f'{gif_center_y}-h/2',  # 垂直居中在结合处
                                                   shortest=1)  # 输出时长由最短的输入决定（游戏视频）
                        print(f"  GIF循环播放设置: 使用loop filter实现无限循环，输出时长由游戏视频决定")
                        print(f"  GIF叠加位置: 水平居中，垂直位置在结合处 (y={gif_center_y})")
                
                # 根据音频模式处理音频
                if audio_mode == "mix":
                    # 混音模式：先检查音频状态
                    print(f"  混音模式 - 素材音量: {material_audio_volume}, 游戏音量: {game_audio_volume}")
                    
                    # 检查素材和游戏视频的音频状态
                    material_info = self.get_video_info(material_path)
                    game_info = self.get_video_info(game_path)
                    
                    if not material_info or not game_info:
                        print("  无法获取视频信息，使用游戏音频")
                        audio_output = game_input.audio
                    elif not material_info['has_audio'] and not game_info['has_audio']:
                        # 两个视频都没有音频
                        print("  错误：素材视频和游戏视频都没有音频，无法进行混音处理")
                        raise ValueError("素材视频和游戏视频都没有音频，无法进行混音处理")
                    elif not material_info['has_audio']:
                        # 只有游戏视频有音频
                        print("  素材视频没有音频，使用游戏音频")
                        audio_output = game_input.audio
                    elif not game_info['has_audio']:
                        # 只有素材视频有音频
                        print("  游戏视频没有音频，使用素材音频")
                        audio_output = material_input.audio
                    else:
                        # 两个视频都有音频，进行混音
                        print("  两个视频都有音频，进行混音处理")
                        # 对素材音频应用音量调整
                        material_audio_adjusted = material_input.audio.filter('volume', material_audio_volume)
                        # 对游戏音频应用音量调整
                        game_audio_adjusted = game_input.audio.filter('volume', game_audio_volume)
                        
                        # 使用amix filter混合音频
                        audio_output = ffmpeg.filter([material_audio_adjusted, game_audio_adjusted], 'amix', inputs=2, duration='longest')
                        print("  混音模式：成功创建混音")
                else:
                    # 只使用游戏音频
                    print("  使用游戏音频")
                    audio_output = game_input.audio
                
                # 输出合并后的视频
                (
                    ffmpeg
                    .output(
                        video_output,
                        audio_output,
                        output_path,
                        vcodec='libx264',
                        acodec='aac',
                        audio_bitrate='128k',
                        preset='medium'
                    )
                    .overwrite_output()
                    .run(quiet=False)  # 显示详细错误信息
                )
            
            return True
            
        except Exception as e:
            print(f"视频合并失败: {str(e)}")
            return False
    
    def merge_videos(self, material_folder, game_folder, position, audio_mode, material_audio_volume, game_audio_volume, output_folder_name, material_path="", game_path="", gif_path=""):
        """
        合并视频文件
        
        Args:
            material_folder: 素材文件夹（下拉框选择）
            game_folder: 游戏视频文件夹（下拉框选择）
            position: 位置关系（up/down）
            audio_mode: 音频模式（game_only/mix）
            material_audio_volume: 素材音频音量占比
            game_audio_volume: 游戏音频音量占比
            output_folder_name: 输出文件夹名字
            material_path: 直接输入的素材文件夹路径（可选，优先级高于下拉框）
            game_path: 直接输入的游戏文件夹路径（可选，优先级高于下拉框）
            gif_path: GIF动态图路径（可选，如果存在则在结合处叠加显示）
        """
        try:
            # 使用ComfyUI的默认输入和输出路径
            base_input_dir = folder_paths.get_input_directory()
            output_folder = folder_paths.get_output_directory()
            
            # 构建完整的输入路径（直接输入的路径优先级更高）
            if material_path and material_path.strip():
                # 使用直接输入的素材路径
                material_input_path = material_path.strip()
            else:
                # 使用下拉框选择的路径
                if material_folder == "input":
                    material_input_path = base_input_dir
                else:
                    material_input_path = os.path.join(base_input_dir, material_folder)
            
            if game_path and game_path.strip():
                # 使用直接输入的游戏路径
                game_input_path = game_path.strip()
            else:
                # 使用下拉框选择的路径
                if game_folder == "input":
                    game_input_path = base_input_dir
                else:
                    game_input_path = os.path.join(base_input_dir, game_folder)
            
            # 验证输入文件夹
            if not os.path.exists(material_input_path):
                raise ValueError(f"素材文件夹不存在: {material_input_path}")
            if not os.path.exists(game_input_path):
                raise ValueError(f"游戏视频文件夹不存在: {game_input_path}")
            
            # 在输出目录下创建指定名字的文件夹
            output_path = os.path.join(output_folder, output_folder_name)
            os.makedirs(output_path, exist_ok=True)
            
            # 支持的视频格式
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']
            
            # 获取所有游戏视频文件
            game_videos = []
            for ext in video_extensions:
                pattern = os.path.join(game_input_path, ext)
                game_videos.extend(glob.glob(pattern))
            
            # 获取所有素材视频文件
            material_videos = []
            for ext in video_extensions:
                pattern = os.path.join(material_input_path, ext)
                material_videos.extend(glob.glob(pattern))
            
            if not game_videos:
                return (f"未找到游戏视频文件",)
            if not material_videos:
                return (f"未找到素材视频文件",)
            
            # 处理每个游戏视频
            processed_count = 0
            material_index = 0
            output_paths = []
            
            for game_video in game_videos:
                if material_index >= len(material_videos):
                    break  # 素材用完则结束
                
                try:
                    # 获取游戏视频信息
                    game_info = self.get_video_info(game_video)
                    if not game_info:
                        continue
                    
                    # 检查游戏视频音频情况
                    game_filename = Path(game_video).stem
                    if audio_mode == "mix" and not game_info['has_audio']:
                        print(f"警告: 游戏视频 {game_filename} 没有音频，在mix模式下可能影响混音效果")
                    
                    game_duration = game_info['duration']
                    used_materials = []
                    current_duration = 0
                    
                    # 为当前游戏视频收集足够的素材
                    while current_duration < game_duration and material_index < len(material_videos):
                        material_video = material_videos[material_index]
                        material_info = self.get_video_info(material_video)
                        
                        if not material_info:
                            material_index += 1
                            continue
                        
                        # 检查素材视频音频情况
                        material_filename = Path(material_video).stem
                        if audio_mode == "mix" and not material_info['has_audio']:
                            print(f"警告: 素材视频 {material_filename} 没有音频，在mix模式下可能影响混音效果")
                        
                        used_materials.append({
                            'path': material_video,
                            'duration': material_info['duration'],
                            'info': material_info
                        })
                        
                        current_duration += material_info['duration']
                        material_index += 1
                    
                    if not used_materials:
                        continue
                    
                    # 生成输出文件名
                    game_filename = Path(game_video).stem
                    output_file = os.path.join(output_path, f"{game_filename}_merged.mp4")
                    
                    # 创建临时合并的素材视频
                    temp_dir = tempfile.mkdtemp()
                    temp_material_path = os.path.join(temp_dir, f"temp_material_{game_filename}.mp4")
                    
                    # 获取游戏视频的宽度
                    game_info = self.get_video_info(game_video)
                    game_width = game_info['width']
                    
                    # 在mix模式下进行最终的音频检查
                    if audio_mode == "mix":
                        # 检查所有使用的素材是否有音频
                        materials_with_audio = [m for m in used_materials if m['info']['has_audio']]
                        materials_without_audio = [m for m in used_materials if not m['info']['has_audio']]
                        
                        if not game_info['has_audio'] and not materials_with_audio:
                            print(f"错误: 游戏视频 {game_filename} 和所有素材视频都没有音频，无法进行混音处理")
                            continue
                        elif not game_info['has_audio']:
                            print(f"警告: 游戏视频 {game_filename} 没有音频，将只使用素材音频")
                        elif not materials_with_audio:
                            print(f"警告: 所有素材视频都没有音频，将只使用游戏音频")
                        elif materials_without_audio:
                            print(f"警告: {len(materials_without_audio)} 个素材视频没有音频，可能影响混音效果")
                    
                    # 如果只有一个素材且长度足够，直接使用
                    if len(used_materials) == 1 and used_materials[0]['duration'] >= game_duration:
                        # 先将素材宽度对齐到游戏宽度，然后截取到游戏长度，兼容音频情况
                        input_stream = ffmpeg.input(used_materials[0]['path'], t=game_duration)
                        material_info = used_materials[0]['info']
                        
                        if material_info['has_audio']:
                            # 有音频的情况
                            (
                                ffmpeg
                                .output(
                                    input_stream.video.filter('scale', game_width, -1),  # 宽度对齐，高度自动计算
                                    input_stream.audio,  # 保留音频
                                    temp_material_path, 
                                    vcodec='libx264', 
                                    acodec='aac', 
                                    preset='medium'
                                )
                                .overwrite_output()
                                .run(quiet=True)
                            )
                        else:
                            # 没有音频的情况
                            (
                                ffmpeg
                                .output(
                                    input_stream.video.filter('scale', game_width, -1),  # 宽度对齐，高度自动计算
                                    temp_material_path, 
                                    vcodec='libx264', 
                                    preset='medium'
                                )
                                .overwrite_output()
                                .run(quiet=True)
                            )
                    else:
                        # 多个素材需要拼接
                        # 先将每个素材宽度对齐到游戏宽度
                        resized_materials = []
                        for i, material in enumerate(used_materials):
                            resized_path = os.path.join(temp_dir, f"resized_material_{i}.mp4")
                            if self.resize_video_to_width(material['path'], game_width, resized_path):
                                resized_materials.append(resized_path)
                        
                        if not resized_materials:
                            continue
                        
                        # 创建concat文件列表
                        concat_file = os.path.join(temp_dir, f"concat_list_{game_filename}.txt")
                        with open(concat_file, 'w') as f:
                            for resized_material in resized_materials:
                                f.write(f"file '{resized_material}'\n")
                        
                        # 使用concat demuxer拼接视频，兼容音频情况
                        # 检查是否有任何素材有音频
                        has_any_audio = any(m['info']['has_audio'] for m in used_materials)
                        
                        if has_any_audio:
                            # 有音频的情况
                            (
                                ffmpeg
                                .input(concat_file, format='concat', safe=0)
                                .output(temp_material_path, vcodec='libx264', acodec='aac', preset='medium')
                                .overwrite_output()
                                .run(quiet=True)
                            )
                        else:
                            # 没有音频的情况
                            (
                                ffmpeg
                                .input(concat_file, format='concat', safe=0)
                                .output(temp_material_path, vcodec='libx264', preset='medium')
                                .overwrite_output()
                                .run(quiet=True)
                            )
                        
                        # 清理concat文件和临时缩放文件
                        try:
                            os.remove(concat_file)
                            for resized_material in resized_materials:
                                os.remove(resized_material)
                        except:
                            pass
                        
                        # 检查合并后的素材时长是否足够支持游戏视频时长
                        temp_material_info = self.get_video_info(temp_material_path)
                        if not temp_material_info:
                            print(f"  警告: 无法获取合并后素材视频信息，跳过游戏视频: {game_filename}")
                            continue
                        
                        temp_material_duration = temp_material_info['duration']
                        if temp_material_duration < game_duration:
                            print(f"  警告: 合并后素材时长 ({temp_material_duration:.2f}秒) 不足以支持游戏视频时长 ({game_duration:.2f}秒)，跳过游戏视频: {game_filename}")
                            continue
                        
                        print(f"  合并后素材时长: {temp_material_duration:.2f}秒，游戏视频时长: {game_duration:.2f}秒")
                        
                        # 截取到游戏长度，兼容音频情况
                        temp_material_cropped = os.path.join(temp_dir, f"temp_material_cropped_{game_filename}.mp4")
                        
                        if temp_material_info and temp_material_info['has_audio']:
                            # 有音频的情况
                            (
                                ffmpeg
                                .input(temp_material_path, t=game_duration)
                                .output(temp_material_cropped, vcodec='libx264', acodec='aac', preset='medium')
                                .overwrite_output()
                                .run(quiet=True)
                            )
                        else:
                            # 没有音频的情况
                            (
                                ffmpeg
                                .input(temp_material_path, t=game_duration)
                                .output(temp_material_cropped, vcodec='libx264', preset='medium')
                                .overwrite_output()
                                .run(quiet=True)
                            )
                        
                        # 替换临时文件
                        os.remove(temp_material_path)
                        temp_material_path = temp_material_cropped
                    
                    # 合并素材和游戏视频
                    if self.merge_videos_vertically(temp_material_path, game_video, output_file, position, audio_mode, material_audio_volume, game_audio_volume, gif_path):
                        processed_count += 1
                        output_paths.append(output_file)
                        print(f"成功合并: {game_filename} -> {output_file}")
                    
                    # 清理临时文件和目录
                    try:
                        shutil.rmtree(temp_dir)
                    except:
                        pass
                    
                except Exception as e:
                    print(f"处理游戏视频 {game_video} 时出错: {str(e)}")
                    continue
            
            if processed_count == 0:
                return ("",)  # 没有可保存的视频时返回空字符串
            else:
                # 返回输出文件的目录路径
                print(f"成功处理 {processed_count} 个游戏视频")
                print(f"输出目录: {output_path}")
                print(f"输出文件: {output_paths}")
                return (output_path,)
                
        except Exception as e:
            print(f"处理过程中出错: {str(e)}")
            return ("",)  # 出错时也返回空字符串

# 节点映射
NODE_CLASS_MAPPINGS = {
    "VideoMergeNode": VideoMergeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoMergeNode": "视频合并"
}