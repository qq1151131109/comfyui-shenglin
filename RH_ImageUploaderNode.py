import requests
from PIL import Image
from io import BytesIO
import torch
import numpy as np
import time  # Add this import

class ImageUploaderNode:
    """
    ComfyUI 节点：ImageUploaderNode
    功能：将输入的图像上传到服务器，并返回服务器返回的文件名。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "apiConfig": ("STRUCT",),  # API 配置参数，必须包含 apiKey 和 base_url
                "image": ("IMAGE",),  # 输入图像张量
            },
        }

    RETURN_TYPES = ("STRING",)  # 输出类型为字符串
    RETURN_NAMES = ("filename",)  # 输出名称为 filename
    CATEGORY = "RunningHub"  # 节点类别
    FUNCTION = "process"  # 指定处理方法

    def process(self, image: torch.Tensor, apiConfig: dict) -> tuple:
        """
        处理方法：将图像上传到服务器并返回文件名。

        参数：
            image (torch.Tensor): 输入的图像张量，形状可能为 [C, H, W]、[H, W, C] 或其他。
            apiConfig (dict): API 配置参数，必须包含 'apiKey' 和 'base_url'。

        返回：
            tuple: 包含上传后返回的文件名。
        """
        # 检查输入的图像类型
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected image to be a torch.Tensor, but got {type(image)}.")
    
        # 将图像张量转换为 NumPy 数组
        image_np = image.detach().cpu().numpy()

        # 打印图像形状以进行调试
        print(f"Original image shape: {image_np.shape}")

        # 处理图像的形状，确保为 [H, W, C]
        if image_np.ndim == 4:
            # 处理批量维度，例如 [B, C, H, W]
            print("Detected 4D tensor. Assuming shape [B, C, H, W]. Taking the first image in the batch.")
            image_np = image_np[0]
            print(f"Image shape after removing batch dimension: {image_np.shape}")

        if image_np.ndim == 3:
            if image_np.shape[0] in [1, 3, 4]:  # [C, H, W]
                image_np = np.transpose(image_np, (1, 2, 0))  # 转换为 [H, W, C]
                print(f"Transposed image shape to [H, W, C]: {image_np.shape}")
            elif image_np.shape[2] in [1, 3, 4]:  # [H, W, C]
                # 已经是 [H, W, C]，无需转置
                print(f"Image already in [H, W, C] format: {image_np.shape}")
            else:
                raise ValueError(f"Unsupported number of channels: {image_np.shape[2]}")
        elif image_np.ndim == 2:
            # 灰度图像 [H, W]
            image_np = np.expand_dims(image_np, axis=-1)  # 转换为 [H, W, 1]
            print(f"Expanded grayscale image to [H, W, 1]: {image_np.shape}")
        else:
            raise ValueError(f"Unsupported image shape: {image_np.shape}")

        # 确定图像模式
        if image_np.shape[2] == 1:
            mode = "L"  # 灰度图像
            image_pil = Image.fromarray((image_np.squeeze(-1) * 255).astype(np.uint8), mode)
            print("Converted to PIL Image with mode 'L'")
        elif image_np.shape[2] == 3:
            mode = "RGB"  # RGB 图像
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8), mode)
            print("Converted to PIL Image with mode 'RGB'")
        elif image_np.shape[2] == 4:
            mode = "RGBA"  # RGBA 图像
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8), mode)
            print("Converted to PIL Image with mode 'RGBA'")
        else:
            raise ValueError(f"Unsupported number of channels: {image_np.shape[2]}")

        # 将 PIL 图像保存到 BytesIO 缓冲区
        buffer = BytesIO()
        image_pil.save(buffer, format='PNG')  # 可以根据需要选择 'JPEG' 或其他格式
        # 先获取缓冲区大小
        buffer_size = buffer.tell()
        # 然后重置指针到开头
        buffer.seek(0)
        print("Saved PIL Image to BytesIO buffer.")

        # 打印图像大小，以 MB 为单位
        buffer_size_mb = buffer_size / (1024 * 1024)
        print(f"Image size: {buffer_size_mb:.2f} MB")

        # 检查图像大小是否超过 10MB
        max_size_bytes = 10 * 1024 * 1024  # 10MB
        if buffer_size > max_size_bytes:
            raise Exception(f"Image size {buffer_size_mb:.2f}MB exceeds the 10MB limit.")

        # 准备 multipart/form-data
        files = {
            'file': ('image.png', buffer, 'image/png')  # 文件名和内容类型
        }
        data = {
            'apiKey': apiConfig.get('apiKey'),
            'fileType': 'image',
        }

        # 获取 base_url，默认为 'https://www.runninghub.cn'
        base_url = apiConfig.get('base_url', 'https://www.runninghub.cn')
        upload_url = f"{base_url}/task/openapi/upload"

        print(f"Uploading image to {upload_url} with apiKey: {data['apiKey']}")

        # 发送 POST 请求，添加重试机制
        max_retries = 5
        retry_delay = 1  # 初始延迟1秒
        
        for attempt in range(max_retries):
            try:
                response = requests.post(upload_url, data=data, files=files)
                print(f"Attempt {attempt + 1}: Received response with status code: {response.status_code}")
                
                if response.status_code == 200:
                    break  # 成功则跳出重试循环
                    
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:  # 最后一次尝试
                    raise Exception(f"Failed to connect to the server after {max_retries} attempts: {e}")
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避，每次失败后延迟时间翻倍
                continue

        # 如果所有重试都失败了
        if response.status_code != 200:
            raise Exception(f"Upload failed with status code {response.status_code} after {max_retries} attempts.")

        # 解析 JSON 响应
        try:
            response_json = response.json()
            print(f"Response JSON: {response_json}")
        except ValueError:
            raise Exception("Failed to parse JSON response from the server.")

        # 检查 API 返回的 code
        if response_json.get('code') != 0:
            raise Exception(f"Upload failed: {response_json.get('msg')}")

        # 提取 filename
        data_field = response_json.get('data', {})
        filename = data_field.get('fileName')

        if not filename:
            raise Exception("Upload succeeded but 'fileName' not found in the response.")

        print(f"Uploaded filename: {filename}")

        return (filename,)
