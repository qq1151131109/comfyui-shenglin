import torch
import numpy as np
from PIL import Image

# https://github.com/melMass/comfy_mtb/blob/501c3301056b2851555cccd75ab3ff15b1ab8e0c/utils.py#L261-L298
def tensor2pil(image):
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil(image[i]))
        return out

    return [
        Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )
    ]

def pil2tensor(image):
    if isinstance(image, list):
        # 检查所有图像是否有相同尺寸
        if len(image) == 0:
            raise ValueError("Empty image list provided")
        
        # 获取第一个图像的尺寸
        first_size = image[0].size
        
        # 检查所有图像尺寸是否一致
        mismatched_images = []
        for i, img in enumerate(image):
            if img.size != first_size:
                mismatched_images.append(f"Image {i}: {img.size} vs expected {first_size}")
        
        if mismatched_images:
            # 统一调整所有图像到相同尺寸
            print(f"⚠️ 检测到图像尺寸不一致，自动调整到 {first_size}")
            print(f"不匹配的图像: {mismatched_images[:3]}...")  # 只显示前3个
            
            # 调整所有图像到第一个图像的尺寸
            resized_images = []
            for img in image:
                if img.size != first_size:
                    resized_img = img.resize(first_size, Image.LANCZOS)
                    resized_images.append(resized_img)
                else:
                    resized_images.append(img)
            
            return torch.cat([pil2tensor(img) for img in resized_images], dim=0)
        
        return torch.cat([pil2tensor(img) for img in image], dim=0)

    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# https://github.com/comfyanonymous/ComfyUI/blob/fc196aac80fd~4bf6c8a39d85d1e809902871cade/comfy_extras/nodes_mask.py#L127
def tensor2Mask(image):
    return image[:, :, :, 0]