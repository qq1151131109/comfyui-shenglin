import os
import numpy as np
import json
import folder_paths
import zipfile
import shutil
import numpy as np
import torch
from PIL import Image, ImageOps

class AllTrue(str):
    def __init__(self, representation=None) -> None:
        self.repr = representation
        pass
    def __ne__(self, __value: object) -> bool:
        return False
    # isinstance, jsonserializable hijack
    def __instancecheck__(self, instance):
        return True
    def __subclasscheck__(self, subclass):
        return True
    def __bool__(self):
        return True
    def __str__(self):
        return self.repr
    # jsonserializable hijack
    def __jsonencode__(self):
        return self.repr
    def __repr__(self) -> str:
        return self.repr
    def __eq__(self, __value: object) -> bool:
        return True
anytype = AllTrue("*") # when a != b is called, it will always return False

class AnyToStringNode:
    def __init__(self):
        # Initialize any necessary parameters for the node
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything": (anytype, {"default": 0.0}),
            }
        }

    RETURN_TYPES = ("STRING",)  # Output type is string
    CATEGORY = "⚙️ Shenglin/RunningHub/Tools"      # Category name is RunningHub
    FUNCTION = "process"         # The processing function is the 'process' method

    def process(self, anything):
        """
        Converts any input type to a string.
        If the input is a string that can be converted to an integer, it performs the conversion.
        Otherwise, it directly converts the input to a string.
        """
        if isinstance(anything, str):
            try:
                # Attempt to convert the string to an integer and then back to string
                return (str(int(anything)),)
            except ValueError:
                # If conversion fails, return the original string
                return (anything,)
        else:
            # For non-string types, directly convert to string
            return (str(anything),)
        
class RH_Extract_Image_From_List():
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images list"}),
                "image_index": ("INT", {"default": 0 }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "rh_extract_image"

    OUTPUT_NODE = False

    CATEGORY = "⚙️ Shenglin/RunningHub/Tools"

    def rh_extract_image(self, images, image_index):
        out = images[int(image_index)].unsqueeze(0)
        return (out,)
    
class RH_Batch_Images_From_List():
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images list"}),
                "image_indices": ("STRING", {"default":"0-3,4,5-7","tooltip": "Some like 0-2, 3, 4-5. Leaving it empty means selecting all."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "rh_batch_images"

    OUTPUT_NODE = False

    CATEGORY = "⚙️ Shenglin/RunningHub/Tools"

    def rh_batch_images(self, images, image_indices):
        image_indices = image_indices.replace(" ", "")
        out = []
        if image_indices == "":
            out = images
        image_indices = image_indices.split(',')
        for index in image_indices:
            if '-' in index:
                sindex = index.split('-')
                out.extend(images[int(sindex[0]):int(sindex[1])+1])
            else:
                out.append(images[int(index)])
        batchsize = len(out)
        max_height = max(image.shape[0] for image in out)
        max_width = max(image.shape[1] for image in out)
        max_channels = max(image.shape[2] for image in out)
        batch_images = torch.zeros([batchsize, max_height, max_width, max_channels])
        for (batch_number, image) in enumerate(out):
            h, w, c = image.shape
            batch_images[batch_number, 0:h, 0:w, 0:c] = image
        return (batch_images,)