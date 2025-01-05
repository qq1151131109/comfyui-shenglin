from .RH_SettingsNode import SettingsNode
from .RH_NodeInfoListNode import NodeInfoListNode
from .RH_ExecuteNode import ExecuteNode
from .RH_ImageUploaderNode import ImageUploaderNode
from .RH_Utils import AnyToStringNode



NODE_CLASS_MAPPINGS = {
    "RH_SettingsNode": SettingsNode,
    "RH_NodeInfoListNode": NodeInfoListNode,
    "RH_ExecuteNode": ExecuteNode,
    "RH_ImageUploaderNode": ImageUploaderNode,
    "RH_Utils": AnyToStringNode,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RH_SettingsNode": "RH Settings",
    "RH_NodeInfoListNode": "RH Node Info List",
    "RH_ExecuteNode": "RH Execute",
    "RH_ImageUploaderNode": "RH Image Uploader",
    "RH_Utils": "RH Anything to String",

}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",]
