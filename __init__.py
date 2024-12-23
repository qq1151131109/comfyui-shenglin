from .RH_SettingsNode import SettingsNode
from .RH_NodeInfoListNode import NodeInfoListNode
from .RH_ExecuteNode import ExecuteNode

NODE_CLASS_MAPPINGS = {
    "RH_SettingsNode": SettingsNode,
    "RH_NodeInfoListNode": NodeInfoListNode,
    "RH_ExecuteNode": ExecuteNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RH_SettingsNode": "RH Settings",
    "RH_NodeInfoListNode": "RH Node Info List",
    "RH_ExecuteNode": "RH Execute",
}
