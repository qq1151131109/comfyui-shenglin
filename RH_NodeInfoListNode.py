class NodeInfoListNode:
    def __init__(self):
        # 初始化一个空的 node_info_list，用于存储所有的 nodeInfo
        self.node_info_list = []
        print(f"NodeInfoListNode __init__ NodeInfoList: {self.node_info_list}")


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nodeId": ("INT", {"default": 0}),
                "fieldName": ("STRING", {"default": ""}),
                "fieldValue": ("STRING", {"default": ""}),
            },
            "optional": {
                "previousNodeInfoList": ("ARRAY", {"default": []}),  # 使其为可选，默认值为空列表
            }
        }

    RETURN_TYPES = ("ARRAY",)  # 输出类型改为 ARRAY
    CATEGORY = "RunningHub"
    FUNCTION = "process"

    def process(self, nodeId, fieldName, fieldValue, previousNodeInfoList=[]):
        """
        该节点允许用户配置多个 nodeId、fieldName 和 fieldValue 参数，
        并将多个 nodeInfoList 输出为数组。支持串联，多个节点将合并成一个数组。
        """
        self.node_info_list = []
        # 输出调试信息，查看 previousNodeInfoList
        print(f"Processing nodeId: {nodeId}, fieldName: {fieldName}, fieldValue: {fieldValue}")
        print(f"previousNodeInfoList: {previousNodeInfoList}")
        # 当前的 node_info
        node_info = {"nodeId": nodeId, "fieldName": fieldName, "fieldValue": fieldValue}

        # 如果前一个节点有输出（previousNodeInfoList），则将其添加到当前 node_info_list 中
        if previousNodeInfoList:
            self.node_info_list.extend(previousNodeInfoList)  # 将前一个节点的输出合并进来

        # 将当前的 node_info 加入 node_info_list 中
        self.node_info_list.append(node_info)

        # 输出调试信息，查看当前的 node_info_list
        print(f"Updated node_info_list: {self.node_info_list}")

        # 返回整个 node_info_list 数组，包含当前节点和之前节点的输出
        return [self.node_info_list]
