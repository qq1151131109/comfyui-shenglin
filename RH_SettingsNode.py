class SettingsNode:
    def __init__(self):
        # 初始化节点的任何必要参数
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": ("STRING", {"default": "https://www.runninghub.cn"}),
                "apiKey": ("STRING", {"default": ""}),
                "workflowId_webappId": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRUCT",)
    CATEGORY = "RunningHub"
    FUNCTION = "process"  # 添加 FUNCTION 属性并指向 process 方法

    def process(self,base_url,apiKey, workflowId_webappId):
        """
        该节点接收 apiKey 和 workflowId，返回结构化数据供后续节点使用
        """
        # 返回一个结构体，包含 apiKey 和 workflowId
        return [{"base_url": base_url, "apiKey": apiKey, "workflowId_webappId": workflowId_webappId}]
