class SettingsNode:
    def __init__(self):
        # Initialize any necessary parameters for the node
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
    CATEGORY = "🔥 Shenglin/工具"
    FUNCTION = "process"  # Add FUNCTION attribute pointing to process method

    def process(self,base_url,apiKey, workflowId_webappId):
        """
        This node receives apiKey and workflowId, returns structured data for use by subsequent nodes
        """
        # Return a structure containing apiKey and workflowId
        return [{"base_url": base_url, "apiKey": apiKey, "workflowId_webappId": workflowId_webappId}]
