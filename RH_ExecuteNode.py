import requests
import time
import json
from PIL import Image, ImageOps
from io import BytesIO
import numpy as np
import torch
import os  # 确保导入 os 库

class ExecuteNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "apiConfig": ("STRUCT",),  # 设置节点的输入
                "nodeInfoList": ("ARRAY", {"default": []}),  # NodeInfoList节点的输出
            },
            "optional": {
                "run_timeout": ("INT", {"default": 600}),       # 最大运行超时时间（秒）
                "query_interval": ("INT", {"default": 10}),      # 查询间隔时间（秒）
            },
        }

    RETURN_TYPES = ("IMAGE",)  # 支持单张和多张图片输出
    RETURN_NAMES = ("images",)  # 定义返回名称，与 ComfyUI 的预期匹配

    CATEGORY = "RunningHub"
    FUNCTION = "process"  # 指向 process 方法

    def process(self, apiConfig, nodeInfoList, run_timeout=600, query_interval=2):
        # Ensure query_interval is not less than 2
        if query_interval < 2:
            print("Query interval is too low, setting to minimum value of 2 seconds.")
            query_interval = 2

        """
        该节点通过调用 RunningHub API 创建任务并返回生成的图片链接。
        """
        # 打印请求数据，方便调试
        print(f"API request data: {apiConfig}")
        print(f"Node info list: {nodeInfoList}")
        print(f"Run timeout: {run_timeout} seconds")
        print(f"Query interval: {query_interval} seconds")

        # 1. 查询账户状态，检查是否可以提交任务
        account_status = self.check_account_status(apiConfig["apiKey"], apiConfig["base_url"])
        if int(account_status["currentTaskCounts"]) > 0:
            print("There are tasks running, waiting for them to finish.")
            # 等待最多 run_timeout 秒，如果任务未完成，则超时
            start_time = time.time()
            while account_status["currentTaskCounts"] > 0 and time.time() - start_time < run_timeout:
                time.sleep(query_interval)  # 每 query_interval 秒查询一次
                account_status = self.check_account_status(apiConfig["apiKey"], apiConfig["base_url"])
            if int(account_status["currentTaskCounts"]) > 0:
                raise Exception(f"Timeout: There are still running tasks after {run_timeout} seconds.")

        # Print nodeInfoList for debugging
        print(f"ExecuteNode NodeInfoList: {nodeInfoList}")
        
        # 2. 创建任务
        task_creation_result = self.create_task(apiConfig, nodeInfoList, apiConfig["base_url"])
        if task_creation_result["code"] != 0:
            raise Exception(f"Task creation failed: {task_creation_result['msg']}")

        task_id = task_creation_result["data"]["taskId"]
        task_status = task_creation_result["data"]["taskStatus"]
        print(f"Task created successfully, taskId: {task_id}, status: {task_status}")

        # 3. 查询任务状态直到任务完成
        task_start_time = time.time()
        while task_status != "success":
            print(f"Task still running, checking again in {query_interval} seconds...")
            time.sleep(query_interval)  # 每 query_interval 秒检查一次任务状态
            task_status_result = self.check_task_status(task_id, apiConfig["apiKey"], apiConfig["base_url"])
            print(f"Task info, taskId: {task_id}, status: {task_status_result}")

            if isinstance(task_status_result, dict):
                task_status = task_status_result.get("taskStatus", "unknown")  # 从结果中获取任务状态
            elif isinstance(task_status_result, list):
                # 假设任务完成后返回的数据列表表示任务成功
                task_status = "success"
            else:
                task_status = "unknown"

            if task_status not in ["RUNNING", "success"]:
                print(f"Task failed or completed with status: {task_status}")
                break

            # 检查是否超过 run_timeout
            if time.time() - task_start_time > run_timeout:
                raise Exception(f"Timeout: Task {task_id} did not complete within {run_timeout} seconds.")

        # 4. 任务完成，处理输出
        return self.process_task_output(task_id, apiConfig["apiKey"], apiConfig["base_url"])

    def check_account_status(self, api_key, base_url):
        """
        查询账户状态，检查是否可以提交新任务
        """
        url = f"{base_url}/uc/openapi/accountStatus"
        headers = {
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            "Content-Type": "application/json",
        }
        data = {
            "apikey": api_key
        }

        response = requests.post(url, json=data, headers=headers)
        result = response.json()
        if result["code"] != 0:
            raise Exception(f"Failed to get account status: {result['msg']}")
        # 检查并确保 currentTaskCounts 是整数
        try:
            current_task_counts = int(result["data"]["currentTaskCounts"])
        except ValueError:
            raise Exception("Invalid value for currentTaskCounts. It should be an integer.")

        result["data"]["currentTaskCounts"] = current_task_counts
        return result["data"]

    def create_task(self, apiConfig, nodeInfoList, base_url):
        """
        创建任务
        """
        url = f"{apiConfig['base_url']}/task/openapi/create"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
        }
        data = {
            "workflowId": apiConfig["workflowId"],
            "apiKey": apiConfig["apiKey"],
            "nodeInfoList": [
                {
                    "nodeId": int(nodeInfo["nodeId"]),  # 确保 nodeId 为整数类型
                    "fieldName": nodeInfo["fieldName"],
                    "fieldValue": nodeInfo["fieldValue"],
                }
                for nodeInfo in nodeInfoList
            ],
        }

        response = requests.post(url, json=data, headers=headers)
        return response.json()

    def check_task_status(self, task_id, api_key, base_url):
        """
        查询任务状态
        """
        url = f"{base_url}/task/openapi/outputs"
        headers = {
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            "Content-Type": "application/json",
        }
        data = {
            "taskId": task_id,
            "apiKey": api_key
        }

        response = requests.post(url, json=data, headers=headers)
        
        # 打印响应以便调试
        print("Response Status Code:", response.status_code)
        try:
            response_json = response.json()
            print("Response JSON:", json.dumps(response_json, indent=4, ensure_ascii=False))
        except ValueError:
            print("Response Text:", response.text)

        if response.status_code != 200:
            raise Exception(f"HTTP request failed with status code: {response.status_code}")

        result = response.json()
        
        # 检查是否有数据结果
        if result.get("data") and isinstance(result["data"], list):
            if len(result["data"]) > 0:
                return result["data"]  # 返回整个列表，表示任务成功并且有结果
            else:
                # data 是空列表，可能是任务还在运行，或出错
                if result.get("code") != 0:
                    msg = result.get("msg")
                    if msg != "APIKEY_TASK_IS_RUNNING":
                        return {"error": msg}  # 出现了错误，返回错误信息
                    else:
                        return {"taskStatus": "RUNNING"}  # 任务仍在运行
                return {"taskStatus": "RUNNING"}  # 如果没有其他信息，认为任务在运行中

        # 如果没有 data 字段或 data 不是列表，检查任务状态
        if result.get("code") != 0:
            msg = result.get("msg")
            if msg != "APIKEY_TASK_IS_RUNNING":
                return {"error": msg}  # 返回错误信息
            else:
                return {"taskStatus": "RUNNING"}  # 任务仍在运行

        return {"taskStatus": "UNKNOWN_ERROR"}

    def process_task_output(self, task_id, api_key, base_url):
        """
        处理任务输出，返回文件链接。
        """
        task_status_result = self.check_task_status(task_id, api_key, base_url)
        
        # 记录任务状态结果以了解其结构
        print("Task Status Result:", json.dumps(task_status_result, indent=4, ensure_ascii=False))
        
        image_urls = []
        
        # 确保 task_status_result 是字典或列表
        if isinstance(task_status_result, dict):
            # 检查 fileUrl 和 fileType
            file_url = task_status_result.get("fileUrl")
            file_type = task_status_result.get("fileType")
            if file_url and file_type.lower() in ["png", "jpg", "jpeg"]:
                image_urls.append(file_url)
        elif isinstance(task_status_result, list):
            for output in task_status_result:
                if isinstance(output, dict):
                    file_url = output.get("fileUrl")
                    file_type = output.get("fileType")
                    if file_url and file_type.lower() in ["png", "jpg", "jpeg"]:
                        image_urls.append(file_url)
        
        if not image_urls:
            raise Exception("No valid image output found.")
        
        # 下载并处理所有图像
        image_data_list = []
        for url in image_urls:
            print("Downloading image from URL:", url)  # 记录图像 URL
            image_tensor = self.download_image(url)  # 下载并处理图像
            print("Image downloaded and processed successfully.")
            
            # 打印张量信息，避免在 uint8 上调用 mean()
            print(f"Image tensor shape: {image_tensor.shape}")
            mean_val = image_tensor.mean()
            print(f"Image tensor min: {image_tensor.min()}, max: {image_tensor.max()}, mean: {mean_val}")
            
            # 确保返回的是 torch.Tensor
            if not isinstance(image_tensor, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor, got {type(image_tensor)}")
            
            image_data_list.append(image_tensor)
        
        print(f"Returning {len(image_data_list)} images.")
        return (image_data_list,)  # 返回一个包含图像列表的元组，与 RETURN_TYPES 和 RETURN_NAMES 匹配

    def download_image(self, image_url):
        """
        从 URL 下载图像并转换为适合预览或保存的 torch.Tensor 格式。
        """
        response = requests.get(image_url)

        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert("RGB")
                        
            # 转换为 numpy 数组并调整数据类型和范围
            img_array = np.array(img).astype(np.float32) / 255.0  # 归一化到 [0, 1]
            
            # **保持形状为 [H, W, C]，不进行 permute 操作**
            img_tensor = torch.from_numpy(img_array)  # 形状 [H, W, C]
            
            # 打印图像尺寸
            print(f"Downloaded image dimensions: {img_tensor.shape}")  # 例如 (高度, 宽度, 3)
            
            return img_tensor

        else:
            raise Exception(f"Failed to download image: {image_url}")

    def download_video(self, video_url):
        """
        从 URL 下载视频。
        根据 ComfyUI 的要求实现此方法。
        """
        response = requests.get(video_url, stream=True)
        if response.status_code == 200:
            # 示例：将视频保存到临时位置并返回路径或数据
            video_content = response.content
            # 您可能需要根据 ComfyUI 的要求处理视频数据
            # 目前，返回原始字节
            return video_content
        else:
            raise Exception(f"Failed to download video: {video_url}")
