import requests
import time
import json
from PIL import Image
from io import BytesIO
import numpy as np
import torch
import os









class ExecuteNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "apiConfig": ("STRUCT",),
            },
            "optional": {
                "nodeInfoList": ("ARRAY", {"default": []}),
                "run_timeout": ("INT", {"default": 600}),
                "query_interval": ("INT", {"default": 10}),
                "concurrency_limit": ("INT", {"default": 1, "min": 1}),  
            },
        }

    RETURN_TYPES = ("IMAGE", "VIDEO")
    RETURN_NAMES = ("images", "videos")

    CATEGORY = "RunningHub"
    FUNCTION = "process"

    def process(self, apiConfig, nodeInfoList=None, run_timeout=600, query_interval=2, concurrency_limit=1):  
        # Ensure query_interval is not less than 2
        if query_interval < 2:
            print("Query interval is too low, setting to minimum value of 2 seconds.")
            query_interval = 2

        print(f"Concurrency limit set to: {concurrency_limit}") 

        # 1. 修改后的账户状态检查逻辑
        account_status = self.check_account_status(apiConfig["apiKey"], apiConfig["base_url"])
        current_tasks = int(account_status["currentTaskCounts"])
        print(f"There are {current_tasks} tasks running")
        # 当当前任务数 >= 并发限制时才等待
        if current_tasks >= concurrency_limit:  
            print(f"There are {current_tasks} tasks running (concurrency limit: {concurrency_limit}), waiting for them to finish.")  
            start_time = time.time()
            while int(account_status["currentTaskCounts"]) >= concurrency_limit and time.time() - start_time < run_timeout:
                time.sleep(query_interval)
                account_status = self.check_account_status(apiConfig["apiKey"], apiConfig["base_url"])
            
            # 超时后再次检查
            if int(account_status["currentTaskCounts"]) >= concurrency_limit:
                raise Exception(f"Timeout: Still have {account_status['currentTaskCounts']} running tasks (limit: {concurrency_limit}) after {run_timeout} seconds.")


        print(f"ExecuteNode NodeInfoList: {nodeInfoList}")
        
        # 2. 创建任务，如果 nodeInfoList 为空则不传递
        task_creation_result = self.create_task(apiConfig, nodeInfoList or [], apiConfig["base_url"])
        if task_creation_result["code"] != 0:
            raise Exception(f"Task creation failed: {task_creation_result['msg']}")

        task_id = task_creation_result["data"]["taskId"]
        task_status = task_creation_result["data"]["taskStatus"]
        print(f"Task created successfully, taskId: {task_id}, status: {task_status}")

        # 3. 查询任务状态直到任务完成
        

        task_start_time = time.time()
        while True:
            print(f"Task still running, checking again in {query_interval} seconds...")
            time.sleep(query_interval)
            task_status_result = self.check_task_status(task_id, apiConfig["apiKey"], apiConfig["base_url"])
            print(f"Task info, taskId: {task_id}, status: {task_status_result}")

            # 处理状态结果
            if isinstance(task_status_result, dict):
                # 特殊处理 APIKEY_TASK_IS_QUEUED 错误状态
                if task_status_result.get('error') == 'APIKEY_TASK_IS_QUEUED':
                    task_status = "QUEUED"
                else:
                    task_status = task_status_result.get("taskStatus", "unknown")
            elif isinstance(task_status_result, list):
                task_status = "success"
            else:
                task_status = "unknown"

            # 检查终止条件
            if task_status == "success":
                print("Task completed successfully")
                break

            if task_status not in ["QUEUED", "RUNNING"]:
                print(f"Task terminated with status: {task_status}")
                break

            # 检查超时
            if time.time() - task_start_time > run_timeout:
                raise Exception(f"Timeout: Task {task_id} did not complete within {run_timeout} seconds.")


        # 4. 任务完成，处理输出
        return self.process_task_output(task_id, apiConfig["apiKey"], apiConfig["base_url"])

    def process_task_output(self, task_id, api_key, base_url):
        """
        处理任务输出，返回文件链接。
        返回图像和视频，若无图像则返回空图像，若无视频则返回空视频。
        """
        task_status_result = self.check_task_status(task_id, api_key, base_url)
        
        print("Task Status Result:", json.dumps(task_status_result, indent=4, ensure_ascii=False))
        
        image_urls = []
        video_urls = []

        # 检查任务是否返回了图像
        if isinstance(task_status_result, dict):
            file_url = task_status_result.get("fileUrl")
            file_type = task_status_result.get("fileType")
            if file_url and file_type.lower() in ["png", "jpg", "jpeg"]:
                image_urls.append(file_url)
            elif file_url and file_type.lower() in ["mp4", "avi", "mov"]:
                video_urls.append(file_url)
        elif isinstance(task_status_result, list):
            for output in task_status_result:
                if isinstance(output, dict):
                    file_url = output.get("fileUrl")
                    file_type = output.get("fileType")
                    if file_url and file_type.lower() in ["png", "jpg", "jpeg"]:
                        image_urls.append(file_url)
                    elif file_url and file_type.lower() in ["mp4", "avi", "mov"]:
                        video_urls.append(file_url)
        
        if not image_urls:
            image_urls = None  # No images found, set to None
        if not video_urls:
            video_urls = None  # No videos found, set to None

        # Download and process images if available
        image_data_list = []
        if image_urls:
            for url in image_urls:
                print("Downloading image from URL:", url)
                image_tensor = self.download_image(url)
                image_data_list.append(image_tensor)
        
        # 下载视频并保存在本地
        video_data_list = []
        if video_urls:
            for url in video_urls:
                print("Downloading video from URL:", url)
                video_path = self.download_video(url)
                video_data_list.append(video_path)
        
        print(f"Returning {len(image_data_list)} images and {len(video_data_list)} videos.")
        return (image_data_list, video_data_list)

    def download_image(self, image_url):
        """
        从 URL 下载图像并转换为适合预览或保存的 torch.Tensor 格式。
        """
        response = requests.get(image_url)

        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img_array = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array)
            return img_tensor
        else:
            raise Exception(f"Failed to download image: {image_url}")

    def download_video(self, video_url):
        """
        从 URL 下载视频并保存到本地。
        """
        response = requests.get(video_url, stream=True)
        if response.status_code == 200:
            # 确保保存视频的目录存在
            output_dir = "output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 获取视频文件名并保存
            video_filename = f"RH_output_video_{str(int(time.time()))}.mp4"
            video_path = os.path.join(output_dir, video_filename)

            with open(video_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            
            print(f"Video saved to {video_path}")
            return video_path
        else:
            raise Exception(f"Failed to download video: {video_url}")

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
            "nodeInfoList": nodeInfoList,  # 如果nodeInfoList为空，传递空列表
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
        
        print("Response Status Code:", response.status_code)
        try:
            response_json = response.json()
            print("Response JSON:", json.dumps(response_json, indent=4, ensure_ascii=False))
        except ValueError:
            print("Response Text:", response.text)

        if response.status_code != 200:
            raise Exception(f"HTTP request failed with status code: {response.status_code}")

        result = response.json()
        
        if result.get("data") and isinstance(result["data"], list):
            if len(result["data"]) > 0:
                return result["data"]
            else:
                if result.get("code") != 0:
                    msg = result.get("msg")
                    if msg != "APIKEY_TASK_IS_RUNNING":
                        return {"error": msg}
                    else:
                        return {"taskStatus": "RUNNING"}
                return {"taskStatus": "RUNNING"}

        if result.get("code") != 0:
            msg = result.get("msg")
            if msg != "APIKEY_TASK_IS_RUNNING":
                return {"error": msg}
            else:
                return {"taskStatus": "RUNNING"}

        return {"taskStatus": "UNKNOWN_ERROR"}
