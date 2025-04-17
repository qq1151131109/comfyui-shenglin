import requests
import time
import json
from PIL import Image
from io import BytesIO
import numpy as np
import torch
import os
import websocket  # 需要安装 websocket-client 包
import threading

# Try importing folder_paths safely
try:
    import folder_paths
    comfyui_progress_available = True
except ImportError:
    comfyui_progress_available = False
    print("ComfyUI folder_paths not found. Progress bar disabled.")


class ExecuteNode:
    # Estimate the typical number of nodes in a workflow for progress calculation
    # Adjust this value if your workflows are significantly larger or smaller
    ESTIMATED_TOTAL_NODES = 10 

    def __init__(self):
        self.ws = None
        self.task_progress = 0
        self.task_completed = False
        self.ws_error = None
        self.progress_callback = None
        self.executed_nodes = set()
        self.prompt_tips = "{}" # Initialize prompt_tips


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
                # Allow overriding the estimated total nodes
                "estimated_total_nodes": ("INT", {"default": cls.ESTIMATED_TOTAL_NODES, "min": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "VIDEO")
    RETURN_NAMES = ("images", "videos")

    CATEGORY = "RunningHub"
    FUNCTION = "process"
    OUTPUT_NODE = True # Indicate support for progress display

    def on_ws_message(self, ws, message):
        """处理 WebSocket 消息，更新进度"""
        try:
            data = json.loads(message)
            message_type = data.get("type")

            # --- Progress Update Logic ---
            if message_type == "executing":
                node_data = data.get("data", {})
                node_id = node_data.get("node")

                if node_id is not None:
                    # Record newly started node
                    if node_id not in self.executed_nodes:
                        self.executed_nodes.add(node_id)
                        
                        # Calculate progress based on unique nodes seen vs estimated total
                        # Use the instance's estimated total nodes
                        estimated_total = self.instance_estimated_total_nodes
                        current_progress = len(self.executed_nodes) * 100 / estimated_total
                        
                        # Cap progress at 99% until the final completion message
                        capped_progress = min(int(current_progress), 99) 
                        
                        # Store and report progress
                        self.task_progress = capped_progress
                        if self.progress_callback:
                            self.progress_callback(capped_progress)
                        print(f"Executing node {node_id}. Progress: {capped_progress}% ({len(self.executed_nodes)}/{estimated_total} estimated nodes)")

                else: # node is null, means completion signal from WS
                    self.task_completed = True
                    self.task_progress = 100
                    if self.progress_callback:
                        self.progress_callback(100)
                    print("Task completed via WebSocket (node: null)")
            
            elif message_type == "execution_success":
                 # Ensure progress hits 100% on success message as a fallback
                 self.task_completed = True
                 self.task_progress = 100
                 if self.progress_callback:
                     self.progress_callback(100)
                 print("Task execution success message received.")

            # We ignore "progress" type messages for the overall bar now

        except json.JSONDecodeError:
            print(f"Received non-JSON message: {message}")
        except Exception as e:
            print(f"Error processing WebSocket message: {e}")

    def on_ws_error(self, ws, error):
        """处理 WebSocket 错误"""
        print(f"WebSocket error: {error}")
        self.ws_error = error

    def on_ws_close(self, ws, close_status_code, close_msg):
        """处理 WebSocket 关闭"""
        print(f"WebSocket closed: {close_status_code} - {close_msg}")

    def on_ws_open(self, ws):
        """处理 WebSocket 连接打开"""
        print("WebSocket connection established")

    def connect_websocket(self, wss_url):
        """建立 WebSocket 连接"""
        websocket.enableTrace(True)
        self.ws = websocket.WebSocketApp(
            wss_url,
            on_message=self.on_ws_message,
            on_error=self.on_ws_error,
            on_close=self.on_ws_close,
            on_open=self.on_ws_open
        )
        
        # 在新线程中运行 WebSocket
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True  # 设置为守护线程，这样主程序退出时会自动结束
        ws_thread.start()

    def process(self, apiConfig, nodeInfoList=None, run_timeout=600, query_interval=2, concurrency_limit=1, estimated_total_nodes=ESTIMATED_TOTAL_NODES):
        # --- Reset State ---
        self.executed_nodes.clear()
        self.task_progress = 0
        self.task_completed = False
        self.ws_error = None
        self.prompt_tips = "{}"
        # Store the estimated total nodes for this instance
        self.instance_estimated_total_nodes = estimated_total_nodes
        print(f"Using estimated total nodes for progress: {self.instance_estimated_total_nodes}")

        # --- Get Progress Callback ---
        self.progress_callback = self.get_progress_callback()
        if self.progress_callback:
            self.progress_callback(0) # Initialize progress bar to 0%
        else:
            print("Progress callback not available. UI progress bar will not be updated.")

        # ... (rest of the setup: query_interval check, concurrency limit print) ...
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
        print(f"Task Creation Result: {json.dumps(task_creation_result, indent=4, ensure_ascii=False)}")
        if task_creation_result["code"] != 0:
            raise Exception(f"Task creation failed: {task_creation_result['msg']}")

        # 在创建任务后，保存 promptTips 信息
        if task_creation_result["code"] == 0:
            self.prompt_tips = task_creation_result["data"].get("promptTips", "{}")
            # Optionally try to parse total nodes from promptTips here if format changes later

        # 获取 WebSocket URL 并建立连接
        wss_url = task_creation_result["data"]["netWssUrl"]
        self.connect_websocket(wss_url)

        task_id = task_creation_result["data"]["taskId"]
        task_status = task_creation_result["data"]["taskStatus"]
        print(f"Task created successfully, taskId: {task_id}, status: {task_status}")

        # --- Task Monitoring Loop ---
        task_start_time = time.time()
        while not self.task_completed: # Simplified loop condition
            if self.ws_error:
                print(f"WebSocket error occurred: {self.ws_error}. Falling back to polling.")
                self.ws = None # Stop relying on WS if it errors

            # Primary check: Rely on WebSocket to set self.task_completed
            if self.task_completed: 
                break

            # Backup Polling (only if WS fails or as a safety net)
            # You might reduce polling frequency if WS is active and stable
            # For simplicity, we keep polling, but the WS is the primary completion signal now.
            print(f"Polling task status (WebSocket active: {self.ws is not None})...")
            time.sleep(query_interval)
            
            try:
                task_status_result = self.check_task_status(task_id, apiConfig["apiKey"], apiConfig["base_url"])
                print(f"Polling result: {task_status_result}")

                # Determine status from polling result
                if isinstance(task_status_result, list):
                    polling_status = "success"
                elif isinstance(task_status_result, dict):
                    if task_status_result.get('error') == 'APIKEY_TASK_IS_QUEUED':
                        polling_status = "QUEUED"
                    elif task_status_result.get('taskStatus') == 'RUNNING':
                        polling_status = "RUNNING"
                    else: # Includes errors reported via polling
                         polling_status = task_status_result.get("error", "failed_or_unknown")
                else:
                    polling_status = "unknown"

                # Check polling termination conditions
                if polling_status == "success":
                    print("Task completed successfully (detected via polling).")
                    self.task_completed = True # Ensure loop terminates
                    if self.progress_callback: self.progress_callback(100) # Set final progress
                    break

                if polling_status not in ["QUEUED", "RUNNING"]:
                    print(f"Task terminated with status: {polling_status} (detected via polling).")
                    self.task_completed = True # Ensure loop terminates
                    # Optionally set progress to 100 or leave as is on failure? Let's set to 100 for now.
                    if self.progress_callback: self.progress_callback(100) 
                    break
            except Exception as poll_error:
                 print(f"Error during polling: {poll_error}")
                 # Decide if polling errors should stop the process or just be logged

            # Check overall timeout
            if time.time() - task_start_time > run_timeout:
                if self.ws:
                    self.ws.close()
                raise Exception(f"Timeout: Task {task_id} did not complete within {run_timeout} seconds.")

        # --- Cleanup and Output ---
        print("Exiting task monitoring loop.")
        if self.ws:
            self.ws.close()

        # Ensure final progress is 100%
        if self.progress_callback:
            self.progress_callback(100)
            
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

        # Check if task_status_result is valid before processing
        if task_status_result is None:
             print("Warning: Task status result is None when processing output.")
             return ([], []) # Return empty lists if status check failed

        #检查任务是否返回了图像
        if isinstance(task_status_result, dict) and task_status_result.get("taskStatus") == "success": # Example check
             file_url = task_status_result.get("fileUrl")
             file_type = task_status_result.get("fileType")
             if file_url and file_type:
                 if file_type.lower() in ["png", "jpg", "jpeg", "webp", "bmp"]: # Expanded image types
                    image_urls.append(file_url)
                 elif file_type.lower() in ["mp4", "avi", "mov", "webm", "gif"]: # Expanded video types
                    video_urls.append(file_url)
        elif isinstance(task_status_result, list):
            for output in task_status_result:
                if isinstance(output, dict):
                    file_url = output.get("fileUrl")
                    file_type = output.get("fileType")
                    if file_url and file_type:
                        if file_type.lower() in ["png", "jpg", "jpeg", "webp", "bmp"]:
                            image_urls.append(file_url)
                        elif file_type.lower() in ["mp4", "avi", "mov", "webm", "gif"]:
                             video_urls.append(file_url)
        else:
             # Handle cases where status is 'failed', 'unknown' or other dict structures
             print(f"Task may not have completed successfully or output format unexpected: {task_status_result}")


        # Download and process images if available
        image_data_list = []
        if image_urls:
            for url in image_urls:
                try:
                    print("Downloading image from URL:", url)
                    image_tensor = self.download_image(url)
                    if image_tensor is not None:
                         image_data_list.append(image_tensor)
                except Exception as img_e:
                     print(f"Error downloading or processing image {url}: {img_e}")
        
        # 下载视频并保存在本地
        video_data_list = []
        if video_urls:
            for url in video_urls:
                 try:
                    print("Downloading video from URL:", url)
                    video_path = self.download_video(url)
                    if video_path is not None:
                        video_data_list.append(video_path)
                 except Exception as vid_e:
                     print(f"Error downloading video {url}: {vid_e}")

        # Ensure we return lists, even if empty
        returned_images = image_data_list if image_data_list else []
        returned_videos = video_data_list if video_data_list else []

        print(f"Returning {len(returned_images)} images and {len(returned_videos)} videos.")
        return (returned_images, returned_videos)


    def download_image(self, image_url):
        """
        从 URL 下载图像并转换为适合预览或保存的 torch.Tensor 格式。
        包含重试机制，最多重试5次。
        """
        max_retries = 5
        retry_delay = 1  # 初始延迟1秒
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = requests.get(image_url, timeout=30) # Add timeout
                print(f"Download image attempt {attempt + 1} ({image_url}): Status code: {response.status_code}")
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                
                # Check content type if possible
                content_type = response.headers.get('Content-Type', '').lower()
                if 'image' not in content_type:
                    print(f"Warning: Content-Type '{content_type}' doesn't look like an image for URL {image_url}")
                
                img = Image.open(BytesIO(response.content)).convert("RGB")
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array)
                # Add batch dimension: [H, W, C] -> [1, H, W, C] (check if ComfyUI needs this)
                # Usually ComfyUI expects [B, H, W, C] or [B, C, H, W]
                # Let's assume [1, H, W, C] which is common for preview nodes
                # return img_tensor.unsqueeze(0) 
                # Or just return [H, W, C] if that's expected by downstream nodes
                return img_tensor # Returning [H, W, C] for now

            except (requests.exceptions.RequestException, IOError, Image.UnidentifiedImageError) as e:
                print(f"Download image attempt {attempt + 1} failed: {e}")
                last_exception = e
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"Failed to download image {image_url} after {max_retries} attempts.")
                    # Raise the last exception or return None/raise specific error
                    # raise Exception(f"Failed to download image {image_url}: {last_exception}") from last_exception
                    return None # Return None to allow processing to continue

        return None # Should not be reached if loop finishes


    def download_video(self, video_url):
        """
        从 URL 下载视频并保存到本地。
        包含重试机制，最多重试5次。
        """
        max_retries = 5
        retry_delay = 1  # 初始延迟1秒
        last_exception = None
        
        for attempt in range(max_retries):
            video_path = None # Define path outside try
            try:
                # Ensure output dir exists
                output_dir = folder_paths.get_output_directory() if comfyui_progress_available else "output"
                temp_dir = folder_paths.get_temp_directory() if comfyui_progress_available else "temp"
                
                # Determine filename and path
                try:
                    # Try to get filename from URL or Content-Disposition
                    # Basic parsing, might need improvement
                    parsed_url = requests.utils.urlparse(video_url)
                    filename_from_url = os.path.basename(parsed_url.path) if parsed_url.path else None
                    
                    # Fallback name
                    base_filename = filename_from_url if filename_from_url and '.' in filename_from_url else f"RH_output_video_{str(int(time.time()))}.mp4"
                    
                    # Use temp dir for download, then move to output? Or save directly?
                    # Saving directly to output for now
                    video_path = os.path.join(output_dir, base_filename) 
                    # Ensure filename uniqueness if needed (e.g., using folder_paths.generate_filename)
                    # video_path = os.path.join(folder_paths.get_output_directory(), folder_paths.generate_filename("RH_Video", ".mp4"))

                except Exception as path_e:
                     print(f"Error determining video path: {path_e}")
                     # Fallback path
                     if not os.path.exists("output"): os.makedirs("output")
                     video_path = os.path.join("output", f"RH_output_video_{str(int(time.time()))}.mp4")

                print(f"Attempting to download video to: {video_path}")
                
                response = requests.get(video_url, stream=True, timeout=60) # Increased timeout for video
                print(f"Download video attempt {attempt + 1} ({video_url}): Status code: {response.status_code}")
                response.raise_for_status()

                with open(video_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192): # Larger chunk size for video
                        if chunk:
                            f.write(chunk)
                
                print(f"Video saved successfully to {video_path}")
                return video_path # Success
                    
            except (requests.exceptions.RequestException, IOError) as e:
                 print(f"Download video attempt {attempt + 1} failed: {e}")
                 last_exception = e
                 # Clean up partially downloaded file if it exists
                 if video_path and os.path.exists(video_path):
                     try:
                         os.remove(video_path)
                         print(f"Removed partially downloaded file: {video_path}")
                     except OSError as rm_e:
                         print(f"Error removing partial file {video_path}: {rm_e}")

                 if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                 else:
                    print(f"Failed to download video {video_url} after {max_retries} attempts.")
                    # raise Exception(f"Failed to download video {video_url}: {last_exception}") from last_exception
                    return None # Return None

        return None # Should not be reached


    def check_account_status(self, api_key, base_url):
        """
        查询账户状态，检查是否可以提交新任务
        """
        url = f"{base_url}/uc/openapi/accountStatus"
        headers = {
            "User-Agent": "ComfyUI-RH-APICall-Node/1.0", # Identify client
            "Content-Type": "application/json",
        }
        data = {"apikey": api_key}
        result = None # Initialize result
        try:
            response = requests.post(url, json=data, headers=headers, timeout=15)
            response.raise_for_status()
            result = response.json()

            if result.get("code") != 0:
                raise Exception(f"API error getting account status: {result.get('msg', 'Unknown error')}")
            
            # Validate data structure
            account_data = result.get("data")
            if not account_data or "currentTaskCounts" not in account_data:
                 raise ValueError("Invalid response structure for account status.")

            # Ensure count is integer
            try:
                current_task_counts = int(account_data["currentTaskCounts"])
                account_data["currentTaskCounts"] = current_task_counts # Store validated int
                return account_data
            except (ValueError, TypeError) as e:
                 raise ValueError(f"Invalid value for currentTaskCounts: {account_data.get('currentTaskCounts')}. Error: {e}")

        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error getting account status: {e}") from e
        except json.JSONDecodeError as e:
             # Log the raw response text if possible
             raw_text = response.text if 'response' in locals() else "N/A"
             print(f"Raw response text on JSON decode error: {raw_text}")
             raise Exception(f"Failed to decode JSON response for account status: {e}") from e
        except Exception as e: # Catch other potential errors (like the custom exceptions raised above)
             # Re-raise the specific exception
             raise e

    def create_task(self, apiConfig, nodeInfoList, base_url):
        """
        创建任务，包含重试机制，最多重试5次
        """
        url = f"{apiConfig.get('base_url', 'https://default.url/if/missing')}/task/openapi/create" # Safer access
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ComfyUI-RH-APICall-Node/1.0",
        }
        data = {
            "workflowId": apiConfig.get("workflowId"),
            "apiKey": apiConfig.get("apiKey"),
            "nodeInfoList": nodeInfoList,
        }
        
        # Validate required config
        if not data["workflowId"] or not data["apiKey"] or not apiConfig.get('base_url'):
             raise ValueError("Missing required apiConfig fields: 'base_url', 'workflowId', 'apiKey'")

        max_retries = 5
        retry_delay = 1
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=data, headers=headers, timeout=30)
                print(f"Create task attempt {attempt + 1}: Status code {response.status_code}")
                response.raise_for_status() # Check for HTTP errors

                result = response.json()
                # Check API-level success code
                if result.get("code") == 0:
                    # Basic validation of expected fields
                    if "data" in result and "taskId" in result["data"] and "netWssUrl" in result["data"]:
                         return result # Success
                    else:
                         # Log the unexpected structure but treat as failure for retry
                         print(f"API success code 0, but response structure invalid: {result}")
                         last_exception = ValueError(f"API success code 0, but response structure invalid.") # Store error for potential raise later
                else:
                    api_msg = result.get('msg', 'Unknown API error')
                    print(f"API error creating task (code {result.get('code')}): {api_msg}")
                    last_exception = Exception(f"API error (code {result.get('code')}): {api_msg}") # Store error

            except requests.exceptions.Timeout as e:
                 print(f"Create task attempt {attempt + 1} timed out.")
                 last_exception = e
            except requests.exceptions.RequestException as e:
                print(f"Create task attempt {attempt + 1} network error: {e}")
                last_exception = e
            except json.JSONDecodeError as e:
                 print(f"Create task attempt {attempt + 1} failed to decode JSON response.")
                 raw_text = response.text if 'response' in locals() else "N/A"
                 print(f"Raw response text: {raw_text}")
                 last_exception = e
            except Exception as e: # Catch other unexpected errors
                 print(f"Create task attempt {attempt + 1} unexpected error: {e}")
                 last_exception = e
            
            # If we got here, it means the attempt failed. Check if we should retry.
            if attempt < max_retries - 1:
                print(f"Retrying task creation in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                # All retries failed
                error_message = f"Failed to create task after {max_retries} attempts."
                if last_exception:
                     error_message += f" Last error: {last_exception}"
                print(error_message)
                raise Exception(error_message) from last_exception

        # Should not be reachable if the loop finishes naturally
        raise Exception("Task creation failed unexpectedly after retry loop.")


    def check_task_status(self, task_id, api_key, base_url):
        """
        查询任务状态。 Returns a dictionary representing status or list of outputs on success.
        """
        url = f"{base_url}/task/openapi/outputs"
        headers = {
            "User-Agent": "ComfyUI-RH-APICall-Node/1.0",
            "Content-Type": "application/json",
        }
        data = { "taskId": task_id, "apiKey": api_key }
        
        try:
            response = requests.post(url, json=data, headers=headers, timeout=20)
            # Don't raise for status immediately, as API might use non-200 codes for status reporting
            print(f"Check status ({task_id}): Response Status Code: {response.status_code}")

            # Handle non-200 responses if they are expected for certain states
            if response.status_code != 200:
                 # Log unexpected HTTP errors, but maybe allow specific codes?
                 # For now, treat non-200 as a potential issue for status check
                 print(f"Warning: Non-200 status code ({response.status_code}) checking task status.")
                 # Depending on API, might return a specific error dict here or raise
                 # return {"taskStatus": "error", "error": f"HTTP Error {response.status_code}"}

            # Try to parse JSON regardless of status code (API might return error details in JSON)
            try:
                result = response.json()
                print(f"Check status ({task_id}): Response JSON: {json.dumps(result, indent=2, ensure_ascii=False)}")
            except json.JSONDecodeError:
                print(f"Check status ({task_id}): Failed to decode JSON. Response Text: {response.text}")
                return {"taskStatus": "error", "error": "Invalid JSON response"}

            # --- Process API Response ---
            api_code = result.get("code")
            api_msg = result.get("msg", "")
            api_data = result.get("data")

            # Success Case: data is a non-empty list
            if api_code == 0 and isinstance(api_data, list) and api_data:
                return api_data # Return the list of output files

            # Running Case: Specific API message
            elif api_msg == "APIKEY_TASK_IS_RUNNING":
                return {"taskStatus": "RUNNING"}
            
            # Queued Case: Specific API message (adjust if different)
            elif api_msg == "APIKEY_TASK_IS_QUEUED": # Assuming this message exists
                 return {"taskStatus": "QUEUED"}

            # Other API Errors reported via code/msg
            elif api_code != 0:
                 print(f"API Error checking status (code {api_code}): {api_msg}")
                 return {"taskStatus": "error", "error": api_msg}
            
            # Handle cases like code=0 but data is empty list or null (might mean still running or just no output yet)
            elif api_code == 0 and (api_data is None or (isinstance(api_data, list) and not api_data)):
                 print("Task status check returned code 0 but no data - assuming still running.")
                 return {"taskStatus": "RUNNING"} 

            # Fallback for unknown states
            else:
                print(f"Unknown task status response: {result}")
                return {"taskStatus": "unknown", "details": result}

        except requests.exceptions.RequestException as e:
            print(f"Network error checking task status: {e}")
            return {"taskStatus": "error", "error": f"Network Error: {e}"}
        except Exception as e:
             print(f"Unexpected error checking task status: {e}")
             return {"taskStatus": "error", "error": f"Unexpected Error: {e}"}


    def get_progress_callback(self):
        """获取并验证进度回调函数"""
        if not comfyui_progress_available:
             print("ComfyUI environment not detected. Progress callback unavailable.")
             return None
        try:
            # Check if the function exists and is callable
            if hasattr(folder_paths, 'get_progress_callback') and callable(folder_paths.get_progress_callback):
                callback = folder_paths.get_progress_callback()
                # Basic check: is it callable?
                if callable(callback):
                    print(f"Progress callback obtained successfully: {type(callback)}")
                    return callback
                else:
                    print(f"Error: folder_paths.get_progress_callback() did not return a callable object. Got: {type(callback)}")
                    return None
            else:
                print("Error: folder_paths.get_progress_callback function not found or not callable.")
                return None
        except Exception as e:
            print(f"Error getting progress callback: {e}")
            return None
