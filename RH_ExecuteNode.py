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
import comfy.utils # Import comfy utils for ProgressBar

# Try importing folder_paths safely
try:
    import folder_paths
    comfyui_env_available = True # Use a more generic name
except ImportError:
    comfyui_env_available = False
    print("ComfyUI folder_paths not found. Some features like specific output paths might use fallbacks.")


class ExecuteNode:
    ESTIMATED_TOTAL_NODES = 10 # Default estimate

    def __init__(self):
        self.ws = None
        self.task_completed = False
        self.ws_error = None
        self.executed_nodes = set()
        self.prompt_tips = "{}"
        self.pbar = None
        self.node_lock = threading.Lock()
        self.total_nodes = None
        self.current_steps = 0 # Track current steps for logging

    def update_progress(self):
        """Increments the progress bar by one step and logs, stopping at total_nodes."""
        # --- Guard Condition ---
        # Use lock to ensure thread safety when checking/updating steps and flag
        with self.node_lock:
            if self.task_completed or (self.pbar and self.current_steps >= self.total_nodes):
                # Print only if trying to update *after* completion for debugging
                if self.task_completed:
                    print(f"Skipping progress update because task is already completed.")
                return

            if self.pbar:
                self.current_steps += 1
                # Increment the ComfyUI progress bar by 1
                self.pbar.update(1)
                # Log the current state
                display_steps = min(self.current_steps, self.total_nodes) # Ensure log doesn't exceed total
                print(f"Progress Update: Step {display_steps}/{self.total_nodes} ({(display_steps/self.total_nodes)*100:.1f}%)")


    def complete_progress(self):
        """Sets the progress bar to 100% and marks task as completed."""
        # --- Use lock for thread safety ---
        with self.node_lock:
            # Check if already completed to prevent redundant calls/logs
            if self.task_completed:
                return

            print(f"Finalizing progress: Setting task_completed = True")
            # --- Set completion flag FIRST ---
            self.task_completed = True

            if self.pbar:
                # Only force update to 100% if update_progress didn't already reach it
                if self.current_steps < self.total_nodes:
                    print(f"Forcing progress bar to 100% as final step.")
                    self.current_steps = self.total_nodes # Ensure internal counter matches
                    self.pbar.update_absolute(1.0)
                    print(f"Progress Finalized: {self.total_nodes}/{self.total_nodes} (100.0%)")
                else:
                    # If current_steps already == total_nodes, update_progress handled the last visual update
                    print(f"Progress already at 100% ({self.current_steps}/{self.total_nodes}). Finalization complete.")
            else:
                 print("Progress bar not available during finalization.")


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "apiConfig": ("STRUCT",),
            },
            "optional": {
                "nodeInfoList": ("ARRAY", {"default": []}),
                "run_timeout": ("INT", {"default": 600}),
                "concurrency_limit": ("INT", {"default": 1, "min": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "VIDEO")
    RETURN_NAMES = ("images", "videos")

    CATEGORY = "RunningHub"
    FUNCTION = "process"
    OUTPUT_NODE = True # Indicate support for progress display

    # --- WebSocket Handlers ---
    def on_ws_message(self, ws, message):
        """处理 WebSocket 消息，更新内部状态和进度条"""
        try:
            # --- Check completion status AT THE START ---
            # This check is implicitly thread-safe due to complete_progress lock
            if self.task_completed:
                 # print("WS Message received after task completion, ignoring.") # Optional: reduce log spam
                 return

            data = json.loads(message)
            message_type = data.get("type")

            if message_type == "executing":
                node_data = data.get("data", {})
                node_id = node_data.get("node")
                if node_id is not None:
                    # Lock is handled within update_progress now
                    # Check if it's a new node before calling update
                    if node_id not in self.executed_nodes:
                         self.executed_nodes.add(node_id) # Add before update call
                         self.update_progress() # This method is now guarded internally
                         print(f"WS: Node {node_id} executed.")
                else:
                    # Null node signal check remains guarded by the top-level check
                    print("WS: Received null node signal, waiting for final success signal...")

            elif message_type == "execution_success":
                 # The internal check in complete_progress handles redundancy
                 print("WS: Task execution success signal received.")
                 self.complete_progress()
                 # No need for sleep or setting task_completed here

        except Exception as e:
            print(f"Error processing WebSocket message: {e}")
            self.ws_error = e
            # Call complete_progress which handles the task_completed flag and lock
            self.complete_progress()

    def on_ws_error(self, ws, error):
        """处理 WebSocket 错误"""
        print(f"WebSocket error: {error}")
        self.ws_error = error
        # Mark task as complete via the centralized method
        self.complete_progress()

    def on_ws_close(self, ws, close_status_code, close_msg):
        """处理 WebSocket 关闭"""
        print(f"WebSocket closed: {close_status_code} - {close_msg}")
        # If closed unexpectedly, mark as complete to end loop
        # Use lock temporarily just to read task_completed safely
        with self.node_lock:
             should_complete = not self.task_completed
        if should_complete:
             print("Warning: WebSocket closed unexpectedly. Forcing task completion.")
             self.ws_error = self.ws_error or IOError(f"WebSocket closed unexpectedly ({close_status_code})")
             # Mark task as complete via the centralized method
             self.complete_progress()

    def on_ws_open(self, ws):
        """处理 WebSocket 连接打开"""
        print("WebSocket connection established")
        # Note: executed_nodes should be cleared at the start of 'process'

    def connect_websocket(self, wss_url):
        """建立 WebSocket 连接"""
        print(f"Connecting to WebSocket: {wss_url}")
        websocket.enableTrace(False) # Keep this false unless debugging WS protocol
        self.ws = websocket.WebSocketApp(
            wss_url,
            on_message=self.on_ws_message,
            on_error=self.on_ws_error,
            on_close=self.on_ws_close,
            on_open=self.on_ws_open
        )
        ws_thread = threading.Thread(target=self.ws.run_forever, name="RH_ExecuteNode_WSThread")
        ws_thread.daemon = True
        ws_thread.start()
        print("WebSocket thread started.")

    def check_and_complete_task(self):
        """If task times out after null node, force completion."""
        # complete_progress now checks the flag internally and uses lock
        print("Task completion timeout after null node signal - attempting forced completion.")
        self.complete_progress()

    def get_workflow_node_count(self, api_key, base_url, workflow_id):
        """Get the total number of nodes from workflow JSON."""
        url = f"{base_url}/api/openapi/getJsonApiFormat"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ComfyUI-RH-APICall-Node/1.0",
        }
        data = {
            "apiKey": api_key,
            "workflowId": workflow_id
        }

        try:
            response = requests.post(url, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()

            if result.get("code") != 0:
                raise Exception(f"API error getting workflow: {result.get('msg', 'Unknown error')}")

            workflow_json = result.get("data", {}).get("prompt")
            if not workflow_json:
                raise Exception("No workflow data found in response")

            # Parse the workflow JSON
            workflow_data = json.loads(workflow_json)
            
            # Count the number of nodes
            node_count = len(workflow_data)
            print(f"Workflow contains {node_count} nodes")
            return node_count

        except Exception as e:
            print(f"Error getting workflow node count: {e}")
            # Return default value if API call fails
            return self.ESTIMATED_TOTAL_NODES

    # --- Main Process Method ---
    def process(self, apiConfig, nodeInfoList=None, run_timeout=600, concurrency_limit=1):
        # Reset state
        with self.node_lock: # Use lock for resetting shared state
            self.executed_nodes.clear()
            self.task_completed = False
            self.ws_error = None
            self.prompt_tips = "{}"
            self.current_steps = 0 # Reset step counter

        # Get workflow node count from API
        try:
            api_key = apiConfig.get("apiKey")
            base_url = apiConfig.get("base_url")
            workflow_id = apiConfig.get("workflowId")
            
            if not all([api_key, base_url, workflow_id]):
                raise ValueError("Missing required apiConfig fields: apiKey, base_url, or workflowId")

            # Get actual node count from workflow
            actual_node_count = self.get_workflow_node_count(api_key, base_url, workflow_id)
            # Use the actual node count directly
            self.total_nodes = actual_node_count
            print(f"Using actual total nodes for progress: {self.total_nodes}")
        except Exception as e:
            print(f"Error getting workflow node count, using default value: {e}")
            self.total_nodes = self.ESTIMATED_TOTAL_NODES
            print(f"Using default total nodes for progress: {self.total_nodes}")

        # Initialize ComfyUI progress bar
        self.pbar = comfy.utils.ProgressBar(self.total_nodes)
        print("Progress bar initialized at 0")

        # --- Concurrency Check ---
        api_key = None
        base_url = None
        try:
            api_key = apiConfig.get("apiKey")
            base_url = apiConfig.get("base_url")
            if not api_key or not base_url:
                 raise ValueError("apiKey and base_url missing from apiConfig")

            account_status = self.check_account_status(api_key, base_url)
            current_tasks = int(account_status["currentTaskCounts"])
            print(f"There are {current_tasks} tasks running")
            
            if current_tasks >= concurrency_limit:
                print(f"Concurrency limit ({concurrency_limit}) reached, waiting...")
                start_wait_time = time.time()
                # Use a shorter sleep interval while waiting for concurrency
                wait_interval = 2 # seconds
                while current_tasks >= concurrency_limit:
                    if time.time() - start_wait_time > run_timeout:
                        if self.pbar: self.pbar.update_absolute(1.0) # Use absolute directly for setup failure
                        raise Exception(f"Timeout waiting for concurrent tasks ({current_tasks}/{concurrency_limit}) to finish.")
                    print(f"Waiting for concurrent tasks... ({current_tasks}/{concurrency_limit})")
                    time.sleep(wait_interval) 
                    account_status = self.check_account_status(api_key, base_url)
                    current_tasks = int(account_status["currentTaskCounts"])
                print("Concurrency slot available.")
        except Exception as e:
             print(f"Error checking account status or waiting: {e}")
             if self.pbar: self.pbar.update_absolute(1.0) # Use absolute directly for setup failure
             raise

        # --- Task Creation & WebSocket ---
        task_id = None
        try:
            print(f"ExecuteNode NodeInfoList: {nodeInfoList}")
            # Pass base_url explicitly from the validated config
            task_creation_result = self.create_task(apiConfig, nodeInfoList or [], base_url) 
            print(f"Task Creation Result: {json.dumps(task_creation_result, indent=2, ensure_ascii=False)}")
            
            # Validate task creation response structure before accessing data
            if not isinstance(task_creation_result.get("data"), dict):
                 raise ValueError("Invalid task creation response data structure.")
            
            self.prompt_tips = task_creation_result["data"].get("promptTips", "{}")
            task_id = task_creation_result["data"].get("taskId")
            wss_url = task_creation_result["data"].get("netWssUrl")
            
            if not task_id or not wss_url:
                 raise ValueError("Missing taskId or netWssUrl in task creation response.")
                 
            print(f"Task created successfully, taskId: {task_id}")
            self.connect_websocket(wss_url)
        except Exception as e:
             print(f"Error creating task or connecting WS: {e}")
             if self.pbar: self.pbar.update_absolute(1.0) # Use absolute directly for setup failure
             raise

        # --- Task Monitoring Loop ---
        task_start_time = time.time()
        loop_sleep_interval = 0.1
        print("Starting task monitoring loop...")

        timeout_timer = None
        try:
            # Setup global timeout timer
            def force_timeout():
                # Use lock to safely check task_completed
                with self.node_lock:
                     is_completed = self.task_completed
                if not is_completed:
                    print("Global timeout reached - forcing task completion.")
                    self.ws_error = Exception("Global timeout reached")
                    # Let the main loop call complete_progress via the finally block or error handling
                    # Just set the flags here to break loop
                    self.task_completed = True # Set flag directly here to break loop

            timeout_timer = threading.Timer(run_timeout, force_timeout)
            timeout_timer.daemon = True
            timeout_timer.start()

            # Main wait loop
            while True:
                # Check completion flags (read safely with lock)
                with self.node_lock:
                     is_completed = self.task_completed
                     current_error = self.ws_error
                if is_completed or current_error:
                     break # Exit loop if completed or error occurred

                # Check for timeout explicitly in loop as backup/alternative to timer
                if time.time() - task_start_time > run_timeout:
                     print("Task monitoring loop timeout check triggered.")
                     # Set flags to exit loop; rely on finally block for completion
                     with self.node_lock:
                         if not self.task_completed: # Avoid overwriting WS error
                             self.ws_error = self.ws_error or Exception(f"Timeout: Task {task_id} did not complete within {run_timeout} seconds.")
                         self.task_completed = True # Ensure loop exit
                     break # Exit loop

                time.sleep(loop_sleep_interval) # Yield CPU

            # Handle exit conditions after loop
            with self.node_lock: # Read error flag safely
                 final_error = self.ws_error

            if final_error:
                print(f"Task ended with error: {final_error}")
                # Only complete progress if not already completed by WS handler
                with self.node_lock:
                    if not self.task_completed:
                        self.complete_progress()
                raise final_error # Re-raise the error
            else: # Task completed normally
                print("Task monitoring completed successfully.")
                # complete_progress should have been called by WS handler

        finally:
            # Cleanup
            if timeout_timer:
                timeout_timer.cancel()
            if self.ws:
                try:
                    self.ws.close()
                except Exception as e:
                    print(f"Error closing WebSocket: {e}")
                self.ws = None

            # Final safety net: Only complete progress if not already completed
            with self.node_lock:
                 is_finally_completed = self.task_completed
            if not is_finally_completed:
                 print("Warning: Monitoring loop ended unexpectedly. Finalizing progress via finally block.")
                 self.complete_progress() # Call the safe completion method


        # --- Process Output ---
        print("Processing task output...")
        # Pass the validated api_key and base_url again
        return self.process_task_output(task_id, api_key, base_url)

    def process_task_output(self, task_id, api_key, base_url):
        """处理任务输出，包含轮询等待机制"""
        max_retries = 30  # 最多等待30次
        retry_interval = 1  # 初始等待1秒
        max_retry_interval = 5  # 最大等待间隔5秒

        for attempt in range(max_retries):
            try:
                task_status_result = self.check_task_status(task_id, api_key, base_url)
                print(f"Check output attempt {attempt + 1}/{max_retries}")
                print("Task Status Result:", json.dumps(task_status_result, indent=2, ensure_ascii=False))

                # 如果任务仍在运行，等待后重试
                if isinstance(task_status_result, dict) and task_status_result.get("taskStatus") in ["RUNNING", "QUEUED"]:
                    wait_time = min(retry_interval * (1.5 ** attempt), max_retry_interval)
                    print(f"Task still running, waiting {wait_time:.1f} seconds before next check...")
                    time.sleep(wait_time)
                    continue

                # 如果获取到了实际的输出结果（文件列表）
                if isinstance(task_status_result, list) and len(task_status_result) > 0:
                    print("Got valid output result, processing files...")
                    image_urls = []
                    video_urls = []

                    for output in task_status_result:
                        if isinstance(output, dict):
                            file_url = output.get("fileUrl")
                            file_type = output.get("fileType")
                            if file_url and file_type:
                                file_type_lower = file_type.lower()
                                if file_type_lower in ["png", "jpg", "jpeg", "webp", "bmp", "gif"]:
                                    image_urls.append(file_url)
                                elif file_type_lower in ["mp4", "avi", "mov", "webm"]:
                                    video_urls.append(file_url)

                    # 处理图片和视频
                    image_data_list = []
                    if image_urls:
                        print(f"Downloading {len(image_urls)} images...")
                        for url in image_urls:
                            try:
                                print(f"Downloading image: {url}")
                                image_tensor = self.download_image(url)
                                if image_tensor is not None:
                                    image_data_list.append(image_tensor)
                            except Exception as img_e:
                                print(f"Error downloading image {url}: {img_e}")

                    video_data_list = []
                    if video_urls:
                        print(f"Downloading {len(video_urls)} videos...")
                        for url in video_urls:
                            try:
                                print(f"Downloading video: {url}")
                                video_path = self.download_video(url)
                                if video_path is not None:
                                    video_data_list.append(video_path)
                            except Exception as vid_e:
                                print(f"Error downloading video {url}: {vid_e}")

                    if image_data_list or video_data_list:
                        print(f"Successfully got results: {len(image_data_list)} images, {len(video_data_list)} videos")
                        return (image_data_list, video_data_list)

                # 如果是错误状态
                if isinstance(task_status_result, dict) and task_status_result.get("taskStatus") == "error":
                    print(f"Task error: {task_status_result.get('error', 'Unknown error')}")
                    return ([], [])

            except Exception as e:
                print(f"Error checking task status (attempt {attempt + 1}): {e}")
                time.sleep(retry_interval)

        print(f"Failed to get valid output after {max_retries} attempts")
        return ([], [])


    def download_image(self, image_url):
        """
        从 URL 下载图像并转换为适合预览或保存的 torch.Tensor 格式。
        包含重试机制，最多重试5次。
        Returns tensor [H, W, C] or None on failure.
        """
        max_retries = 5
        retry_delay = 1
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = requests.get(image_url, timeout=30) 
                print(f"Download image attempt {attempt + 1} ({image_url}): Status code: {response.status_code}")
                response.raise_for_status()
                
                content_type = response.headers.get('Content-Type', '').lower()
                
                img = Image.open(BytesIO(response.content)).convert("RGB")
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array) # Shape: [H, W, C]
                return img_tensor 

            except (requests.exceptions.RequestException, IOError, Image.UnidentifiedImageError) as e:
                print(f"Download image attempt {attempt + 1} failed: {e}")
                last_exception = e
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"Failed to download image {image_url} after {max_retries} attempts.")
                    return None

        return None 


    def download_video(self, video_url):
        """
        从 URL 下载视频并保存到本地。
        包含重试机制，最多重试5次。
        Returns the local file path string or None on failure.
        """
        max_retries = 5
        retry_delay = 1
        last_exception = None
        
        for attempt in range(max_retries):
            video_path = None 
            try:
                output_dir = "output" 
                if comfyui_env_available and hasattr(folder_paths, 'get_output_directory'):
                    try:
                         output_dir = folder_paths.get_output_directory()
                    except Exception as e_dir:
                         print(f"Warning: Could not get output directory from folder_paths: {e_dir}. Using default 'output'.")
                
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    print(f"Created output directory: {output_dir}")

                try:
                    parsed_url = requests.utils.urlparse(video_url)
                    filename_from_url = os.path.basename(parsed_url.path) if parsed_url.path and '.' in os.path.basename(parsed_url.path) else None
                    
                    safe_filename_from_url = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in filename_from_url) if filename_from_url else None

                    base_filename = safe_filename_from_url if safe_filename_from_url else f"RH_output_video_{str(int(time.time()*1000))}.mp4"

                    counter = 0
                    video_path_base, video_ext = os.path.splitext(base_filename)
                    max_base_len = 100 
                    video_path_base = video_path_base[:max_base_len]
                    
                    video_path = os.path.join(output_dir, f"{video_path_base}{video_ext}")
                    while os.path.exists(video_path):
                         counter += 1
                         video_path = os.path.join(output_dir, f"{video_path_base}_{counter}{video_ext}")
                         if counter > 100: 
                              print("Warning: Could not find unique filename after 100 attempts.")
                              video_path = os.path.join(output_dir, f"{video_path_base}_{str(int(time.time()*1000))}{video_ext}")
                              break
                except Exception as path_e:
                     print(f"Error determining video path: {path_e}")
                     fallback_filename = f"RH_output_video_fallback_{str(int(time.time()*1000))}.mp4"
                     video_path = os.path.join(output_dir, fallback_filename)

                print(f"Attempting to download video to: {video_path}")
                
                response = requests.get(video_url, stream=True, timeout=60) 
                print(f"Download video attempt {attempt + 1} ({video_url}): Status code: {response.status_code}")
                response.raise_for_status()

                content_type = response.headers.get('Content-Type', '').lower()
                if not any(vid_type in content_type for vid_type in ['video/', 'octet-stream']):
                     print(f"Warning: Content-Type '{content_type}' may not be a video for URL {video_url}")

                downloaded_size = 0
                with open(video_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=65536): 
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                
                if downloaded_size > 0:
                     print(f"Video saved successfully to {video_path} ({downloaded_size / (1024*1024):.2f} MB)")
                     return video_path 
                else:
                     print(f"Warning: Downloaded video file is empty: {video_path}")
                     if os.path.exists(video_path): os.remove(video_path) 
                     last_exception = IOError("Downloaded video file is empty.")

            except (requests.exceptions.RequestException, IOError) as e:
                 print(f"Download video attempt {attempt + 1} failed: {e}")
                 last_exception = e
                 if video_path and os.path.exists(video_path):
                     try:
                         os.remove(video_path)
                         print(f"Removed partial/failed download file: {video_path}")
                     except OSError as rm_e:
                         print(f"Error removing partial file {video_path}: {rm_e}")
            
            # Retry logic
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"Failed to download video {video_url} after {max_retries} attempts.")
                return None

        return None 


    def check_account_status(self, api_key, base_url):
        """
        查询账户状态，检查是否可以提交新任务。包含重试机制。
        """
        if not api_key or not base_url:
            raise ValueError("API Key and Base URL are required for checking account status.")
        
        url = f"{base_url}/uc/openapi/accountStatus"
        headers = {
            "User-Agent": "ComfyUI-RH-APICall-Node/1.0", 
            "Content-Type": "application/json",
        }
        data = {"apikey": api_key}
        
        max_retries = 5
        retry_delay = 1
        last_exception = None

        for attempt in range(max_retries):
            response = None 
            try:
                print(f"Attempt {attempt + 1}/{max_retries} to check account status...")
                response = requests.post(url, json=data, headers=headers, timeout=15)
                response.raise_for_status()
                
                result = response.json()

                if result.get("code") != 0:
                    api_msg = result.get('msg', 'Unknown API error')
                    print(f"API error on attempt {attempt + 1}: {api_msg}")
                    raise Exception(f"API error getting account status: {api_msg}") 
                
                account_data = result.get("data")
                if not account_data or "currentTaskCounts" not in account_data:
                    raise ValueError("Invalid response structure for account status.")

                try:
                    current_task_counts = int(account_data["currentTaskCounts"])
                    account_data["currentTaskCounts"] = current_task_counts 
                    print("Account status check successful.")
                    return account_data # Success
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid value for currentTaskCounts: {account_data.get('currentTaskCounts')}. Error: {e}")

            except (requests.exceptions.RequestException, json.JSONDecodeError, ValueError, Exception) as e:
                print(f"Error on attempt {attempt + 1}/{max_retries}: {e}")
                last_exception = e
                if isinstance(e, json.JSONDecodeError) and response is not None:
                     print(f"Raw response text on JSON decode error: {response.text}")

                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2 
                else:
                    print("Max retries reached for checking account status.")
                    raise Exception(f"Failed to check account status after {max_retries} attempts. Last error: {last_exception}") from last_exception
            
        raise Exception(f"Failed to check account status after {max_retries} attempts (unexpected loop end). Last error: {last_exception}")


    def create_task(self, apiConfig, nodeInfoList, base_url):
        """
        创建任务，包含重试机制，最多重试5次
        """
        safe_base_url = apiConfig.get('base_url')
        safe_workflow_id = apiConfig.get("workflowId")
        safe_api_key = apiConfig.get("apiKey")

        if not safe_base_url or not safe_workflow_id or not safe_api_key:
             raise ValueError("Missing required apiConfig fields: 'base_url', 'workflowId', 'apiKey'")
        
        url = f"{safe_base_url}/task/openapi/create" 
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ComfyUI-RH-APICall-Node/1.0",
        }
        data = {
            "workflowId": safe_workflow_id,
            "apiKey": safe_api_key,
            "nodeInfoList": nodeInfoList, 
        }

        max_retries = 5
        retry_delay = 1
        last_exception = None
        
        for attempt in range(max_retries):
            response = None 
            current_last_exception = None 
            try:
                print(f"Create task attempt {attempt + 1}/{max_retries}...")
                response = requests.post(url, json=data, headers=headers, timeout=30)
                print(f"Create task attempt {attempt + 1}: Status code {response.status_code}")
                response.raise_for_status() 

                result = response.json()
                
                if result.get("code") == 0:
                    if "data" in result and "taskId" in result["data"] and "netWssUrl" in result["data"]:
                         print("Task created successfully.")
                         return result 
                    else:
                         print(f"API success code 0, but response structure invalid: {result}")
                         current_last_exception = ValueError(f"API success code 0, but response structure invalid.") 
                else:
                    api_msg = result.get('msg', 'Unknown API error')
                    print(f"API error creating task (code {result.get('code')}): {api_msg}")
                    current_last_exception = Exception(f"API error (code {result.get('code')}): {api_msg}") 

            except requests.exceptions.Timeout as e:
                 print(f"Create task attempt {attempt + 1} timed out.")
                 current_last_exception = e
            except requests.exceptions.RequestException as e:
                print(f"Create task attempt {attempt + 1} network error: {e}")
                current_last_exception = e
            except json.JSONDecodeError as e:
                 print(f"Create task attempt {attempt + 1} failed to decode JSON response.")
                 if response is not None: print(f"Raw response text: {response.text}")
                 current_last_exception = e
            except Exception as e: 
                 print(f"Create task attempt {attempt + 1} unexpected error: {e}")
                 current_last_exception = e
            
            if current_last_exception is not None: 
                 last_exception = current_last_exception 
                 if attempt < max_retries - 1:
                     print(f"Retrying task creation in {retry_delay} seconds...")
                     time.sleep(retry_delay)
                     retry_delay *= 2
                 else:
                     error_message = f"Failed to create task after {max_retries} attempts."
                     if last_exception: 
                          error_message += f" Last error: {last_exception}"
                     print(error_message)
                     raise Exception(error_message) from last_exception
            else:
                 if attempt == max_retries - 1:
                      error_message = f"Failed to create task after {max_retries} attempts (unknown reason)."
                      if last_exception: error_message += f" Last error: {last_exception}"
                      print(error_message)
                      raise Exception(error_message) from last_exception

        raise Exception("Task creation failed unexpectedly after retry loop.")


    def check_task_status(self, task_id, api_key, base_url):
        """
        查询任务状态。 Returns a dictionary representing status or list of outputs on success.
        """
        if not task_id or not api_key or not base_url:
             raise ValueError("Task ID, API Key, and Base URL are required for checking task status.")
        url = f"{base_url}/task/openapi/outputs"
        headers = {
            "User-Agent": "ComfyUI-RH-APICall-Node/1.0",
            "Content-Type": "application/json",
        }
        data = { "taskId": task_id, "apiKey": api_key }
        
        try:
            response = requests.post(url, json=data, headers=headers, timeout=20)
            print(f"Check status ({task_id}): Response Status Code: {response.status_code}")

            try:
                result = response.json()
                print(f"Check status ({task_id}): Response JSON: {json.dumps(result, indent=2, ensure_ascii=False)}")
            except json.JSONDecodeError:
                print(f"Check status ({task_id}): Failed to decode JSON. Response Text: {response.text}")
                error_msg = f"HTTP Error {response.status_code} and Invalid JSON" if response.status_code != 200 else "Invalid JSON response"
                return {"taskStatus": "error", "error": error_msg}

            api_code = result.get("code")
            api_msg = result.get("msg", "")
            api_data = result.get("data")

            if response.status_code != 200:
                 error_detail = api_msg if api_msg else f"HTTP Error {response.status_code}"
                 print(f"Warning: Non-200 status code ({response.status_code}). API Message: {api_msg}")
                 return {"taskStatus": "error", "error": error_detail}

            if api_code == 0 and isinstance(api_data, list) and api_data:
                return api_data 

            elif api_msg == "APIKEY_TASK_IS_RUNNING":
                return {"taskStatus": "RUNNING"}
            
            elif api_msg == "APIKEY_TASK_IS_QUEUED":
                 return {"taskStatus": "QUEUED"}

            elif api_code != 0:
                 print(f"API Error checking status (code {api_code}): {api_msg}")
                 return {"taskStatus": "error", "error": api_msg}
            
            elif api_code == 0 and (api_data is None or (isinstance(api_data, list) and not api_data)):
                 print("Task status check returned code 0 but no data - assuming still running.")
                 return {"taskStatus": "RUNNING"} 

            else:
                print(f"Unknown task status response: {result}")
                return {"taskStatus": "unknown", "details": result}

        except requests.exceptions.Timeout:
            print(f"Network timeout checking task status for {task_id}")
            return {"taskStatus": "error", "error": "Network Timeout"}
        except requests.exceptions.RequestException as e:
            print(f"Network error checking task status: {e}")
            return {"taskStatus": "error", "error": f"Network Error: {e}"}
        except Exception as e:
             print(f"Unexpected error checking task status: {e}")
             return {"taskStatus": "error", "error": f"Unexpected Error: {e}"}

