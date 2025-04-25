import requests
import time
import json
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import numpy as np
import torch
import os
import websocket  # 需要安装 websocket-client 包
import threading
import comfy.utils # Import comfy utils for ProgressBar
import cv2 # <<< Added import for OpenCV
import safetensors.torch # <<< Added safetensors import
import torchaudio 
import torch.nn.functional as F # <<< Add F for padding

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
        with self.node_lock:
            # Guard 1: Check completion status first
            if self.task_completed:
                # Optional: Log if needed, but return silently to avoid spam
                # print(f"Skipping progress update because task is already completed.")
                return

            # Guard 2: Check if progress bar exists AND if we are already at or beyond the total
            if not self.pbar or self.current_steps >= self.total_nodes:
                 # Optional: Log if trying to update when already >= total for debugging
                 # if self.pbar and self.current_steps >= self.total_nodes:
                 #     print(f"Debug: update_progress called when steps ({self.current_steps}) >= total ({self.total_nodes}). Skipping update.")
                 return

            # --- If guards passed, proceed with increment and update --- 
            self.current_steps += 1
            # Increment the ComfyUI progress bar by 1
            self.pbar.update(1)
            # Log the current state
            # Use min for logging safety, although current_steps should now never exceed total_nodes here
            display_steps = min(self.current_steps, self.total_nodes) 
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
                # Check if task finished before reaching 100% naturally
                if self.current_steps < self.total_nodes:
                    # Option 1: Update counter but DON'T force bar to 100%
                    print(f"Task completed early at step {self.current_steps}/{self.total_nodes}. Setting counter to total.")
                    self.current_steps = self.total_nodes # Ensure internal counter matches total for consistency
                    # Option 2: (Commented out) Force bar to 100% - REMOVED
                    # print(f"Forcing progress bar to 100% as final step.")
                    # self.pbar.update_absolute(1.0)
                    # print(f"Progress Finalized: {self.total_nodes}/{self.total_nodes} (100.0%)")
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
                "is_webapp_task": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "LATENT", "STRING", "AUDIO")
    RETURN_NAMES = ("images", "video_frames", "latent", "text", "audio")

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

        max_retries = 5
        retry_delay = 1
        last_exception = None
        node_count = None

        for attempt in range(max_retries):
            response = None
            try:
                print(f"Attempt {attempt + 1}/{max_retries} to get workflow node count...")
                response = requests.post(url, json=data, headers=headers, timeout=30)
                response.raise_for_status()

                result = response.json()

                if result.get("code") != 0:
                    api_msg = result.get('msg', 'Unknown API error')
                    print(f"API error on attempt {attempt + 1}: {api_msg}")
                    raise Exception(f"API error getting workflow node count: {api_msg}")

                workflow_json = result.get("data", {}).get("prompt")
                if not workflow_json:
                    raise Exception("No workflow data found in response")

                # Parse the workflow JSON
                workflow_data = json.loads(workflow_json)
                
                # Count the number of nodes
                node_count = len(workflow_data)
                print(f"Workflow contains {node_count} nodes")
                return node_count

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
                    print("Max retries reached for getting workflow node count.")
                    raise Exception(f"Failed to get workflow node count after {max_retries} attempts. Last error: {last_exception}") from last_exception

        # This should ideally not be reached if the loop logic is correct
        raise Exception(f"Failed to get workflow node count after {max_retries} attempts (unexpected loop end). Last error: {last_exception}")

    # --- Main Process Method ---
    def process(self, apiConfig, nodeInfoList=None, run_timeout=600, concurrency_limit=1, is_webapp_task=False):
        # Reset state
        with self.node_lock: # Use lock for resetting shared state
            self.executed_nodes.clear()
            self.task_completed = False
            self.ws_error = None
            self.prompt_tips = "{}"
            self.current_steps = 0 # Reset step counter

        # Get config values
        api_key = apiConfig.get("apiKey")
        base_url = apiConfig.get("base_url")

        if not api_key or not base_url:
            raise ValueError("Missing required apiConfig fields: apiKey, base_url")

        # Get workflow node count from API (only for non-AI App tasks)
        self.total_nodes = self.ESTIMATED_TOTAL_NODES # Default
        retrieved_workflow_id = apiConfig.get("workflowId_webappId") # <<< Changed key here

        if not is_webapp_task:
            # --- Standard ComfyUI Task --- 
            print("Standard ComfyUI Task mode enabled.")
            try:
                # workflow_id = apiConfig.get("workflowId_webappId") # Already retrieved
                if not retrieved_workflow_id:
                    print("Warning: workflowId_webappId missing in apiConfig for standard task. Using default node estimate.")
                    # Fall through to use default estimate
                else:
                    # Get actual node count from workflow
                    actual_node_count = self.get_workflow_node_count(api_key, base_url, retrieved_workflow_id)
                    # Use the actual node count directly
                    self.total_nodes = actual_node_count
                    print(f"Using actual total nodes for progress: {self.total_nodes}")
            except Exception as e:
                print(f"Error getting workflow node count, using default value: {e}")
                # self.total_nodes is already set to default
                print(f"Using default total nodes for progress: {self.total_nodes}")
        else:
            # --- AI App Task --- 
            # Rename print log message to reflect webapp task
            print(f"Webapp Task mode enabled. Using default estimated nodes for progress: {self.total_nodes}")
            # Validate that workflowId (acting as webappId) is provided in config
            if not retrieved_workflow_id:
                 # Update ValueError message
                 raise ValueError("workflowId_webappId (acting as webappId) must be provided in apiConfig when is_webapp_task is True.")
            # Optional: Add validation if webappId must be numeric, though API might handle string conversion
            try:
                # Attempt conversion to int, but keep it as string for the API call if needed
                int(retrieved_workflow_id)
                # Update print log message
                print(f"Using workflowId_webappId from apiConfig as webappId: {retrieved_workflow_id}")
            except ValueError:
                 # Update print log message
                 print(f"Warning: workflowId_webappId '{retrieved_workflow_id}' provided for Webapp Task is not purely numeric, but proceeding.")


        # Initialize ComfyUI progress bar
        self.pbar = comfy.utils.ProgressBar(self.total_nodes)
        print("Progress bar initialized at 0")

        # --- Concurrency Check ---
        # api_key and base_url are already validated
        try:
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
        wss_url = None # <<< Initialize wss_url
        try:
            print(f"ExecuteNode NodeInfoList: {nodeInfoList}")

            # <<< Decide which creation function to call >>>
            if is_webapp_task:
                # Call AI App Task creation, passing the retrieved ID as webappId
                webappId_to_pass = retrieved_workflow_id # Use the ID from config
                # Update print log message
                print(f"Creating Webapp task with webappId: {webappId_to_pass}...")
                task_creation_result = self.create_ai_app_task(apiConfig, nodeInfoList or [], webappId_to_pass)
            else:
                # Call standard ComfyUI Task creation
                print("Creating standard ComfyUI task...")
                # <<< Add base_url back to the create_task call >>>
                task_creation_result = self.create_task(apiConfig, nodeInfoList or [], base_url)

            print(f"Task Creation Result: {json.dumps(task_creation_result, indent=2, ensure_ascii=False)}")

            # Validate task creation response structure before accessing data
            if not isinstance(task_creation_result.get("data"), dict):
                 raise ValueError("Invalid task creation response data structure.")

            task_data = task_creation_result["data"]
            self.prompt_tips = task_data.get("promptTips", "{}")
            task_id = task_data.get("taskId")
            initial_status = task_data.get("taskStatus")
            wss_url = task_data.get("netWssUrl") # <<< Get initial WSS URL

            if not task_id:
                 raise ValueError("Missing taskId in task creation response.")

            print(f"Task created, taskId: {task_id}, Initial Status: {initial_status}")

            # --- Handle QUEUED state ---
            if initial_status == "QUEUED" and not wss_url:
                print("Task is QUEUED. Polling for RUNNING status and WebSocket URL...")
                queue_start_time = time.time()
                poll_interval = 2 # seconds
                while True:
                    # Check timeout while waiting in queue
                    if time.time() - queue_start_time > run_timeout:
                         raise TimeoutError(f"Timeout waiting for task {task_id} to leave QUEUED state.")

                    # Check task status
                    status_result = self.check_task_status(task_id, api_key, base_url)
                    current_status = status_result.get("taskStatus")
                    print(f"  Polling status for queued task {task_id}: {current_status}")

                    if current_status == "RUNNING":
                        # Task is running, try to get WSS URL from status check
                        wss_url = status_result.get("netWssUrl")
                        if wss_url:
                            print(f"Task {task_id} is RUNNING. WebSocket URL obtained: {wss_url}")
                            break # Exit queue polling loop
                        else:
                            # This case might indicate an API design issue or a transient state
                            print(f"Warning: Task {task_id} is RUNNING but WebSocket URL not yet available from status check. Retrying check...")
                            # Keep polling, maybe the URL will appear shortly
                    elif current_status == "error":
                         error_msg = status_result.get('error', 'Unknown error during queue polling')
                         raise Exception(f"Task {task_id} failed while in queue: {error_msg}")
                    elif isinstance(status_result, list): # Task completed while polling queue status
                         print(f"Task {task_id} completed while polling queue status. Skipping WebSocket connection.")
                         # Set wss_url to a non-None dummy value to skip connection attempt later
                         wss_url = "skipped_completed_in_queue" 
                         break # Exit queue polling loop
                    elif current_status != "QUEUED":
                        # Handle unexpected status if necessary
                        print(f"Warning: Task {task_id} transitioned to unexpected status '{current_status}' while polling queue.")
                        # Decide if we should break or continue polling based on the status

                    # Wait before next poll
                    time.sleep(poll_interval)

            # --- Connect WebSocket if URL is available and not skipped ---
            if wss_url and wss_url != "skipped_completed_in_queue":
                print(f"Attempting to connect WebSocket: {wss_url}")
                self.connect_websocket(wss_url)
            elif not wss_url:
                # If still no WSS URL after potential polling (e.g., finished directly, or RUNNING but no URL provided)
                # Raise error or proceed without WS? Let's raise error for now.
                raise ValueError(f"Failed to obtain WebSocket URL for task {task_id} after creation/polling.")
            else: # wss_url == "skipped_completed_in_queue"
                 print("WebSocket connection skipped as task already completed.")

        except Exception as e:
             print(f"Error during task creation, queue polling, or WS connection: {e}")
             if self.pbar: self.pbar.update_absolute(1.0) # Use absolute directly for setup failure
             raise

        # --- Task Monitoring Loop ---
        task_start_time = time.time()
        loop_sleep_interval = 0.1
        print("Starting task monitoring loop...")

        timeout_timer = None
        final_error = None # <<< Define final_error outside try/finally

        try:
            # Setup global timeout timer
            def force_timeout():
                # Use lock to safely check task_completed
                with self.node_lock:
                     is_completed = self.task_completed
                if not is_completed:
                    print("Global timeout reached - forcing task completion.")
                    # Use lock to set error safely
                    with self.node_lock:
                        self.ws_error = Exception("Global timeout reached")
                        # Just set the flags here to break loop
                        self.task_completed = True # Set flag directly here to break loop

            timeout_timer = threading.Timer(run_timeout, force_timeout)
            timeout_timer.daemon = True
            timeout_timer.start()

            # Main wait loop
            while True: # <<< Added loop structure
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
                 final_error = self.ws_error # Assign to outer scope variable

            if final_error:
                print(f"Task ended with error: {final_error}")
                # Only complete progress if not already completed by WS handler
                # (complete_progress handles internal check)
                self.complete_progress() # Call safe completion method
                # Error will be raised outside the finally block if needed
            else: # Task completed normally
                print("Task monitoring completed successfully.")
                # complete_progress should have been called by WS handler or the loop exit condition
                # Ensure completion even if WS didn't send success (e.g., timeout)
                self.complete_progress()

        finally: # <<< Added finally clause
            # Cleanup
            if timeout_timer:
                timeout_timer.cancel()
            if self.ws:
                try:
                    self.ws.close()
                except Exception as e:
                    print(f"Error closing WebSocket: {e}")
                self.ws = None

            # Final safety net: Ensure progress is marked complete.
            # complete_progress has internal checks, so calling it again is safe.
            self.complete_progress()

        # If an error occurred during the loop, raise it now after cleanup
        if final_error:
            raise final_error

        # --- Process Output ---
        print("Processing task output...")
        # Pass the validated api_key and base_url again
        return self.process_task_output(task_id, api_key, base_url)

    def process_task_output(self, task_id, api_key, base_url):
        """Handles task output, separating images, video frames, audio, etc."""
        max_retries = 30
        retry_interval = 1
        max_retry_interval = 5
        image_data_list = [] # <<< For regular images
        frame_data_list = [] # <<< For video frames
        latent_data = None
        text_data = None
        audio_data = None # <<< For audio data

        for attempt in range(max_retries):
            task_status_result = None
            try:
                task_status_result = self.check_task_status(task_id, api_key, base_url)
                print(f"Check output attempt {attempt + 1}/{max_retries}")

                if isinstance(task_status_result, dict) and task_status_result.get("taskStatus") in ["RUNNING", "QUEUED"]:
                    wait_time = min(retry_interval * (1.5 ** attempt), max_retry_interval)
                    print(f"Task still running ({task_status_result.get('taskStatus')}), waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    continue # <<< Continue within loop

                if isinstance(task_status_result, list) and len(task_status_result) > 0:
                    print("Got valid output result, processing files...")
                    image_urls = []
                    video_urls = []
                    latent_urls = []
                    text_urls = []
                    audio_urls = [] # <<< Add list for audio urls

                    for output in task_status_result: # <<< Indent loop correctly
                        if isinstance(output, dict):
                            file_url = output.get("fileUrl")
                            file_type = output.get("fileType")
                            if file_url and file_type:
                                file_type_lower = file_type.lower()
                                if file_type_lower in ["png", "jpg", "jpeg", "webp", "bmp", "gif"]:
                                    image_urls.append(file_url)
                                elif file_type_lower in ["mp4", "avi", "mov", "webm"]:
                                    video_urls.append(file_url)
                                elif file_type_lower == "latent":
                                    latent_urls.append(file_url)
                                elif file_type_lower == "txt":
                                    text_urls.append(file_url)
                                # <<< Add common audio types
                                elif file_type_lower in ["wav", "mp3", "flac", "ogg"]: 
                                    audio_urls.append(file_url)

                    # Process Images -> Add to image_data_list
                    if image_urls:
                        print(f"Processing {len(image_urls)} images...")
                        # Download all images first
                        downloaded_images = []
                        for url in image_urls:
                            try:
                                img_tensor = self.download_image(url)
                                if img_tensor is not None:
                                    downloaded_images.append(img_tensor)
                                    print(f"Successfully downloaded image from {url} (Shape: {img_tensor.shape})")
                                # Remove the break to process all images
                                # print(f"Successfully processed first image from {url}. Skipping remaining images.")
                                # break # Process only the first image
                            except Exception as img_e:
                                print(f"Error downloading image {url}: {img_e}")
                        
                        # If images were downloaded, find max dimensions and pad
                        if downloaded_images:
                            if len(downloaded_images) > 1:
                                print("Multiple images found. Checking dimensions and padding if necessary...")
                                max_h = 0
                                max_w = 0
                                for img in downloaded_images:
                                    # Shape is [1, H, W, C]
                                    max_h = max(max_h, img.shape[1])
                                    max_w = max(max_w, img.shape[2])
                                print(f"Max dimensions found: Height={max_h}, Width={max_w}")

                                padded_images = []
                                for i, img_tensor in enumerate(downloaded_images):
                                    _, h, w, _ = img_tensor.shape
                                    if h < max_h or w < max_w:
                                        pad_h_total = max_h - h
                                        pad_w_total = max_w - w
                                        pad_top = pad_h_total // 2
                                        pad_bottom = pad_h_total - pad_top
                                        pad_left = pad_w_total // 2
                                        pad_right = pad_w_total - pad_left
                                        
                                        # Permute [1, H, W, C] -> [1, C, H, W] for F.pad
                                        img_permuted = img_tensor.permute(0, 3, 1, 2)
                                        # Pad (pad is specified for last dimensions first: W, then H)
                                        padded_permuted = F.pad(img_permuted, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)
                                        # Permute back [1, C, H, W] -> [1, H, W, C]
                                        padded_img = padded_permuted.permute(0, 2, 3, 1)
                                        print(f"  Padded image {i+1} from {h}x{w} to {max_h}x{max_w}")
                                        padded_images.append(padded_img)
                                    else:
                                        print(f"  Image {i+1} already has max dimensions.")
                                        padded_images.append(img_tensor) # Already max size
                                image_data_list = padded_images # Use the padded list for concatenation
                            else:
                                # Only one image, no padding needed, just use it
                                print("Only one image found, no padding needed.")
                                image_data_list = downloaded_images
                        # else: image_data_list remains empty

                    # Process Videos (extract frames) -> Add to frame_data_list
                    if video_urls:
                        print(f"Processing {len(video_urls)} videos for frames...")
                        for url in video_urls:
                            try:
                                frame_tensors = self.download_video(url)
                                if frame_tensors:
                                    frame_data_list.extend(frame_tensors) # <<< Add to frame_data_list
                                    print(f"Extracted {len(frame_tensors)} frames from video {url}")
                            except Exception as vid_e:
                                print(f"Error processing video {url}: {vid_e}")

                    # Process Latents (load the first one found)
                    if latent_urls and latent_data is None:
                        print(f"Processing {len(latent_urls)} latent file(s)...")
                        for url in latent_urls:
                             try:
                                 loaded_latent = self.download_and_load_latent(url)
                                 if loaded_latent is not None:
                                     latent_data = loaded_latent
                                     print(f"Successfully loaded latent from {url}")
                                     break # Process only the first successful latent
                             except Exception as lat_e:
                                 print(f"Error processing latent {url}: {lat_e}")

                    # Process Text Files (read the first one found)
                    if text_urls and text_data is None:
                        print(f"Processing {len(text_urls)} text file(s)...")
                        for url in text_urls:
                             try:
                                 loaded_text = self.download_and_read_text(url)
                                 if loaded_text is not None:
                                     text_data = loaded_text
                                     print(f"Successfully read text from {url}")
                                     break # Process only the first successful text file
                             except Exception as txt_e:
                                 print(f"Error processing text file {url}: {txt_e}")

                    # <<< Process Audio Files (load the first one found)
                    if audio_urls and audio_data is None:
                        print(f"Processing {len(audio_urls)} audio file(s)...")
                        for url in audio_urls:
                             try:
                                 loaded_audio = self.download_and_process_audio(url)
                                 if loaded_audio is not None:
                                     audio_data = loaded_audio
                                     print(f"Successfully processed audio from {url}")
                                     break # Process only the first successful audio file
                             except Exception as aud_e:
                                 print(f"Error processing audio file {url}: {aud_e}")

                    # Task processing complete, break the retry loop
                    break # <<< Break within loop

                elif isinstance(task_status_result, dict) and task_status_result.get("taskStatus") == "error": # <<< Use elif
                    print(f"Task failed with error: {task_status_result.get('error', 'Unknown error')}")
                    break # <<< Break within loop

                else: # <<< Handle other cases or unexpected results
                    print(f"Unexpected task status or empty result, waiting...")
                    time.sleep(min(retry_interval * (1.5 ** attempt), max_retry_interval))

            except Exception as e: # <<< Added except clause
                print(f"Error checking/processing task status (attempt {attempt + 1}): {e}")
                # Check if the result indicates an error, even if an exception occurred during processing
                if isinstance(task_status_result, dict) and task_status_result.get("taskStatus") == "error":
                     print("Stopping retries due to reported task error.")
                     break # <<< Break within loop
                # Simple exponential backoff for retries
                time.sleep(min(retry_interval * (1.5 ** attempt), max_retry_interval))

        # --- Final Output Generation ---

        # Placeholder for regular images
        if not image_data_list:
            print("No regular images generated, creating placeholder.")
            image_data_list.append(self.create_placeholder_image(text="No image output"))

        # Placeholder for video frames
        if not frame_data_list:
            print("No video frames generated, creating placeholder.")
            frame_data_list.append(self.create_placeholder_image(text="No video frame output"))

        # Placeholder for latent
        if latent_data is None:
            print("No latent generated, creating placeholder.")
            latent_data = self.create_placeholder_latent()

        # Default for text
        if text_data is None:
             print("No text file processed, returning 'null' string.")
             text_data = "null"
             
        # <<< Placeholder for audio
        if audio_data is None:
            print("No audio generated, creating placeholder.")
            audio_data = self.create_placeholder_audio()

        # Batch images and frames separately
        final_image_batch = torch.cat(image_data_list, dim=0) if image_data_list else None
        final_frame_batch = torch.cat(frame_data_list, dim=0) if frame_data_list else None # <<< Batch frames

        # Ensure we return a tuple matching RETURN_TYPES
        # <<< Add audio_data to the return tuple
        return (final_image_batch, final_frame_batch, latent_data, text_data, audio_data) 

    def create_placeholder_image(self, text="No image/video output", width=256, height=64):
        """Creates a placeholder image tensor with text."""
        img = Image.new('RGB', (width, height), color = (50, 50, 50)) # Dark gray background
        d = ImageDraw.Draw(img)
        try:
            # Attempt to load a simple default font (may vary by system)
            # A small default size to fit the image
            fontsize = 15
            # Try common system font names/paths
            font_paths = ["arial.ttf", "LiberationSans-Regular.ttf", "DejaVuSans.ttf"]
            font = None
            for fp in font_paths:
                try:
                    font = ImageFont.truetype(fp, fontsize)
                    break
                except IOError:
                    continue
            if font is None:
                 font = ImageFont.load_default() # Fallback to PIL default bitmap font
                 print("Warning: Could not load system font, using PIL default.")

            # Calculate text position for centering
            text_bbox = d.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = (width - text_width) / 2
            text_y = (height - text_height) / 2
            d.text((text_x, text_y), text, fill=(200, 200, 200), font=font) # Light gray text
        except Exception as e:
            print(f"Error adding text to placeholder image: {e}. Returning image without text.")
        
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,] # Shape: [1, H, W, C]
        return img_tensor

    def create_placeholder_latent(self, batch_size=1, channels=4, height=64, width=64):
        """Creates a placeholder latent tensor dictionary."""
        latent = torch.zeros([batch_size, channels, height, width])
        return {"samples": latent}
        
    # <<< Add placeholder audio function
    def create_placeholder_audio(self, sample_rate=44100, duration_sec=0.01):
        """Creates a placeholder silent audio dictionary."""
        print(f"Creating silent placeholder audio: {duration_sec}s @ {sample_rate}Hz")
        num_samples = int(sample_rate * duration_sec)
        waveform = torch.zeros((1, num_samples), dtype=torch.float32) # Mono silence
        return {"waveform": waveform, "sample_rate": sample_rate}

    def download_image(self, image_url):
        """
        从 URL 下载图像并转换为适合预览或保存的 torch.Tensor 格式。
        包含重试机制，最多重试5次。
        Returns tensor [1, H, W, C] or None on failure.
        """
        max_retries = 5
        retry_delay = 1
        last_exception = None
        img_tensor = None # Define img_tensor outside try

        for attempt in range(max_retries):
            try:
                response = requests.get(image_url, timeout=30)
                print(f"Download image attempt {attempt + 1} ({image_url}): Status code: {response.status_code}")
                response.raise_for_status()

                content_type = response.headers.get('Content-Type', '').lower()
                # Consider validating content_type if needed

                img = Image.open(BytesIO(response.content)).convert("RGB")
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array)[None,] # Shape: [1, H, W, C]
                return img_tensor # Return on success

            except (requests.exceptions.RequestException, IOError, Image.UnidentifiedImageError) as e: # <<< Correct except clause
                print(f"Download image attempt {attempt + 1} failed: {e}")
                last_exception = e
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                # else: # <<< Implicitly handled by loop ending
                #     print(f"Failed to download image {image_url} after {max_retries} attempts.")
                #     # Keep img_tensor as None

        # If loop finishes without returning, it means all retries failed
        print(f"Failed to download image {image_url} after {max_retries} attempts.")
        return None


    def download_video(self, video_url):
        """
        Downloads a video, extracts all frames, converts them to tensors,
        deletes the video file, and returns a list of image tensors.
        Requires opencv-python (cv2).
        Returns list[torch.Tensor] or None on failure. Each tensor shape [1, H, W, C]. <<< Updated shape comment
        """
        max_retries = 5
        retry_delay = 1
        last_exception = None
        video_path = None
        output_dir = "temp" # Use a temp directory for downloaded videos

        # --- Ensure temp directory exists ---
        if not os.path.exists(output_dir): # <<< Correct indentation
            try:
                os.makedirs(output_dir)
                print(f"Created temporary directory: {output_dir}")
            except OSError as e:
                print(f"Error creating temporary directory {output_dir}: {e}")
                return None # Cannot proceed without temp dir

        # --- Download the video file ---
        for attempt in range(max_retries):
            video_path = None # Reset path for each attempt
            try:
                # Generate a unique temporary filename
                try:
                    safe_filename = f"temp_video_{os.path.basename(video_url)}_{str(int(time.time()*1000))}.tmp"
                    safe_filename = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in safe_filename)[:150] # Basic sanitization and length limit
                    video_path = os.path.join(output_dir, safe_filename)
                except Exception as path_e:
                    print(f"Error creating temporary video path: {path_e}")
                    # Fallback filename
                    video_path = os.path.join(output_dir, f"temp_video_{str(int(time.time()*1000))}.tmp")

                print(f"Attempt {attempt + 1}/{max_retries} to download video to temp path: {video_path}")
                response = requests.get(video_url, stream=True, timeout=60)
                response.raise_for_status()

                downloaded_size = 0
                with open(video_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)

                if downloaded_size > 0:
                    print(f"Temporary video downloaded successfully: {video_path}")
                    break # Exit retry loop on successful download
                else:
                    print(f"Warning: Downloaded video file is empty: {video_path}")
                    if os.path.exists(video_path):
                        try: os.remove(video_path)
                        except OSError: pass
                    last_exception = IOError("Downloaded video file is empty.")
                    # Continue to retry

            except (requests.exceptions.RequestException, IOError) as e: # <<< Correct except clause
                 print(f"Download video attempt {attempt + 1} failed: {e}")
                 last_exception = e
                 if video_path and os.path.exists(video_path):
                     try: os.remove(video_path)
                     except OSError: pass # Ignore error removing partial file
                 # Continue to retry unless it's the last attempt

            if attempt < max_retries - 1:
                print(f"Retrying download in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            # else: # Implicitly handled by loop ending
            #     print(f"Failed to download video {video_url} after {max_retries} attempts.")
            #     # video_path will likely be None or point to a non-existent/empty file

        # Check if download succeeded (video_path exists and is not empty)
        if not video_path or not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
             print(f"Failed to download video {video_url} successfully after {max_retries} attempts.")
             # Clean up potentially empty file if it exists
             if video_path and os.path.exists(video_path):
                 try: os.remove(video_path)
                 except OSError: pass
             return None

        # --- Extract frames if download was successful ---
        frame_tensors = []
        cap = None
        try:
            print(f"Extracting frames from {video_path}...")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video file: {video_path}")

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break # End of video

                # Convert frame (BGR) to RGB, then to Tensor [1, H, W, C] (float32, 0-1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Reuse PIL conversion for consistency? Or keep cv2->numpy path
                img_array = frame_rgb.astype(np.float32) / 255.0 # Direct conversion
                # img = Image.fromarray(frame_rgb)
                # img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array)[None,] # <<< Added batch dimension
                frame_tensors.append(img_tensor)
                frame_count += 1
                # Optional: Add progress logging for long videos
                # if frame_count % 100 == 0: print(f"  Extracted {frame_count} frames...")

            print(f"Finished extracting {frame_count} frames.")
        except Exception as e:
            print(f"Error extracting frames from video {video_path}: {e}")
            # Return None or potentially partially extracted frames? Let's return None for consistency.
            frame_tensors = None # Indicate failure
        finally:
            # --- Cleanup ---
            if cap:
                cap.release()
            # Delete the temporary video file regardless of extraction success/failure
            if video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    print(f"Deleted temporary video file: {video_path}")
                except OSError as e:
                    print(f"Error deleting temporary video file {video_path}: {e}")

        return frame_tensors

    def download_and_load_latent(self, latent_url):
        """
        Downloads a .latent file, loads it using safetensors, applies multiplier,
        cleans up the temp file, and returns the latent dictionary.
        Returns dict { "samples": tensor } or None on failure.
        """
        max_retries = 5
        retry_delay = 1
        last_exception = None
        latent_path = None
        output_dir = "temp" # Use temp directory

        # Ensure temp directory exists
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except OSError as e:
                print(f"Error creating temporary directory {output_dir}: {e}")
                return None

        # --- Download the latent file ---
        for attempt in range(max_retries):
            latent_path = None # Reset path for each attempt
            try:
                # Generate a unique temporary filename
                try:
                    safe_filename = f"temp_latent_{os.path.basename(latent_url)}_{str(int(time.time()*1000))}.latent"
                    safe_filename = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in safe_filename)[:150]
                    latent_path = os.path.join(output_dir, safe_filename)
                except Exception as path_e:
                    print(f"Error creating temporary latent path: {path_e}")
                    latent_path = os.path.join(output_dir, f"temp_latent_{str(int(time.time()*1000))}.latent")

                print(f"Attempt {attempt + 1}/{max_retries} to download latent to temp path: {latent_path}")
                response = requests.get(latent_url, stream=True, timeout=30)
                response.raise_for_status()

                downloaded_size = 0
                with open(latent_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk: # <<< Correct indent
                            f.write(chunk)
                            downloaded_size += len(chunk) # <<< Correct indent

                if downloaded_size > 0:
                    print(f"Temporary latent downloaded successfully: {latent_path}")
                    break # Exit retry loop on successful download
                else:
                    print(f"Warning: Downloaded latent file is empty: {latent_path}")
                    if os.path.exists(latent_path):
                        try: os.remove(latent_path)
                        except OSError: pass
                    last_exception = IOError("Downloaded latent file is empty.")
                    # Continue retry loop

            except (requests.exceptions.RequestException, IOError) as e:
                 print(f"Download latent attempt {attempt + 1} failed: {e}")
                 last_exception = e
                 if latent_path and os.path.exists(latent_path):
                     try: os.remove(latent_path)
                     except OSError: pass
                 # Continue retry loop

            if attempt < max_retries - 1:
                print(f"Retrying download in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            # else: # Implicitly handled by loop ending
            #     print(f"Failed to download latent {latent_url} after {max_retries} attempts.")

        # Check if download succeeded
        if not latent_path or not os.path.exists(latent_path) or os.path.getsize(latent_path) == 0:
             print(f"Failed to download latent {latent_url} successfully after {max_retries} attempts.")
             if latent_path and os.path.exists(latent_path):
                 try: os.remove(latent_path)
                 except OSError: pass
             return None

        # --- Load the latent file ---
        loaded_latent_dict = None
        try:
            print(f"Loading latent from {latent_path}...")
            # Use safetensors.torch.load_file
            latent_content = safetensors.torch.load_file(latent_path, device="cpu")

            if "latent_tensor" not in latent_content:
                 raise ValueError("'latent_tensor' key not found in the loaded latent file.")

            # Apply multiplier based on LoadLatent logic
            multiplier = 1.0
            if "latent_format_version_0" not in latent_content:
                multiplier = 1.0 / 0.18215
                print(f"Applying multiplier {multiplier:.5f} (old latent format detected)")

            samples_tensor = latent_content["latent_tensor"].float() * multiplier
            loaded_latent_dict = {"samples": samples_tensor}
            print("Latent loaded successfully.")

        except Exception as e:
            print(f"Error loading latent file {latent_path}: {e}")
            # Ensure loaded_latent_dict remains None on error
            loaded_latent_dict = None
        finally:
            # --- Cleanup ---
            if latent_path and os.path.exists(latent_path):
                try:
                    os.remove(latent_path)
                    print(f"Deleted temporary latent file: {latent_path}")
                except OSError as e:
                    print(f"Error deleting temporary latent file {latent_path}: {e}")

        return loaded_latent_dict

    def download_and_read_text(self, text_url):
        """
        Downloads a .txt file, reads its content as UTF-8,
        cleans up the temp file, and returns the text content.
        Returns str or None on failure.
        """
        max_retries = 5
        retry_delay = 1
        last_exception = None
        text_path = None
        output_dir = "temp"

        if not os.path.exists(output_dir):
            try: os.makedirs(output_dir)
            except OSError as e: print(f"Error creating temp dir {output_dir}: {e}"); return None

        # --- Download the text file ---
        for attempt in range(max_retries):
            text_path = None
            try:
                try:
                    safe_filename = f"temp_text_{os.path.basename(text_url)}_{str(int(time.time()*1000))}.txt"
                    safe_filename = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in safe_filename)[:150]
                    text_path = os.path.join(output_dir, safe_filename)
                except Exception as path_e:
                    print(f"Error creating temporary text path: {path_e}")
                    text_path = os.path.join(output_dir, f"temp_text_{str(int(time.time()*1000))}.txt")

                print(f"Attempt {attempt + 1}/{max_retries} to download text to temp path: {text_path}")
                response = requests.get(text_url, stream=True, timeout=20) # Shorter timeout for text
                response.raise_for_status()

                downloaded_size = 0
                with open(text_path, "wb") as f: # Write in binary first
                    for chunk in response.iter_content(chunk_size=4096):
                        if chunk: f.write(chunk); downloaded_size += len(chunk)

                if downloaded_size > 0:
                    print(f"Temporary text file downloaded: {text_path}")
                    break # Success
                else:
                    if os.path.exists(text_path):
                         try: os.remove(text_path)
                         except OSError: pass
                    last_exception = IOError("Downloaded text file is empty.")
                    # Continue retries

            except (requests.exceptions.RequestException, IOError) as e:
                 print(f"Download text attempt {attempt + 1} failed: {e}")
                 last_exception = e
                 if text_path and os.path.exists(text_path):
                     try: os.remove(text_path)
                     except OSError: pass
                 # Continue retries

            if attempt < max_retries - 1:
                print(f"Retrying download in {retry_delay} seconds...")
                time.sleep(retry_delay); retry_delay *= 2
            # else: # Implicitly handled by loop ending
            #     print(f"Failed to download text {text_url} after {max_retries} attempts.")

        # Check download success
        if not text_path or not os.path.exists(text_path) or os.path.getsize(text_path) == 0:
             print(f"Failed to download text {text_url} successfully after {max_retries} attempts.")
             if text_path and os.path.exists(text_path):
                 try: os.remove(text_path)
                 except OSError: pass
             return None

        # --- Read the text file ---
        read_content = None
        try:
            print(f"Reading text from {text_path}...")
            # Read with UTF-8 encoding, handle potential errors
            with open(text_path, "r", encoding="utf-8", errors="replace") as f:
                read_content = f.read()
            print("Text read successfully.")
        except Exception as e:
            print(f"Error reading text file {text_path}: {e}")
            read_content = None
        finally:
            # --- Cleanup ---
            if text_path and os.path.exists(text_path):
                try:
                    os.remove(text_path)
                    print(f"Deleted temporary text file: {text_path}")
                except OSError as e:
                    print(f"Error deleting temporary text file {text_path}: {e}")

        return read_content

    # <<< Add audio download and processing function
    def download_and_process_audio(self, audio_url):
        """
        Downloads an audio file, processes it using torchaudio,
        cleans up the temp file, and returns the audio dictionary.
        Returns dict { "waveform": tensor [Channels, Samples], "sample_rate": int } or None on failure.
        """
        max_retries = 5
        retry_delay = 1
        last_exception = None
        audio_path = None
        output_dir = "temp"

        if not os.path.exists(output_dir):
            try: os.makedirs(output_dir)
            except OSError as e: print(f"Error creating temp dir {output_dir}: {e}"); return None

        # --- Download the audio file ---
        for attempt in range(max_retries):
            audio_path = None
            try:
                # Generate temp filename based on URL extension if possible
                try:
                    basename = os.path.basename(audio_url)
                    _, ext = os.path.splitext(basename)
                    if not ext: ext = ".audio" # Default if no extension
                    safe_filename = f"temp_audio_{str(int(time.time()*1000))}{ext}"
                    safe_filename = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in safe_filename)[:150]
                    audio_path = os.path.join(output_dir, safe_filename)
                except Exception as path_e:
                    print(f"Error creating temporary audio path: {path_e}")
                    audio_path = os.path.join(output_dir, f"temp_audio_{str(int(time.time()*1000))}.tmp")

                print(f"Attempt {attempt + 1}/{max_retries} to download audio to temp path: {audio_path}")
                response = requests.get(audio_url, stream=True, timeout=60) # Longer timeout for audio/video
                response.raise_for_status()

                downloaded_size = 0
                with open(audio_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk: f.write(chunk); downloaded_size += len(chunk)

                if downloaded_size > 0:
                    print(f"Temporary audio file downloaded: {audio_path} ({downloaded_size} bytes)")
                    break # Success
                else:
                    if os.path.exists(audio_path):
                        try: os.remove(audio_path)
                        except OSError: pass
                    last_exception = IOError("Downloaded audio file is empty.")
                    # Continue retries

            except (requests.exceptions.RequestException, IOError) as e:
                 print(f"Download audio attempt {attempt + 1} failed: {e}")
                 last_exception = e
                 if audio_path and os.path.exists(audio_path):
                     try: os.remove(audio_path)
                     except OSError: pass
                 # Continue retries

            if attempt < max_retries - 1:
                print(f"Retrying download in {retry_delay} seconds...")
                time.sleep(retry_delay); retry_delay *= 2
            # else:
            #     print(f"Failed to download audio {audio_url} after {max_retries} attempts.")

        # Check download success
        if not audio_path or not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
             print(f"Failed to download audio {audio_url} successfully after {max_retries} attempts.")
             if audio_path and os.path.exists(audio_path):
                 try: os.remove(audio_path)
                 except OSError: pass
             return None

        # --- Process the audio file ---
        processed_audio = None
        try:
            print(f"Processing audio from {audio_path} using torchaudio...")
            # Use torchaudio.load to get waveform and sample rate
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Ensure waveform is float32, which is common for ComfyUI audio nodes
            if waveform.dtype != torch.float32:
                print(f"Converting waveform from {waveform.dtype} to float32.")
                waveform = waveform.to(torch.float32)
                
            # <<< Ensure the tensor is contiguous <<<
            if not waveform.is_contiguous():
                print("Audio waveform is not contiguous. Making it contiguous.")
                waveform = waveform.contiguous()

            # <<< ADD BATCH DIMENSION TO MATCH STANDARD COMFYUI AUDIO FORMAT <<<
            waveform = waveform.unsqueeze(0)

            # Most nodes seem to work with [channels, samples] or just [samples] if mono.
            # torchaudio.load returns [channels, samples]. Let's stick with that.
            print(f"Audio loaded successfully: Shape={waveform.shape}, Sample Rate={sample_rate} Hz, dtype={waveform.dtype}, Contiguous={waveform.is_contiguous()}") # <<< Added contiguous log
            processed_audio = {"waveform": waveform, "sample_rate": sample_rate}

        except Exception as e:
            print(f"Error processing audio file {audio_path} with torchaudio: {e}")
            processed_audio = None # Ensure it's None on error
        finally:
            # --- Cleanup ---
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    print(f"Deleted temporary audio file: {audio_path}")
                except OSError as e:
                    print(f"Error deleting temporary audio file {audio_path}: {e}")

        return processed_audio


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
            try: # <<< Added try block
                print(f"Attempt {attempt + 1}/{max_retries} to check account status...")
                response = requests.post(url, json=data, headers=headers, timeout=15)
                response.raise_for_status()

                result = response.json()

                if result.get("code") != 0: # <<< Correct indent
                    api_msg = result.get('msg', 'Unknown API error')
                    print(f"API error on attempt {attempt + 1}: {api_msg}")
                    raise Exception(f"API error getting account status: {api_msg}")

                account_data = result.get("data")
                if not account_data or "currentTaskCounts" not in account_data:
                    raise ValueError("Invalid response structure for account status.")

                try: # <<< Correct indent (inner try for int conversion)
                    current_task_counts = int(account_data["currentTaskCounts"])
                    account_data["currentTaskCounts"] = current_task_counts
                    print("Account status check successful.")
                    return account_data # Success
                except (ValueError, TypeError) as e: # <<< Correct indent
                    raise ValueError(f"Invalid value for currentTaskCounts: {account_data.get('currentTaskCounts')}. Error: {e}")

            except (requests.exceptions.RequestException, json.JSONDecodeError, ValueError, Exception) as e: # <<< Correct indent
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

        # This should ideally not be reached if the loop logic is correct
        raise Exception(f"Failed to check account status after {max_retries} attempts (unexpected loop end). Last error: {last_exception}")


    def create_task(self, apiConfig, nodeInfoList, base_url):
        """
        创建任务，包含重试机制，最多重试5次
        """
        safe_base_url = apiConfig.get('base_url')
        # Use the updated key name here
        safe_workflow_id = apiConfig.get("workflowId_webappId") 
        safe_api_key = apiConfig.get("apiKey")

        if not safe_base_url or not safe_workflow_id or not safe_api_key:
             # Update the error message to reflect the new key
             raise ValueError("Missing required apiConfig fields: 'base_url', 'workflowId_webappId', 'apiKey'")

        url = f"{safe_base_url}/task/openapi/create"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ComfyUI-RH-APICall-Node/1.0",
        }
        # Also update the key used in the API payload
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
            success = False # Flag to indicate success within try block
            try:
                print(f"Create task attempt {attempt + 1}/{max_retries}...")
                response = requests.post(url, json=data, headers=headers, timeout=30)
                print(f"Create task attempt {attempt + 1}: Status code {response.status_code}")
                response.raise_for_status()

                result = response.json()

                if result.get("code") == 0:
                    if "data" in result and "taskId" in result["data"] and "netWssUrl" in result["data"]:
                         print("Task created successfully.")
                         success = True # Mark as success
                         return result # Return successful result
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

            # If successful, we already returned. If not successful, process the error.
            if not success:
                 last_exception = current_last_exception # Store the most recent error
                 if attempt < max_retries - 1:
                     print(f"Retrying task creation in {retry_delay} seconds...")
                     time.sleep(retry_delay)
                     retry_delay *= 2
                 else: # Max retries reached
                     error_message = f"Failed to create task after {max_retries} attempts."
                     if last_exception:
                          error_message += f" Last error: {last_exception}"
                     print(error_message)
                     raise Exception(error_message) from last_exception

        # Should not be reachable if logic is correct
        raise Exception("Task creation failed unexpectedly after retry loop.")

    # <<< Add new function for creating AI App tasks >>>
    def create_ai_app_task(self, apiConfig, nodeInfoList, webappId):
        """
        创建 AI 应用任务 (using /task/openapi/ai-app/run), 包含重试机制.
        """
        safe_base_url = apiConfig.get('base_url')
        safe_api_key = apiConfig.get("apiKey")

        if not safe_base_url or not safe_api_key:
             raise ValueError("Missing required apiConfig fields: 'base_url', 'apiKey'")

        # <<< Use the AI App endpoint >>>
        url = f"{safe_base_url}/task/openapi/ai-app/run"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ComfyUI-RH-APICall-Node/1.0",
            # Host header is typically handled by requests library
        }
        # <<< Construct payload for AI App task, converting webappId to int >>>
        try:
            webappId_int = int(webappId)
        except ValueError:
            # Handle error if the ID from config cannot be converted to int
            raise ValueError(f"Invalid webappId provided: '{webappId}'. It must be convertible to an integer.")

        data = {
            "webappId": webappId_int,
            "apiKey": safe_api_key,
            "nodeInfoList": nodeInfoList,
        }

        max_retries = 5
        retry_delay = 1
        last_exception = None

        for attempt in range(max_retries):
            response = None
            current_last_exception = None
            success = False # Flag to indicate success within try block
            try:
                print(f"Create AI App task attempt {attempt + 1}/{max_retries}...")
                response = requests.post(url, json=data, headers=headers, timeout=30)
                print(f"Create AI App task attempt {attempt + 1}: Status code {response.status_code}")
                response.raise_for_status()

                result = response.json()

                # Response structure seems identical to standard task, check code and data fields
                if result.get("code") == 0:
                    if "data" in result and "taskId" in result["data"]: # Don't strictly require netWssUrl here
                         print("AI App Task created/queued successfully.")
                         success = True # Mark as success
                         return result # Return successful result
                    else:
                         print(f"AI App Task API success code 0, but response structure invalid: {result}")
                         current_last_exception = ValueError(f"API success code 0, but response structure invalid.")
                else:
                    api_msg = result.get('msg', 'Unknown API error')
                    print(f"API error creating AI App task (code {result.get('code')}): {api_msg}")
                    current_last_exception = Exception(f"API error (code {result.get('code')}): {api_msg}")

            except requests.exceptions.Timeout as e:
                 print(f"Create AI App task attempt {attempt + 1} timed out.")
                 current_last_exception = e
            except requests.exceptions.RequestException as e:
                print(f"Create AI App task attempt {attempt + 1} network error: {e}")
                current_last_exception = e
            except json.JSONDecodeError as e:
                 print(f"Create AI App task attempt {attempt + 1} failed to decode JSON response.")
                 if response is not None: print(f"Raw response text: {response.text}")
                 current_last_exception = e
            except Exception as e:
                 print(f"Create AI App task attempt {attempt + 1} unexpected error: {e}")
                 current_last_exception = e

            # If successful, we already returned. If not successful, process the error.
            if not success:
                 last_exception = current_last_exception # Store the most recent error
                 if attempt < max_retries - 1:
                     print(f"Retrying AI App task creation in {retry_delay} seconds...")
                     time.sleep(retry_delay)
                     retry_delay *= 2
                 else: # Max retries reached
                     error_message = f"Failed to create AI App task after {max_retries} attempts."
                     if last_exception:
                          error_message += f" Last error: {last_exception}"
                     print(error_message)
                     raise Exception(error_message) from last_exception

        # Should not be reachable if logic is correct
        raise Exception("AI App Task creation failed unexpectedly after retry loop.")


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

        response = None # Define response outside try
        result = None # Define result outside try
        
        # <<< Add retry loop for the network request itself <<<
        max_retries = 5
        retry_delay = 1
        last_exception = None

        for attempt in range(max_retries):
            try: # <<< Outer try block for requests/JSON processing
                print(f"Check status attempt {attempt + 1}/{max_retries} (TaskID: {task_id})...")
                response = requests.post(url, json=data, headers=headers, timeout=20)
                print(f"Check status ({task_id}): Response Status Code: {response.status_code}")

                # Process the response (JSON decoding, status checks) only if request succeeded
                try: # <<< Inner try block for JSON decoding
                    result = response.json()
                    print(f"Check status ({task_id}): Response JSON: {json.dumps(result, indent=2, ensure_ascii=False)}")
                except json.JSONDecodeError: # <<< Correct indent
                    print(f"Check status ({task_id}): Failed to decode JSON. Response Text: {response.text}")
                    error_msg = f"HTTP Error {response.status_code} and Invalid JSON" if response.status_code != 200 else "Invalid JSON response"
                    # Consider this a failure for retry purposes if status code indicates error
                    if response.status_code != 200:
                         raise requests.exceptions.RequestException(f"HTTP Error {response.status_code} with Invalid JSON")
                    else: # If status 200 but bad JSON, treat as terminal error for this check
                         return {"taskStatus": "error", "error": error_msg}

                # Process the decoded JSON result
                api_code = result.get("code") # <<< Correct indent
                api_msg = result.get("msg", "") # <<< Correct indent
                api_data = result.get("data") # <<< Correct indent

                # Handle Non-200 status codes AFTER potential JSON decoding
                if response.status_code != 200: # <<< Correct indent
                    error_detail = api_msg if api_msg else f"HTTP Error {response.status_code}"
                    print(f"Warning: Non-200 status code ({response.status_code}). API Message: {api_msg}")
                    # Raise exception to trigger retry for server-side issues (e.g., 5xx)
                    if 500 <= response.status_code < 600:
                         raise requests.exceptions.RequestException(f"Server Error {response.status_code}: {error_detail}")
                    else: # Treat other non-200 codes (like 4xx) as terminal for this check
                         return {"taskStatus": "error", "error": error_detail} # <<< Correct indent

                # --- If we got here, the request was successful (status 200, valid JSON) --- 
                # Now interpret the API result
                # 1. Check for successful completion (code 0, list data)
                if api_code == 0 and isinstance(api_data, list) and api_data:
                    return api_data # SUCCESS, return output data

                # 2. Check for explicit QUEUED message
                elif api_msg == "APIKEY_TASK_IS_QUEUED":
                    print(f"Check status ({task_id}): Task QUEUED.")
                    return {"taskStatus": "QUEUED"}

                # 3. Check for explicit RUNNING message
                elif api_msg == "APIKEY_TASK_IS_RUNNING":
                    possible_wss_url = None
                    if isinstance(api_data, dict): # Check if api_data is a dict
                        possible_wss_url = api_data.get("netWssUrl")
                    # No need to check result["data"] separately here, as api_data holds it

                    if possible_wss_url:
                        print(f"Check status ({task_id}): Task RUNNING, found netWssUrl.")
                        return {"taskStatus": "RUNNING", "netWssUrl": possible_wss_url}
                    else:
                        print(f"Check status ({task_id}): Task RUNNING, but netWssUrl not found in response data.")
                        return {"taskStatus": "RUNNING"} # Return RUNNING status without URL

                # 4. Check for API-reported errors (non-zero code, excluding specific handled messages)
                elif api_code != 0:
                    print(f"API Error checking status (code {api_code}): {api_msg}")
                    return {"taskStatus": "error", "error": api_msg}

                # 5. Check for code 0 but no data/output list (likely still running/initializing)
                elif api_code == 0 and (api_data is None or (isinstance(api_data, list) and not api_data)):
                    print(f"Check status ({task_id}): Task RUNNING (code 0, no output data yet).")
                    # Check if WSS url is available even in this state
                    possible_wss_url = None
                    if isinstance(api_data, dict):
                        possible_wss_url = api_data.get("netWssUrl")

                    if possible_wss_url:
                         print(f"Check status ({task_id}): Task RUNNING (code 0, no output), found netWssUrl.")
                         return {"taskStatus": "RUNNING", "netWssUrl": possible_wss_url}
                    else:
                         print(f"Check status ({task_id}): Task RUNNING (code 0, no output), netWssUrl not found.")
                         return {"taskStatus": "RUNNING"}

                # 6. Fallback for unknown successful response structure
                else:
                    print(f"Unknown task status response structure: {result}")
                    return {"taskStatus": "unknown", "details": result}

            except requests.exceptions.Timeout as e: # <<< Correct except clause indent
                print(f"Network timeout on attempt {attempt + 1}/{max_retries} for task {task_id}")
                last_exception = e
                # Continue to retry loop
            except requests.exceptions.RequestException as e: # <<< Correct except clause indent
                print(f"Network error on attempt {attempt + 1}/{max_retries}: {e}")
                last_exception = e
                # Continue to retry loop
            # Note: json.JSONDecodeError or other processing errors after successful request
            # are handled inside the try block and return specific statuses without retry here.

            # If exception occurred and not the last attempt, wait and retry
            if last_exception is not None and attempt < max_retries - 1:
                print(f"Retrying status check in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            elif last_exception is not None: # Max retries reached after an exception
                 print(f"Max retries ({max_retries}) reached for status check due to network errors. Last error: {last_exception}")
                 return {"taskStatus": "error", "error": f"Network Error after retries: {last_exception}"} # <<< Return error after retries
        
        # This point should theoretically not be reached if the loop handles all cases, but as a fallback:
        print(f"Status check loop completed unexpectedly after {max_retries} attempts.")
        return {"taskStatus": "error", "error": f"Status check failed after {max_retries} attempts. Last error: {last_exception}"}


# <<< Add NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "RH_ExecuteNode": ExecuteNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RH_ExecuteNode": "RunningHub Execute Task"
}

# <<< Standard Python entry point check (optional but good practice)
if __name__ == "__main__":
    # Example usage or testing code could go here
    pass

