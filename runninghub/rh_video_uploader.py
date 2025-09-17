import requests
import json
import os
import time

# Try importing folder_paths safely for potential future use, though not strictly needed for this node's core logic
try:
    import folder_paths
    comfyui_env_available = True
except ImportError:
    folder_paths = None # Set to None if not available
    comfyui_env_available = False
    print("RH_VideoUploader: ComfyUI folder_paths not found. Cannot determine input file path.")


class RH_VideoUploader:
    # This node relies heavily on its JavaScript counterpart for the actual upload.
    # The Python part primarily retrieves the filename returned by the JS upload process.

    @classmethod
    def INPUT_TYPES(cls):
        # Add a required input for the filename provided by the JS widget after ComfyUI upload
        # This widget type should allow JS to set its value.
        # A standard STRING input where JS sets the value seems appropriate.
        return {
            "required": {
                "apiConfig": ("STRUCT",),
                # This input receives the filename assigned by ComfyUI's upload endpoint
                # It's made visible and editable, but primarily set by the JS interaction.
                "video": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "upload_and_get_filename" # Renamed function to reflect action
    CATEGORY = "🔥 Shenglin/工具" # Or your preferred category
    OUTPUT_NODE = False # This node outputs data, not a final image/video display

    def upload_and_get_filename(self, apiConfig, video):
        """
        Reads the video file from ComfyUI's input directory based on the 'video' filename,
        uploads it to the RunningHub API, and returns the resulting filename/ID.
        """
        # 1. Validate inputs
        if not comfyui_env_available or not folder_paths:
            raise ImportError("folder_paths module is required for RH_VideoUploader to find input files.")

        if not isinstance(apiConfig, dict) or not apiConfig.get("apiKey") or not apiConfig.get("base_url"):
            raise ValueError("Invalid or missing apiConfig structure provided to RH_VideoUploader.")
        
        if not video or video.strip() == "":
             raise ValueError("No video filename provided. Please select and upload a video using the node's widget.")

        apiKey = apiConfig["apiKey"]
        baseUrl = apiConfig["base_url"]

        # 2. Get the full path to the uploaded file in ComfyUI's input directory
        # The 'video' input contains the relative path within the input directory
        try:
            video_path = folder_paths.get_annotated_filepath(video)
            if not video_path or not os.path.exists(video_path):
                 # Check subdirectories common for uploads if get_annotated_filepath fails directly
                 # (get_annotated_filepath usually handles this, but as a fallback)
                 potential_path = os.path.join(folder_paths.get_input_directory(), video)
                 if os.path.exists(potential_path):
                     video_path = potential_path
                 else: # Check common subfolders like 'uploads'
                      potential_path = os.path.join(folder_paths.get_input_directory(), 'uploads', video)
                      if os.path.exists(potential_path):
                          video_path = potential_path
                      else:
                         raise FileNotFoundError(f"Video file not found in input directory: {video}")

            print(f"RH_VideoUploader: Found video file at: {video_path}")

        except Exception as e:
            raise FileNotFoundError(f"Error finding video file '{video}': {e}")

        # 3. Prepare for RunningHub API upload
        # *** Use the same endpoint and data structure as ImageUploader ***
        upload_api_url = f"{baseUrl}/task/openapi/upload" # Corrected endpoint
        # Headers should likely NOT contain the apiKey based on ImageUploader
        headers = {
            # 'X-API-Key': apiKey, # REMOVED based on ImageUploader
            'User-Agent': 'ComfyUI-RH_VideoUploader/1.0'
        }
        # Data payload containing apiKey and fileType
        data = {
            'apiKey': apiKey,
            'fileType': 'video' # Added fileType, assuming 'video' is correct
        }

        # 4. Read the file and upload
        print(f"RH_VideoUploader: Uploading {video_path} to {upload_api_url}...")
        
        # Add retry logic
        max_retries = 5
        retry_delay = 1 # Initial delay in seconds
        last_exception = None
        response = None

        for attempt in range(max_retries):
            try:
                with open(video_path, 'rb') as f:
                    files = {
                        'file': (os.path.basename(video_path), f)
                    }
                    # Send request with apiKey only in data, files attached
                    response = requests.post(upload_api_url, headers=headers, data=data, files=files)
                    print(f"RH_VideoUploader: Upload attempt {attempt + 1}/{max_retries} - Status Code: {response.status_code}")
                    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                
                # If successful, break the loop
                break 

            except requests.exceptions.Timeout as e:
                last_exception = e
                print(f"RH_VideoUploader: Upload attempt {attempt + 1} timed out.")
            except requests.exceptions.ConnectionError as e:
                 last_exception = e
                 print(f"RH_VideoUploader: Upload attempt {attempt + 1} connection error: {e}")
            except requests.exceptions.RequestException as e:
                last_exception = e
                print(f"RH_VideoUploader: Upload attempt {attempt + 1} failed: {e}")
                # Check if response exists for potential non-2xx status codes handled by raise_for_status
                if e.response is not None:
                     print(f"RH_VideoUploader: Response content on error: {e.response.text}")
                     # Potentially break early for client errors (4xx) that won't be fixed by retrying?
                     # For now, we retry all RequestExceptions
                
            # Wait before retrying, unless it's the last attempt
            if attempt < max_retries - 1:
                print(f"RH_VideoUploader: Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2 # Exponential backoff
            else:
                 print(f"RH_VideoUploader: Max retries ({max_retries}) reached.")
                 # Raise the last exception encountered after all retries fail
                 raise ConnectionError(f"Failed to upload video to RunningHub API after {max_retries} attempts. Last error: {last_exception}") from last_exception

        # If loop completes without breaking (shouldn't happen if success breaks), or response is None
        if response is None:
             raise ConnectionError(f"Upload failed after {max_retries} attempts, no response received. Last error: {last_exception}")

        # 5. Parse successful response (outside the retry loop)
        try:
            response_json = response.json()
            print(f"RH_VideoUploader: Upload API Response JSON: {response_json}")
        except json.JSONDecodeError as e:
             print(f"RH_VideoUploader: Failed to decode JSON response: {response.text}")
             raise ValueError(f"Failed to decode API response after successful upload: {e}") from e

        # Check API-level success code
        if response_json.get('code') != 0:
            raise ValueError(f"RunningHub API reported an error after upload: {response_json.get('msg', 'Unknown API error')}")

        # Extract filename using the correct key 'fileName'
        rh_data = response_json.get("data", {})
        uploaded_filename = None # Initialize
        if isinstance(rh_data, dict):
            uploaded_filename = rh_data.get("fileName") # Corrected key: fileName
            # Add fallbacks if needed, though fileName seems primary based on logs
            # uploaded_filename = uploaded_filename or rh_data.get("fileId") or rh_data.get("url") 
        elif isinstance(rh_data, str):
             uploaded_filename = rh_data
        
        if not isinstance(uploaded_filename, str) or not uploaded_filename:
            raise ValueError("Upload succeeded but 'fileName' (or compatible field) not found in RunningHub API response.data.")

        print(f"RH_VideoUploader: Upload successful. RunningHub filename/ID: {uploaded_filename}")
        return (uploaded_filename,)

        # Removed generic Exception catch block to let specific errors propagate
        # except Exception as e:
        #     print(f"RH_VideoUploader: An unexpected error occurred during upload: {e}")
        #     raise RuntimeError(f"Unexpected error during video upload: {e}") from e

# Mappings for ComfyUI
# Moved to __init__.py typically, but included here for completeness if run standalone initially
# NODE_CLASS_MAPPINGS = {
#     "VideoUploaderNode_RH": VideoUploaderNode
# }
# NODE_DISPLAY_NAME_MAPPINGS = {
#     "VideoUploaderNode_RH": "RunningHub Video Uploader"
# } 