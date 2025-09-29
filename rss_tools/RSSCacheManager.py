import os
import json
import pickle
import time
import gzip
from pathlib import Path

class RSSCacheManager:
    """
    RSS Cache Manager Node
    
    Manages caching for RSS content to improve performance and reduce network requests.
    """
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("cached_content",)
    FUNCTION = "execute"
    CATEGORY = "RSS Content Processing"

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_content": ("STRING", {
                    "forceInput": True
                }),
                "cache_folder": ("STRING", {
                    "default": "./cache/rss",
                    "multiline": False
                }),
                "cache_duration": ("INT", {
                    "default": 3600,
                    "min": 60,
                    "max": 86400,
                    "step": 60
                }),
                "max_cache_items": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "auto_cleanup": ("BOOLEAN", {"default": True}),
                "cache_key": ("STRING", {
                    "default": "default",
                    "multiline": False
                }),
                "force_refresh": ("BOOLEAN", {"default": False}),
                "cache_format": (["json", "pickle", "text"], {"default": "json"}),
                "compress_cache": ("BOOLEAN", {"default": False}),
            }
        }

    def execute(self, input_content, cache_folder, cache_duration, max_cache_items,
                auto_cleanup, cache_key, force_refresh, cache_format, compress_cache):
        return (self.manage_cache(
            input_content, cache_folder, cache_duration, max_cache_items,
            auto_cleanup, cache_key, force_refresh, cache_format, compress_cache
        ),)
    
    def manage_cache(self, input_content, cache_folder, cache_duration, max_cache_items,
                    auto_cleanup, cache_key, force_refresh, cache_format, compress_cache):
        try:
            # Ensure cache directory exists
            cache_path = Path(cache_folder)
            cache_path.mkdir(parents=True, exist_ok=True)
            
            # Generate cache filename
            extension = "gz" if compress_cache else cache_format
            cache_file = cache_path / f"{cache_key}.{extension}"
            
            current_time = time.time()
            
            # Check if we should use cached content
            if not force_refresh and cache_file.exists():
                cache_age = current_time - cache_file.stat().st_mtime
                if cache_age < cache_duration:
                    try:
                        cached_content = self._read_cache(cache_file, cache_format, compress_cache)
                        if cached_content:
                            return f"[CACHED] {cached_content}"
                    except Exception:
                        pass
            
            # Save new content to cache
            self._write_cache(cache_file, input_content, cache_format, compress_cache)
            
            # Auto cleanup if enabled
            if auto_cleanup:
                self._cleanup_cache(cache_path, cache_duration, max_cache_items)
            
            return input_content
            
        except Exception as e:
            return f"Cache Error: {str(e)}\n\nOriginal Content:\n{input_content}"
    
    def _read_cache(self, cache_file, cache_format, compress_cache):
        """Read content from cache file"""
        if compress_cache:
            with gzip.open(cache_file, 'rt', encoding='utf-8') as f:
                content = f.read()
        else:
            with open(cache_file, 'r', encoding='utf-8') as f:
                content = f.read()
        
        if cache_format == "json":
            data = json.loads(content)
            return data.get('content', '')
        elif cache_format == "pickle":
            # For pickle format, we need to read as binary
            if compress_cache:
                with gzip.open(cache_file, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
            return data.get('content', '')
        else:  # text format
            return content
    
    def _write_cache(self, cache_file, content, cache_format, compress_cache):
        """Write content to cache file"""
        if cache_format == "json":
            data = {
                'content': content,
                'timestamp': time.time()
            }
            content_to_write = json.dumps(data, ensure_ascii=False, indent=2)
        elif cache_format == "pickle":
            data = {
                'content': content,
                'timestamp': time.time()
            }
            if compress_cache:
                with gzip.open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
                return
            else:
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
                return
        else:  # text format
            content_to_write = content
        
        if compress_cache:
            with gzip.open(cache_file, 'wt', encoding='utf-8') as f:
                f.write(content_to_write)
        else:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(content_to_write)
    
    def _cleanup_cache(self, cache_path, cache_duration, max_cache_items):
        """Clean up old cache files"""
        try:
            current_time = time.time()
            cache_files = []
            
            # Collect all cache files with their modification times
            for file_path in cache_path.iterdir():
                if file_path.is_file():
                    mtime = file_path.stat().st_mtime
                    cache_files.append((file_path, mtime))
            
            # Remove expired files
            for file_path, mtime in cache_files:
                if current_time - mtime > cache_duration:
                    file_path.unlink(missing_ok=True)
            
            # Remove excess files (keep only the newest ones)
            remaining_files = [(f, m) for f, m in cache_files if f.exists()]
            if len(remaining_files) > max_cache_items:
                # Sort by modification time (newest first)
                remaining_files.sort(key=lambda x: x[1], reverse=True)
                # Remove oldest files
                for file_path, _ in remaining_files[max_cache_items:]:
                    file_path.unlink(missing_ok=True)
                    
        except Exception:
            pass  # Ignore cleanup errors

NODE_CLASS_MAPPINGS = {
    "RSSCacheManager": RSSCacheManager
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RSSCacheManager": "RSS缓存管理器"
}