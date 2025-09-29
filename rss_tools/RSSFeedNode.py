import feedparser
import time
import re
import json
import os
from urllib.parse import urlparse

class RSSFeedNode:
    """
    RSS Feed Node

    Fetches and parses RSS feeds, producing a script output containing news titles and descriptions.
    """
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("script_output", "article_links",)
    FUNCTION = "execute"
    CATEGORY = "RSS Content Processing"

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feed_url": ("STRING", {
                    "multiline": False, 
                    "dynamicPrompts": False, 
                    "default": "http://example.com/rss"
                }),
                "extract_mode": (["feed_only", "links_only", "feed_and_links"], {
                    "default": "feed_only"
                }),
                "feed_count": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 50,
                    "step": 1
                }),
                "include_title": ("BOOLEAN", {"default": True}),
                "include_content": ("BOOLEAN", {"default": True}),
                "include_link": ("BOOLEAN", {"default": False}),
                "include_publish_date": ("BOOLEAN", {"default": False}),
                "include_author": ("BOOLEAN", {"default": False}),
                "text_separator": ("STRING", {
                    "default": "\n---\n",
                    "multiline": False
                }),
                "timeout": ("INT", {
                    "default": 30,
                    "min": 5,
                    "max": 120,
                    "step": 1
                }),
                "retry_count": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 5,
                    "step": 1
                }),
            },
            "optional": {
                "filter_keywords": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "keyword1,keyword2"
                }),
                "exclude_keywords": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "ad,spam"
                }),
            }
        }

    def execute(self, extract_mode, feed_url, feed_count, include_title, include_content, 
                include_link, include_publish_date, include_author, text_separator, timeout, 
                retry_count, filter_keywords="", exclude_keywords=""):
        script_output, article_links = self.fetch_and_parse_rss(
            extract_mode, feed_url, feed_count, include_title, include_content, include_link,
            include_publish_date, include_author, text_separator, timeout, retry_count, 
            filter_keywords, exclude_keywords
        )
        return (script_output, article_links)
    
    def fetch_and_parse_rss(self, extract_mode, feed_url, feed_count, include_title, include_content, 
                           include_link, include_publish_date, include_author, text_separator, 
                           timeout, retry_count, filter_keywords, exclude_keywords):
        filter_list = [k.strip().lower() for k in filter_keywords.split(",") if k.strip()]
        exclude_list = [k.strip().lower() for k in exclude_keywords.split(",") if k.strip()]
        
        for attempt in range(retry_count + 1):
            try:
                feed = feedparser.parse(feed_url)
                if hasattr(feed, 'status') and feed.status >= 400:
                    if attempt < retry_count:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        return f"Error: Failed to fetch RSS feed after {retry_count + 1} attempts", ""
                
                if not feed.entries:
                    return "Error: No entries found in RSS feed", ""
                
                script_output = "News Update:\n"
                article_links = []
                processed_count = 0
                
                for entry in feed.entries:
                    if processed_count >= feed_count:
                        break
                    
                    title = getattr(entry, 'title', '')
                    description = getattr(entry, 'description', getattr(entry, 'summary', ''))
                    link = getattr(entry, 'link', '')
                    publish_date = getattr(entry, 'published', getattr(entry, 'updated', ''))
                    author = getattr(entry, 'author', '')
                    
                    content_text = f"{title} {description}".lower()
                    
                    if filter_list and not any(keyword in content_text for keyword in filter_list):
                        continue
                    
                    if exclude_list and any(keyword in content_text for keyword in exclude_list):
                        continue
                    
                    # 根据模式处理输出
                    if extract_mode in ["links_only", "feed_and_links"]:
                        if link:
                            article_links.append(link)
                    
                    if extract_mode in ["feed_only", "feed_and_links"]:
                        entry_output = []
                        
                        if include_title and title:
                            entry_output.append(f"Title: {title}")
                        
                        if include_content and description:
                            clean_desc = re.sub(r'<[^>]+>', '', description)
                            entry_output.append(f"Description: {clean_desc}")
                        
                        if include_link and link:
                            entry_output.append(f"Link: {link}")
                        
                        if include_publish_date and publish_date:
                            entry_output.append(f"Published: {publish_date}")
                        
                        if include_author and author:
                            entry_output.append(f"Author: {author}")
                        
                        if entry_output:
                            script_output += "\n".join(entry_output) + text_separator
                    
                    processed_count += 1
                
                # 处理不同模式的输出
                if extract_mode == "links_only":
                    script_output = "Article Links:\n" + "\n".join(article_links)
                elif extract_mode == "feed_only":
                    script_output = script_output.rstrip(text_separator)
                elif extract_mode == "feed_and_links":
                    script_output = script_output.rstrip(text_separator)
                
                return script_output, "\n".join(article_links)
                
            except Exception as e:
                if attempt < retry_count:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return f"Error: {str(e)}", ""

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "RSSFeedNode": RSSFeedNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "RSSFeedNode": "RSS订阅解析器"
}
