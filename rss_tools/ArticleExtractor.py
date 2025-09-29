import os
import time
import nltk
from pathlib import Path

class ArticleContentExtractor:
    """
    Article Content Extractor Node
    
    Uses newspaper4k to extract full article content from URLs with intelligent text processing.
    """
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("extracted_content",)
    FUNCTION = "execute"
    CATEGORY = "RSS Content Processing"

    def __init__(self):
        self._setup_nltk()
    
    def _setup_nltk(self):
        """Setup NLTK data path and download required resources"""
        try:
            current_dir = Path(__file__).parent
            nltk_data_path = current_dir / 'nltk_data'
            nltk_data_path.mkdir(exist_ok=True)
            
            # Add local nltk data path
            if str(nltk_data_path) not in nltk.data.path:
                nltk.data.path.insert(0, str(nltk_data_path))
            
            # Check and download punkt resources if needed
            resources_to_check = ['tokenizers/punkt', 'tokenizers/punkt_tab']
            for resource in resources_to_check:
                try:
                    nltk.data.find(resource)
                except LookupError:
                    resource_name = resource.split('/')[-1]
                    print(f"正在下载NLTK {resource_name}资源...")
                    try:
                        nltk.download(resource_name, download_dir=str(nltk_data_path), quiet=True)
                    except Exception as download_error:
                        print(f"下载{resource_name}失败: {download_error}")
                
        except Exception as e:
            print(f"NLTK setup warning: {e}")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "article_links": ("STRING", {
                    "forceInput": True,
                    "multiline": True
                }),
                "extract_title": ("BOOLEAN", {"default": True}),
                "extract_summary": ("BOOLEAN", {"default": True}),
                "extract_text": ("BOOLEAN", {"default": True}),
                "extract_authors": ("BOOLEAN", {"default": False}),
                "extract_keywords": ("BOOLEAN", {"default": False}),
                "extract_publish_date": ("BOOLEAN", {"default": False}),
                "max_text_length": ("INT", {
                    "default": 1000,
                    "min": 100,
                    "max": 10000,
                    "step": 100
                }),
                "timeout": ("INT", {
                    "default": 30,
                    "min": 5,
                    "max": 120,
                    "step": 1
                }),
                "language": (["auto", "zh", "en", "ja", "ko", "es", "fr", "de"], {
                    "default": "auto"
                }),
            },
            "optional": {
                "custom_headers": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "User-Agent: Mozilla/5.0...\nReferer: https://example.com"
                }),
                "clear_nltk_cache": ("BOOLEAN", {"default": False}),
            }
        }

    def execute(self, article_links, extract_title, extract_summary, extract_text,
                extract_authors, extract_keywords, extract_publish_date, max_text_length,
                timeout, language, custom_headers="", clear_nltk_cache=False):
        return (self.extract_articles(
            article_links, extract_title, extract_summary, extract_text,
            extract_authors, extract_keywords, extract_publish_date, max_text_length,
            timeout, language, custom_headers, clear_nltk_cache
        ),)
    
    def extract_articles(self, article_links, extract_title, extract_summary, extract_text,
                        extract_authors, extract_keywords, extract_publish_date, max_text_length,
                        timeout, language, custom_headers, clear_nltk_cache):
        try:
            # Import newspaper4k
            from newspaper import Article, Config
            
            # Setup custom headers
            config = Config()
            if custom_headers.strip():
                headers = {}
                for line in custom_headers.strip().split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        headers[key.strip()] = value.strip()
                config.headers = headers
            
            config.request_timeout = timeout
            if language != "auto":
                config.language = language
            
            # Clean NLTK cache if requested
            if clear_nltk_cache:
                self._clean_nltk_cache()
            
            # Process article links
            links = [link.strip() for link in article_links.split('\n') if link.strip()]
            if not links:
                return "Error: No valid article links provided"
            
            results = []
            
            for i, link in enumerate(links):
                try:
                    print(f"正在处理文章 {i+1}/{len(links)}: {link[:50]}...")
                    
                    article = Article(link, config=config)
                    article.download()
                    article.parse()
                    
                    # Generate NLP data if needed
                    if extract_summary or extract_keywords:
                        article.nlp()
                    
                    # Build content output
                    content_parts = []
                    content_parts.append(f"URL: {link}")
                    
                    if extract_title and article.title:
                        content_parts.append(f"Title: {article.title}")
                    
                    if extract_authors and article.authors:
                        authors_str = ", ".join(article.authors)
                        content_parts.append(f"Authors: {authors_str}")
                    
                    if extract_publish_date and article.publish_date:
                        content_parts.append(f"Published: {article.publish_date}")
                    
                    if extract_text and article.text:
                        text = article.text[:max_text_length]
                        if len(article.text) > max_text_length:
                            text += "..."
                        content_parts.append(f"Text: {text}")
                    
                    if extract_summary and article.summary:
                        content_parts.append(f"Summary: {article.summary}")
                    
                    if extract_keywords and article.keywords:
                        keywords_str = ", ".join(article.keywords[:10])  # Limit to 10 keywords
                        content_parts.append(f"Keywords: {keywords_str}")
                    
                    results.append("\n".join(content_parts))
                    
                except Exception as e:
                    error_msg = f"Error processing {link}: {str(e)}"
                    print(error_msg)
                    results.append(error_msg)
                    continue
            
            if not results:
                return "Error: No articles could be processed successfully"
            
            return "\n\n" + "="*80 + "\n\n".join(results)
            
        except ImportError:
            return "Error: newspaper4k not installed. Please run: pip install newspaper4k"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _clean_nltk_cache(self):
        """Clean NLTK cache data"""
        try:
            current_dir = Path(__file__).parent
            nltk_data_path = current_dir / 'nltk_data'
            
            if nltk_data_path.exists():
                import shutil
                punkt_path = nltk_data_path / 'tokenizers' / 'punkt'
                if punkt_path.exists():
                    shutil.rmtree(punkt_path)
                    print(f"已清理NLTK缓存: {punkt_path}")
        except Exception as e:
            print(f"清理NLTK缓存失败: {e}")

NODE_CLASS_MAPPINGS = {
    "ArticleContentExtractor": ArticleContentExtractor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArticleContentExtractor": "文章内容提取器"
}