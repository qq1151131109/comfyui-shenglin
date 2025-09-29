import json
import time
import os
import nltk
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

class BatchArticleExtractor:
    """
    Batch Article Extractor Node
    
    Processes multiple articles simultaneously with intelligent content extraction and progress tracking.
    """
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("extracted_articles_json", "extraction_report")
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
            
            if str(nltk_data_path) not in nltk.data.path:
                nltk.data.path.insert(0, str(nltk_data_path))
            
            resources_to_check = ['tokenizers/punkt', 'tokenizers/punkt_tab']
            for resource in resources_to_check:
                try:
                    nltk.data.find(resource)
                except LookupError:
                    resource_name = resource.split('/')[-1]
                    try:
                        nltk.download(resource_name, download_dir=str(nltk_data_path), quiet=True)
                    except Exception:
                        pass
                        
        except Exception as e:
            print(f"NLTK setup warning: {e}")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "article_data": ("STRING", {
                    "forceInput": True,
                    "multiline": True
                }),
                "extract_strategy": (["fast", "detailed", "smart"], {
                    "default": "smart"
                }),
                "max_articles": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 200,
                    "step": 1
                }),
                "parallel_workers": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 8,
                    "step": 1
                }),
                "timeout_per_article": ("INT", {
                    "default": 20,
                    "min": 5,
                    "max": 60,
                    "step": 1
                }),
                "max_text_length": ("INT", {
                    "default": 2000,
                    "min": 200,
                    "max": 10000,
                    "step": 100
                }),
            },
            "optional": {
                "custom_headers": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "User-Agent: Mozilla/5.0...\nReferer: https://example.com"
                }),
                "skip_extraction_errors": ("BOOLEAN", {"default": True}),
            }
        }

    def execute(self, article_data, extract_strategy, max_articles, parallel_workers,
                timeout_per_article, max_text_length, custom_headers="", skip_extraction_errors=True):
        return self.batch_extract_articles(
            article_data, extract_strategy, max_articles, parallel_workers,
            timeout_per_article, max_text_length, custom_headers, skip_extraction_errors
        )
    
    def batch_extract_articles(self, article_data, extract_strategy, max_articles, parallel_workers,
                              timeout_per_article, max_text_length, custom_headers, skip_extraction_errors):
        try:
            # Import newspaper4k
            from newspaper import Article, Config
            
            # Parse input data
            try:
                if article_data.startswith('['):
                    articles = json.loads(article_data)
                else:
                    # Handle simple link list format
                    links = [link.strip() for link in article_data.split('\n') if link.strip()]
                    articles = [{"link": link, "title": "", "source_url": ""} for link in links]
            except json.JSONDecodeError:
                # Fallback: treat as simple text with links
                links = [link.strip() for link in article_data.split('\n') if link.strip()]
                articles = [{"link": link, "title": "", "source_url": ""} for link in links]
            
            if not articles:
                return "Error: No articles to process", ""
            
            # Limit articles
            articles = articles[:max_articles]
            
            # Setup newspaper config
            config = Config()
            if custom_headers.strip():
                headers = {}
                for line in custom_headers.strip().split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        headers[key.strip()] = value.strip()
                config.headers = headers
            
            config.request_timeout = timeout_per_article
            
            # Setup extraction strategy
            extract_options = self._get_extract_options(extract_strategy)
            
            # Process articles
            extracted_articles = []
            extraction_stats = {
                "total": len(articles),
                "successful": 0,
                "failed": 0,
                "errors": [],
                "start_time": datetime.now()
            }
            
            print(f"开始批量提取 {len(articles)} 篇文章，使用 {parallel_workers} 个并发...")
            
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                # Submit tasks
                future_to_article = {
                    executor.submit(
                        self._extract_single_article,
                        article, config, extract_options, max_text_length
                    ): article for article in articles
                }
                
                # Collect results
                for i, future in enumerate(as_completed(future_to_article)):
                    article = future_to_article[future]
                    try:
                        extracted_article = future.result()
                        if extracted_article:
                            extracted_articles.append(extracted_article)
                            extraction_stats["successful"] += 1
                        else:
                            extraction_stats["failed"] += 1
                            
                        # Progress update
                        progress = (i + 1) / len(articles) * 100
                        print(f"进度: {progress:.1f}% ({i+1}/{len(articles)})")
                        
                    except Exception as e:
                        extraction_stats["failed"] += 1
                        error_msg = f"Article {article.get('link', 'unknown')}: {str(e)}"
                        extraction_stats["errors"].append(error_msg)
                        
                        if not skip_extraction_errors:
                            print(f"❌ {error_msg}")
            
            # Generate report
            extraction_stats["end_time"] = datetime.now()
            processing_time = (extraction_stats["end_time"] - extraction_stats["start_time"]).total_seconds()
            
            extraction_report = f"""批量文章提取报告:
总文章数: {extraction_stats['total']}
成功提取: {extraction_stats['successful']}
提取失败: {extraction_stats['failed']}
成功率: {extraction_stats['successful']/extraction_stats['total']*100:.1f}%
处理时间: {processing_time:.1f}秒
平均耗时: {processing_time/extraction_stats['total']:.1f}秒/篇

提取策略: {extract_strategy}
并发数: {parallel_workers}
最大文本长度: {max_text_length}字符

错误详情:
{chr(10).join(extraction_stats['errors'][:10])}  # 只显示前10个错误
"""

            if extraction_stats["failed"] > 10:
                extraction_report += f"\n... 还有 {extraction_stats['failed'] - 10} 个错误未显示"
            
            # Format output
            extracted_articles_json = json.dumps(extracted_articles, ensure_ascii=False, indent=2)
            
            return extracted_articles_json, extraction_report
            
        except ImportError:
            return "Error: newspaper4k not installed. Please run: pip install newspaper4k", ""
        except Exception as e:
            return f"Error: {str(e)}", ""
    
    def _extract_single_article(self, article_info, config, extract_options, max_text_length):
        """Extract content from a single article"""
        try:
            from newspaper import Article
            
            link = article_info.get("link", "")
            if not link:
                return None
            
            article = Article(link, config=config)
            article.download()
            article.parse()
            
            # Generate NLP data if needed
            if extract_options["summary"] or extract_options["keywords"]:
                try:
                    article.nlp()
                except Exception:
                    pass  # NLP failure shouldn't stop the extraction
            
            # Build extracted article
            extracted = {
                "original_data": article_info,
                "url": link,
                "link": link,  # 保持与RSS格式一致
                "source_url": article_info.get("source_url", ""),  # 传递RSS源URL
                "extraction_time": datetime.now().isoformat()
            }
            
            if extract_options["title"] and article.title:
                extracted["title"] = article.title
            
            if extract_options["text"] and article.text:
                text = article.text[:max_text_length]
                if len(article.text) > max_text_length:
                    text += "..."
                extracted["text"] = text
                extracted["text_length"] = len(article.text)
            
            if extract_options["summary"] and hasattr(article, 'summary') and article.summary:
                extracted["summary"] = article.summary
            
            if extract_options["authors"] and article.authors:
                extracted["authors"] = list(article.authors)
            
            if extract_options["keywords"] and hasattr(article, 'keywords') and article.keywords:
                extracted["keywords"] = list(article.keywords)[:15]  # Limit keywords
            
            if extract_options["publish_date"] and article.publish_date:
                extracted["publish_date"] = article.publish_date.isoformat()
            
            # Only return if we got meaningful content
            if extracted.get("title") or extracted.get("text"):
                return extracted
            else:
                return None
                
        except Exception as e:
            raise Exception(f"Extract failed: {str(e)}")
    
    def _get_extract_options(self, strategy):
        """Get extraction options based on strategy"""
        if strategy == "fast":
            return {
                "title": True,
                "text": True,
                "summary": False,
                "authors": False,
                "keywords": False,
                "publish_date": False
            }
        elif strategy == "detailed":
            return {
                "title": True,
                "text": True,
                "summary": True,
                "authors": True,
                "keywords": True,
                "publish_date": True
            }
        else:  # smart
            return {
                "title": True,
                "text": True,
                "summary": True,
                "authors": False,
                "keywords": True,
                "publish_date": True
            }

NODE_CLASS_MAPPINGS = {
    "BatchArticleExtractor": BatchArticleExtractor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchArticleExtractor": "批量文章提取器"
}