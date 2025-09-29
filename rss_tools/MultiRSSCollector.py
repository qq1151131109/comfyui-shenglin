import feedparser
import time
import re
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

class MultiRSSCollector:
    """
    Multi RSS Collector Node
    
    Collects articles from multiple RSS sources simultaneously with intelligent filtering.
    """
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("all_articles_json", "source_summary", "all_links")
    FUNCTION = "execute"
    CATEGORY = "RSS Content Processing"

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rss_urls": ("STRING", {
                    "multiline": True,
                    "default": "https://feeds.bbci.co.uk/news/rss.xml\nhttps://rss.cnn.com/rss/edition.rss",
                    "placeholder": "每行一个RSS URL"
                }),
                "articles_per_source": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "step": 1
                }),
                "time_filter": (["all", "24h", "48h", "7d"], {
                    "default": "24h"
                }),
                "language_filter": (["all", "zh", "en", "auto"], {
                    "default": "all"
                }),
                "timeout": ("INT", {
                    "default": 30,
                    "min": 5,
                    "max": 120,
                    "step": 1
                }),
                "max_workers": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
            },
            "optional": {
                "global_filter_keywords": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "keyword1,keyword2"
                }),
                "global_exclude_keywords": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "ad,spam,promotion"
                }),
            }
        }

    def execute(self, rss_urls, articles_per_source, time_filter, language_filter,
                timeout, max_workers, global_filter_keywords="", global_exclude_keywords=""):
        return self.collect_multi_rss(
            rss_urls, articles_per_source, time_filter, language_filter,
            timeout, max_workers, global_filter_keywords, global_exclude_keywords
        )
    
    def collect_multi_rss(self, rss_urls, articles_per_source, time_filter, language_filter,
                          timeout, max_workers, global_filter_keywords, global_exclude_keywords):
        try:
            # Parse RSS URLs
            urls = [url.strip() for url in rss_urls.split('\n') if url.strip()]
            if not urls:
                return "Error: No valid RSS URLs provided", "", ""
            
            # Setup filters
            filter_list = [k.strip().lower() for k in global_filter_keywords.split(",") if k.strip()]
            exclude_list = [k.strip().lower() for k in global_exclude_keywords.split(",") if k.strip()]
            
            # Setup time filter
            time_threshold = self._get_time_threshold(time_filter)
            
            # Collect articles from all sources
            all_articles = []
            source_stats = {}
            all_links = []
            
            print(f"开始收集 {len(urls)} 个RSS源的内容...")
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit tasks
                future_to_url = {
                    executor.submit(
                        self._process_single_rss, 
                        url, articles_per_source, time_threshold, 
                        filter_list, exclude_list, language_filter, timeout
                    ): url for url in urls
                }
                
                # Collect results
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        articles, stats, links = future.result()
                        all_articles.extend(articles)
                        all_links.extend(links)
                        source_stats[url] = stats
                        print(f"✅ {url}: 获取到 {len(articles)} 篇文章")
                    except Exception as e:
                        source_stats[url] = {"error": str(e), "articles_count": 0}
                        print(f"❌ {url}: {str(e)}")
            
            # Generate summary
            total_articles = len(all_articles)
            successful_sources = sum(1 for stats in source_stats.values() if "error" not in stats)
            
            source_summary = f"RSS源统计:\n"
            source_summary += f"- 总源数: {len(urls)}\n"
            source_summary += f"- 成功源数: {successful_sources}\n"
            source_summary += f"- 总文章数: {total_articles}\n"
            source_summary += f"- 处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            for url, stats in source_stats.items():
                if "error" in stats:
                    source_summary += f"❌ {url[:50]}...: {stats['error']}\n"
                else:
                    source_summary += f"✅ {url[:50]}...: {stats['articles_count']} 篇文章\n"
            
            # Format output
            all_articles_json = json.dumps(all_articles, ensure_ascii=False, indent=2)
            all_links_text = "\n".join(all_links)
            
            return all_articles_json, source_summary, all_links_text
            
        except Exception as e:
            return f"Error: {str(e)}", "", ""
    
    def _process_single_rss(self, url, articles_per_source, time_threshold, 
                           filter_list, exclude_list, language_filter, timeout):
        """Process a single RSS source"""
        articles = []
        links = []
        
        try:
            feed = feedparser.parse(url)
            if hasattr(feed, 'status') and feed.status >= 400:
                raise Exception(f"HTTP {feed.status}")
            
            if not feed.entries:
                raise Exception("No entries found")
            
            processed_count = 0
            for entry in feed.entries:
                if processed_count >= articles_per_source:
                    break
                
                # Extract basic info
                title = getattr(entry, 'title', '')
                description = getattr(entry, 'description', getattr(entry, 'summary', ''))
                link = getattr(entry, 'link', '')
                published = getattr(entry, 'published_parsed', None) or getattr(entry, 'updated_parsed', None)
                author = getattr(entry, 'author', '')
                
                # Time filter
                if time_threshold and published:
                    try:
                        pub_time = datetime(*published[:6])
                        if pub_time < time_threshold:
                            continue
                    except:
                        pass  # Skip time filtering if parsing fails
                
                # Content filter
                content_text = f"{title} {description}".lower()
                
                if filter_list and not any(keyword in content_text for keyword in filter_list):
                    continue
                
                if exclude_list and any(keyword in content_text for keyword in exclude_list):
                    continue
                
                # Language filter (basic implementation)
                if language_filter != "all":
                    if language_filter == "zh" and not self._contains_chinese(content_text):
                        continue
                    elif language_filter == "en" and self._contains_chinese(content_text):
                        continue
                
                # Clean description
                clean_desc = re.sub(r'<[^>]+>', '', description)
                
                # Create article object
                article = {
                    "source_url": url,
                    "title": title,
                    "description": clean_desc,
                    "link": link,
                    "published": published,
                    "author": author,
                    "timestamp": datetime.now().isoformat()
                }
                
                articles.append(article)
                if link:
                    links.append(link)
                processed_count += 1
            
            stats = {"articles_count": len(articles), "success": True}
            return articles, stats, links
            
        except Exception as e:
            raise Exception(f"Failed to process RSS: {str(e)}")
    
    def _get_time_threshold(self, time_filter):
        """Get time threshold based on filter setting"""
        if time_filter == "all":
            return None
        
        now = datetime.now()
        if time_filter == "24h":
            return now - timedelta(hours=24)
        elif time_filter == "48h":
            return now - timedelta(hours=48)
        elif time_filter == "7d":
            return now - timedelta(days=7)
        return None
    
    def _contains_chinese(self, text):
        """Check if text contains Chinese characters"""
        return bool(re.search(r'[\u4e00-\u9fff]', text))

NODE_CLASS_MAPPINGS = {
    "MultiRSSCollector": MultiRSSCollector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiRSSCollector": "多源RSS收集器"
}