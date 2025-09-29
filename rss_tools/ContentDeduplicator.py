import json
import re
import hashlib
from difflib import SequenceMatcher
from datetime import datetime

class ContentDeduplicator:
    """
    Content Deduplicator Node
    
    Removes duplicate articles using multiple similarity detection methods.
    """
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("unique_articles_json", "deduplication_report")
    FUNCTION = "execute"
    CATEGORY = "RSS Content Processing"

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "articles_json": ("STRING", {
                    "forceInput": True,
                    "multiline": True
                }),
                "similarity_threshold": ("FLOAT", {
                    "default": 0.80,
                    "min": 0.50,
                    "max": 0.95,
                    "step": 0.05
                }),
                "dedup_method": (["title", "content", "smart", "hash"], {
                    "default": "smart"
                }),
                "keep_strategy": (["newest", "longest", "first", "most_complete"], {
                    "default": "most_complete"
                }),
                "min_title_length": ("INT", {
                    "default": 10,
                    "min": 5,
                    "max": 100,
                    "step": 1
                }),
                "min_content_length": ("INT", {
                    "default": 100,
                    "min": 50,
                    "max": 1000,
                    "step": 10
                }),
            },
            "optional": {
                "preserve_sources": ("BOOLEAN", {"default": True}),
                "merge_similar": ("BOOLEAN", {"default": False}),
            }
        }

    def execute(self, articles_json, similarity_threshold, dedup_method, keep_strategy,
                min_title_length, min_content_length, preserve_sources=True, merge_similar=False):
        return self.deduplicate_content(
            articles_json, similarity_threshold, dedup_method, keep_strategy,
            min_title_length, min_content_length, preserve_sources, merge_similar
        )
    
    def deduplicate_content(self, articles_json, similarity_threshold, dedup_method, keep_strategy,
                           min_title_length, min_content_length, preserve_sources, merge_similar):
        try:
            # Parse articles
            try:
                articles = json.loads(articles_json)
            except json.JSONDecodeError:
                return "Error: Invalid JSON format for articles", ""
            
            if not articles:
                return "Error: No articles to process", ""
            
            print(f"开始去重处理 {len(articles)} 篇文章...")
            
            # Filter articles by minimum requirements
            filtered_articles = []
            for article in articles:
                title = article.get("title", "")
                content = article.get("text", "") or article.get("description", "")
                
                if len(title) >= min_title_length and len(content) >= min_content_length:
                    filtered_articles.append(article)
            
            print(f"过滤后剩余 {len(filtered_articles)} 篇文章")
            
            # Deduplication
            if dedup_method == "hash":
                unique_articles = self._hash_deduplication(filtered_articles)
            elif dedup_method == "title":
                unique_articles = self._title_deduplication(filtered_articles, similarity_threshold, keep_strategy)
            elif dedup_method == "content":
                unique_articles = self._content_deduplication(filtered_articles, similarity_threshold, keep_strategy)
            else:  # smart
                unique_articles = self._smart_deduplication(filtered_articles, similarity_threshold, keep_strategy, merge_similar)
            
            # Generate statistics
            original_count = len(articles)
            filtered_count = len(filtered_articles)
            unique_count = len(unique_articles)
            removed_count = filtered_count - unique_count
            
            dedup_report = f"""内容去重报告:
原始文章数: {original_count}
过滤后文章数: {filtered_count} (移除了 {original_count - filtered_count} 篇不符合长度要求的文章)
去重后文章数: {unique_count}
移除重复: {removed_count} 篇
去重率: {removed_count/filtered_count*100:.1f}%

去重方法: {dedup_method}
相似度阈值: {similarity_threshold}
保留策略: {keep_strategy}
最小标题长度: {min_title_length}字符
最小内容长度: {min_content_length}字符

处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

            # Add source distribution if preserve_sources
            if preserve_sources:
                source_dist = {}
                for article in unique_articles:
                    source = article.get("source_url", "unknown")
                    source_dist[source] = source_dist.get(source, 0) + 1
                
                dedup_report += "\n各源文章分布:\n"
                for source, count in sorted(source_dist.items(), key=lambda x: x[1], reverse=True):
                    source_name = source.split("//")[-1].split("/")[0] if "//" in source else source[:30]
                    dedup_report += f"- {source_name}: {count}篇\n"
            
            # Format output
            unique_articles_json = json.dumps(unique_articles, ensure_ascii=False, indent=2)
            
            return unique_articles_json, dedup_report
            
        except Exception as e:
            return f"Error: {str(e)}", ""
    
    def _hash_deduplication(self, articles):
        """Simple hash-based deduplication"""
        seen_hashes = set()
        unique_articles = []
        
        for article in articles:
            content = f"{article.get('title', '')}{article.get('text', '')}"
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_articles.append(article)
        
        return unique_articles
    
    def _title_deduplication(self, articles, threshold, keep_strategy):
        """Title similarity-based deduplication"""
        unique_articles = []
        
        for article in articles:
            title = article.get("title", "")
            is_duplicate = False
            
            for existing in unique_articles:
                existing_title = existing.get("title", "")
                similarity = self._calculate_similarity(title, existing_title)
                
                if similarity >= threshold:
                    is_duplicate = True
                    # Replace if current article is better
                    if self._is_better_article(article, existing, keep_strategy):
                        unique_articles.remove(existing)
                        unique_articles.append(article)
                    break
            
            if not is_duplicate:
                unique_articles.append(article)
        
        return unique_articles
    
    def _content_deduplication(self, articles, threshold, keep_strategy):
        """Content similarity-based deduplication"""
        unique_articles = []
        
        for article in articles:
            content = article.get("text", "") or article.get("description", "")
            is_duplicate = False
            
            for existing in unique_articles:
                existing_content = existing.get("text", "") or existing.get("description", "")
                similarity = self._calculate_similarity(content, existing_content)
                
                if similarity >= threshold:
                    is_duplicate = True
                    if self._is_better_article(article, existing, keep_strategy):
                        unique_articles.remove(existing)
                        unique_articles.append(article)
                    break
            
            if not is_duplicate:
                unique_articles.append(article)
        
        return unique_articles
    
    def _smart_deduplication(self, articles, threshold, keep_strategy, merge_similar):
        """Smart deduplication using multiple factors"""
        unique_articles = []
        
        for article in articles:
            title = article.get("title", "")
            content = article.get("text", "") or article.get("description", "")
            is_duplicate = False
            
            for existing in unique_articles:
                existing_title = existing.get("title", "")
                existing_content = existing.get("text", "") or existing.get("description", "")
                
                # Calculate multiple similarities
                title_sim = self._calculate_similarity(title, existing_title)
                content_sim = self._calculate_similarity(content, existing_content)
                
                # Smart similarity scoring
                if title_sim >= 0.9:  # Very similar titles
                    final_sim = max(title_sim, content_sim)
                elif title_sim >= 0.7 and content_sim >= 0.5:  # Similar title + some content overlap
                    final_sim = (title_sim * 0.7 + content_sim * 0.3)
                else:
                    final_sim = content_sim
                
                if final_sim >= threshold:
                    is_duplicate = True
                    
                    if merge_similar:
                        # Merge information from both articles
                        merged = self._merge_articles(existing, article)
                        unique_articles.remove(existing)
                        unique_articles.append(merged)
                    elif self._is_better_article(article, existing, keep_strategy):
                        unique_articles.remove(existing)
                        unique_articles.append(article)
                    break
            
            if not is_duplicate:
                unique_articles.append(article)
        
        return unique_articles
    
    def _calculate_similarity(self, text1, text2):
        """Calculate text similarity using SequenceMatcher"""
        if not text1 or not text2:
            return 0.0
        
        # Clean text
        clean_text1 = re.sub(r'\s+', ' ', text1.lower().strip())
        clean_text2 = re.sub(r'\s+', ' ', text2.lower().strip())
        
        return SequenceMatcher(None, clean_text1, clean_text2).ratio()
    
    def _is_better_article(self, article1, article2, strategy):
        """Determine which article is better based on strategy"""
        if strategy == "newest":
            time1 = article1.get("extraction_time", "")
            time2 = article2.get("extraction_time", "")
            return time1 > time2
        
        elif strategy == "longest":
            len1 = len(article1.get("text", "") or article1.get("description", ""))
            len2 = len(article2.get("text", "") or article2.get("description", ""))
            return len1 > len2
        
        elif strategy == "first":
            return False  # Keep existing
        
        else:  # most_complete
            score1 = self._calculate_completeness_score(article1)
            score2 = self._calculate_completeness_score(article2)
            return score1 > score2
    
    def _calculate_completeness_score(self, article):
        """Calculate article completeness score"""
        score = 0
        
        if article.get("title"):
            score += 1
        if article.get("text") and len(article["text"]) > 200:
            score += 2
        if article.get("summary"):
            score += 1
        if article.get("authors"):
            score += 1
        if article.get("keywords"):
            score += 1
        if article.get("publish_date"):
            score += 1
        
        return score
    
    def _merge_articles(self, article1, article2):
        """Merge two similar articles"""
        merged = article1.copy()
        
        # Keep the longer text
        text1_len = len(article1.get("text", ""))
        text2_len = len(article2.get("text", ""))
        if text2_len > text1_len:
            merged["text"] = article2["text"]
        
        # Combine authors
        authors1 = set(article1.get("authors", []))
        authors2 = set(article2.get("authors", []))
        if authors1 or authors2:
            merged["authors"] = list(authors1.union(authors2))
        
        # Combine keywords
        keywords1 = set(article1.get("keywords", []))
        keywords2 = set(article2.get("keywords", []))
        if keywords1 or keywords2:
            merged["keywords"] = list(keywords1.union(keywords2))[:20]  # Limit to 20
        
        # Add merge info
        merged["merged_from"] = [
            article1.get("url", "unknown"),
            article2.get("url", "unknown")
        ]
        merged["merge_time"] = datetime.now().isoformat()
        
        return merged

NODE_CLASS_MAPPINGS = {
    "ContentDeduplicator": ContentDeduplicator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ContentDeduplicator": "内容去重器"
}