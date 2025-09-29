import json
import re
from datetime import datetime

class SelectedTopicViewer:
    """
    Selected Topic Viewer Node
    
    从话题聚类结果中选择特定话题，提取该话题下所有文章的详细内容供写作使用。
    """
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("topic_articles_json", "writing_content", "topic_summary", "article_links")
    FUNCTION = "execute"
    CATEGORY = "RSS Content Processing"

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clustered_topics_json": ("STRING", {
                    "forceInput": True,
                    "multiline": True
                }),
                "topic_selection_method": (["topic_id", "topic_label", "keywords_match"], {
                    "default": "topic_label"
                }),
                "topic_identifier": ("STRING", {
                    "default": "",
                    "placeholder": "话题ID、标签关键词或话题标签"
                }),
                "content_format": (["detailed", "writing_ready", "research_notes", "simple"], {
                    "default": "writing_ready"
                }),
                "include_metadata": ("BOOLEAN", {"default": True}),
                "sort_articles_by": (["relevance", "time", "length", "source"], {
                    "default": "relevance"
                }),
            },
            "optional": {
                "max_content_length": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 100,
                    "tooltip": "0表示不限制长度"
                }),
                "include_article_links": ("BOOLEAN", {"default": True}),
                "writing_style": (["formal", "casual", "news", "analysis"], {
                    "default": "news"
                }),
                "language_preference": (["auto", "zh", "en"], {
                    "default": "auto"
                }),
            }
        }

    def execute(self, clustered_topics_json, topic_selection_method, topic_identifier,
                content_format, include_metadata, sort_articles_by, max_content_length=0,
                include_article_links=True, writing_style="news", language_preference="auto"):
        return self.extract_selected_topic(
            clustered_topics_json, topic_selection_method, topic_identifier,
            content_format, include_metadata, sort_articles_by, max_content_length,
            include_article_links, writing_style, language_preference
        )
    
    def extract_selected_topic(self, clustered_topics_json, topic_selection_method, topic_identifier,
                              content_format, include_metadata, sort_articles_by, max_content_length,
                              include_article_links, writing_style, language_preference):
        try:
            # Parse clustered topics
            try:
                clustered_data = json.loads(clustered_topics_json)
            except json.JSONDecodeError:
                return "Error: Invalid JSON format for clustered topics", "", "", ""
            
            if not clustered_data:
                return "Error: No clustered topics to process", "", "", ""
            
            if not topic_identifier.strip():
                return "Error: Please specify topic identifier", "", "", ""
            
            print(f"正在查找话题: {topic_identifier}")
            
            # Find matching topic
            selected_topic = self._find_matching_topic(clustered_data, topic_selection_method, topic_identifier)
            
            if not selected_topic:
                available_topics = self._list_available_topics(clustered_data)
                return f"Error: Topic not found. Available topics:\n{available_topics}", "", "", ""
            
            topic_data = selected_topic["data"]
            topic_key = selected_topic["key"]
            
            print(f"找到话题: {topic_data.get('topic_label', 'Unknown')} ({topic_data.get('articles_count', 0)}篇文章)")
            
            # Extract and format articles
            articles = topic_data.get("articles", [])
            
            # Sort articles
            sorted_articles = self._sort_articles(articles, sort_articles_by)
            
            # Generate different content formats
            topic_articles_json = json.dumps({
                "topic_info": {
                    "topic_id": topic_data.get("topic_id"),
                    "topic_label": topic_data.get("topic_label"),
                    "keywords": topic_data.get("keywords", []),
                    "articles_count": len(sorted_articles)
                },
                "articles": sorted_articles
            }, ensure_ascii=False, indent=2)
            
            # Generate writing-ready content
            writing_content = self._format_writing_content(
                topic_data, sorted_articles, content_format, max_content_length, 
                include_metadata, writing_style, language_preference
            )
            
            # Generate topic summary
            topic_summary = self._generate_detailed_summary(topic_data, sorted_articles)
            
            # Generate article links
            article_links = self._extract_article_links(sorted_articles, include_article_links)
            
            return topic_articles_json, writing_content, topic_summary, article_links
            
        except Exception as e:
            return f"Error: {str(e)}", "", "", ""
    
    def _find_matching_topic(self, clustered_data, method, identifier):
        """Find topic based on selection method"""
        identifier = identifier.strip().lower()
        
        for topic_key, topic_data in clustered_data.items():
            if method == "topic_id":
                topic_id = str(topic_data.get("topic_id", ""))
                if topic_id.lower() == identifier:
                    return {"key": topic_key, "data": topic_data}
            
            elif method == "topic_label":
                topic_label = topic_data.get("topic_label", "").lower()
                # 支持部分匹配
                if identifier in topic_label or topic_label in identifier:
                    return {"key": topic_key, "data": topic_data}
                
                # 去除emoji和括号后匹配
                clean_label = re.sub(r'[^\w\s\u4e00-\u9fff]', '', topic_label)
                if identifier in clean_label:
                    return {"key": topic_key, "data": topic_data}
            
            elif method == "keywords_match":
                keywords = [kw.lower() for kw in topic_data.get("keywords", [])]
                if any(identifier in kw or kw in identifier for kw in keywords):
                    return {"key": topic_key, "data": topic_data}
        
        return None
    
    def _list_available_topics(self, clustered_data):
        """List all available topics"""
        topics_list = []
        for i, (topic_key, topic_data) in enumerate(clustered_data.items(), 1):
            topic_id = topic_data.get("topic_id", "unknown")
            topic_label = topic_data.get("topic_label", "Unknown")
            articles_count = topic_data.get("articles_count", 0)
            topics_list.append(f"{i}. ID:{topic_id} | {topic_label} ({articles_count}篇)")
        
        return "\n".join(topics_list)
    
    def _sort_articles(self, articles, sort_by):
        """Sort articles based on criteria"""
        if sort_by == "time":
            return sorted(articles, key=lambda x: x.get("extraction_time", ""), reverse=True)
        elif sort_by == "length":
            return sorted(articles, key=lambda x: len(x.get("text", "")), reverse=True)
        elif sort_by == "source":
            return sorted(articles, key=lambda x: x.get("source_url", ""))
        else:  # relevance - keep original order from clustering
            return articles
    
    def _format_writing_content(self, topic_data, articles, format_type, max_length, 
                               include_metadata, style, language):
        """Format content for writing purposes"""
        topic_label = topic_data.get("topic_label", "Unknown Topic")
        keywords = topic_data.get("keywords", [])
        
        if format_type == "writing_ready":
            content = self._format_writing_ready(topic_label, keywords, articles, style, max_length)
        elif format_type == "research_notes":
            content = self._format_research_notes(topic_label, keywords, articles, max_length)
        elif format_type == "detailed":
            content = self._format_detailed_content(topic_label, keywords, articles, max_length)
        else:  # simple
            content = self._format_simple_content(topic_label, keywords, articles, max_length)
        
        if include_metadata:
            metadata = f"""话题: {topic_label}
关键词: {', '.join(keywords)}
文章数量: {len(articles)}
生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}

---

"""
            content = metadata + content
        
        return content
    
    def _format_writing_ready(self, topic_label, keywords, articles, style, max_length):
        """Format content ready for writing"""
        content = f"# 写作素材：{topic_label}\n\n"
        
        # 话题背景
        content += f"## 📋 话题背景\n"
        content += f"**核心关键词**: {', '.join(keywords[:5])}\n"
        content += f"**相关文章数**: {len(articles)}篇\n\n"
        
        # 主要观点和角度
        content += f"## 💡 主要观点和角度\n\n"
        
        for i, article in enumerate(articles, 1):
            title = article.get("title", "无标题")
            text = article.get("text", article.get("description", ""))
            source = self._extract_domain(article.get("source_url", ""))
            link = article.get("link", "")
            
            # 限制长度
            if max_length > 0 and len(text) > max_length:
                text = text[:max_length] + "..."
            
            content += f"### {i}. {title}\n"
            content += f"**来源**: {source}\n"
            if link:
                content += f"**原文**: {link}\n"
            content += f"\n**内容要点**:\n{text}\n\n"
            content += "---\n\n"
        
        # 写作建议
        content += f"## ✍️ 写作建议\n"
        content += self._generate_writing_suggestions(topic_label, keywords, articles, style)
        
        return content
    
    def _format_research_notes(self, topic_label, keywords, articles, max_length):
        """Format as research notes"""
        content = f"# 研究笔记：{topic_label}\n\n"
        
        # 信息源分析
        sources = {}
        for article in articles:
            source = self._extract_domain(article.get("source_url", ""))
            sources[source] = sources.get(source, 0) + 1
        
        content += f"## 📊 信息源分析\n"
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            content += f"- {source}: {count}篇\n"
        content += "\n"
        
        # 关键信息提取
        content += f"## 🔍 关键信息提取\n\n"
        
        for i, article in enumerate(articles, 1):
            title = article.get("title", "无标题")
            text = article.get("text", article.get("description", ""))
            
            if max_length > 0 and len(text) > max_length:
                text = text[:max_length] + "..."
            
            content += f"**#{i} {title}**\n"
            content += f"核心内容: {text}\n\n"
        
        return content
    
    def _format_detailed_content(self, topic_label, keywords, articles, max_length):
        """Format detailed content with all information"""
        content = f"# 详细内容：{topic_label}\n\n"
        
        for i, article in enumerate(articles, 1):
            content += f"## 文章 {i}\n\n"
            
            # 基本信息
            title = article.get("title", "无标题")
            link = article.get("link", "")
            source = self._extract_domain(article.get("source_url", ""))
            author = article.get("author", article.get("authors", ""))
            pub_date = article.get("publish_date", article.get("published", ""))
            
            content += f"**标题**: {title}\n"
            content += f"**来源**: {source}\n"
            if link:
                content += f"**链接**: {link}\n"
            if author:
                content += f"**作者**: {author}\n"
            if pub_date:
                content += f"**发布时间**: {pub_date}\n"
            
            # 内容
            text = article.get("text", article.get("description", ""))
            summary = article.get("summary", "")
            keywords_art = article.get("keywords", [])
            
            if summary:
                content += f"\n**摘要**: {summary}\n"
            
            if keywords_art:
                content += f"**关键词**: {', '.join(keywords_art[:10])}\n"
            
            if text:
                if max_length > 0 and len(text) > max_length:
                    text = text[:max_length] + "..."
                content += f"\n**正文内容**:\n{text}\n"
            
            content += "\n" + "="*50 + "\n\n"
        
        return content
    
    def _format_simple_content(self, topic_label, keywords, articles, max_length):
        """Format simple content list"""
        content = f"{topic_label}\n\n"
        
        for i, article in enumerate(articles, 1):
            title = article.get("title", "无标题")
            text = article.get("text", article.get("description", ""))[:200]
            
            content += f"{i}. {title}\n"
            content += f"   {text}...\n\n"
        
        return content
    
    def _generate_detailed_summary(self, topic_data, articles):
        """Generate detailed topic summary"""
        topic_label = topic_data.get("topic_label", "Unknown")
        keywords = topic_data.get("keywords", [])
        
        summary = f"话题: {topic_label}\n"
        summary += f"关键词: {', '.join(keywords)}\n"
        summary += f"文章总数: {len(articles)}\n\n"
        
        # 时间分布
        time_dist = {}
        for article in articles:
            extract_time = article.get("extraction_time", "")
            if extract_time:
                date = extract_time.split("T")[0] if "T" in extract_time else extract_time[:10]
                time_dist[date] = time_dist.get(date, 0) + 1
        
        if time_dist:
            summary += "时间分布:\n"
            for date, count in sorted(time_dist.items(), reverse=True):
                summary += f"- {date}: {count}篇\n"
            summary += "\n"
        
        # 来源分布
        source_dist = {}
        for article in articles:
            source = self._extract_domain(article.get("source_url", ""))
            source_dist[source] = source_dist.get(source, 0) + 1
        
        summary += "来源分布:\n"
        for source, count in sorted(source_dist.items(), key=lambda x: x[1], reverse=True):
            summary += f"- {source}: {count}篇\n"
        
        # 内容统计
        total_chars = sum(len(article.get("text", "")) for article in articles)
        avg_length = total_chars / len(articles) if articles else 0
        
        summary += f"\n内容统计:\n"
        summary += f"- 总字符数: {total_chars:,}\n"
        summary += f"- 平均长度: {avg_length:.0f}字符/篇\n"
        
        return summary
    
    def _extract_article_links(self, articles, include_links):
        """Extract article links"""
        if not include_links:
            return ""
        
        links = []
        for i, article in enumerate(articles, 1):
            title = article.get("title", "无标题")
            link = article.get("link", "")
            source = self._extract_domain(article.get("source_url", ""))
            
            if link:
                links.append(f"{i}. {title}")
                links.append(f"   链接: {link}")
                links.append(f"   来源: {source}")
                links.append("")
        
        return "\n".join(links)
    
    def _generate_writing_suggestions(self, topic_label, keywords, articles, style):
        """Generate writing suggestions based on content"""
        suggestions = ""
        
        # 分析关键词模式
        keywords_text = " ".join(keywords).lower()
        
        if style == "news":
            suggestions += "📰 新闻写作建议:\n"
            suggestions += "- 使用倒金字塔结构，重要信息在前\n"
            suggestions += "- 引用多个信息源增加可信度\n"
            suggestions += "- 突出最新发展和影响\n\n"
        elif style == "analysis":
            suggestions += "📊 分析写作建议:\n"
            suggestions += "- 深入分析事件背景和原因\n"
            suggestions += "- 比较不同来源的观点\n"
            suggestions += "- 预测未来发展趋势\n\n"
        
        # 基于关键词的角度建议
        if any(word in keywords_text for word in ['科技', 'tech', 'ai', '技术']):
            suggestions += "🔬 科技角度:\n- 技术创新点分析\n- 行业影响评估\n- 未来发展预测\n\n"
        elif any(word in keywords_text for word in ['市场', 'market', '股票', '金融']):
            suggestions += "💰 市场角度:\n- 市场反应分析\n- 投资影响评估\n- 经济数据解读\n\n"
        elif any(word in keywords_text for word in ['政策', 'policy', '政府', '法律']):
            suggestions += "🏛️ 政策角度:\n- 政策背景分析\n- 实施影响预测\n- 相关法律解读\n\n"
        
        # 基于文章数量的建议
        article_count = len(articles)
        if article_count >= 5:
            suggestions += "📚 内容丰富，建议:\n- 选择3-4个最具代表性的案例\n- 构建完整的故事线\n- 避免信息过载\n\n"
        elif article_count <= 2:
            suggestions += "📝 内容较少，建议:\n- 深入挖掘现有内容\n- 添加背景信息补充\n- 寻找相关话题扩展\n\n"
        
        return suggestions
    
    def _extract_domain(self, url):
        """Extract domain from URL"""
        if not url:
            return "未知来源"
        
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            return domain.replace("www.", "") if domain else "未知来源"
        except:
            return url.split("//")[-1].split("/")[0] if "//" in url else "未知来源"

NODE_CLASS_MAPPINGS = {
    "SelectedTopicViewer": SelectedTopicViewer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelectedTopicViewer": "选定话题查看器"
}