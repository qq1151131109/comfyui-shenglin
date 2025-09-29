import json
import os
from datetime import datetime
from collections import Counter

class TopicReportGenerator:
    """
    Topic Report Generator Node
    
    Generates comprehensive topic reports with multiple output formats for news topic analysis.
    """
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("formatted_report", "topic_overview", "selection_guide")
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
                "report_format": (["markdown", "json", "html", "text"], {
                    "default": "markdown"
                }),
                "include_summary": ("BOOLEAN", {"default": True}),
                "include_keywords": ("BOOLEAN", {"default": True}),
                "include_articles": ("BOOLEAN", {"default": True}),
                "sort_by": (["热度", "时间", "文章数量", "关键词"], {
                    "default": "文章数量"
                }),
                "max_articles_per_topic": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 1
                }),
            },
            "optional": {
                "custom_template": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "自定义报告模板，留空使用默认模板"
                }),
                "generate_selection_tips": ("BOOLEAN", {"default": True}),
                "include_trending_analysis": ("BOOLEAN", {"default": True}),
            }
        }

    def execute(self, clustered_topics_json, report_format, include_summary, include_keywords,
                include_articles, sort_by, max_articles_per_topic, custom_template="", 
                generate_selection_tips=True, include_trending_analysis=True):
        return self.generate_topic_report(
            clustered_topics_json, report_format, include_summary, include_keywords,
            include_articles, sort_by, max_articles_per_topic, custom_template,
            generate_selection_tips, include_trending_analysis
        )
    
    def generate_topic_report(self, clustered_topics_json, report_format, include_summary,
                             include_keywords, include_articles, sort_by, max_articles_per_topic,
                             custom_template, generate_selection_tips, include_trending_analysis):
        try:
            # Parse clustered topics
            try:
                clustered_data = json.loads(clustered_topics_json)
            except json.JSONDecodeError:
                return "Error: Invalid JSON format for clustered topics", "", ""
            
            if not clustered_data:
                return "Error: No clustered topics to process", "", ""
            
            print(f"开始生成话题报告，包含 {len(clustered_data)} 个话题...")
            
            # Sort topics
            topics_list = list(clustered_data.values())
            topics_list = self._sort_topics(topics_list, sort_by)
            
            # Generate different format reports
            if report_format == "markdown":
                formatted_report = self._generate_markdown_report(
                    topics_list, include_summary, include_keywords, include_articles, max_articles_per_topic
                )
            elif report_format == "json":
                formatted_report = json.dumps(topics_list, ensure_ascii=False, indent=2)
            elif report_format == "html":
                formatted_report = self._generate_html_report(
                    topics_list, include_summary, include_keywords, include_articles, max_articles_per_topic
                )
            else:  # text
                formatted_report = self._generate_text_report(
                    topics_list, include_summary, include_keywords, include_articles, max_articles_per_topic
                )
            
            # Generate topic overview
            topic_overview = self._generate_topic_overview(topics_list, include_trending_analysis)
            
            # Generate selection guide
            selection_guide = ""
            if generate_selection_tips:
                selection_guide = self._generate_selection_guide(topics_list)
            
            return formatted_report, topic_overview, selection_guide
            
        except Exception as e:
            return f"Error: {str(e)}", "", ""
    
    def _sort_topics(self, topics, sort_by):
        """Sort topics based on criteria"""
        if sort_by == "文章数量":
            return sorted(topics, key=lambda x: x["articles_count"], reverse=True)
        elif sort_by == "热度":
            # Calculate hotness score based on articles count and recency
            for topic in topics:
                hotness = topic["articles_count"] * 10  # Base score
                # Add recency bonus (simplified)
                hotness += len([a for a in topic["articles"] if "今天" in str(a.get("published", ""))])
                topic["hotness_score"] = hotness
            return sorted(topics, key=lambda x: x.get("hotness_score", 0), reverse=True)
        elif sort_by == "关键词":
            return sorted(topics, key=lambda x: len(x.get("keywords", [])), reverse=True)
        else:  # 时间
            return sorted(topics, key=lambda x: x.get("topic_id", 0))
    
    def _generate_markdown_report(self, topics, include_summary, include_keywords, include_articles, max_articles):
        """Generate Markdown format report"""
        report = f"# 📰 新闻话题分析报告\n\n"
        report += f"**生成时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n"
        report += f"**话题总数**: {len(topics)}\n"
        report += f"**文章总数**: {sum(topic['articles_count'] for topic in topics)}\n\n"
        
        report += "---\n\n"
        
        for i, topic in enumerate(topics, 1):
            report += f"## {i}. {topic['topic_label']}\n\n"
            report += f"**文章数量**: {topic['articles_count']}篇\n\n"
            
            if include_keywords and topic.get("keywords"):
                report += f"**关键词**: {', '.join(topic['keywords'][:8])}\n\n"
            
            if include_summary:
                report += f"**话题摘要**: {self._generate_topic_summary(topic)}\n\n"
            
            if include_articles:
                report += "**相关文章**:\n\n"
                articles_to_show = topic["articles"][:max_articles]
                
                for j, article in enumerate(articles_to_show, 1):
                    title = article.get("title", "无标题")
                    link = article.get("link", "")
                    source = self._extract_domain(article.get("source_url", ""))
                    
                    report += f"{j}. **{title}**\n"
                    if link:
                        report += f"   - 链接: {link}\n"
                    if source:
                        report += f"   - 来源: {source}\n"
                    
                    # Add brief description if available
                    description = article.get("description", "")
                    if description and len(description) > 20:
                        short_desc = description[:150] + "..." if len(description) > 150 else description
                        report += f"   - 摘要: {short_desc}\n"
                    
                    report += "\n"
                
                if len(topic["articles"]) > max_articles:
                    report += f"   ... 还有 {len(topic['articles']) - max_articles} 篇相关文章\n\n"
            
            report += "---\n\n"
        
        return report
    
    def _generate_html_report(self, topics, include_summary, include_keywords, include_articles, max_articles):
        """Generate HTML format report"""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>新闻话题分析报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .topic {{ margin-bottom: 30px; padding: 15px; border-left: 4px solid #007acc; }}
        .topic-title {{ color: #007acc; margin-bottom: 10px; }}
        .keywords {{ background: #f0f8ff; padding: 8px; border-radius: 4px; }}
        .article {{ margin: 10px 0; padding: 8px; background: #f9f9f9; }}
        .stats {{ background: #e8f4f8; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>📰 新闻话题分析报告</h1>
    <div class="stats">
        <strong>生成时间</strong>: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}<br>
        <strong>话题总数</strong>: {len(topics)}<br>
        <strong>文章总数</strong>: {sum(topic['articles_count'] for topic in topics)}
    </div>
"""
        
        for i, topic in enumerate(topics, 1):
            html += f'<div class="topic">\n'
            html += f'<h2 class="topic-title">{i}. {topic["topic_label"]}</h2>\n'
            html += f'<p><strong>文章数量</strong>: {topic["articles_count"]}篇</p>\n'
            
            if include_keywords and topic.get("keywords"):
                keywords_html = ", ".join(topic["keywords"][:8])
                html += f'<div class="keywords"><strong>关键词</strong>: {keywords_html}</div>\n'
            
            if include_summary:
                summary = self._generate_topic_summary(topic)
                html += f'<p><strong>话题摘要</strong>: {summary}</p>\n'
            
            if include_articles:
                html += '<h3>相关文章:</h3>\n'
                articles_to_show = topic["articles"][:max_articles]
                
                for article in articles_to_show:
                    title = article.get("title", "无标题")
                    link = article.get("link", "")
                    source = self._extract_domain(article.get("source_url", ""))
                    
                    html += f'<div class="article">\n'
                    html += f'<strong>{title}</strong><br>\n'
                    if link:
                        html += f'<a href="{link}" target="_blank">查看原文</a> | '
                    if source:
                        html += f'来源: {source}'
                    html += f'</div>\n'
            
            html += '</div>\n'
        
        html += '</body></html>'
        return html
    
    def _generate_text_report(self, topics, include_summary, include_keywords, include_articles, max_articles):
        """Generate plain text format report"""
        report = f"新闻话题分析报告\n"
        report += f"=" * 50 + "\n"
        report += f"生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n"
        report += f"话题总数: {len(topics)}\n"
        report += f"文章总数: {sum(topic['articles_count'] for topic in topics)}\n\n"
        
        for i, topic in enumerate(topics, 1):
            report += f"{i}. {topic['topic_label']}\n"
            report += f"   文章数量: {topic['articles_count']}篇\n"
            
            if include_keywords and topic.get("keywords"):
                report += f"   关键词: {', '.join(topic['keywords'][:6])}\n"
            
            if include_summary:
                summary = self._generate_topic_summary(topic)
                report += f"   摘要: {summary}\n"
            
            if include_articles:
                report += "   相关文章:\n"
                articles_to_show = topic["articles"][:max_articles]
                
                for j, article in enumerate(articles_to_show, 1):
                    title = article.get("title", "无标题")
                    source = self._extract_domain(article.get("source_url", ""))
                    report += f"   {j}) {title}"
                    if source:
                        report += f" [{source}]"
                    report += "\n"
            
            report += "\n" + "-" * 40 + "\n\n"
        
        return report
    
    def _generate_topic_overview(self, topics, include_trending):
        """Generate topic overview"""
        total_articles = sum(topic['articles_count'] for topic in topics)
        
        overview = f"话题概览 ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n\n"
        
        # Top topics by article count
        overview += "📊 热门话题排行:\n"
        for i, topic in enumerate(topics[:5], 1):
            percentage = topic['articles_count'] / total_articles * 100
            overview += f"{i}. {topic['topic_label']}: {topic['articles_count']}篇 ({percentage:.1f}%)\n"
        
        overview += "\n"
        
        # Keywords analysis
        all_keywords = []
        for topic in topics:
            all_keywords.extend(topic.get("keywords", [])[:3])  # Top 3 keywords per topic
        
        keyword_freq = Counter(all_keywords)
        if keyword_freq:
            overview += "🔥 热门关键词:\n"
            for keyword, freq in keyword_freq.most_common(10):
                overview += f"- {keyword}: {freq}次\n"
            overview += "\n"
        
        # Trending analysis
        if include_trending:
            overview += "📈 趋势分析:\n"
            
            # Identify high-activity topics (more than average articles)
            avg_articles = total_articles / len(topics)
            hot_topics = [t for t in topics if t['articles_count'] > avg_articles * 1.5]
            
            if hot_topics:
                overview += f"高热度话题 ({len(hot_topics)}个):\n"
                for topic in hot_topics[:3]:
                    overview += f"- {topic['topic_label']}: {topic['articles_count']}篇文章\n"
            else:
                overview += "- 暂无特别突出的热点话题\n"
            
            overview += "\n"
        
        # Source diversity
        source_count = set()
        for topic in topics:
            for article in topic.get("articles", []):
                source_url = article.get("source_url", "")
                if source_url:
                    source_count.add(self._extract_domain(source_url))
        
        overview += f"📡 信息源多样性: {len(source_count)}个不同来源\n"
        
        return overview
    
    def _generate_selection_guide(self, topics):
        """Generate content selection guide for creators"""
        guide = f"📝 选题建议指南\n\n"
        
        # Categorize topics by potential
        high_potential = []
        medium_potential = []
        low_potential = []
        
        for topic in topics:
            score = self._calculate_selection_score(topic)
            topic["selection_score"] = score
            
            if score >= 70:
                high_potential.append(topic)
            elif score >= 40:
                medium_potential.append(topic)
            else:
                low_potential.append(topic)
        
        # High potential topics
        if high_potential:
            guide += "🌟 推荐选题 (高潜力):\n"
            for i, topic in enumerate(high_potential[:3], 1):
                guide += f"{i}. {topic['topic_label']} (评分: {topic['selection_score']})\n"
                guide += f"   推荐原因: {self._get_selection_reason(topic)}\n"
                guide += f"   建议角度: {self._suggest_writing_angles(topic)}\n\n"
        
        # Medium potential topics
        if medium_potential:
            guide += "✨ 备选选题 (中等潜力):\n"
            for i, topic in enumerate(medium_potential[:3], 1):
                guide += f"{i}. {topic['topic_label']} (评分: {topic['selection_score']})\n"
                guide += f"   特点: {self._get_topic_characteristics(topic)}\n\n"
        
        # Writing suggestions
        guide += "💡 写作建议:\n"
        guide += "- 选择文章数量适中(3-8篇)的话题，信息充分但不冗余\n"
        guide += "- 优先选择有明确关键词的话题，便于深入挖掘\n"
        guide += "- 关注不同来源的报道角度，寻找独特视角\n"
        guide += "- 考虑话题的时效性和读者兴趣度\n\n"
        
        # Topic combination suggestions
        if len(topics) > 1:
            guide += "🔗 话题组合建议:\n"
            combinations = self._suggest_topic_combinations(topics)
            for combo in combinations[:3]:
                guide += f"- {combo}\n"
        
        return guide
    
    def _calculate_selection_score(self, topic):
        """Calculate topic selection potential score"""
        score = 0
        
        # Article count factor (optimal range: 3-8 articles)
        article_count = topic["articles_count"]
        if 3 <= article_count <= 8:
            score += 30
        elif 2 <= article_count <= 10:
            score += 20
        elif article_count > 10:
            score += 10
        
        # Keywords quality
        keywords = topic.get("keywords", [])
        if len(keywords) >= 5:
            score += 20
        elif len(keywords) >= 3:
            score += 15
        
        # Topic label quality (longer, more descriptive labels score higher)
        label = topic.get("topic_label", "")
        if len(label) > 15 and any(char in label for char in "📈💻🏛️⚽🎬🌍₿"):
            score += 15
        elif len(label) > 10:
            score += 10
        
        # Content diversity (different sources)
        sources = set()
        for article in topic.get("articles", []):
            source_url = article.get("source_url", "")
            if source_url:
                sources.add(self._extract_domain(source_url))
        
        if len(sources) > 1:
            score += 15
        
        return min(score, 100)  # Cap at 100
    
    def _get_selection_reason(self, topic):
        """Get reason why topic is recommended"""
        reasons = []
        
        if 3 <= topic["articles_count"] <= 8:
            reasons.append("信息量适中")
        
        if len(topic.get("keywords", [])) >= 5:
            reasons.append("关键词丰富")
        
        # Check source diversity
        sources = set()
        for article in topic.get("articles", []):
            source_url = article.get("source_url", "")
            if source_url:
                sources.add(self._extract_domain(source_url))
        
        if len(sources) > 1:
            reasons.append("多源报道")
        
        return ", ".join(reasons) if reasons else "内容质量良好"
    
    def _suggest_writing_angles(self, topic):
        """Suggest writing angles for a topic"""
        keywords = topic.get("keywords", [])
        
        # Analyze keywords to suggest angles
        if any(word in str(keywords).lower() for word in ['市场', '价格', '股票', '投资']):
            return "市场影响分析、投资机会解读"
        elif any(word in str(keywords).lower() for word in ['技术', '科技', 'ai', '创新']):
            return "技术趋势分析、行业影响评估"
        elif any(word in str(keywords).lower() for word in ['政策', '法律', '监管']):
            return "政策解读、法律影响分析"
        else:
            return "事件背景分析、未来发展预测"
    
    def _get_topic_characteristics(self, topic):
        """Get topic characteristics"""
        characteristics = []
        
        if topic["articles_count"] > 10:
            characteristics.append("高关注度")
        elif topic["articles_count"] < 3:
            characteristics.append("小众话题")
        
        keywords_count = len(topic.get("keywords", []))
        if keywords_count > 8:
            characteristics.append("关键词丰富")
        elif keywords_count < 3:
            characteristics.append("信息集中")
        
        return ", ".join(characteristics) if characteristics else "一般话题"
    
    def _suggest_topic_combinations(self, topics):
        """Suggest topic combinations"""
        combinations = []
        
        # Find related topics based on keyword overlap
        for i, topic1 in enumerate(topics):
            for topic2 in topics[i+1:]:
                overlap = self._calculate_keyword_overlap(topic1, topic2)
                if 0.2 <= overlap <= 0.6:  # Some overlap but not too much
                    combo = f"{topic1['topic_label']} + {topic2['topic_label']} (关联度: {overlap:.1%})"
                    combinations.append(combo)
        
        return combinations[:5]  # Return top 5 combinations
    
    def _calculate_keyword_overlap(self, topic1, topic2):
        """Calculate keyword overlap between topics"""
        keywords1 = set(topic1.get("keywords", []))
        keywords2 = set(topic2.get("keywords", []))
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        return intersection / union if union > 0 else 0.0
    
    def _generate_topic_summary(self, topic):
        """Generate a brief summary for a topic"""
        articles_count = topic["articles_count"]
        keywords = topic.get("keywords", [])[:5]
        
        if not keywords:
            return f"包含{articles_count}篇相关文章的话题"
        
        keywords_text = "、".join(keywords)
        return f"围绕{keywords_text}等关键词的{articles_count}篇文章，涵盖该领域的最新动态"
    
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
    "TopicReportGenerator": TopicReportGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TopicReportGenerator": "话题报告生成器"
}