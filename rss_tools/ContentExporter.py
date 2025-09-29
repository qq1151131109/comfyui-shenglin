import json
import os
import csv
import html
from datetime import datetime
from pathlib import Path

class ContentExporter:
    """
    Content Exporter Node
    
    Exports processed content to various local formats including JSON, CSV, HTML, and Markdown files.
    """
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("export_path", "export_summary", "export_status")
    FUNCTION = "execute"
    CATEGORY = "RSS Content Processing"

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "content_data": ("STRING", {
                    "forceInput": True,
                    "multiline": True
                }),
                "export_format": (["json", "csv", "html", "markdown", "all"], {
                    "default": "json"
                }),
                "output_directory": ("STRING", {
                    "default": "./exports",
                    "placeholder": "导出文件夹路径"
                }),
                "filename_prefix": ("STRING", {
                    "default": "rss_export",
                    "placeholder": "文件名前缀"
                }),
                "include_timestamp": ("BOOLEAN", {"default": True}),
                "create_directory": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "compress_output": ("BOOLEAN", {"default": False}),
                "split_by_topic": ("BOOLEAN", {"default": False}),
                "include_metadata": ("BOOLEAN", {"default": True}),
                "custom_template": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "自定义导出模板"
                }),
                "max_articles_per_file": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "step": 10,
                    "tooltip": "0表示不限制"
                }),
            }
        }

    def execute(self, content_data, export_format, output_directory, filename_prefix,
                include_timestamp, create_directory, compress_output=False, split_by_topic=False,
                include_metadata=True, custom_template="", max_articles_per_file=0):
        return self.export_content(
            content_data, export_format, output_directory, filename_prefix,
            include_timestamp, create_directory, compress_output, split_by_topic,
            include_metadata, custom_template, max_articles_per_file
        )
    
    def export_content(self, content_data, export_format, output_directory, filename_prefix,
                      include_timestamp, create_directory, compress_output, split_by_topic,
                      include_metadata, custom_template, max_articles_per_file):
        try:
            # Parse content data
            try:
                if content_data.startswith('{'):
                    # Clustered topics format
                    data = json.loads(content_data)
                    data_type = "topics"
                elif content_data.startswith('['):
                    # Articles list format
                    data = json.loads(content_data)
                    data_type = "articles"
                else:
                    return "Error: Invalid content format", "", "failed"
            except json.JSONDecodeError:
                return "Error: Invalid JSON format", "", "failed"
            
            if not data:
                return "Error: No content to export", "", "failed"
            
            # Prepare output directory
            if create_directory:
                Path(output_directory).mkdir(parents=True, exist_ok=True)
            elif not os.path.exists(output_directory):
                return f"Error: Directory {output_directory} does not exist", "", "failed"
            
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if include_timestamp else ""
            
            exported_files = []
            export_stats = {
                "total_files": 0,
                "total_articles": 0,
                "export_formats": [],
                "file_sizes": {},
                "start_time": datetime.now()
            }
            
            print(f"开始导出内容到 {output_directory}...")
            
            # Export based on format
            if export_format == "all":
                formats = ["json", "csv", "html", "markdown"]
            else:
                formats = [export_format]
            
            for fmt in formats:
                if split_by_topic and data_type == "topics":
                    files = self._export_by_topic(
                        data, fmt, output_directory, filename_prefix, timestamp,
                        include_metadata, custom_template, max_articles_per_file
                    )
                else:
                    files = self._export_unified(
                        data, data_type, fmt, output_directory, filename_prefix, timestamp,
                        include_metadata, custom_template, max_articles_per_file
                    )
                
                exported_files.extend(files)
                export_stats["export_formats"].append(fmt)
            
            # Compress if requested
            if compress_output and exported_files:
                compressed_file = self._compress_files(exported_files, output_directory, filename_prefix, timestamp)
                if compressed_file:
                    exported_files = [compressed_file]
            
            # Calculate statistics
            export_stats["total_files"] = len(exported_files)
            export_stats["end_time"] = datetime.now()
            
            for file_path in exported_files:
                if os.path.exists(file_path):
                    size_mb = os.path.getsize(file_path) / 1024 / 1024
                    export_stats["file_sizes"][os.path.basename(file_path)] = f"{size_mb:.2f}MB"
                    
                    # Count articles
                    if data_type == "topics":
                        export_stats["total_articles"] = sum(
                            topic.get("articles_count", 0) for topic in data.values()
                        )
                    elif data_type == "articles":
                        export_stats["total_articles"] = len(data)
            
            # Generate summary
            processing_time = (export_stats["end_time"] - export_stats["start_time"]).total_seconds()
            
            export_summary = f"""内容导出完成报告:
导出格式: {', '.join(export_stats['export_formats'])}
导出文件数: {export_stats['total_files']}
处理文章数: {export_stats['total_articles']}
处理时间: {processing_time:.1f}秒
输出目录: {output_directory}

导出文件列表:"""
            
            for file_path in exported_files:
                filename = os.path.basename(file_path)
                file_size = export_stats["file_sizes"].get(filename, "未知大小")
                export_summary += f"\n✅ {filename} ({file_size})"
            
            # Return primary export path
            primary_export = exported_files[0] if exported_files else ""
            
            return primary_export, export_summary, "success"
            
        except Exception as e:
            return f"Error: {str(e)}", "", "failed"
    
    def _export_unified(self, data, data_type, format_type, output_dir, prefix, timestamp,
                       include_metadata, custom_template, max_articles_per_file):
        """Export all content to a single file"""
        exported_files = []
        
        # Generate filename
        ts_suffix = f"_{timestamp}" if timestamp else ""
        filename = f"{prefix}{ts_suffix}.{format_type}"
        file_path = os.path.join(output_dir, filename)
        
        try:
            if format_type == "json":
                content = self._format_json_export(data, data_type, include_metadata)
            elif format_type == "csv":
                content = self._format_csv_export(data, data_type, include_metadata)
            elif format_type == "html":
                content = self._format_html_export(data, data_type, include_metadata, custom_template)
            elif format_type == "markdown":
                content = self._format_markdown_export(data, data_type, include_metadata, custom_template)
            else:
                return []
            
            # Handle file splitting if needed
            if max_articles_per_file > 0 and data_type == "articles" and len(data) > max_articles_per_file:
                exported_files.extend(self._split_and_export(
                    data, format_type, output_dir, prefix, timestamp, 
                    max_articles_per_file, include_metadata, custom_template
                ))
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                exported_files.append(file_path)
            
            return exported_files
            
        except Exception as e:
            print(f"Export failed for {format_type}: {e}")
            return []
    
    def _export_by_topic(self, topics_data, format_type, output_dir, prefix, timestamp,
                        include_metadata, custom_template, max_articles_per_file):
        """Export content split by topics"""
        exported_files = []
        
        for topic_key, topic_data in topics_data.items():
            topic_label = topic_data.get("topic_label", topic_key)
            safe_topic_name = re.sub(r'[^\w\s-]', '', topic_label).strip()[:50]
            safe_topic_name = re.sub(r'\s+', '_', safe_topic_name)
            
            ts_suffix = f"_{timestamp}" if timestamp else ""
            filename = f"{prefix}_{safe_topic_name}{ts_suffix}.{format_type}"
            file_path = os.path.join(output_dir, filename)
            
            try:
                if format_type == "json":
                    content = json.dumps(topic_data, ensure_ascii=False, indent=2)
                elif format_type == "csv":
                    content = self._format_topic_csv(topic_data, include_metadata)
                elif format_type == "html":
                    content = self._format_topic_html(topic_data, include_metadata, custom_template)
                elif format_type == "markdown":
                    content = self._format_topic_markdown(topic_data, include_metadata, custom_template)
                else:
                    continue
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                exported_files.append(file_path)
                
            except Exception as e:
                print(f"Topic export failed for {topic_label}: {e}")
        
        return exported_files
    
    def _format_json_export(self, data, data_type, include_metadata):
        """Format data as JSON"""
        export_data = {
            "export_metadata": {
                "export_time": datetime.now().isoformat(),
                "data_type": data_type,
                "exporter": "ComfyUI RSS Tool v1.0"
            } if include_metadata else {},
            "content": data
        }
        
        return json.dumps(export_data, ensure_ascii=False, indent=2)
    
    def _format_csv_export(self, data, data_type, include_metadata):
        """Format data as CSV"""
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        
        if data_type == "topics":
            # CSV for topics
            headers = ["topic_id", "topic_label", "articles_count", "keywords", "article_titles", "article_links"]
            writer.writerow(headers)
            
            for topic_key, topic_data in data.items():
                articles = topic_data.get("articles", [])
                titles = " | ".join([a.get("title", "") for a in articles[:5]])
                links = " | ".join([a.get("link", "") for a in articles[:5]])
                keywords = ", ".join(topic_data.get("keywords", []))
                
                writer.writerow([
                    topic_data.get("topic_id", topic_key),
                    topic_data.get("topic_label", ""),
                    topic_data.get("articles_count", 0),
                    keywords,
                    titles,
                    links
                ])
        else:
            # CSV for articles
            headers = ["title", "link", "source_url", "published", "author", "description", "text_length"]
            writer.writerow(headers)
            
            for article in data:
                text_len = len(article.get("text", ""))
                writer.writerow([
                    article.get("title", ""),
                    article.get("link", ""),
                    article.get("source_url", ""),
                    article.get("published", ""),
                    article.get("author", ""),
                    article.get("description", "")[:200],
                    text_len
                ])
        
        return output.getvalue()
    
    def _format_html_export(self, data, data_type, include_metadata, custom_template):
        """Format data as HTML"""
        if custom_template:
            return self._apply_custom_template(data, custom_template, "html")
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>RSS内容导出</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background: #f0f8ff; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .topic {{ margin-bottom: 30px; padding: 20px; border-left: 4px solid #007acc; background: #fafafa; }}
        .topic-title {{ color: #007acc; margin-bottom: 15px; font-size: 1.5em; }}
        .keywords {{ background: #e8f4fd; padding: 10px; border-radius: 4px; margin: 10px 0; }}
        .article {{ margin: 15px 0; padding: 15px; background: white; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .article-title {{ font-weight: bold; color: #333; margin-bottom: 8px; }}
        .article-meta {{ color: #666; font-size: 0.9em; margin-bottom: 10px; }}
        .article-content {{ color: #444; }}
        .export-info {{ background: #f9f9f9; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
    </style>
</head>
<body>"""
        
        if include_metadata:
            html_content += f"""
    <div class="export-info">
        <h2>📊 导出信息</h2>
        <p><strong>导出时间</strong>: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
        <p><strong>数据类型</strong>: {'话题聚类数据' if data_type == 'topics' else '文章列表数据'}</p>
        <p><strong>生成工具</strong>: ComfyUI RSS Tool</p>
    </div>"""
        
        if data_type == "topics":
            html_content += "<h1>📰 新闻话题分析结果</h1>"
            for topic_key, topic_data in data.items():
                html_content += f"""
    <div class="topic">
        <h2 class="topic-title">{topic_data.get('topic_label', topic_key)}</h2>
        <p><strong>文章数量</strong>: {topic_data.get('articles_count', 0)}篇</p>
        
        <div class="keywords">
            <strong>关键词</strong>: {', '.join(topic_data.get('keywords', []))}
        </div>
        
        <h3>相关文章:</h3>"""
                
                for article in topic_data.get("articles", []):
                    html_content += f"""
        <div class="article">
            <div class="article-title">{html.escape(article.get('title', '无标题'))}</div>
            <div class="article-meta">
                来源: {html.escape(self._extract_domain(article.get('source_url', '')))} |
                链接: <a href="{article.get('link', '')}" target="_blank">查看原文</a>
            </div>
            <div class="article-content">{html.escape(article.get('description', '')[:300])}...</div>
        </div>"""
                
                html_content += "</div>"
        else:
            html_content += "<h1>📄 文章内容导出</h1>"
            for article in data:
                html_content += f"""
    <div class="article">
        <div class="article-title">{html.escape(article.get('title', '无标题'))}</div>
        <div class="article-meta">
            来源: {html.escape(self._extract_domain(article.get('source_url', '')))} |
            <a href="{article.get('link', '')}" target="_blank">查看原文</a>
        </div>
        <div class="article-content">{html.escape(article.get('text', article.get('description', ''))[:500])}...</div>
    </div>"""
        
        html_content += """
</body>
</html>"""
        
        return html_content
    
    def _format_markdown_export(self, data, data_type, include_metadata, custom_template):
        """Format data as Markdown"""
        if custom_template:
            return self._apply_custom_template(data, custom_template, "markdown")
        
        md_content = "# 📰 RSS内容导出\n\n"
        
        if include_metadata:
            md_content += f"""## 导出信息

- **导出时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
- **数据类型**: {'话题聚类数据' if data_type == 'topics' else '文章列表数据'}
- **生成工具**: ComfyUI RSS Tool

---

"""
        
        if data_type == "topics":
            for topic_key, topic_data in data.items():
                md_content += f"""## {topic_data.get('topic_label', topic_key)}

**文章数量**: {topic_data.get('articles_count', 0)}篇

**关键词**: {', '.join(topic_data.get('keywords', []))}

### 相关文章

"""
                for i, article in enumerate(topic_data.get("articles", []), 1):
                    title = article.get('title', '无标题')
                    link = article.get('link', '')
                    source = self._extract_domain(article.get('source_url', ''))
                    description = article.get('description', '')[:200]
                    
                    md_content += f"""{i}. **{title}**
   - 链接: {link}
   - 来源: {source}
   - 摘要: {description}...

"""
                md_content += "---\n\n"
        else:
            md_content += "## 文章列表\n\n"
            for i, article in enumerate(data, 1):
                title = article.get('title', '无标题')
                link = article.get('link', '')
                source = self._extract_domain(article.get('source_url', ''))
                
                md_content += f"""{i}. **{title}**
   - 链接: {link}
   - 来源: {source}

"""
        
        return md_content
    
    def _format_topic_csv(self, topic_data, include_metadata):
        """Format single topic as CSV"""
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        
        if include_metadata:
            writer.writerow([f"Topic: {topic_data.get('topic_label', 'Unknown')}", "", "", ""])
            writer.writerow([f"Articles: {topic_data.get('articles_count', 0)}", "", "", ""])
            writer.writerow([f"Keywords: {', '.join(topic_data.get('keywords', []))}", "", "", ""])
            writer.writerow([])
        
        writer.writerow(["标题", "链接", "来源", "描述"])
        
        for article in topic_data.get("articles", []):
            writer.writerow([
                article.get("title", ""),
                article.get("link", ""),
                self._extract_domain(article.get("source_url", "")),
                article.get("description", "")[:200]
            ])
        
        return output.getvalue()
    
    def _format_topic_html(self, topic_data, include_metadata, custom_template):
        """Format single topic as HTML"""
        topic_label = topic_data.get('topic_label', 'Unknown Topic')
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{topic_label}</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 20px; }}
        .topic-header {{ background: #f0f8ff; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .article {{ margin: 15px 0; padding: 15px; border: 1px solid #ddd; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="topic-header">
        <h1>{topic_label}</h1>
        <p><strong>文章数量</strong>: {topic_data.get('articles_count', 0)}篇</p>
        <p><strong>关键词</strong>: {', '.join(topic_data.get('keywords', []))}</p>
    </div>
"""
        
        for article in topic_data.get("articles", []):
            html += f"""
    <div class="article">
        <h3>{html.escape(article.get('title', '无标题'))}</h3>
        <p><strong>来源</strong>: {html.escape(self._extract_domain(article.get('source_url', '')))}</p>
        <p><strong>链接</strong>: <a href="{article.get('link', '')}" target="_blank">查看原文</a></p>
        <p>{html.escape(article.get('description', ''))}</p>
    </div>"""
        
        html += "</body></html>"
        return html
    
    def _format_topic_markdown(self, topic_data, include_metadata, custom_template):
        """Format single topic as Markdown"""
        topic_label = topic_data.get('topic_label', 'Unknown Topic')
        
        md = f"# {topic_label}\n\n"
        
        if include_metadata:
            md += f"""## 话题信息

- **文章数量**: {topic_data.get('articles_count', 0)}篇
- **关键词**: {', '.join(topic_data.get('keywords', []))}

---

## 相关文章

"""
        
        for i, article in enumerate(topic_data.get("articles", []), 1):
            title = article.get('title', '无标题')
            link = article.get('link', '')
            source = self._extract_domain(article.get('source_url', ''))
            description = article.get('description', '')
            
            md += f"""{i}. **{title}**
   - 链接: {link}
   - 来源: {source}
   - 摘要: {description}

"""
        
        return md
    
    def _split_and_export(self, articles, format_type, output_dir, prefix, timestamp,
                         max_articles, include_metadata, custom_template):
        """Split large dataset into multiple files"""
        exported_files = []
        
        for i in range(0, len(articles), max_articles):
            chunk = articles[i:i + max_articles]
            chunk_num = i // max_articles + 1
            
            ts_suffix = f"_{timestamp}" if timestamp else ""
            filename = f"{prefix}_part{chunk_num}{ts_suffix}.{format_type}"
            file_path = os.path.join(output_dir, filename)
            
            try:
                if format_type == "json":
                    content = self._format_json_export(chunk, "articles", include_metadata)
                elif format_type == "csv":
                    content = self._format_csv_export(chunk, "articles", include_metadata)
                elif format_type == "html":
                    content = self._format_html_export(chunk, "articles", include_metadata, custom_template)
                elif format_type == "markdown":
                    content = self._format_markdown_export(chunk, "articles", include_metadata, custom_template)
                else:
                    continue
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                exported_files.append(file_path)
                
            except Exception as e:
                print(f"Split export failed for part {chunk_num}: {e}")
        
        return exported_files
    
    def _compress_files(self, file_paths, output_dir, prefix, timestamp):
        """Compress exported files"""
        try:
            import zipfile
            
            ts_suffix = f"_{timestamp}" if timestamp else ""
            zip_filename = f"{prefix}_export{ts_suffix}.zip"
            zip_path = os.path.join(output_dir, zip_filename)
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in file_paths:
                    if os.path.exists(file_path):
                        arcname = os.path.basename(file_path)
                        zipf.write(file_path, arcname)
                        # Remove original file after adding to zip
                        os.remove(file_path)
            
            return zip_path
            
        except Exception as e:
            print(f"Compression failed: {e}")
            return None
    
    def _apply_custom_template(self, data, template, format_type):
        """Apply custom template to data"""
        try:
            # Simple template variable replacement
            template_vars = {
                "{{data}}": json.dumps(data, ensure_ascii=False, indent=2),
                "{{timestamp}}": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "{{total_articles}}": str(self._count_total_articles(data)),
                "{{export_tool}}": "ComfyUI RSS Tool"
            }
            
            formatted_template = template
            for var, value in template_vars.items():
                formatted_template = formatted_template.replace(var, value)
            
            return formatted_template
            
        except Exception as e:
            return f"Template error: {e}"
    
    def _count_total_articles(self, data):
        """Count total articles in data"""
        if isinstance(data, list):
            return len(data)
        elif isinstance(data, dict):
            if "articles" in data:
                return len(data["articles"])
            else:
                # Topics format
                return sum(topic.get("articles_count", 0) for topic in data.values())
        return 0
    
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
    "ContentExporter": ContentExporter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ContentExporter": "内容导出器"
}