#!/usr/bin/env python3
"""
完整工作流测试脚本
测试从多源RSS收集到最终导出的完整流程
"""

import json
import time
import tempfile
import os

def test_step_1_multi_rss_collector():
    """测试步骤1: Multi RSS Collector"""
    print("🔄 步骤1: 测试Multi RSS Collector...")
    
    from MultiRSSCollector import MultiRSSCollector
    
    collector = MultiRSSCollector()
    
    # 使用稳定的RSS源进行测试
    test_rss_urls = """https://feeds.bbci.co.uk/news/rss.xml
https://rss.cnn.com/rss/edition.rss"""
    
    result = collector.collect_multi_rss(
        rss_urls=test_rss_urls,
        articles_per_source=3,  # 每个源3篇文章，减少测试时间
        time_filter="all",
        language_filter="all", 
        timeout=15,
        max_workers=2,
        global_filter_keywords="",
        global_exclude_keywords=""
    )
    
    articles_json, source_summary, all_links = result
    
    if articles_json.startswith("Error"):
        print(f"❌ Multi RSS Collector失败: {articles_json}")
        return None, None, None
    
    try:
        articles_data = json.loads(articles_json)
        print(f"✅ Multi RSS Collector成功: 收集到 {len(articles_data)} 篇文章")
        print(f"   - 来源统计: {source_summary.count('✅')} 个成功源")
        print(f"   - 文章链接: {len(all_links.split())} 个链接")
        return articles_json, source_summary, all_links
    except json.JSONDecodeError:
        print(f"❌ Multi RSS Collector JSON解析失败")
        return None, None, None

def test_step_2_batch_extractor(articles_json):
    """测试步骤2: Batch Article Extractor"""
    print("\n🔄 步骤2: 测试Batch Article Extractor...")
    
    if not articles_json:
        print("❌ 跳过Batch Article Extractor - 无输入数据")
        return None, None
    
    from BatchArticleExtractor import BatchArticleExtractor
    
    extractor = BatchArticleExtractor()
    
    result = extractor.batch_extract_articles(
        article_data=articles_json,
        extract_strategy="fast",  # 使用快速模式减少测试时间
        max_articles=5,  # 限制文章数量
        parallel_workers=2,
        timeout_per_article=10,
        max_text_length=1000,
        custom_headers="",
        skip_extraction_errors=True
    )
    
    extracted_json, extraction_report = result
    
    if extracted_json.startswith("Error"):
        print(f"❌ Batch Article Extractor失败: {extracted_json}")
        return None, None
    
    try:
        extracted_data = json.loads(extracted_json)
        print(f"✅ Batch Article Extractor成功: 提取到 {len(extracted_data)} 篇完整文章")
        success_rate = extraction_report.split("成功率: ")[1].split("%")[0] if "成功率:" in extraction_report else "未知"
        print(f"   - 提取成功率: {success_rate}%")
        return extracted_json, extraction_report
    except json.JSONDecodeError:
        print(f"❌ Batch Article Extractor JSON解析失败")
        return None, None

def test_step_3_deduplicator(extracted_json):
    """测试步骤3: Content Deduplicator"""
    print("\n🔄 步骤3: 测试Content Deduplicator...")
    
    if not extracted_json:
        print("❌ 跳过Content Deduplicator - 无输入数据")
        return None, None
    
    from ContentDeduplicator import ContentDeduplicator
    
    deduplicator = ContentDeduplicator()
    
    result = deduplicator.deduplicate_content(
        articles_json=extracted_json,
        similarity_threshold=0.80,
        dedup_method="smart",
        keep_strategy="most_complete",
        min_title_length=5,  # 降低要求以适应测试
        min_content_length=50,  # 降低要求以适应测试
        preserve_sources=True,
        merge_similar=False
    )
    
    unique_json, dedup_report = result
    
    if unique_json.startswith("Error"):
        print(f"❌ Content Deduplicator失败: {unique_json}")
        return None, None
    
    try:
        unique_data = json.loads(unique_json)
        print(f"✅ Content Deduplicator成功: 去重后剩余 {len(unique_data)} 篇文章")
        if "去重率:" in dedup_report:
            dedup_rate = dedup_report.split("去重率: ")[1].split("%")[0]
            print(f"   - 去重率: {dedup_rate}%")
        return unique_json, dedup_report
    except json.JSONDecodeError:
        print(f"❌ Content Deduplicator JSON解析失败")
        return None, None

def test_step_4_clustering(unique_json):
    """测试步骤4: Topic Clustering Engine"""
    print("\n🔄 步骤4: 测试Topic Clustering Engine...")
    
    if not unique_json:
        print("❌ 跳过Topic Clustering Engine - 无输入数据")
        return None, None, None
    
    from TopicClusteringEngine import TopicClusteringEngine
    
    clustering = TopicClusteringEngine()
    
    result = clustering.cluster_topics(
        articles_json=unique_json,
        clustering_method="auto",
        num_topics=0,  # 自动确定
        vectorizer="tfidf",
        min_articles_per_topic=1,  # 降低要求以适应测试
        language="auto",
        custom_stopwords="",
        topic_keywords_count=5
    )
    
    clustered_json, topic_labels, clustering_report = result
    
    if clustered_json.startswith("Error"):
        print(f"❌ Topic Clustering Engine失败: {clustered_json}")
        return None, None, None
    
    try:
        clustered_data = json.loads(clustered_json)
        num_topics = len(clustered_data)
        total_articles = sum(topic.get("articles_count", 0) for topic in clustered_data.values())
        print(f"✅ Topic Clustering Engine成功: 识别出 {num_topics} 个话题")
        print(f"   - 总文章数: {total_articles}")
        print(f"   - 话题标签: {len(topic_labels.split(chr(10)))} 个")
        return clustered_json, topic_labels, clustering_report
    except json.JSONDecodeError:
        print(f"❌ Topic Clustering Engine JSON解析失败")
        return None, None, None

def test_step_5_report_generator(clustered_json):
    """测试步骤5: Topic Report Generator"""
    print("\n🔄 步骤5: 测试Topic Report Generator...")
    
    if not clustered_json:
        print("❌ 跳过Topic Report Generator - 无输入数据")
        return None, None, None
    
    from TopicReportGenerator import TopicReportGenerator
    
    generator = TopicReportGenerator()
    
    result = generator.generate_topic_report(
        clustered_topics_json=clustered_json,
        report_format="markdown",
        include_summary=True,
        include_keywords=True,
        include_articles=True,
        sort_by="文章数量",
        max_articles_per_topic=3,
        custom_template="",
        generate_selection_tips=True,
        include_trending_analysis=True
    )
    
    formatted_report, topic_overview, selection_guide = result
    
    if formatted_report.startswith("Error"):
        print(f"❌ Topic Report Generator失败: {formatted_report}")
        return None, None, None
    
    print(f"✅ Topic Report Generator成功:")
    print(f"   - 报告长度: {len(formatted_report)} 字符")
    print(f"   - 概览长度: {len(topic_overview)} 字符") 
    print(f"   - 建议长度: {len(selection_guide)} 字符")
    
    return formatted_report, topic_overview, selection_guide

def test_step_6_content_exporter(clustered_json):
    """测试步骤6: Content Exporter"""
    print("\n🔄 步骤6: 测试Content Exporter...")
    
    if not clustered_json:
        print("❌ 跳过Content Exporter - 无输入数据")
        return None, None, None
    
    from ContentExporter import ContentExporter
    
    exporter = ContentExporter()
    
    # 使用临时目录
    temp_dir = tempfile.mkdtemp()
    
    result = exporter.export_content(
        content_data=clustered_json,
        export_format="json",
        output_directory=temp_dir,
        filename_prefix="test_export",
        include_timestamp=True,
        create_directory=True,
        compress_output=False,
        split_by_topic=False,
        include_metadata=True,
        custom_template="",
        max_articles_per_file=0
    )
    
    export_path, export_summary, export_status = result
    
    if export_status != "success":
        print(f"❌ Content Exporter失败: {export_path}")
        return None, None, None
    
    # 检查文件是否存在
    if os.path.exists(export_path):
        file_size = os.path.getsize(export_path) / 1024  # KB
        print(f"✅ Content Exporter成功:")
        print(f"   - 导出文件: {os.path.basename(export_path)}")
        print(f"   - 文件大小: {file_size:.1f}KB")
        print(f"   - 状态: {export_status}")
        
        # 清理临时文件
        os.remove(export_path)
        os.rmdir(temp_dir)
        
        return export_path, export_summary, export_status
    else:
        print(f"❌ Content Exporter文件未创建")
        return None, None, None

def run_full_workflow_test():
    """运行完整工作流测试"""
    print("🧪 开始完整工作流测试...")
    print("=" * 60)
    
    start_time = time.time()
    
    # 步骤1: Multi RSS Collector
    articles_json, source_summary, all_links = test_step_1_multi_rss_collector()
    
    # 步骤2: Batch Article Extractor  
    extracted_json, extraction_report = test_step_2_batch_extractor(articles_json)
    
    # 步骤3: Content Deduplicator
    unique_json, dedup_report = test_step_3_deduplicator(extracted_json)
    
    # 步骤4: Topic Clustering Engine
    clustered_json, topic_labels, clustering_report = test_step_4_clustering(unique_json)
    
    # 步骤5: Topic Report Generator
    formatted_report, topic_overview, selection_guide = test_step_5_report_generator(clustered_json)
    
    # 步骤6: Content Exporter
    export_path, export_summary, export_status = test_step_6_content_exporter(clustered_json)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 60)
    print("📊 完整工作流测试结果:")
    print(f"总测试时间: {total_time:.1f}秒")
    
    # 检查每一步是否成功
    steps_status = [
        ("Multi RSS Collector", articles_json is not None),
        ("Batch Article Extractor", extracted_json is not None),
        ("Content Deduplicator", unique_json is not None), 
        ("Topic Clustering Engine", clustered_json is not None),
        ("Topic Report Generator", formatted_report is not None),
        ("Content Exporter", export_path is not None)
    ]
    
    success_count = sum(1 for _, success in steps_status if success)
    total_steps = len(steps_status)
    
    print(f"成功步骤: {success_count}/{total_steps}")
    
    for step_name, success in steps_status:
        status = "✅" if success else "❌"
        print(f"{status} {step_name}")
    
    if success_count == total_steps:
        print("\n🎉 完整工作流测试通过！所有节点工作正常。")
        
        # 显示最终结果预览
        if formatted_report:
            print("\n📰 最终报告预览:")
            print(formatted_report[:300] + "..." if len(formatted_report) > 300 else formatted_report)
            
        return True
    else:
        print(f"\n❌ 工作流测试失败，{total_steps - success_count} 个步骤出现问题。")
        return False

if __name__ == "__main__":
    print("ComfyUI RSS Tool - 完整工作流测试")
    print("=" * 60)
    
    try:
        success = run_full_workflow_test()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ 测试被用户中断")
        exit(1)
    except Exception as e:
        print(f"\n💥 测试过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
        exit(1)