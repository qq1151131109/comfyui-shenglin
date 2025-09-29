import json
import re
import jieba
import numpy as np
from datetime import datetime
from collections import Counter

class TopicClusteringEngine:
    """
    Topic Clustering Engine Node
    
    Performs intelligent topic clustering using TF-IDF and machine learning algorithms.
    """
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("clustered_topics_json", "topic_labels", "clustering_report")
    FUNCTION = "execute"
    CATEGORY = "RSS Content Processing"

    def __init__(self):
        # Initialize Chinese stopwords
        self.chinese_stopwords = self._load_chinese_stopwords()
    
    def _load_chinese_stopwords(self):
        """Load Chinese stopwords"""
        default_stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这',
            '那', '它', '我们', '他们', '她们', '什么', '怎么', '为什么', '哪里', 'when', 'where', 'what', 'why', 'how', 'who', 'which', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'
        }
        return default_stopwords
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "articles_json": ("STRING", {
                    "forceInput": True,
                    "multiline": True
                }),
                "clustering_method": (["kmeans", "dbscan", "auto"], {
                    "default": "auto"
                }),
                "num_topics": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                    "tooltip": "0表示自动确定话题数量"
                }),
                "vectorizer": (["tfidf", "tfidf_bigram", "simple_count"], {
                    "default": "tfidf"
                }),
                "min_articles_per_topic": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
                "language": (["auto", "zh", "en", "mixed"], {
                    "default": "auto"
                }),
            },
            "optional": {
                "custom_stopwords": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "自定义停用词，每行一个"
                }),
                "topic_keywords_count": ("INT", {
                    "default": 5,
                    "min": 3,
                    "max": 15,
                    "step": 1
                }),
            }
        }

    def execute(self, articles_json, clustering_method, num_topics, vectorizer,
                min_articles_per_topic, language, custom_stopwords="", topic_keywords_count=5):
        return self.cluster_topics(
            articles_json, clustering_method, num_topics, vectorizer,
            min_articles_per_topic, language, custom_stopwords, topic_keywords_count
        )
    
    def cluster_topics(self, articles_json, clustering_method, num_topics, vectorizer,
                      min_articles_per_topic, language, custom_stopwords, topic_keywords_count):
        try:
            # Import required libraries
            from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
            from sklearn.cluster import KMeans, DBSCAN
            from sklearn.metrics import silhouette_score
            
            # Parse articles
            try:
                articles = json.loads(articles_json)
            except json.JSONDecodeError:
                return "Error: Invalid JSON format for articles", "", ""
            
            if not articles:
                return "Error: No articles to process", "", ""
            
            if len(articles) < min_articles_per_topic:
                return "Error: Not enough articles for clustering", "", ""
            
            print(f"开始聚类分析 {len(articles)} 篇文章...")
            
            # Prepare text data
            texts = []
            for article in articles:
                title = article.get("title", "")
                content = article.get("text", "") or article.get("description", "")
                combined_text = f"{title} {content}"
                texts.append(combined_text)
            
            # Preprocess texts
            processed_texts = [self._preprocess_text(text, language, custom_stopwords) for text in texts]
            
            # Vectorization
            if vectorizer == "tfidf":
                vectorizer_obj = TfidfVectorizer(
                    max_features=1000,
                    ngram_range=(1, 1),
                    min_df=1,
                    max_df=0.8,
                    stop_words=None  # We handle stopwords in preprocessing
                )
            elif vectorizer == "tfidf_bigram":
                vectorizer_obj = TfidfVectorizer(
                    max_features=1500,
                    ngram_range=(1, 2),  # Include bigrams
                    min_df=1,
                    max_df=0.8
                )
            else:  # simple_count
                vectorizer_obj = CountVectorizer(
                    max_features=800,
                    ngram_range=(1, 1),
                    min_df=1,
                    max_df=0.8
                )
            
            # Transform texts to vectors
            X = vectorizer_obj.fit_transform(processed_texts)
            feature_names = vectorizer_obj.get_feature_names_out()
            
            print(f"文本向量化完成，特征维度: {X.shape}")
            
            # Determine optimal clustering
            if clustering_method == "auto":
                best_method, best_clusters, best_score = self._find_optimal_clustering(X, num_topics, min_articles_per_topic)
                clustering_method = best_method
                if num_topics == 0:
                    num_topics = best_clusters
            
            # Perform clustering
            if clustering_method == "kmeans":
                if num_topics == 0:
                    num_topics = min(max(2, len(articles) // 5), 8)  # Auto determine
                
                # Ensure we don't have more clusters than samples
                num_topics = min(num_topics, len(articles))
                
                clusterer = KMeans(n_clusters=num_topics, random_state=42, n_init=10)
                cluster_labels = clusterer.fit_predict(X)
                
            else:  # dbscan
                # Estimate eps based on data
                from sklearn.neighbors import NearestNeighbors
                neighbors = NearestNeighbors(n_neighbors=min_articles_per_topic)
                neighbors_fit = neighbors.fit(X)
                distances, indices = neighbors_fit.kneighbors(X)
                distances = np.sort(distances, axis=0)
                eps = np.percentile(distances[:, min_articles_per_topic-1], 75)
                
                clusterer = DBSCAN(eps=eps, min_samples=min_articles_per_topic)
                cluster_labels = clusterer.fit_predict(X)
            
            # Process clustering results
            clustered_topics, topic_labels, report = self._process_clustering_results(
                articles, cluster_labels, X, feature_names, topic_keywords_count, clustering_method
            )
            
            # Format outputs - clean data for JSON serialization
            cleaned_topics = self._clean_for_json(clustered_topics)
            clustered_json = json.dumps(cleaned_topics, ensure_ascii=False, indent=2)
            topic_labels_text = "\n".join([f"话题{i}: {label}" for i, label in enumerate(topic_labels)])
            
            return clustered_json, topic_labels_text, report
            
        except ImportError as e:
            missing_lib = str(e).split("'")[1] if "'" in str(e) else str(e)
            return f"Error: Missing library {missing_lib}. Please run: pip install scikit-learn jieba", "", ""
        except Exception as e:
            return f"Error: {str(e)}", "", ""
    
    def _preprocess_text(self, text, language, custom_stopwords):
        """Preprocess text for clustering"""
        # Clean text
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)  # Keep only alphanumeric and Chinese
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization
        if language in ["zh", "auto"] and self._contains_chinese(text):
            # Chinese text processing
            words = list(jieba.cut(text))
            words = [word for word in words if len(word.strip()) > 1]
        else:
            # English text processing  
            words = text.lower().split()
        
        # Remove stopwords
        stopwords = self.chinese_stopwords.copy()
        if custom_stopwords.strip():
            custom_stops = set(custom_stopwords.strip().split('\n'))
            stopwords.update(custom_stops)
        
        filtered_words = [word for word in words if word.lower() not in stopwords and len(word) > 1]
        
        return ' '.join(filtered_words)
    
    def _contains_chinese(self, text):
        """Check if text contains Chinese characters"""
        return bool(re.search(r'[\u4e00-\u9fff]', text))
    
    def _find_optimal_clustering(self, X, num_topics, min_articles_per_topic):
        """Find optimal clustering method and parameters"""
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.metrics import silhouette_score
        
        best_score = -1
        best_method = "kmeans"
        best_clusters = 3
        
        # Try different k-means configurations
        if num_topics == 0:
            k_range = range(2, min(8, X.shape[0] // min_articles_per_topic + 1))
        else:
            k_range = [num_topics]
        
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                score = silhouette_score(X, labels)
                
                if score > best_score:
                    best_score = score
                    best_method = "kmeans"
                    best_clusters = k
            except:
                continue
        
        return best_method, best_clusters, best_score
    
    def _process_clustering_results(self, articles, cluster_labels, X, feature_names, 
                                   topic_keywords_count, clustering_method):
        """Process clustering results and generate topic labels"""
        # Group articles by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            # Convert numpy int32 to Python int for JSON serialization
            label = int(label) if label != -1 else "uncategorized"
            
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(articles[i])
        
        # Generate topic keywords and labels
        topic_labels = []
        clustered_topics = {}
        
        for cluster_id, cluster_articles in clusters.items():
            # Extract keywords for this cluster
            cluster_texts = []
            for article in cluster_articles:
                title = article.get("title", "")
                content = article.get("text", "") or article.get("description", "")
                cluster_texts.append(f"{title} {content}")
            
            # Get top keywords for this cluster
            keywords = self._extract_cluster_keywords(cluster_texts, topic_keywords_count)
            
            # Generate topic label
            if cluster_id == "uncategorized":
                topic_label = "未分类内容"
            else:
                topic_label = self._generate_topic_label(keywords, cluster_articles)
            
            topic_labels.append(topic_label)
            
            # Store clustered data
            clustered_topics[f"topic_{cluster_id}"] = {
                "topic_id": int(cluster_id) if hasattr(cluster_id, 'dtype') else cluster_id,
                "topic_label": topic_label,
                "keywords": keywords,
                "articles_count": len(cluster_articles),
                "articles": cluster_articles
            }
        
        # Generate clustering report
        total_articles = len(articles)
        num_clusters = len([k for k in clusters.keys() if k != "uncategorized"])
        uncategorized_count = len(clusters.get("uncategorized", []))
        
        report = f"""话题聚类分析报告:
总文章数: {total_articles}
聚类方法: {clustering_method}
识别话题数: {num_clusters}
未分类文章: {uncategorized_count}

各话题统计:
"""
        
        # Sort topics by article count
        sorted_topics = sorted(clustered_topics.items(), 
                              key=lambda x: x[1]["articles_count"], reverse=True)
        
        for topic_key, topic_data in sorted_topics:
            report += f"📊 {topic_data['topic_label']}: {topic_data['articles_count']}篇\n"
            report += f"   关键词: {', '.join(topic_data['keywords'][:5])}\n"
        
        report += f"\n处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return clustered_topics, topic_labels, report
    
    def _extract_cluster_keywords(self, texts, top_k):
        """Extract top keywords for a cluster"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Preprocess texts
            processed_texts = []
            for text in texts:
                processed = self._preprocess_for_keywords(text)
                if processed:
                    processed_texts.append(processed)
            
            if not processed_texts:
                return ["未知话题"]
            
            # TF-IDF extraction
            # Adjust parameters based on text count
            min_df = 1
            max_df = min(0.8, max(0.5, len(processed_texts) - 1)) if len(processed_texts) > 1 else 1.0
            
            vectorizer = TfidfVectorizer(
                max_features=200,
                ngram_range=(1, 2),
                min_df=min_df,
                max_df=max_df,
                stop_words=None
            )
            
            tfidf_matrix = vectorizer.fit_transform(processed_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Calculate average TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Get top keywords
            top_indices = np.argsort(mean_scores)[-top_k:][::-1]
            keywords = [feature_names[i] for i in top_indices if mean_scores[i] > 0]
            
            return keywords[:top_k] if keywords else ["通用话题"]
            
        except Exception as e:
            print(f"关键词提取失败: {e}")
            return ["话题提取失败"]
    
    def _preprocess_for_keywords(self, text):
        """Preprocess text for keyword extraction"""
        # Clean text
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text:
            return ""
        
        # Tokenization
        if self._contains_chinese(text):
            words = list(jieba.cut(text))
        else:
            words = text.lower().split()
        
        # Filter words
        filtered_words = []
        for word in words:
            word = word.strip()
            if (len(word) > 1 and 
                word.lower() not in self.chinese_stopwords and
                not word.isdigit() and
                len(word) < 20):  # Avoid very long words
                filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    def _contains_chinese(self, text):
        """Check if text contains Chinese characters"""
        return bool(re.search(r'[\u4e00-\u9fff]', text))
    
    def _generate_topic_label(self, keywords, articles):
        """Generate human-readable topic label"""
        if not keywords:
            return "未知话题"
        
        # Simple rule-based topic labeling
        main_keywords = keywords[:3]
        
        # Check for common topic patterns
        keywords_text = ' '.join(keywords).lower()
        
        if any(word in keywords_text for word in ['股票', '股价', '投资', '市场', '金融']):
            return f"📈 金融市场 ({', '.join(main_keywords)})"
        elif any(word in keywords_text for word in ['科技', '技术', 'ai', '人工智能', '数字']):
            return f"💻 科技创新 ({', '.join(main_keywords)})"
        elif any(word in keywords_text for word in ['政治', '政府', '政策', '法律', '选举']):
            return f"🏛️ 政治政策 ({', '.join(main_keywords)})"
        elif any(word in keywords_text for word in ['健康', '医疗', '疫情', '病毒', '疫苗']):
            return f"🏥 健康医疗 ({', '.join(main_keywords)})"
        elif any(word in keywords_text for word in ['体育', '足球', '篮球', '奥运', '运动']):
            return f"⚽ 体育运动 ({', '.join(main_keywords)})"
        elif any(word in keywords_text for word in ['娱乐', '电影', '音乐', '明星', '演员']):
            return f"🎬 娱乐文化 ({', '.join(main_keywords)})"
        elif any(word in keywords_text for word in ['环境', '气候', '环保', '污染', '能源']):
            return f"🌍 环境能源 ({', '.join(main_keywords)})"
        elif any(word in keywords_text for word in ['bitcoin', 'crypto', '比特币', '加密', '区块链']):
            return f"₿ 加密货币 ({', '.join(main_keywords)})"
        else:
            # Use most frequent keyword as topic name
            return f"📰 {main_keywords[0]} ({', '.join(main_keywords)})"
    
    def _clean_for_json(self, data):
        """Clean data to ensure JSON serialization compatibility"""
        if isinstance(data, dict):
            return {key: self._clean_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._clean_for_json(item) for item in data]
        elif hasattr(data, 'dtype'):  # numpy types
            return int(data) if 'int' in str(data.dtype) else float(data)
        elif isinstance(data, (np.integer, np.int32, np.int64)):
            return int(data)
        elif isinstance(data, (np.floating, np.float32, np.float64)):
            return float(data)
        else:
            return data

NODE_CLASS_MAPPINGS = {
    "TopicClusteringEngine": TopicClusteringEngine
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TopicClusteringEngine": "话题聚类引擎"
}