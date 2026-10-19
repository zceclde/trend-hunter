import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
import jieba

from src.velocity import analyze_cluster_velocity

STOPWORDS = set([
    '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
    '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
    '自己', '这', '那', '还', '个', '把', '过', '我们', '大家', '就是', '可以', '什么',
    '不是', '怎么', '觉得', '这个', '那个', '因为', '所以', '如果', '但是', '而且', '感觉',
    '知道', '其实', '现在', '还是', '直接', '真的', '已经', '比较', '应该', '一下', '出来',
    '还是', '或者', '很多', '这种', '还有', '为了', '之后', '通过', '可能', '结果', '今天',
    '回复', '楼主', '大佬', '萌新', '请问', '求问', '萌新问一下', '老玩家说句话', '家人们',
    '有人', '一起', '多多关照', '日常', '水一贴', '快乐', '第一次', '发帖', '哈哈哈哈哈', '2333',
    'image', 'emoji', 'https', 'http', 'com', 'cn', 'doge'
])

CHIT_CHAT_FILLERS = set([
    "早", "晚安", "签到", "打卡", "前排", "沙发", "发错群了", "666", "hhh", "hhhh", "哈哈哈哈哈",
    "有人一起玩吗", "今天也是快乐的一天", "求个好友", "哇啊啊啊啊啊啊啊啊啊", "我勒个复读机"
])

def jieba_tokenizer(text: str) -> List[str]:
    return [w.strip() for w in jieba.cut(text) if len(w.strip()) > 1 and w.strip() not in STOPWORDS]

class TrendClusterer:
    def __init__(self, window_days: int = 14):
        self.window_days = window_days

    def cluster_messages(
        self,
        messages: List[Dict[str, Any]],
        as_of: str,
        versions: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs unsupervised clustering on messages within the active temporal window
        and tracks historical trajectory across the entire dataset.
        """
        as_of_dt = datetime.strptime(as_of, "%Y-%m-%d")
        window_start_dt = as_of_dt - timedelta(days=self.window_days)
        window_start_str = window_start_dt.strftime("%Y-%m-%d")
        
        # Split into active window messages and historical messages
        active_msgs = [m for m in messages if m["date"] >= window_start_str]
        if not active_msgs:
            active_msgs = messages[-100:] if len(messages) >= 100 else messages
            
        texts = [m["stitched_text"] for m in active_msgs]
        
        # 1. TF-IDF feature extraction
        tfidf = TfidfVectorizer(
            tokenizer=jieba_tokenizer,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.4,
            sublinear_tf=True
        )
        X_tfidf = tfidf.fit_transform(texts)
        
        # 2. Dimensionality reduction
        n_components = min(50, max(5, X_tfidf.shape[1] - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        X_dense = svd.fit_transform(X_tfidf)
        X_dense = normalize(X_dense)
        
        # 3. Agglomerative clustering with Euclidean metric on L2-normalized vectors
        agg = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.85,
            metric="euclidean",
            linkage="ward"
        )
        labels = agg.fit_predict(X_dense)
        
        # Group messages by cluster label
        cluster_groups = defaultdict(list)
        cluster_dense_vectors = defaultdict(list)
        for idx, lbl in enumerate(labels):
            cluster_groups[lbl].append(active_msgs[idx])
            cluster_dense_vectors[lbl].append(X_dense[idx])
            
        candidate_clusters = []
        
        for lbl, c_msgs in cluster_groups.items():
            cnt = len(c_msgs)
            if cnt < 8:
                continue # Prune tiny noise groups
                
            # Compute average text length
            avg_len = sum(len(m["cleaned_text"]) for m in c_msgs) / cnt
            
            # Check vocabulary diversity
            content_tokens = set()
            for m in c_msgs:
                content_tokens.update(w for w in jieba.cut(m["cleaned_text"]) if len(w) > 1 and w not in STOPWORDS)
                
            is_bot_or_spam = any(
                m.get("is_bot") or any(s in m["cleaned_text"] for s in ["代练", "出号", "破解版", "Q群管家"])
                for m in c_msgs
            )
            
            # Prune trivial greetings and low-diversity filler ("早", "晚安", "签到", "我也是")
            if (len(content_tokens) < 5 or avg_len <= 5.0) and not is_bot_or_spam:
                continue
                
            # Trajectory velocity analysis
            metrics = analyze_cluster_velocity(c_msgs, as_of=as_of, versions=versions)
            
            # Find representative medoid messages (closest to centroid)
            vecs = np.array(cluster_dense_vectors[lbl])
            centroid = np.mean(vecs, axis=0)
            dists = np.linalg.norm(vecs - centroid, axis=1)
            medoid_indices = np.argsort(dists)[:5]
            
            exemplar_msgs = [c_msgs[i] for i in medoid_indices]
            voice_ids = [m["id"] for m in exemplar_msgs]
            sample_texts = [m["cleaned_text"] for m in exemplar_msgs]
            
            # Extract distinctive keywords to trace true earliest historical origin
            cluster_tokens = []
            for s in sample_texts:
                cluster_tokens.extend([w for w in jieba.cut(s) if len(w) > 1 and w not in STOPWORDS])
            top_kw = [w for w, _ in Counter(cluster_tokens).most_common(2)]
            
            true_first_seen = metrics["first_seen"]
            if top_kw:
                for m in messages:
                    c_txt = m.get("cleaned_text", "")
                    if all(k in c_txt for k in top_kw) or (len(top_kw) >= 1 and top_kw[0] in c_txt):
                        true_first_seen = m["date"]
                        break

            # Calculate momentum surge score
            recent_3d = metrics.get("recent_msgs_3d", 0)
            n_plats = metrics.get("num_platforms", 1)
            status_priority = {
                "growing": 50,
                "new": 45,
                "one-off": 30,
                "not real": 20,
                "fading": 10
            }
            momentum_score = (
                status_priority.get(metrics["status"], 10)
                + metrics.get("recent_weighted", recent_3d) * 2.5
                + n_plats * 8
                + metrics.get("cascade_score", 0) * 15
                + math.log(max(1, cnt))
            )

            candidate_clusters.append({
                "cluster_id": lbl,
                "count": cnt,
                "status": metrics["status"],
                "first_seen": true_first_seen,
                "last_seen": metrics["last_seen"],
                "metrics": metrics,
                "voice_ids": voice_ids,
                "sample_texts": sample_texts,
                "exemplar_msgs": exemplar_msgs,
                "top_kw": top_kw,
                "momentum_score": momentum_score
            })
            
        # Rank candidate clusters by momentum score
        candidate_clusters.sort(key=lambda c: c["momentum_score"], reverse=True)
        
        # Select up to 8 diverse trends ensuring semantic diversity
        selected = []
        seen_statuses = Counter()
        seen_topic_keys = set()
        
        for c in candidate_clusters:
            st = c["status"]
            # Deduplicate by primary keyword (e.g. avoid duplicate Iron Pot trends)
            primary_kw = c["top_kw"][0] if c.get("top_kw") else str(c["cluster_id"])
            if primary_kw in seen_topic_keys and len(selected) < 7:
                continue
                
            # Diversity guardrails per status
            if st == "growing" and seen_statuses[st] >= 3:
                continue
            if st == "new" and seen_statuses[st] >= 2:
                continue
            if st == "one-off" and seen_statuses[st] >= 2:
                continue
            if st == "not real" and seen_statuses[st] >= 1:
                continue
            if st == "fading" and seen_statuses[st] >= 2:
                continue
                
            selected.append(c)
            seen_statuses[st] += 1
            seen_topic_keys.add(primary_kw)
            if len(selected) >= 8:
                break
                
        # Backfill if needed
        if len(selected) < 8:
            selected_ids = set(c["cluster_id"] for c in selected)
            for c in candidate_clusters:
                if c["cluster_id"] not in selected_ids:
                    selected.append(c)
                    if len(selected) >= 8:
                        break
                        
        return selected[:8]

