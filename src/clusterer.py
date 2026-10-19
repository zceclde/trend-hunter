import re
import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Set, Callable
from collections import Counter, defaultdict
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
import jieba

from src.velocity import analyze_cluster_velocity, build_account_stats

STOPWORDS = set([
    '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
    '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
    '自己', '这', '那', '还', '个', '把', '过', '我们', '大家', '就是', '可以', '什么',
    '不是', '怎么', '觉得', '这个', '那个', '因为', '所以', '如果', '但是', '而且', '感觉',
    '知道', '其实', '现在', '还是', '直接', '真的', '已经', '比较', '应该', '一下', '出来',
    '或者', '很多', '这种', '还有', '为了', '之后', '通过', '可能', '结果', '今天',
    '回复', '楼主', '大佬', '萌新', '请问', '求问', '萌新问一下', '老玩家说句话', '家人们',
    '有人', '一起', '多多关照', '日常', '水一贴', '快乐', '第一次', '发帖', '哈哈哈哈哈', '2333',
    'image', 'emoji', 'https', 'http', 'com', 'cn', 'doge',
    '不是我说', '说真的', '讲道理', '有一说一', '我真的会谢', '老玩家', '说句话', '萌新问',
    '笑死', '绷不住', '真是', '简直', '到底', '最近', '不会', '这么', '那么', '有点',
    '完全', '实在', '果然', '终于', '是不是', '能不能', '有没有', '为什么', '一直',
    '啊啊啊', '哈哈哈', '哈哈', 'orz', 'qaq', 'hhhh', '小丑', '流汗', '大哭', '吃瓜',
    '滑稽', '人们', '会谢', '真滴', '介个', '那种', '看到', '时候', '地方', '东西',
    '问题', '根本', '以前', '越来越', '不能', '不过', '起来', '下去', '不要', '非常',
    '特别', '肯定', '绝对'
])

CHIT_CHAT_FILLERS = set([
    "早", "晚安", "签到", "打卡", "前排", "沙发", "发错群了", "666", "hhh", "hhhh", "哈哈哈哈哈",
    "有人一起玩吗", "今天也是快乐的一天", "求个好友", "哇啊啊啊啊啊啊啊啊啊", "我勒个复读机",
    "顶", "已三连", "三连", "求抱大腿", "求带", "蹲个后续", "插眼", "吃瓜",
    "有一说一确实", "确实", "+1", "我也是", "先马再看", "火钳刘明", "路过",
    "看看", "支持", "加油", "好家伙", "绝了", "无语", "离谱",
    "你说得对", "说得对", "附议", "同意楼上", "顶楼上"
])


def register_dynamic_entities(
    messages: List[Dict[str, Any]],
    versions: List[Dict[str, Any]] = None
) -> Set[str]:
    """
    Dynamically discovers game-specific proper nouns (quoted entities 「...」 and
    campaign hashtags #...#) from the corpus and registers them in jieba so novel
    skin/character names are never split into single characters. Also identifies
    ubiquitous franchise-wide hashtags (>40% of active days) so they do not merge
    unrelated Weibo/Bilibili posts into a single mega-cluster.
    """
    hashtag_days = defaultdict(set)
    quoted_counts = Counter()
    all_days = set()

    for m in messages:
        d = m.get("date", "")
        if d:
            all_days.add(d)
        txt = m.get("cleaned_text", "")
        for h in re.findall(r"#([^#\s]{2,15})#", txt):
            if re.match(r"^[bB]\d+:\d+$", h):
                continue
            hashtag_days[h].add(d)
        for q in re.findall(r"「([^」]{2,8})」", txt):
            quoted_counts[q] += 1

    if versions:
        for v in versions:
            for q in re.findall(r"「([^」]{2,8})」", v.get("notes", "")):
                quoted_counts[q] += 10

    num_days = max(1, len(all_days))
    global_tags = {
        h for h, ds in hashtag_days.items()
        if len(ds) / num_days > 0.40 and len(ds) >= 5
    }

    for q, c in quoted_counts.items():
        if c >= 3:
            jieba.add_word(q, freq=500000)

    for h in hashtag_days:
        if h not in global_tags:
            jieba.add_word(h, freq=500000)
            for g in global_tags:
                if h.startswith(g) and len(h) > len(g):
                    suffix = h[len(g):]
                    if len(suffix) >= 2:
                        jieba.add_word(suffix, freq=500000)

    return global_tags


def make_tokenizer(global_tags: Set[str]) -> Callable[[str], List[str]]:
    """Builds a tokenizer that strips global franchise tags, Bahamut floor refs, and bracketed emoji macros."""
    def tokenize(text: str) -> List[str]:
        text = re.sub(r"#[bB]\d+:\d+#", " ", text)
        for g in global_tags:
            text = re.sub(rf"#{re.escape(g)}([^#\s]*)#", r" \1 ", text)
            text = text.replace(g, " ")
        text = re.sub(r"\[[^\]]+\]", " ", text)
        tokens = []
        for w in jieba.cut(text.lower()):
            w = w.strip()
            if len(w) > 1 and w not in STOPWORDS and w not in CHIT_CHAT_FILLERS and not w.isdigit():
                tokens.append(w)
        return tokens
    return tokenize


def is_substantive(m: Dict[str, Any], tokenizer: Callable[[str], List[str]]) -> bool:
    """Filters out zero-content greetings, +1 replies, and pure emoji reactions while keeping bot posts."""
    if m.get("is_bot"):
        return True
    txt = re.sub(r"#[bB]\d+:\d+#", " ", m.get("cleaned_text", "")).strip()
    if len(txt) <= 5:
        return False
    for f in CHIT_CHAT_FILLERS:
        if txt == f or (len(txt) <= 12 and f in txt):
            return False
    toks = [w for w in tokenizer(txt) if w not in CHIT_CHAT_FILLERS]
    return len(toks) >= 2


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
        and tracks historical trajectory across the entire time-gated dataset.
        """
        if not messages:
            return []

        as_of_dt = datetime.strptime(as_of, "%Y-%m-%d")
        account_stats = build_account_stats(messages)
        global_tags = register_dynamic_entities(messages, versions)
        tokenizer = make_tokenizer(global_tags)

        substantive_msgs = [m for m in messages if is_substantive(m, tokenizer)]
        if not substantive_msgs:
            substantive_msgs = messages

        data_start = min(m["date"] for m in substantive_msgs)
        data_start_dt = datetime.strptime(data_start, "%Y-%m-%d")
        corpus_span_days = (as_of_dt - data_start_dt).days

        # Establish baseline vocabulary from the first 5 days of observation (when history >= 10 days)
        # so newly emerged entities, slang, and patches stand out against ambient chatter.
        baseline_cutoff = (data_start_dt + timedelta(days=4)).strftime("%Y-%m-%d")
        baseline_counts = Counter()
        if corpus_span_days >= 10:
            for m in substantive_msgs:
                if m["date"] <= baseline_cutoff:
                    for w in set(tokenizer(m["cleaned_text"])):
                        baseline_counts[w] += 1
        baseline_vocab = {w for w, c in baseline_counts.items() if c >= 2}

        window_start_str = (as_of_dt - timedelta(days=self.window_days)).strftime("%Y-%m-%d")
        active_msgs = [m for m in substantive_msgs if m["date"] >= window_start_str]
        hist_msgs = [m for m in substantive_msgs if m["date"] < window_start_str]
        if not active_msgs:
            active_msgs = substantive_msgs[-100:] if len(substantive_msgs) >= 100 else substantive_msgs
            hist_msgs = []

        # 1. TF-IDF feature extraction on active window
        tfidf = TfidfVectorizer(
            tokenizer=tokenizer,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.35,
            sublinear_tf=True
        )
        X_act = tfidf.fit_transform([m["stitched_text"] for m in active_msgs])
        nonzero_mask = np.asarray(X_act.sum(axis=1)).ravel() > 0
        if np.any(nonzero_mask):
            active_msgs = [active_msgs[i] for i in range(len(active_msgs)) if nonzero_mask[i]]
            X_act = X_act[nonzero_mask]

        X_hist = tfidf.transform([m["stitched_text"] for m in hist_msgs]) if hist_msgs else None

        # 2. Dimensionality reduction via TruncatedSVD
        n_comp = min(50, max(2, min(X_act.shape[1] - 1, X_act.shape[0] - 1)))
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        X_dense = normalize(svd.fit_transform(X_act))

        # 3. Agglomerative clustering with Ward linkage on L2-normalized vectors
        agg = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.90,
            metric="euclidean",
            linkage="ward"
        )
        labels = agg.fit_predict(X_dense)

        raw_groups = defaultdict(list)
        for idx, lbl in enumerate(labels):
            raw_groups[lbl].append(idx)

        feat_vocab = tfidf.get_feature_names_out()
        unigram_mask = np.array([" " not in w for w in feat_vocab])
        novelty_mask = np.array([
            1.0 if (" " not in w and w not in baseline_vocab) else 0.0
            for w in feat_vocab
        ])

        # 4. Anchor each micro-cluster by its top novel unigram (or top overall unigram)
        merged_groups = defaultdict(list)
        for lbl, idxs in raw_groups.items():
            if len(idxs) < 5:
                continue
            mean_tfidf = np.asarray(X_act[idxs].mean(axis=0)).ravel()
            uni_tfidf = mean_tfidf * unigram_mask
            nov_tfidf = mean_tfidf * novelty_mask
            if nov_tfidf.max() >= 0.08:
                key_uni = feat_vocab[np.argmax(nov_tfidf)]
            else:
                key_uni = feat_vocab[np.argmax(uni_tfidf)]
            merged_groups[key_uni].extend(idxs)

        # 5. Rescue nascent novel entities (1-4 day old emerging trends with >= 5 recent messages)
        if baseline_vocab:
            nov_token_idxs = defaultdict(list)
            for i, m in enumerate(active_msgs):
                for w in set(tokenizer(m["cleaned_text"])):
                    if w not in baseline_vocab and len(w) >= 2:
                        nov_token_idxs[w].append(i)
            for w, idxs in sorted(nov_token_idxs.items(), key=lambda x: -len(x[1])):
                if len(idxs) >= 5 and w not in merged_groups:
                    recent_cnt = sum(
                        1 for i in idxs
                        if (as_of_dt - datetime.strptime(active_msgs[i]["date"], "%Y-%m-%d")).days <= 4
                    )
                    if recent_cnt >= 5:
                        already_covered = any(
                            len(set(idxs) & set(g_idxs)) >= len(idxs) * 0.5
                            for g_w, g_idxs in merged_groups.items()
                            if g_w not in baseline_vocab
                        )
                        if not already_covered:
                            merged_groups[w] = idxs

        # 6. Sub-cluster absorption: merge fragmented sub-clusters into their parent topic
        # Guard: Never allow a baseline topic (e.g. '艾琳') to swallow a newly emerged novel topic (e.g. '铁锅').
        sorted_keys = sorted(merged_groups.keys(), key=lambda k: -len(merged_groups[k]))
        absorbed = set()
        final_groups = {}
        for i, k_big in enumerate(sorted_keys):
            if k_big in absorbed:
                continue
            big_idxs = set(merged_groups[k_big])
            big_is_baseline = k_big in baseline_vocab
            best_key = k_big
            for k_small in sorted_keys[i + 1:]:
                if k_small in absorbed:
                    continue
                small_is_baseline = k_small in baseline_vocab
                if big_is_baseline and not small_is_baseline:
                    continue
                small_idxs = merged_groups[k_small]
                hits_big_in_small = sum(1 for idx in small_idxs if best_key in active_msgs[idx]["cleaned_text"].lower())
                hits_small_in_big = sum(1 for idx in big_idxs if k_small in active_msgs[idx]["cleaned_text"].lower())
                if hits_big_in_small >= len(small_idxs) * 0.42 or hits_small_in_big >= len(big_idxs) * 0.42:
                    big_idxs.update(small_idxs)
                    absorbed.add(k_small)
                    # Keep whichever novel anchor appears in more messages of the merged cluster
                    if not small_is_baseline:
                        cnt_best = sum(1 for idx in big_idxs if best_key in active_msgs[idx]["cleaned_text"].lower())
                        cnt_small = sum(1 for idx in big_idxs if k_small in active_msgs[idx]["cleaned_text"].lower())
                        if cnt_small > cnt_best:
                            best_key = k_small
            final_groups[best_key] = sorted(big_idxs)

        # 7. Evaluate trajectory velocity and rank candidate clusters
        candidate_clusters = []
        for cluster_id, (key_uni, idxs) in enumerate(final_groups.items()):
            if len(idxs) < 5:
                continue
            c_msgs = [active_msgs[i] for i in idxs]

            tok_counts = Counter()
            for m in c_msgs:
                for w in set(tokenizer(m["cleaned_text"])):
                    tok_counts[w] += 1
            core_vocab_size = sum(1 for w, c in tok_counts.items() if c >= 3)

            c_vec = normalize(np.asarray(X_act[idxs].mean(axis=0)))
            mean_uni = c_vec.ravel() * unigram_mask
            top_uni_idxs = np.argsort(mean_uni)[-4:][::-1]
            top_kw = [feat_vocab[i] for i in top_uni_idxs if mean_uni[i] > 0]
            if key_uni not in top_kw:
                top_kw.insert(0, key_uni)
            else:
                top_kw.remove(key_uni)
                top_kw.insert(0, key_uni)

            # Historical projection across pre-window messages
            full_msgs = list(c_msgs)
            if X_hist is not None and X_hist.shape[0] > 0:
                sims = (X_hist @ c_vec.T).ravel()
                matched_hist = [
                    hist_msgs[i] for i in np.where(sims >= 0.30)[0]
                    if key_uni in hist_msgs[i]["cleaned_text"].lower()
                ]
                full_msgs = matched_hist + c_msgs

            metrics = analyze_cluster_velocity(
                cluster_msgs=full_msgs,
                as_of=as_of,
                versions=versions,
                account_stats=account_stats,
                active_msgs=c_msgs,
                data_start=data_start
            )

            if len(idxs) < 8 and metrics["status"] != "new":
                continue
            if core_vocab_size <= 4 and metrics["status"] not in ("not real", "new"):
                continue

            # Select 5 representative medoid messages (preferring messages containing key_uni)
            anchor_sub = [pos for pos, i in enumerate(idxs) if key_uni in active_msgs[i]["cleaned_text"].lower()]
            if len(anchor_sub) >= 5:
                sub_idxs = [idxs[pos] for pos in anchor_sub]
                sub_msgs = [c_msgs[pos] for pos in anchor_sub]
            else:
                sub_idxs = idxs
                sub_msgs = c_msgs

            vecs = X_dense[sub_idxs]
            centroid = np.mean(vecs, axis=0)
            dists = np.linalg.norm(vecs - centroid, axis=1)
            medoid_indices = np.argsort(dists)[:5]

            exemplar_msgs = [sub_msgs[i] for i in medoid_indices]
            voice_ids = [m["id"] for m in exemplar_msgs]
            sample_texts = [m["cleaned_text"] for m in exemplar_msgs]

            st = metrics["status"]
            status_priority = {
                "new": 105,
                "one-off": 90,
                "growing": 55,
                "fading": 42,
                "not real": 32
            }
            if corpus_span_days >= 14:
                if metrics.get("emerged_after_start"):
                    novelty_bonus = 55
                elif metrics.get("accel_ratio", 1.0) >= 1.30:
                    novelty_bonus = 15
                else:
                    novelty_bonus = -20
            else:
                novelty_bonus = 0

            ambient_penalty = -45 if (metrics.get("is_stationary_ambient") and st == "growing") else 0
            astroturf_bonus = 45 if metrics.get("is_astroturf") else 0

            momentum_score = (
                status_priority.get(st, 10)
                + novelty_bonus
                + ambient_penalty
                + astroturf_bonus
                + min(70.0, metrics.get("recent_weighted", 0) * 1.1)
                + metrics.get("num_platforms", 1) * 5
                + metrics.get("cascade_score", 0) * 10
                + math.log(max(1, len(c_msgs))) * 5
            )

            candidate_clusters.append({
                "cluster_id": cluster_id,
                "count": len(c_msgs),
                "status": st,
                "first_seen": metrics["first_seen"],
                "last_seen": metrics["last_seen"],
                "metrics": metrics,
                "voice_ids": voice_ids,
                "sample_texts": sample_texts,
                "exemplar_msgs": exemplar_msgs,
                "top_kw": top_kw,
                "c_vec": c_vec.ravel(),
                "momentum_score": momentum_score
            })

        # Rank candidate clusters by momentum score
        candidate_clusters.sort(key=lambda c: c["momentum_score"], reverse=True)

        # Select up to 8 semantically distinct trends with status diversity guardrails
        selected = []
        seen_statuses = Counter()

        for c in candidate_clusters:
            st = c["status"]
            is_dup = False
            for s in selected:
                sim = float(np.dot(c["c_vec"], s["c_vec"]))
                kw_overlap = len(set(c["top_kw"][:3]) & set(s["top_kw"][:3]))
                if (
                    c["top_kw"][0] == s["top_kw"][0]
                    or sim >= 0.14
                    or (kw_overlap >= 1 and st != "new")
                    or kw_overlap >= 2
                ):
                    is_dup = True
                    break
            if is_dup:
                continue

            if st == "not real" and seen_statuses[st] >= 1:
                continue
            if st == "growing" and seen_statuses[st] >= 4:
                continue
            if st == "new" and seen_statuses[st] >= 3:
                continue
            if st == "one-off" and seen_statuses[st] >= 1:
                continue
            if st == "fading" and seen_statuses[st] >= 2:
                continue

            selected.append(c)
            seen_statuses[st] += 1
            if len(selected) >= 8:
                break

        # Backfill non-duplicate clusters if fewer than 8 selected
        if len(selected) < 8:
            selected_ids = {c["cluster_id"] for c in selected}
            for c in candidate_clusters:
                if c["cluster_id"] in selected_ids:
                    continue
                if c["status"] == "not real" and seen_statuses["not real"] >= 1:
                    continue
                is_dup = any(
                    c["top_kw"][0] == s["top_kw"][0]
                    or float(np.dot(c["c_vec"], s["c_vec"])) >= 0.14
                    or len(set(c["top_kw"][:3]) & set(s["top_kw"][:3])) >= 1
                    for s in selected
                )
                if not is_dup:
                    selected.append(c)
                    seen_statuses[c["status"]] += 1
                    if len(selected) >= 8:
                        break

        return selected[:8]




