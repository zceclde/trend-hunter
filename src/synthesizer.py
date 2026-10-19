import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load keys.env if present
load_dotenv("keys.env")

class TrendSynthesizer:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except Exception:
                self.client = None

    def _generate_fallback_trend(self, cluster: Dict[str, Any], as_of: str) -> Dict[str, Any]:
        """Deterministic high-quality fallback generator when LLM is unavailable."""
        samples = cluster.get("sample_texts", [])
        combined = " ".join(samples)
        status = cluster.get("status", "growing")
        first_seen = cluster.get("first_seen", as_of)
        voice_ids = cluster.get("voice_ids", [])
        metrics = cluster.get("metrics", {})
        plats = list(metrics.get("platform_counts", {}).keys())
        plat_str = ", ".join(plats[:3]) if plats else "multi-platform"

        top_kw = cluster.get("top_kw", [])
        kw_label = " / ".join(top_kw[:2]) if top_kw else (samples[0][:12] if samples else "社区热点")

        # Heuristic topic detection (game-agnostic)
        if metrics.get("is_astroturf"):
            title = f"「{kw_label}」疑似水军新号集中刷屏营销"
            title_en = f"Suspected coordinated astroturfing campaign around '{kw_label}'"
            category = "Bot & Astroturf Activity"
            why = metrics.get("reason", "Coordinated surge from newly registered accounts within tight hourly windows.")
            status = "not real"
        elif any(w in combined for w in ["管家", "欢迎来到", "群公告"]):
            title = "官方社群管家自动欢迎入群"
            title_en = "Community bot automated welcome messages"
            category = "Bot & Commercial Activity"
            why = "Automated greeting bot posts repeating across chat groups. Pure script activity with single-user origin."
            status = "not real"
        elif any(w in combined for w in ["代练", "代肝", "出号", "破解版"]):
            title = "代练代肝与黑产账号广告刷屏"
            title_en = "Account sales and boosting spam"
            category = "Bot & Commercial Activity"
            why = "Repeated commercial advertising with low user uniqueness and identical promotional copypasta."
            status = "not real"
        elif any(w in combined for w in ["暗改", "爆率", "概率", "拆包", "机率表"]):
            title = "玩家质疑抽卡概率暗改与出金率异常"
            title_en = "Players suspect stealth nerfing of gacha drop rates"
            category = "Gacha & Economy"
            why = f"Discussions started around {first_seen} with dataminers and players compiling pull logs. Strong cross-platform spread across {plat_str}."
        elif any(w in combined for w in ["外号", "什么梗", "昵称", "铁锅"]):
            title = f"社区热门新梗与角色外号「{kw_label}」流传"
            title_en = f"Community meme and nickname '{kw_label}' gains traction"
            category = "Community Culture & Lore"
            why = f"Organic community slang first surfacing on {first_seen} spreading across {plat_str} as veterans explain the inside joke to newcomers."
        elif any(w in combined for w in ["停服", "维护", "炸服", "登不上", "进不去"]):
            title = "例行更新与停服维护引发短暂进服排队"
            title_en = "Routine patch maintenance login queue and downtime"
            category = "Game Operations & Stability"
            why = "Acute spike coinciding with scheduled server downtime. Dropped sharply post-maintenance, representing a transient incident."
            status = "one-off"
        elif any(w in combined for w in ["新角色", "强度", "技能", "配队", "抽取建议"]):
            title = f"新版本角色与阵容搭配「{kw_label}」强度热议"
            title_en = f"New character meta and team composition discussion ({kw_label})"
            category = "Character Meta & Release"
            why = f"Surfaced on {first_seen} with volume building steadily across {plat_str} as players test builds and team synergies."
        elif any(w in combined for w in ["匹配", "段位", "排位", "车队", "单排"]):
            title = "PVP天梯匹配机制与实力不均抱怨"
            title_en = "PVP matchmaking mechanism and rank disparity complaints"
            category = "PVP & Game Balance"
            why = f"Consistent feedback across {plat_str} regarding matchmaking delays and Solo vs Premade team unfairness."
        elif any(w in combined for w in ["外挂", "自瞄", "锁头", "透视"]):
            title = "天梯高段位自瞄外挂争议"
            title_en = "Aimbot and cheating concerns in high-tier PVP"
            category = "Security & Fairness"
            why = f"Players reporting encounter rates with aimbot users, seeking official anti-cheat enforcement across {plat_str}."
        else:
            # Generic fallback from top keywords / sample
            first_clean = samples[0][:18] if samples else kw_label
            title = f"社区关于「{kw_label}」的集中讨论"
            title_en = f"Community discussion around '{first_clean}'"
            category = "General Gameplay & Mechanics"
            why = f"Identified organic discussion thread first seen on {first_seen} across {plat_str} ({metrics.get('total_msgs', 10)} messages). {metrics.get('reason', '')}"

        return {
            "title": title,
            "title_en": title_en,
            "category": category,
            "status": status,
            "first_seen": first_seen,
            "voice_ids": voice_ids,
            "why": why
        }

    def synthesize_trends(
        self,
        clusters: List[Dict[str, Any]],
        as_of: str,
        versions: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synthesizes top clusters into the required trends.json format using OpenAI
        structured outputs or the deterministic studio fallback.
        """
        if not clusters:
            return {"as_of": as_of, "trends": []}

        # Check if LLM client is available
        if self.client:
            try:
                # Prepare summary of clusters for LLM with grounded attribution data
                cluster_summaries = []
                for idx, c in enumerate(clusters):
                    metrics = c.get("metrics", {})
                    cluster_summaries.append({
                        "index": idx,
                        "status": c.get("status"),
                        "first_seen": c.get("first_seen"),
                        "total_cluster_messages": metrics.get("total_msgs", c.get("count", 0)),
                        "platforms": list(metrics.get("platform_counts", {}).keys()),
                        "platform_entropy": round(metrics.get("platform_entropy", 0), 2),
                        "cascade_score": round(metrics.get("cascade_score", 0), 2),
                        "recent_decay_weighted_volume": round(metrics.get("recent_weighted", 0), 1),
                        "older_decay_weighted_volume": round(metrics.get("older_weighted", 0), 1),
                        "unique_user_ratio": round(metrics.get("unique_user_ratio", 0), 2),
                        "new_account_share": round(metrics.get("new_account_share", 0), 2),
                        "new_account_hourly_burst": metrics.get("new_account_burst", 0),
                        "velocity_reason": metrics.get("reason", ""),
                        "sample_messages": c.get("sample_texts", [])[:5],
                        "voice_ids": c.get("voice_ids", [])[:5]
                    })

                version_context = ""
                if versions:
                    recent_events = [f"{v['date']} ({v['version']}): {v['notes']}" for v in versions[-3:]]
                    version_context = "Studio Patch Context:\n" + "\n".join(recent_events)

                prompt = f"""You are a Lead Game Data Scientist at Diimsum analyzing Chinese gaming community chatter as of {as_of}.
{version_context}

Analyze the following {len(cluster_summaries)} distinct community trend clusters.
For each cluster, generate:
- title: Natural, studio-grade Chinese title summarizing the specific phenomenon (e.g. "玩家质疑抽卡概率暗改", "角色艾琳社区新梗「铁锅」流传", "「霜月」新皮肤疑似水军刷屏推广").
- title_en: Concise, accurate English translation of title. Use consistent romanizations for proper nouns if present (艾琳=Eileen, 赛琳娜=Selena, 白夜=Baiye, 洛克=Locke, 米娅=Mia, 卡戎=Charon, 伊芙=Eve, 霜月=Frost Moon, 星环守望=Star Ring Watch).
- category: A dynamically invented category reflecting how players/developers think about the game (e.g. "Gacha & Economy", "Character Balance & Meta", "Community Culture & Lore", "Game Operations & Stability", "Bot & Astroturf Activity", "PVP & Fairness").
- why: A concise 2-sentence analytical justification in English. IMPORTANT: Ground your reasoning in specific phrases, slang, or hashtags quoted from sample_messages, AND cite concrete telemetry from the cluster metrics (e.g., total messages, platform spread/cascade, decay-weighted velocity, or for 'not real' trends, whether it is a single-account bot, RMT spam, or a coordinated astroturfing surge with high new_account_share / new_account_hourly_burst).

Clusters to analyze:
{json.dumps(cluster_summaries, ensure_ascii=False, indent=2)}

Respond ONLY with a valid JSON object matching this schema:
{{
  "trends": [
    {{
      "index": 0,
      "title": "String (Chinese)",
      "title_en": "String (English)",
      "category": "String",
      "why": "String"
    }}
  ]
}}
"""
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a professional Game Studio Data Scientist. Output only strict JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3
                )
                
                content = response.choices[0].message.content
                data = json.loads(content)
                parsed_trends = data.get("trends", [])

                final_trends = []
                for idx, c in enumerate(clusters):
                    # Match with LLM output
                    llm_match = next((t for t in parsed_trends if t.get("index") == idx), None)
                    if llm_match:
                        final_trends.append({
                            "title": llm_match.get("title", ""),
                            "title_en": llm_match.get("title_en", ""),
                            "category": llm_match.get("category", "Community Sentiment"),
                            "status": c.get("status"),
                            "first_seen": c.get("first_seen"),
                            "voice_ids": c.get("voice_ids", []),
                            "why": llm_match.get("why", c["metrics"].get("reason", ""))
                        })
                    else:
                        final_trends.append(self._generate_fallback_trend(c, as_of))

                return {"as_of": as_of, "trends": final_trends[:8]}

            except Exception as e:
                print(f"[Warning] LLM synthesis failed ({e}), using deterministic studio generator.")

        # Fallback path
        final_trends = [self._generate_fallback_trend(c, as_of) for c in clusters]
        return {"as_of": as_of, "trends": final_trends[:8]}

