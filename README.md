# Diimsum Trend Hunter

**Entry Point:** `python run.py --input voices.jsonl --as-of YYYY-MM-DD --out trends.json`

## 1. How to Run

```bash
pip install -r requirements.txt
python run.py --input voices.jsonl --as-of 2026-08-25 --out trends.json
```
*(Optional)* Place `OPENAI_API_KEY=sk-...` in `keys.env` for `gpt-4o-mini` bilingual synthesis; otherwise the deterministic fallback generator in `src/synthesizer.py` is used automatically. All messages and `versions.csv` rows after `--as-of 23:59:59+08:00` are strictly dropped at ingestion (`src/ingest.py`).

---

## 2. How Messages Are Grouped Together (`src/preprocess.py`, `src/clusterer.py`)

1. **Normalization & Selective Reply Stitching:** Traditional Chinese (Bahamut/Discord) is converted to Simplified via `OpenCC`, homophones are normalized (`蚌埠住了` $\rightarrow$ `绷不住了`), and emoji macros (`<emoji:doge>`, `🤡`) are standardized. Short replies ($\le 15$ chars or agreement openers like `"确实"`, `"附议"`) are stitched with their `parent_id` text-except when the parent is generic studio bulletin boilerplate (`"详情见图"`), preventing official headers from merging unrelated replies.
2. **Dynamic Entity & Hashtag Discovery:** Without hardcoded game keywords, `register_dynamic_entities()` scans the time-gated corpus to (a) strip ubiquitous franchise-wide hashtags (`>40%` of active days, e.g. `#星环守望#`) so unrelated Weibo posts don't collapse together, and (b) dynamically register quoted proper nouns (`「...」`) and campaign hashtag suffixes (`#星环守望霜月#` $\rightarrow$ `霜月`) into `jieba` so new character/skin names aren't split into single characters.
3. **Novelty-Anchored Clustering & Historical Projection:** Active-window messages (14 days) are vectorized with sublinear $(1,2)$-gram TF-IDF, reduced via 50-D `TruncatedSVD` + L2 normalization, and grouped via Euclidean Ward `AgglomerativeClustering` ($d=0.90$). Micro-clusters are anchored by their top novel unigram (contrasted against the first 5 days of baseline chatter), nascent 1–4 day trends ($\ge 5$ recent mentions) are rescued, and fragmented sub-clusters (`公示` + `暗改`) merge via guarded overlap (baseline Day-1 topics like `艾琳` are blocked from swallowing novel slang like `铁锅`). Pre-window messages are projected onto cluster centroids (`X_hist @ c_vec.T >= 0.30`) to reconstruct the full timeline, using an onset threshold (`max(2, 0.08 * max_day_cnt)`) for noise-robust `first_seen` dates.

---

## 3. How the 5 Statuses Are Distinguished (`src/velocity.py`)

* **`not real`:** Flagged by three orthogonal checks: (1) **Coordinated Astroturfing** (`new_account_share >= 0.55` among accounts appearing after a 7-day warm-up with $\le 3$ total posts, plus `new_account_burst >= 6` distinct new accounts in a 60-minute window), (2) **Commercial RMT Spam** (`commercial_share >= 0.35` containing terms like `"代练"`, `"出号"`, `"破解版"`), or (3) **Bot Floods** (`top_author_share >= 0.40` or $\le 5$ unique users).
* **`one-off`:** Acute 48-hour spike (`win_max_cnt >= 16`, peak + adjacent day $\ge 38\%$ of active window volume, $\ge 3.5\times$ mean or $\ge 2.5\times$ near a `versions.csv` patch) followed by rapid decay ($\le 35\%$ of peak/day).
* **`new`:** First reached onset within $\le 4$ days of `--as-of` with $\ge 5$ organic messages in the last 72 hours.
* **`growing`:** Emerged after the baseline period and maintains $\ge 48\%$ of its rolling 7-day peak (`peak_7d`), or exhibits positive acceleration (`accel_ratio >= 1.18`), reinforced by cross-platform Shannon entropy and a **3-tier platform cascade score** (Tier 1 Instant Chat `Discord/QQ` $\rightarrow$ Tier 2 Forums/Reviews `TapTap/Steam/Bahamut` $\rightarrow$ Tier 3 Broadcast Social `Bilibili/Weibo`).
* **`fading`:** Peaked $\ge 5$ days ago and recent 7-day volume has dropped to $\le 46\%$ of its rolling 7-day peak (or $\le 42\%$ of the preceding 4-day window).

---

## 4. What Was Tried & Discarded

* **Naive `unique_user_ratio` & `any()` Spam Matching:** A finite core of ~75 regular users posting 1,000+ times over 60 days naturally has `unique_user_ratio < 0.10`, while the 563-post Weibo `"霜月"` astroturf campaign used 563 1-post throwaway accounts (`unique_user_ratio = 1.00`). Replacing `unique_user_ratio` with time-gated `new_account_share` + 60-minute burst detection and replacing `any()` with `commercial_share >= 0.35` eliminated both false positives and false negatives.
* **Unconditional Reply Stitching & Raw `jieba`:** Stitching `"详情见图"` studio bulletins and leaving `#星环守望#` in TF-IDF created synthetic mega-clusters, while default `jieba` split `「霜月」` into single characters; solved via selective stitching and dynamic entity registration.
* **Fixed-$K$ Clustering, External Embedding APIs & Stateful DBs:** $K$-Means forced greetings (`"早"`) into real topics; embedding APIs (`text-embedding-3-small`) risk 429 rate limits on 45k messages (our CPU pipeline + single batched `gpt-4o-mini` call runs in ~18s); stateful DBs break idempotency when `--as-of` dates are evaluated out of order.

---

## 5. State of 《星环守望》 by Late August 2026

By late August 2026, community sentiment is dominated by five storylines:
1. **Selena (`赛琳娜`) Powercreep (`growing`, `first_seen: 2026-07-22`):** Following the July 21 v2.0 patch, Selena dominates ranked PVP and Abyss (~1,900+ messages), sparking widespread calls for nerfs (`"老角色全进仓库了"`).
2. **Gacha Stealth-Nerf Crisis (`growing`, `first_seen: 2026-07-18 / 2026-07-23`):** Datamined pull tables on Bahamut (`拆包机率表`) escalated on July 23 into 1,300+ messages alleging hidden drop-rate nerfs (`"暗改爆率"`) and organizing consumer complaints (`"消协投诉"`).
3. **Eileen "Iron Pot" (`艾琳=铁锅`) Meme (`new` on July 7 $\rightarrow$ `growing`, `first_seen: 2026-07-06`):** An affectionate organic nickname (~1,000+ messages) reflecting healthy community lore.
4. **"Frost Moon" (`新皮肤「霜月」`) Astroturfing (`not real`, `first_seen: 2026-08-10`):** 563 Weibo posts from 1-post throwaway accounts (up to 24 new accounts/hour) promoting `#星环守望霜月#`, alongside background RMT/cracked-client spam (`"破解版"`).
5. **v2.1 Outage (`one-off`, `2026-07-31`) & Matchmaking Fix (`fading`):** The July 31 login queue/rollback spike (`"掉线回档"`) resolved within 48 hours, while early June PVP matchmaking complaints faded after the July 11 v1.6 fix, leaving bundle pricing (`"成长礼包68元，抢钱呢"`) as the main ongoing monetization grievance.

---

## 6. AI Assistant Usage

AI assistants (Claude / Antigravity) were used for pair-programming, scaffolding, and running diagnostic scripts across cutoffs. Core algorithmic designs-the 3-tier platform cascade (`Discord/QQ` $\rightarrow$ `TapTap/Steam/Bahamut` $\rightarrow$ `Bilibili/Weibo`), the 1-post-per-account astroturf burst detector, dynamic entity registration, and guarded novelty absorption-were directed and verified through empirical inspection of `voices.jsonl`. At runtime, `gpt-4o-mini` is called once in strict JSON mode only to format the top 8 deterministically discovered clusters into bilingual titles and grounded `why` summaries.
