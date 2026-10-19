# Diimsum Trend Hunter Solution: Early-Warning Community Radar

> **Candidate:** Luke  
> **Target:** PhD Data Scientist Fellowship in Hong Kong (Diimsum / Vivacity Limited)  
> **Entry Point CLI:** `python run.py --input voices.jsonl --as-of YYYY-MM-DD --out trends.json`

---

## 1. Executive Summary & Design Philosophy

Gaming community chatter across Asian platforms (QQ, Discord, Bilibili, Weibo, TapTap, Steam, Bahamut) is noisy, highly fragmented, and laden with irony, slang, and sarcasm.

Standard statistical clustering fails because:
1. **Volume $\neq$ Trend:** A routine server downtime or hotfix causes 150+ angry messages in 24 hours before vanishing (`one-off`). Throwaway accounts flood identical promotional hashtags or account-sales copypasta (`not real`). Meanwhile, 15 whispers on Bahamut and Discord about datamined drop-rate tables explode into a studio-threatening consumer crisis two weeks later (`growing`).
2. **Context Loss:** Fragmented replies like `"确实"` ("Indeed"), `"+1"`, or `"附议"` carry zero semantic value without their parent message—yet blindly stitching official studio bulletin headers (`"详情见图"`) pollutes player replies with announcement boilerplate.
3. **Keyword Rigidity:** Studios cannot anticipate what players will care about next month. Topics like *"Eileen = Iron Pot"* (`艾琳=铁锅`), *"Selena"* (`赛琳娜`), or *"Frost Moon"* (`霜月`) cannot rely on hardcoded game-specific keyword lists.

The Diimsum Trend Hunter pipeline is built on a **Game-Agnostic Hybrid Hierarchical Architecture**:
* **Strict Chronological Gating (`src/ingest.py`):** Zero future leakage; only messages and patch notes with `created_at <= --as-of 23:59:59+08:00` are loaded into memory.
* **Subcultural Normalization & Selective Context Stitching (`src/preprocess.py`):** Unifies Traditional/Simplified Chinese via OpenCC, normalizes homophonic slang (`"蚌埠住了"` $\rightarrow$ `"绷不住了"`), and stitches short child replies (`parent_id`) while filtering out generic studio bulletin headers.
* **Dynamic Corpus Entity Discovery & Novelty-Anchored Clustering (`src/clusterer.py`):** Automatically registers corpus-discovered quoted entities (`「...」`) and campaign hashtag suffixes into `jieba`, strips ubiquitous franchise-wide hashtags (`>40%` of active days), and contrasts the 14-day active window against an early baseline vocabulary to surface newly emerging storylines alongside macro trends.
* **Multi-Signal Trajectory & Authenticity Engine (`src/velocity.py`):** Combines noise-robust onset detection (`first_seen`), continuous exponential decay ($t_{1/2}=7\text{d}$), rolling 7-day peak comparison, a 3-tier platform cascade model, and account-history authenticity telemetry (detecting both 1-post-per-account astroturf bursts and single-author/RMT spam).
* **Grounded Studio-Grade Synthesis (`src/synthesizer.py`):** Generates bilingual titles, player-centric categories, and telemetry-backed justifications via `gpt-4o-mini` structured JSON output (with a 100% deterministic offline fallback).

---

## 2. Methodology: How the Solution Decides What Belongs Together

### 2.1 Selective Context Stitching (`parent_id`) & Subcultural Normalization
* **Reply Graph Reconstruction with Bulletin Guard:** Approximately 40% of community messages are replies. When a child reply is short ($\le 15$ characters) or begins with a conversational agreement marker (`"确实"`, `"+1"`, `"附议"`, `"笑死"`), we prepend the parent message: `【回复: [Parent Text]】 [Child Text]`. Crucially, if the parent message is an official studio announcement placeholder (`"详情见图"`, `"具体时间另行通知"`) or exceeds 80 characters, we only stitch if the child reply itself has no standalone content ($\le 6$ chars). This prevents official patch titles from artificially merging distinct player grievances.
* **Script Unification (OpenCC `t2s`):** Taiwanese and Hong Kong players on Bahamut (`哈啦板`) and Discord write in Traditional Chinese (`"外掛"`, `"機率表"`, `"伺服器"`), while Mainland players on QQ, TapTap, Bilibili, and Weibo write in Simplified Chinese (`"外挂"`, `"概率"`, `"服务器"`). We unify scripts to Simplified Chinese for vectorization while preserving raw message IDs (`voice_ids`).
* **Sarcasm & Emoji Macro Handling:** In `preprocess.py`, raw emoji codes (`<emoji:doge>`, `🤡`, `😅`) are normalized into standardized bracketed tags (`[doge]`, `[滑稽]`, `[小丑]`, `[流汗]`) so the LLM synthesizer can read the emotional tone, while `make_tokenizer()` strips bracketed macros and Bahamut floor markers (`#B47:5173781#`) before TF-IDF vectorization so cross-topic posts sharing `[流汗]` or `[大哭]` never cluster together by punctuation.

### 2.2 Dynamic Entity Registration & Two-Stage Novelty-Anchored Clustering
To remain 100% game-agnostic while achieving high precision on domain-specific proper nouns:
1. **Dynamic Entity & Global Hashtag Discovery (`register_dynamic_entities`):**
   * Scans the time-gated corpus for quoted proper nouns (`「...」` appearing $\ge 3$ times or in `versions.csv`) and hashtags (`#...#`).
   * Any hashtag appearing on $>40\%$ of active corpus days (e.g., `#星环守望#`) is flagged as a **global franchise tag** and stripped during tokenization so unrelated Weibo/Bilibili posts do not collapse into a single hashtag cluster.
   * Campaign-specific hashtags (`#星环守望霜月#`) have their global prefix stripped (`霜月`) and are dynamically injected into `jieba` (`freq=500000`) alongside quoted entities (`「霜月」`, `「赛琳娜」`), preventing `jieba` from splitting novel multi-character names into single characters.
2. **Baseline Vocabulary & Active Window Projection:**
   * When the observed corpus spans $\ge 10$ days, the pipeline builds a `baseline_vocab` from the first 5 days of community chatter ($t_0$ to $t_0 + 4\text{d}$).
   * Messages in the 14-day active window are vectorized via sublinear $(1, 2)$-gram `TfidfVectorizer` (`min_df=3`, `max_df=0.35`), projected to 50 latent dimensions via `TruncatedSVD`, L2-normalized, and grouped via Euclidean Ward `AgglomerativeClustering` ($d_{\text{threshold}} = 0.90$).
3. **Novelty-Boosted Anchor Consolidation & Nascent Trend Rescue:**
   * Each Ward micro-cluster is anchored to its highest-TF-IDF novel unigram (`w not in baseline_vocab` if $\text{TF-IDF} \ge 0.08$, else its top unigram).
   * **Nascent Entity Rescue:** Emerging topics that are only 1–4 days old (e.g., `铁锅` on July 7 with 15 messages, or `暗改` on July 24 with 8 messages) are rescued directly if they have $\ge 5$ mentions in the past 4 days and are not yet covered by a novel cluster.
   * **Guarded Bidirectional Sub-Cluster Absorption:** Fragmented sub-clusters sharing $\ge 42\%$ lexical overlap with a larger topic (e.g., `公示` and `暗改`, both discussing stealth gacha rate nerfs) are merged into a unified storyline, subject to a strict guard: **a baseline Day-1 topic (`k_big in baseline_vocab`, e.g., `艾琳`) is never allowed to absorb a newly emerged novel topic (`k_small not in baseline_vocab`, e.g., `铁锅`)**.
4. **Historical Trajectory Projection & Noise-Robust `first_seen`:**
   * Pre-window historical messages are projected onto each active cluster centroid (`X_hist @ c_vec.T >= 0.30` plus anchor verification) to reconstruct the full 90-day time series.
   * Rather than taking `min(dates)` blindly—where a single coincidental baseline mention on June 1 would corrupt `first_seen`—we compute an onset threshold (`max(2, int(0.08 * max_day_cnt))` for sustained topics, or `max(3, int(0.20 * win_max_cnt))` for acute spikes) to pinpoint the exact date the phenomenon ignited.

---

## 3. The 5 Trend Lifecycles: Mathematical & Heuristic Signatures

For each discovered cluster, `analyze_cluster_velocity()` in [`src/velocity.py`](file:///c:/Users/ldeme/OneDrive/Desktop/New%20folder/trend-hunter/src/velocity.py) evaluates authenticity telemetry, spike concentration, rolling 7-day peak decay, and cross-platform cascade flow:

| Status | Mathematical & Behavioral Signature | Real-World Community Manifestation |
| :--- | :--- | :--- |
| **`not real`** | Triggered by any of three orthogonal authenticity detectors:<br>1. **Coordinated Astroturfing:** `new_account_share >= 0.55` (accounts first seen after a 7-day warm-up with $\le 3$ lifetime posts) AND `new_account_burst >= 6` distinct new accounts within a 60-minute sliding window.<br>2. **Commercial RMT Spam:** `commercial_share >= 0.35` of messages contain boosting/account-trading terms (`"代练"`, `"出号"`, `"破解版"`).<br>3. **Bot Flood:** `top_author_share >= 0.40` or $\le 5$ unique users across $\ge 10$ messages. | Coordinated Weibo marketing campaigns (`新皮肤「霜月」` with 563 1-post accounts), illicit RMT/cracked-client spam, or automated group-welcome bots. |
| **`one-off`** | • Active-window peak day $\ge 16$ messages ($\ge 3.5\times$ window mean, or $\ge 2.5\times$ within $\pm 1\text{d}$ of a `versions.csv` patch).<br>• **48h Concentration:** Peak day + adjacent day account for $\ge 38\%$ of active window volume (`two_day_conc >= 0.38`).<br>• **Rapid Post-Incident Decay:** Post-peak daily average drops to $\le 35\%$ of peak day volume. | Emergency server outages, login queues (`排队12000人`), and disconnect rollbacks (`掉线回档`) during the July 31 v2.1 update. |
| **`new`** | • Ignited within $\le 4$ days of `--as-of` (`days_since_first <= 4`) with $\ge 5$ messages in the last 72 hours.<br>• Passes authenticity checks (organic user base). | Day-2 emergence of the *"Eileen = Iron Pot"* meme (`2026-07-07`) or initial eruption of the gacha stealth-nerf allegations (`2026-07-24`). |
| **`growing`** | • Emerged after the baseline period with recent 7-day volume maintaining $\ge 48\%$ of its rolling 7-day peak (`peak_7d`), OR accelerating over prior weeks (`accel_ratio >= 1.18`).<br>• High **Platform Entropy** ($H = -\sum p_i \ln p_i$) and positive exponential decay velocity ($t_{1/2} = 7\text{d}$). | Version 2.0 Selena (`赛琳娜`) balance backlash, sustained gacha rate-nerf outrage (`暗改爆率`), and viral community lore (`铁锅`). |
| **`fading`** | • Peaked $\ge 5$ days before `--as-of` (`days_since_peak >= 5`) and recent 7-day volume has fallen to $\le 46\%$ of its rolling 7-day peak (`peak_7d`) or $\le 42\%$ of the prior 4-day window (or $<78\%$ of `peak_7d` for baseline topics). | Resolved issues (e.g., PVP matchmaking complaints fading after the July 11 v1.6 patch fix) or cooling post-update crash reports. |

### 3.1 Directed Platform Propagation Cascade Model
Rather than treating platforms as an unordered bag of counts, [`src/velocity.py`](file:///c:/Users/ldeme/OneDrive/Desktop/New%20folder/trend-hunter/src/velocity.py) models the **Information Flow Topology** across three empirical tiers of Asian gaming communities:
* **Tier 1 — Ground Zero / Instant Peer Chat (Discord, QQ):** When players encounter a bug, stealth nerf, or server outage, they immediately cross-check with peers in real-time chat channels (`#bug-report`, `#綜合聊天`, QQ groups).
* **Tier 2 — Community Forum & Review Hubs (TapTap, Steam, Bahamut):** Verified grievances and meta debates crystallize into structured reviews, datamining threads, and forum guides (TapTap reviews, Steam discussions, Bahamut `哈啦板`).
* **Tier 3 — Viral Public Broadcast (Bilibili, Weibo):** Major controversies and sticky memes erupt into creator videos, trending hashtags (`#星环守望#`), and public PR pressure.

---

## 4. Failed Experiments, Engineering Tradeoffs & Discarded Ideas

During iterative testing against `voices.jsonl`, several initial hypotheses failed under empirical inspection and were redesigned:

1. **The "1-Post-Per-Account" Astroturfing Paradox vs. Finite Community Pool:**
   * *Initial Approach:* Flagged clusters as `not real` solely when `unique_user_ratio < 0.22` OR when `any()` message in the cluster contained a spam keyword (`"代练"`, `"破解版"`).
   * *Why it failed:* Two opposite failure modes occurred simultaneously:
     1. Organic communities have a finite core of active regulars (~75 named accounts in `voices.jsonl`). Over a 60-day storyline like `赛琳娜` (1,900+ messages) or `铁锅` (1,000+ messages), each regular posts 10–25 times, driving `unique_user_ratio` down to `0.05–0.12` and falsely flagging major organic trends as bots—especially if a single stray spam reply happened to land in the cluster via `any()`.
     2. Conversely, the August 10–20 `"新皮肤「霜月」"` (`#星环守望霜月#`) campaign consisted of **563 Weibo posts from 563 distinct throwaway accounts**, each posting exactly once! Its `unique_user_ratio` was `1.00`, completely bypassing traditional user-uniqueness filters.
   * *Solution:* Replaced `any()` keyword tainting with proportional `commercial_share >= 0.35`, restricted low-uniqueness bot checks to high single-author dominance (`top_author_share >= 0.40`) or tiny user pools (`len(user_counts) < 15`), and built a **time-gated account-history & 60-minute sliding burst detector** (`build_account_stats` + `compute_authenticity_signals`). This catches 1-post-per-account astroturfing (`91–100%` new-account share, up to 24 new accounts/hour) with zero false positives on large organic threads.
2. **Official Studio Bulletin Reply-Stitching Pollution:**
   * *Initial Approach:* Unconditionally prepended the parent message text (`【回复: ...】`) to every reply with `parent_id`.
   * *Why it failed:* Official studio announcements (`is_official=true`) frequently end with generic boilerplate like `"详情见图"` ("See image for details") or `"具体时间另行通知"`. When hundreds of players replied to different studio announcements with self-contained feedback, unconditional stitching injected `"详情见图"` into all of their TF-IDF vectors, creating a synthetic cluster of unrelated replies.
   * *Solution:* Restricted context stitching in [`src/preprocess.py`](file:///c:/Users/ldeme/OneDrive/Desktop/New%20folder/trend-hunter/src/preprocess.py) to genuinely context-dependent replies ($\le 15$ chars or conversational openers) and blocked generic studio bulletin headers from being stitched onto self-contained comments.
3. **`jieba` Out-of-Vocabulary (OOV) Splitting & Franchise Hashtag Collapse:**
   * *Initial Approach:* Used default `jieba.cut()` directly on cleaned text.
   * *Why it failed:* `jieba`'s default dictionary does not know newly released skin names like `「霜月」`, splitting them into single characters `'霜'` and `'月'` (which are then discarded by `len(w) > 1` token filtering). Meanwhile, the ubiquitous `#星环守望#` hashtag caused unrelated Weibo posts to cluster together.
   * *Solution:* Added `register_dynamic_entities()` to automatically extract quoted entities (`「...」`) and campaign hashtag suffixes from the corpus into `jieba` while stripping ubiquitous franchise tags (`>40%` of active days).
4. **Failure of Fixed-$K$ Clustering & Cosine Zero-Vector Traps:**
   * Fixed $K$-Means forced unrelated greetings (`"早"`, `"打卡"`) into substantive gameplay clusters, while raw Cosine distance on sparse TF-IDF matrices raised zero-vector division errors on short messages. Projecting onto 50-D L2-normalized `TruncatedSVD` components and applying Euclidean Ward linkage ($d^2 = 2 - 2\cos\theta$) produced clean, density-adaptive clusters.
5. **External Embedding APIs & Heavy Local GPU Models (Discarded):**
   * Calling external embedding APIs (`text-embedding-3-small`) on 10,000–45,000 messages per run risks HTTP 429 rate limits and multi-minute network latency, while multi-gigabyte local PyTorch models risk out-of-memory failures in CPU grading containers. Our local CPU vectorization + clustering pipeline + single batched `gpt-4o-mini` synthesis call executes end-to-end in **~15–20 seconds**.
6. **Stateful Streaming Databases (Discarded):**
   * Persisting cluster centroids across runs violates the stateless CLI requirement (`run.py --as-of YYYY-MM-DD`) when evaluators test historical dates out of order. Reconstructing historical trajectories on-the-fly via `X_hist @ c_vec.T` guarantees 100% idempotency.

---

## 5. Studio Executive Briefing: The State of 《星环守望》 by Late August 2026

**To:** Executive Producer & Operations Director, 《星环守望》 (*Star Ring Watch*)  
**From:** Lead Data Scientist, Community Intelligence (Diimsum)  
**Date:** Late August 2026

### The Strategic Picture
Over the 90-day observation window from June 1 to August 30, 2026, 《星环守望》 evolved from a healthy launch community debating PVP matchmaking and story chapters into a **dual-front crisis of character powercreep and gacha drop-rate distrust**, complicated by an inauthentic Weibo marketing campaign in mid-August.

```
[June 2026]            [July 6]               [July 11 - v1.6]     [July 21 - v2.0]        [July 31 - v2.1]     [Aug 10 - Aug 25]
PVP matchmaking &      "Eileen = Iron Pot"    Matchmaking patched  Selena Banner launches  Server outage &      "Frost Moon" astroturf
monetization grumbles  Organic meme ignites   PVP complaints fade  & Stealth Nerf Crisis   rollback (one-off)   & consumer complaints
```

### Key Community Developments Discovered by the Radar:
1. **Version 2.0 Character Powercreep: Selena (`赛琳娜`) Balance Backlash (`growing`, `first_seen: 2026-07-22`):**
   * Following the Version 2.0 update (July 21), the new character **Selena (`赛琳娜`)** triggered the largest sustained discussion in the dataset (**1,937 messages** by August 25 across all 7 platforms). Players across Discord, TapTap, and Bilibili report that Selena completely invalidates older characters in both Abyss and PVP (*"排位十把九把对面有赛琳娜"*, *"老角色全进仓库了"*, *"赛琳娜什么时候削？平衡性呢"*).
2. **The Existential Trust Crisis: Gacha Drop-Rate Stealth-Nerf Allegations (`new` on July 24 $\rightarrow$ `growing`, `first_seen: 2026-07-18 / 2026-07-23`):**
   * Starting with Bahamut datamining posts on July 18 (`"有人拆包看了机率表吗"`) and exploding across all platforms on **July 23** (**1,317+ messages** by August 25), players began aggregating pull histories alleging that 5-star drop rates and soft pity were stealthily nerfed in v2.0 (*"实锤了，暗改爆率！公示的概率根本对不上"*, *"180抽没出金"*). By mid-August, discourse escalated into organized consumer protection complaints (`"消协投诉走起"`).
3. **Organic Community Lore: Eileen's "Iron Pot" (`艾琳=铁锅`) Meme (`new` on July 7 $\rightarrow$ `growing`, `first_seen: 2026-07-06`):**
   * Igniting on **July 6** (**1,011 messages** by August 25), players affectionately dubbed the shield-bearing character Eileen *"铁锅"* (Iron Pot / Wok). Our radar catches this trend as `[new]` on July 7 (`first_seen: 2026-07-06`) when veterans begin onboarding newcomers (*"为什么大家都叫艾琳铁锅啊"*, *"艾琳=铁锅，新人别懵"*), and tracks its growth into the community's defining piece of organic folklore.
4. **Coordinated Astroturfing Detection: "Frost Moon" (`新皮肤「霜月」`) Campaign (`not real`, `first_seen: 2026-08-10`):**
   * Between August 10 and August 20, **563 Weibo posts** promoted `#星环守望霜月#` (*"新皮肤「霜月」也太美了吧！必须拿下"*). While naive unique-user metrics failed because every post came from a different username, our authenticity engine flagged that **91–100% of authors were brand-new 1-post accounts** arriving in coordinated bursts of up to **24 new accounts per hour**, isolating the campaign as `not real` alongside background RMT/cracked-client spam (`"破解版"`, `"代练"`).
5. **Operational Incidents vs. Resolved Fixes (`one-off` & `fading`):**
   * **July 31 v2.1 Server Outage & Rollback (`one-off`, `first_seen: 2026-07-31`):** Extended maintenance, 12,000-player login queues, and character rollbacks (`"掉线回档了，刚抽的角色没了"`) caused an acute 48-hour spike on July 31–August 1 before decaying rapidly once servers stabilized.
   * **PVP Matchmaking & Optimization Fixes (`fading`):** Heavy June complaints over solo-vs-premade PVP matchmaking (`"单排总被匹配到车队"`) peaked in early July and transitioned to `fading` after the July 11 v1.6 matchmaking patch (`"匹配机制改了之后舒服多了"`). Meanwhile, ongoing complaints over bundle pricing (`"成长礼包68元，抢钱呢"`) and numerical inflation (`"数值膨胀太快"`) remain a persistent monetization headwind.

### Actionable Studio Recommendations:
1. **Address the Gacha Rate Controversy Transparently:** Publish audited server-side pull distribution logs for the v2.0 banner immediately; unaddressed `"暗改爆率"` accusations and `"消协投诉"` threads pose an immediate regulatory and retention risk.
2. **Rebalance PVP Around Selena via Counter-Play Rather Than Direct Nerfs:** Buff older roster counters or introduce ban/pick protection in ranked PVP so F2P/veteran rosters aren't obsoleted by Selena.
3. **Terminate Third-Party Weibo Astroturf Vendors & Embrace Organic Lore:** Halt the synthetic `#星环守望霜月#` bot promotion (which risks community ridicule if exposed) and instead lean into organic community attachment around Eileen's *"铁锅"* meme with official in-game cosmetics or stickers.

---

## 6. How to Run

### Installation
```bash
pip install -r requirements.txt
```
*(Optional)* Place your OpenAI API key in `keys.env` (`OPENAI_API_KEY=sk-...`) in the project root. If no API key is present or network access is unavailable, the pipeline automatically falls back to its deterministic studio synthesis engine.

### Execution
Run the CLI specifying any historical cut-off date:
```bash
python run.py --input voices.jsonl --as-of 2026-08-25 --out trends.json
```
Options:
* `--input`: Path to `voices.jsonl` (default: `voices.jsonl`)
* `--as-of`: Evaluation cutoff date in `YYYY-MM-DD` format (required; strictly filters to `<= YYYY-MM-DD 23:59:59+08:00`)
* `--out`: Output path for JSON results (default: `trends.json`)
* `--versions`: Studio patch calendar (default: `versions.csv`)

---

## 7. AI Assistant Usage Disclosure

In accordance with the challenge guidelines on transparency around AI tool usage:
* **Development & Debugging Collaboration:** AI coding assistants (Claude / Antigravity) were used during development as a pair-programming partner to scaffold the modular CLI structure (`run.py`, `src/*.py`), draft initial docstrings, and run diagnostic scripts across the corpus.
* **Human-Directed Architecture & Empirical Verification:** All core algorithmic decisions and iterative fixes were driven by empirical inspection of the dataset across multiple `--as-of` cutoffs (`2026-06-03`, `2026-07-07`, `2026-07-24`, `2026-08-01`, `2026-08-15`, `2026-08-25`):
  * Designing the **3-tier platform cascade topology** (Tier 1 Instant Chat `Discord/QQ` $\rightarrow$ Tier 2 Forums/Reviews `TapTap/Steam/Bahamut` $\rightarrow$ Tier 3 Broadcast Social `Bilibili/Weibo`) based on how gamers actually report bugs vs. amplify controversies.
  * Diagnosing why standard `unique_user_ratio` failed on the 563-account Weibo `"霜月"` astroturfing campaign (where each throwaway account posted once) and designing the **time-gated account-history & 60-minute burst detector**.
  * Discovering and fixing **official bulletin reply-stitching pollution** (`"详情见图"`), **`jieba` OOV splitting** on novel quoted/hashtag entities, and **baseline-vs-novel sub-cluster absorption** so emerging 2-day-old trends (`铁锅` on July 7) are never swallowed by Day-1 character chatter (`艾琳`).
* **Runtime LLM Usage (`src/synthesizer.py`):** At runtime, the pipeline makes a **single batched call** to `gpt-4o-mini` (`temperature=0.3`, strict JSON mode) passing only the top 8 discovered cluster summaries, telemetry metrics, and 5 medoid exemplar quotes (`voice_ids`) to synthesize natural bilingual titles (`title`, `title_en`), dynamic `category` labels, and grounded `why` explanations. All clustering, temporal gating, `first_seen` onset calculation, and `status` classification are 100% deterministic Python math prior to the LLM call, and a complete offline fallback (`_generate_fallback_trend`) ensures the CLI runs even without API access.
