import math
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple

# Platform cascade tiers:
# Trends typically ignite in real-time instant chat groups (Discord/QQ),
# crystallize into community forum reviews and technical boards (TapTap/Steam/Bahamut),
# and finally erupt into public broadcast social media (Bilibili/Weibo).
PLATFORM_TIERS = {
    "Discord": 1,   # Tier 1: Instant peer-to-peer chat & immediate bug-reports (ground zero)
    "QQ": 1,        # Tier 1: Fast real-time player group discussions
    "TapTap": 2,    # Tier 2: Structured game reviews and community complaints
    "Steam": 2,     # Tier 2: Technical reviews and platform discussions
    "Bahamut": 2,   # Tier 2: Long-form community forum threads (哈啦板)
    "Bilibili": 3,  # Tier 3: Creator videos, guides, and amplified public comment sections
    "Weibo": 3,     # Tier 3: Macro viral social broadcast and studio public PR pressure
}

def compute_platform_entropy(platform_counts: Dict[str, int]) -> float:
    """Computes Shannon entropy across platforms: H = -sum(p * ln(p))"""
    total = sum(platform_counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for cnt in platform_counts.values():
        if cnt > 0:
            p = cnt / total
            entropy -= p * math.log(p)
    return entropy

def compute_cascade_score(cluster_msgs: List[Dict[str, Any]]) -> float:
    """
    Measures whether a trend follows a healthy information cascade:
    originating on review/bug channels (Tier 1) before spreading to
    mass social channels (Tier 3).
    
    Returns a cascade score from 0.0 (no cascade) to 1.0 (textbook cascade).
    """
    if not cluster_msgs:
        return 0.0
        
    # Find the earliest date each tier was observed
    tier_first_seen = {}
    for m in cluster_msgs:
        platform = m.get("platform", "Unknown")
        tier = PLATFORM_TIERS.get(platform, 2)
        date = m.get("date", "9999-99-99")
        if tier not in tier_first_seen or date < tier_first_seen[tier]:
            tier_first_seen[tier] = date
    
    if len(tier_first_seen) <= 1:
        return 0.0
        
    # Check if Tier 1 appeared before or same day as Tier 2/3
    tier1_date = tier_first_seen.get(1, "9999-99-99")
    tier2_date = tier_first_seen.get(2, "9999-99-99")
    tier3_date = tier_first_seen.get(3, "9999-99-99")
    
    score = 0.0
    # Reward: Tier 1 (instant peer chat) appeared first or same day as Tier 2 (reviews/forums)
    if tier1_date <= tier2_date:
        score += 0.4
    # Reward: Tier 2 (reviews/forums) appeared before or same day as Tier 3 (broadcast social media)
    if tier2_date <= tier3_date:
        score += 0.3
    # Reward: at least 3 tiers are represented (broad organic cascade)
    if len(tier_first_seen) >= 3:
        score += 0.3
        
    return score

def compute_decay_weighted_volume(
    daily_counts: Dict[str, int],
    as_of: str,
    half_life_days: float = 7.0
) -> float:
    """
    Computes time-weighted message volume using continuous exponential decay:
        w(Δt) = e^(-λ * Δt)   where λ = ln(2) / half_life
    
    Recent messages contribute more than older ones, but long-tail grievances
    (e.g. 40-day stamina fatigue buildup) retain cumulative context rather
    than being hard-cut by a boxcar window.
    """
    as_of_dt = datetime.strptime(as_of, "%Y-%m-%d")
    decay_lambda = math.log(2) / half_life_days
    
    weighted_volume = 0.0
    for date_str, count in daily_counts.items():
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        delta_days = (as_of_dt - dt).days
        if delta_days < 0:
            continue
        weight = math.exp(-decay_lambda * delta_days)
        weighted_volume += count * weight
        
    return weighted_volume

# Generic commercial-spam vocabulary shared across Chinese game communities
# (boosting services, account sales, cracked clients, "DM me" solicitations).
# Deliberately game-agnostic: none of these terms refer to a specific title.
COMMERCIAL_SPAM_TERMS = [
    "代练", "代肝", "代打", "出号", "收号", "全图鉴", "破解版", "低价", "加群", "私聊",
]

# An account only counts as "new" if we have been observing the community for
# at least this many days before it first posted. Otherwise every account seen
# at the start of the data window would look brand-new.
NEW_ACCOUNT_OBSERVATION_DAYS = 7
NEW_ACCOUNT_RECENCY_DAYS = 14
NEW_ACCOUNT_MAX_POSTS = 3

def build_account_stats(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Builds per-account history from the time-gated message set only, so that no
    information from after --as-of can leak into authenticity checks.
    """
    first_seen = {}
    post_counts = Counter()
    data_start = None
    for m in messages:
        user = m.get("user_name") or "anon"
        date = m.get("date", "")
        if not date:
            continue
        if data_start is None or date < data_start:
            data_start = date
        if user not in first_seen or date < first_seen[user]:
            first_seen[user] = date
        post_counts[user] += 1
    return {"first_seen": first_seen, "post_counts": post_counts, "data_start": data_start}

def is_new_account(user: str, account_stats: Dict[str, Any], as_of_dt: datetime) -> bool:
    """A low-history account that first appeared recently, after an observation warm-up."""
    if not account_stats or not account_stats.get("data_start"):
        return False
    first = account_stats["first_seen"].get(user)
    if not first:
        return False
    first_dt = datetime.strptime(first, "%Y-%m-%d")
    data_start_dt = datetime.strptime(account_stats["data_start"], "%Y-%m-%d")
    if (first_dt - data_start_dt).days < NEW_ACCOUNT_OBSERVATION_DAYS:
        return False
    if (as_of_dt - first_dt).days > NEW_ACCOUNT_RECENCY_DAYS:
        return False
    return account_stats["post_counts"].get(user, 0) <= NEW_ACCOUNT_MAX_POSTS

def compute_authenticity_signals(
    cluster_msgs: List[Dict[str, Any]],
    account_stats: Dict[str, Any],
    as_of_dt: datetime
) -> Dict[str, float]:
    """
    Signals that separate coordinated or automated activity from organic chatter.

    - new_account_share: share of messages posted by brand-new, low-history accounts.
      Organic topics are driven by established community members; astroturf
      campaigns are driven by throwaway accounts.
    - new_account_burst: most distinct new accounts posting into the cluster
      within any 60-minute window.
    - top_author_share: share of messages from the single most active author
      (automated bots such as group-management scripts).
    - commercial_share: share of messages containing commercial-spam vocabulary.
      Uses a share rather than "any message matches", so a single stray spam
      reply cannot taint a genuine discussion.
    """
    total = len(cluster_msgs)
    if total == 0:
        return {"new_account_share": 0.0, "new_account_burst": 0,
                "top_author_share": 0.0, "commercial_share": 0.0}

    users = [m.get("user_name") or "anon" for m in cluster_msgs]
    top_author_share = Counter(users).most_common(1)[0][1] / total

    new_msgs = [m for m, u in zip(cluster_msgs, users) if is_new_account(u, account_stats, as_of_dt)]
    new_account_share = len(new_msgs) / total

    # Sliding 60-minute window over new-account posts, counting distinct accounts
    new_account_burst = 0
    if new_msgs:
        stamped = []
        for m in new_msgs:
            try:
                stamped.append((datetime.fromisoformat(m["created_at"]), m.get("user_name") or "anon"))
            except (ValueError, KeyError):
                continue
        stamped.sort(key=lambda x: x[0])
        left = 0
        for right in range(len(stamped)):
            while (stamped[right][0] - stamped[left][0]).total_seconds() > 3600:
                left += 1
            new_account_burst = max(new_account_burst, len({u for _, u in stamped[left:right + 1]}))

    commercial_hits = sum(
        1 for m in cluster_msgs
        if any(term in m.get("cleaned_text", "") for term in COMMERCIAL_SPAM_TERMS)
    )
    commercial_share = commercial_hits / total

    return {
        "new_account_share": new_account_share,
        "new_account_burst": new_account_burst,
        "top_author_share": top_author_share,
        "commercial_share": commercial_share,
    }

def analyze_cluster_velocity(
    cluster_msgs: List[Dict[str, Any]],
    as_of: str,
    versions: List[Dict[str, Any]] = None,
    account_stats: Dict[str, Any] = None,
    active_msgs: List[Dict[str, Any]] = None,
    data_start: str = None
) -> Dict[str, Any]:
    """
    Computes trajectory metrics for a cluster of messages up to as_of date:
    - Noise-robust onset detection for first_seen (separating true topic emergence from stray baseline mentions)
    - Acute 48-hour incident spike concentration vs post-peak decay for 'one-off' detection
    - Rolling 7-day peak comparison for 'fading' vs 'growing' trajectories
    - Exponentially-decayed volume (half-life = 7 days)
    - User uniqueness ratio and account-history authenticity signals (astroturfing, RMT spam, bots)
    - Cross-platform dispersion entropy & platform cascade score (instant chat -> forums -> broadcast social)
    """
    as_of_dt = datetime.strptime(as_of, "%Y-%m-%d")
    if not cluster_msgs:
        return {"status": "fading", "velocity": 0, "first_seen": as_of}

    c_msgs = active_msgs if active_msgs else cluster_msgs
    if not data_start:
        if account_stats and account_stats.get("data_start"):
            data_start = account_stats["data_start"]
        else:
            data_start = min(m["date"] for m in cluster_msgs)
    data_start_dt = datetime.strptime(data_start, "%Y-%m-%d")

    # 1. Active-window acute spike check (for 24-48h transient server/patch incidents)
    win_dc = Counter(m["date"] for m in c_msgs)
    win_max_cnt = max(win_dc.values())
    win_peak_date = max(win_dc, key=win_dc.get)
    win_peak_dt = datetime.strptime(win_peak_date, "%Y-%m-%d")
    win_days_since_peak = (as_of_dt - win_peak_dt).days
    win_mean_cnt = len(c_msgs) / max(1, len(win_dc))

    adj_cnt = max(
        win_dc[(win_peak_dt - timedelta(days=1)).strftime("%Y-%m-%d")],
        win_dc[(win_peak_dt + timedelta(days=1)).strftime("%Y-%m-%d")]
    )
    two_day_conc = (win_max_cnt + adj_cnt) / max(1, len(c_msgs))

    post_peak_msgs = sum(
        win_dc[(as_of_dt - timedelta(days=d)).strftime("%Y-%m-%d")]
        for d in range(win_days_since_peak)
    ) if win_days_since_peak >= 1 else 0
    post_peak_daily_avg = post_peak_msgs / max(1, win_days_since_peak)

    near_studio_event = False
    if versions:
        for v in versions:
            v_date = v.get("date", "")
            if v_date and abs((datetime.strptime(v_date, "%Y-%m-%d") - win_peak_dt).days) <= 1:
                near_studio_event = True
                break

    is_acute_spike = (
        win_max_cnt >= 16
        and win_days_since_peak >= 1
        and two_day_conc >= 0.38
        and (win_max_cnt >= 3.5 * win_mean_cnt or (near_studio_event and win_max_cnt >= 2.5 * win_mean_cnt))
        and post_peak_daily_avg <= win_max_cnt * 0.35
    )

    # 2. Noise-robust onset detection for first_seen
    dc = Counter(m["date"] for m in cluster_msgs)
    max_day_cnt = max(dc.values())
    if is_acute_spike:
        spike_thresh = max(3, int(0.20 * win_max_cnt))
        spike_dates = [d for d, c in win_dc.items() if c >= spike_thresh and d <= win_peak_date]
        first_seen = min(spike_dates) if spike_dates else win_peak_date
    else:
        onset_thresh = max(2, int(0.08 * max_day_cnt))
        onset_dates = [d for d, c in dc.items() if c >= onset_thresh]
        first_seen = min(onset_dates) if onset_dates else min(dc.keys())

    trend_msgs = [m for m in cluster_msgs if m["date"] >= first_seen]
    if len(trend_msgs) < 5:
        trend_msgs = cluster_msgs

    total_msgs = len(trend_msgs)
    t_dc = Counter(m["date"] for m in trend_msgs)
    last_seen = max(t_dc.keys())
    first_seen_dt = datetime.strptime(first_seen, "%Y-%m-%d")
    days_since_first = (as_of_dt - first_seen_dt).days
    emerged_after_start = (first_seen_dt - data_start_dt).days >= 5

    peak_date = max(t_dc, key=t_dc.get)
    peak_dt = datetime.strptime(peak_date, "%Y-%m-%d")
    days_since_peak = (as_of_dt - peak_dt).days

    # Rolling window counts
    d_0_to_2 = sum(t_dc[(as_of_dt - timedelta(days=d)).strftime("%Y-%m-%d")] for d in range(3))
    d_3_to_6 = sum(t_dc[(as_of_dt - timedelta(days=d)).strftime("%Y-%m-%d")] for d in range(3, 7))
    d_0_to_6 = d_0_to_2 + d_3_to_6
    d_7_to_13 = sum(t_dc[(as_of_dt - timedelta(days=d)).strftime("%Y-%m-%d")] for d in range(7, 14))

    span_days = max(1, days_since_first + 1)
    all_dates = [(first_seen_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(span_days)]
    rolling_7d = [
        sum(t_dc[all_dates[j]] for j in range(max(0, i - 6), i + 1))
        for i in range(len(all_dates))
    ]
    peak_7d = max(rolling_7d) if rolling_7d else d_0_to_6

    # User uniqueness & authenticity signals
    auth = compute_authenticity_signals(trend_msgs, account_stats, as_of_dt)
    users = [m.get("user_name") or "anon" for m in trend_msgs]
    user_counts = Counter(users)
    unique_user_ratio = len(user_counts) / total_msgs

    # Platform dispersion & cascade
    platforms = [m.get("platform", "Unknown") for m in trend_msgs]
    platform_counts = Counter(platforms)
    platform_entropy = compute_platform_entropy(platform_counts)
    num_platforms = len(platform_counts)
    cascade_score = compute_cascade_score(trend_msgs)

    # Exponentially-decayed volume comparison (recent vs older)
    recent_weighted = compute_decay_weighted_volume(
        {d: c for d, c in t_dc.items()
         if (as_of_dt - datetime.strptime(d, "%Y-%m-%d")).days <= 3},
        as_of, half_life_days=7.0
    )
    older_weighted = compute_decay_weighted_volume(
        {d: c for d, c in t_dc.items()
         if 3 < (as_of_dt - datetime.strptime(d, "%Y-%m-%d")).days <= 10},
        as_of, half_life_days=7.0
    )

    is_astroturf = auth["new_account_share"] >= 0.55 and auth["new_account_burst"] >= 6
    is_commercial_spam = auth["commercial_share"] >= 0.35
    is_bot_flood = (
        auth["top_author_share"] >= 0.40
        or (total_msgs >= 10 and len(user_counts) <= 5)
        or (unique_user_ratio < 0.20 and len(user_counts) < 15)
    )

    lifetime_daily_avg = total_msgs / max(1, span_days)
    recent_7d_daily_avg = d_0_to_6 / min(7, span_days)
    accel_ratio = recent_7d_daily_avg / max(0.5, lifetime_daily_avg)

    # Status determination logic:
    is_stationary_ambient = False

    # 1. NOT REAL (Coordinated astroturfing / Bot script / Commercial RMT spam)
    if is_astroturf or is_commercial_spam or is_bot_flood:
        status = "not real"
        if is_astroturf:
            top_plat = max(platform_counts, key=platform_counts.get)
            reason = (
                f"Inauthentic coordinated astroturfing campaign: {auth['new_account_share']*100:.0f}% of posts "
                f"originate from newly created/low-activity accounts with up to {auth['new_account_burst']} distinct "
                f"new accounts posting within a single 60-minute window (primarily on {top_plat})."
            )
        elif is_commercial_spam:
            reason = (
                f"Commercial RMT/account-boosting spam ({auth['commercial_share']*100:.0f}% of messages contain "
                f"solicitation keywords) with repetitive templated copy."
            )
        else:
            reason = (
                f"Automated bot or script activity: low unique user diversity (unique ratio {unique_user_ratio:.2f}, "
                f"top author accounts for {auth['top_author_share']*100:.0f}% of cluster volume)."
            )

    # 2. ONE-OFF (Acute incident spike, followed by immediate decay)
    elif is_acute_spike:
        status = "one-off"
        event_note = " Correlates with a scheduled studio patch/maintenance window." if near_studio_event else ""
        reason = (
            f"Acute volume spike concentrated around {win_peak_date} ({win_max_cnt} msgs on peak day, "
            f"{two_day_conc*100:.0f}% of active window volume within 48h) followed by rapid post-incident decay "
            f"({post_peak_daily_avg:.1f} msgs/day after peak).{event_note}"
        )

    # 3. NEW (Freshly emerged in the past 0-4 days)
    elif days_since_first <= 4 and d_0_to_2 >= 5:
        status = "new"
        reason = (
            f"Freshly emerged topic first detected on {first_seen} ({days_since_first} days before cutoff), "
            f"building initial momentum with {d_0_to_2} messages in the past 72 hours across {num_platforms} platforms."
        )

    # 4. FADING (Past peak, significant drop from rolling 7-day peak or immediate 3-day drop)
    elif days_since_peak >= 5 and (d_0_to_6 <= peak_7d * 0.46 or d_0_to_2 <= d_3_to_6 * 0.42):
        status = "fading"
        reason = (
            f"Declining discussion volume: topic peaked around {peak_date} (7-day rolling peak of {peak_7d} msgs) "
            f"and recent 7-day volume has dropped to {d_0_to_6} msgs (decay-weighted velocity {recent_weighted:.1f} vs {older_weighted:.1f} prior)."
        )

    # 5. GROWING (Emerged after baseline with sustained momentum, or accelerating vs prior weeks)
    elif (emerged_after_start and d_0_to_6 >= peak_7d * 0.48 and d_0_to_2 >= 4) or (
        accel_ratio >= 1.18 or (d_0_to_6 >= d_7_to_13 * 1.18 and d_0_to_2 >= 6)
    ):
        status = "growing"
        cascade_note = (
            " Follows an organic cascade from instant peer chat groups (Discord/QQ) to community review boards and public social media."
            if cascade_score >= 0.6 else ""
        )
        reason = (
            f"Sustained organic momentum expanding across {num_platforms} platforms (entropy: {platform_entropy:.2f}, "
            f"{total_msgs} total msgs since {first_seen}). Recent 7-day volume ({d_0_to_6} msgs, decay-weighted {recent_weighted:.1f}) remains strong.{cascade_note}"
        )

    # 6. Ambient baseline / fallback comparison
    else:
        is_stationary_ambient = (not emerged_after_start and span_days >= 14)
        if d_0_to_6 < peak_7d * 0.78:
            status = "fading"
            reason = (
                f"Gradually receding conversation volume from earlier 7-day peak of {peak_7d} msgs down to {d_0_to_6} msgs "
                f"in the past week (decay-weighted: {recent_weighted:.1f} recent vs {older_weighted:.1f} prior)."
            )
        else:
            status = "growing"
            reason = (
                f"Steady multi-platform discussion across {num_platforms} platforms (decay-weighted volume "
                f"{recent_weighted:.1f}, {d_0_to_6} msgs in the past 7 days)."
            )

    return {
        "status": status,
        "first_seen": first_seen,
        "last_seen": last_seen,
        "total_msgs": total_msgs,
        "recent_msgs_3d": d_0_to_2,
        "recent_weighted": recent_weighted,
        "older_weighted": older_weighted,
        "unique_users": len(user_counts),
        "unique_user_ratio": unique_user_ratio,
        "new_account_share": auth["new_account_share"],
        "new_account_burst": auth["new_account_burst"],
        "top_author_share": auth["top_author_share"],
        "commercial_share": auth["commercial_share"],
        "is_astroturf": is_astroturf,
        "is_stationary_ambient": is_stationary_ambient,
        "emerged_after_start": emerged_after_start,
        "accel_ratio": accel_ratio,
        "num_platforms": num_platforms,
        "platform_entropy": platform_entropy,
        "platform_counts": dict(platform_counts),
        "cascade_score": cascade_score,
        "daily_counts": dict(t_dc),
        "reason": reason
    }

