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

def analyze_cluster_velocity(
    cluster_msgs: List[Dict[str, Any]],
    as_of: str,
    versions: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Computes trajectory metrics for a cluster of messages up to as_of date:
    - Exponentially-decayed volume (half-life = 7 days)
    - User uniqueness ratio
    - Cross-platform dispersion entropy
    - Platform cascade score (review → influencer → mass social)
    - Trend status: 'new' | 'growing' | 'one-off' | 'fading' | 'not real'
    """
    as_of_dt = datetime.strptime(as_of, "%Y-%m-%d")
    total_msgs = len(cluster_msgs)
    if total_msgs == 0:
        return {"status": "fading", "velocity": 0, "first_seen": as_of}
        
    dates = [m["date"] for m in cluster_msgs]
    first_seen = min(dates)
    last_seen = max(dates)
    
    first_seen_dt = datetime.strptime(first_seen, "%Y-%m-%d")
    days_since_first = (as_of_dt - first_seen_dt).days
    
    # User uniqueness
    users = [m.get("user_name") or "anon" for m in cluster_msgs]
    user_counts = Counter(users)
    unique_user_ratio = len(user_counts) / total_msgs
    
    # Platform dispersion
    platforms = [m.get("platform", "Unknown") for m in cluster_msgs]
    platform_counts = Counter(platforms)
    platform_entropy = compute_platform_entropy(platform_counts)
    num_platforms = len(platform_counts)
    
    # Platform cascade score
    cascade_score = compute_cascade_score(cluster_msgs)
    
    # Daily counts
    daily_counts = Counter(dates)
    
    # Exponentially-decayed volume comparison (recent vs older)
    recent_weighted = compute_decay_weighted_volume(
        {d: c for d, c in daily_counts.items()
         if (as_of_dt - datetime.strptime(d, "%Y-%m-%d")).days <= 3},
        as_of, half_life_days=7.0
    )
    older_weighted = compute_decay_weighted_volume(
        {d: c for d, c in daily_counts.items()
         if 3 < (as_of_dt - datetime.strptime(d, "%Y-%m-%d")).days <= 10},
        as_of, half_life_days=7.0
    )
    
    # Check for acute single-day spike (one-off)
    max_day_cnt = max(daily_counts.values())
    peak_date = max(daily_counts, key=daily_counts.get)
    peak_dt = datetime.strptime(peak_date, "%Y-%m-%d")
    days_since_peak = (as_of_dt - peak_dt).days
    spike_concentration = max_day_cnt / total_msgs
    
    # Raw volume in last 3 days vs preceding 3-7 days
    d_0_to_2 = sum(daily_counts[(as_of_dt - timedelta(days=d)).strftime("%Y-%m-%d")] for d in range(3))
    d_3_to_6 = sum(daily_counts[(as_of_dt - timedelta(days=d)).strftime("%Y-%m-%d")] for d in range(3, 7))
    
    # Check spam signatures
    all_text = " ".join(m.get("cleaned_text", "") for m in cluster_msgs)
    is_spam_text = any(term in all_text for term in ["代练", "代肝", "出号", "全图鉴", "破解版", "Q群管家", "群管家", "私聊"])
    
    # Check if cluster coincides with a known studio event (for one-off detection)
    near_studio_event = False
    if versions:
        for v in versions:
            v_date = v.get("date", "")
            if v_date and abs((datetime.strptime(v_date, "%Y-%m-%d") - peak_dt).days) <= 1:
                near_studio_event = True
                break
    
    # Status determination logic:
    
    # 1. NOT REAL (Bot / Astroturfing / Commercial spam)
    if is_spam_text or unique_user_ratio < 0.22 or (total_msgs > 20 and len(user_counts) <= 2):
        status = "not real"
        reason = f"Identified commercial spam or automated bot activity. Unique user ratio is low ({unique_user_ratio:.2f}) with repetitive templated posts."
        
    # 2. ONE-OFF (Acute incident spike, followed by immediate decay)
    elif spike_concentration >= 0.55 and days_since_peak >= 2 and d_0_to_2 <= (max_day_cnt * 0.25):
        status = "one-off"
        event_note = " Correlates with a known studio patch/event." if near_studio_event else ""
        reason = f"Acute volume spike concentrated on {peak_date} ({max_day_cnt} msgs, {spike_concentration*100:.1f}% of cluster volume) followed by rapid post-event decay.{event_note}"
        
    # 3. NEW (Freshly emerged in the past 1-3 days)
    elif days_since_first <= 3 and d_0_to_2 >= 5:
        status = "new"
        reason = f"Freshly emerged topic first detected on {first_seen} ({days_since_first} days before cutoff). Initial momentum building with {d_0_to_2} recent messages."
        
    # 4. GROWING (Sustained cross-platform expansion and momentum)
    elif d_0_to_2 >= max(4, int(d_3_to_6 * 0.6)) and (num_platforms >= 3 or platform_entropy >= 0.8) and d_0_to_2 > 0:
        status = "growing"
        cascade_note = " Follows an organic cascade from instant peer chat groups (Discord/QQ) to community review boards and public social media." if cascade_score >= 0.6 else ""
        reason = f"Sustained organic interest expanding across {num_platforms} platforms (entropy: {platform_entropy:.2f}). Recent decay-weighted volume ({recent_weighted:.1f}) remains strong.{cascade_note}"
        
    # 5. FADING (Past peak, decaying interest)
    elif days_since_peak >= 4 and d_0_to_2 < (d_3_to_6 * 0.5):
        status = "fading"
        reason = f"Declining discussion volume. Topic peaked on {peak_date} and recent decay-weighted volume ({recent_weighted:.1f}) has fallen well below prior levels ({older_weighted:.1f})."
        
    else:
        # Default fallback using decay-weighted momentum comparison
        if recent_weighted > older_weighted:
            status = "growing"
            reason = f"Positive decay-weighted velocity ({recent_weighted:.1f} recent vs {older_weighted:.1f} prior) across {num_platforms} platforms."
        else:
            status = "fading"
            reason = f"Plateaued or gradually receding conversation volume (decay-weighted: {recent_weighted:.1f} recent vs {older_weighted:.1f} prior)."

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
        "num_platforms": num_platforms,
        "platform_entropy": platform_entropy,
        "platform_counts": dict(platform_counts),
        "cascade_score": cascade_score,
        "daily_counts": dict(daily_counts),
        "reason": reason
    }
