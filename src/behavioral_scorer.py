# src/behavioral_scorer.py
#
# WHY THIS FILE EXISTS:
# A perfect-on-paper candidate who hasn't logged in for 6 months
# and ignores recruiter messages is not actually hirable.
# The redrob_signals doc says these behavioral signals are "often
# more predictive of whether a candidate can actually be hired
# than their static profile."
#
# This file builds an availability multiplier from 0.0 to 1.0.
# It MULTIPLIES the final score — it can't make a bad candidate
# good, but it can make an unavailable great candidate rank lower
# than an available good candidate.
#
# WHY MULTIPLICATIVE NOT ADDITIVE:
# If we added availability as another score component, a candidate
# with perfect skills but 0% response rate could still score high
# because skills dominate. Multiplication means availability acts
# as a ceiling — bad availability drags the whole score down
# regardless of how strong the profile is.

from datetime import datetime, date


# Reference date for recency calculations
# Using the dataset's most recent activity dates we observed
REFERENCE_DATE = date(2026, 6, 24)


def score_recency(last_active_date_str: str) -> float:
    """
    WHY: Someone who logged in yesterday is available.
    Someone who logged in 8 months ago probably found a job
    or stopped looking. We penalize inactivity progressively.

    Thresholds chosen based on typical job search cycles:
    - 0-30 days: actively looking, full score
    - 30-60 days: still warm, small penalty
    - 60-90 days: cooling off, moderate penalty
    - 90+ days: likely gone, significant penalty
    """
    try:
        last_active = datetime.strptime(
            last_active_date_str, "%Y-%m-%d"
        ).date()
        days_inactive = (REFERENCE_DATE - last_active).days
    except (ValueError, TypeError):
        # If date is malformed, assume worst case
        return 0.5

    if days_inactive <= 30:
        return 1.0
    elif days_inactive <= 60:
        return 0.85
    elif days_inactive <= 90:
        return 0.70
    else:
        return 0.50


def score_responsiveness(
    response_rate: float,
    avg_response_time_hours: float
) -> float:
    """
    WHY: Two signals together tell you if someone will actually
    reply when a recruiter reaches out.

    response_rate is the fraction of recruiter messages they replied to.
    avg_response_time_hours is how long they take to reply.

    A candidate with 0.05 response rate is a ghost — even if their
    profile is perfect, recruiters can't reach them.
    A candidate who takes 200 hours to reply is nearly as bad.

    We combine both into one responsiveness score.
    """
    # Response rate scoring
    if response_rate >= 0.7:
        rate_score = 1.0
    elif response_rate >= 0.5:
        rate_score = 0.85
    elif response_rate >= 0.3:
        rate_score = 0.70
    elif response_rate >= 0.15:
        rate_score = 0.55
    else:
        # Below 15% response rate — effectively unreachable
        rate_score = 0.35

    # Response time scoring
    # JD wants someone who moves fast and communicates well
    if avg_response_time_hours <= 24:
        time_score = 1.0
    elif avg_response_time_hours <= 72:
        time_score = 0.85
    elif avg_response_time_hours <= 120:
        time_score = 0.70
    else:
        time_score = 0.55

    # Response rate matters more than speed
    return (rate_score * 0.7) + (time_score * 0.3)


def score_reliability(
    interview_completion_rate: float,
    offer_acceptance_rate: float
) -> float:
    """
    WHY: These two signals tell you if the candidate follows through.

    interview_completion_rate — do they show up to interviews they agreed to?
    Low rate means they schedule and ghost. Recruiters hate this.

    offer_acceptance_rate — do they accept offers or waste everyone's time?
    -1 means no offer history, which is neutral (not penalized).

    These are reliability signals — they don't say if the candidate is
    good at their job, they say if the hiring process will complete.
    """
    # Interview completion
    if interview_completion_rate >= 0.8:
        interview_score = 1.0
    elif interview_completion_rate >= 0.6:
        interview_score = 0.85
    elif interview_completion_rate >= 0.4:
        interview_score = 0.70
    else:
        interview_score = 0.55

    # Offer acceptance — -1 means no history, treat as neutral
    if offer_acceptance_rate == -1:
        offer_score = 0.80  # neutral, slightly below perfect
    elif offer_acceptance_rate >= 0.7:
        offer_score = 1.0
    elif offer_acceptance_rate >= 0.5:
        offer_score = 0.85
    else:
        offer_score = 0.70

    return (interview_score * 0.6) + (offer_score * 0.4)


def score_notice_period(notice_period_days: int) -> float:
    """
    WHY: The JD explicitly says they want sub-30 day notice.
    They can buy out up to 30 days. 30+ day candidates are
    "still in scope but the bar gets higher."

    We implement exactly what the JD says — not our opinion,
    their stated preference.
    """
    if notice_period_days <= 30:
        return 1.0
    elif notice_period_days <= 60:
        return 0.85
    elif notice_period_days <= 90:
        return 0.70
    else:
        # 90+ days notice — significant friction for hiring
        return 0.55


def score_open_signals(signals: dict) -> float:
    """
    WHY: Some signals are simple boolean boosts.

    open_to_work_flag — they explicitly said they're looking.
      Not having this doesn't mean they're not looking, but
      having it is a strong positive signal.

    verified_email + verified_phone — basic trust signals.
      Unverified contact info means the recruiter can't reach them
      even if they respond.

    profile_completeness_score — incomplete profiles suggest
      low engagement with the platform.
    """
    score = 0.70  # baseline — neutral candidate

    if signals.get("open_to_work_flag", False):
        score += 0.15

    # Both contact methods verified
    if signals.get("verified_email", False) and \
       signals.get("verified_phone", False):
        score += 0.10
    elif signals.get("verified_email", False):
        score += 0.05

    # Profile completeness
    completeness = signals.get("profile_completeness_score", 0)
    if completeness >= 90:
        score += 0.05
    elif completeness < 50:
        score -= 0.10

    return min(score, 1.0)


def compute_behavioral_multiplier(candidate: dict) -> dict:
    """
    WHY THIS IS THE MAIN FUNCTION:
    Combines all behavioral signals into one multiplier between
    0.0 and 1.0 that gets multiplied against the candidate's
    composite skill + semantic score.

    Component weights:
    - Recency (0.25): are they still active on the platform?
    - Responsiveness (0.25): will they actually reply?
    - Reliability (0.20): will they show up and follow through?
    - Notice period (0.15): can they start soon?
    - Open signals (0.15): are they actively engaged?

    WHY THESE WEIGHTS:
    Recency and responsiveness together are 50% because a candidate
    who is inactive or unreachable is functionally unavailable,
    regardless of every other signal. The JD makes this explicit.
    """
    signals = candidate.get("redrob_signals", {})

    recency = score_recency(
        signals.get("last_active_date", "2020-01-01")
    )

    responsiveness = score_responsiveness(
        signals.get("recruiter_response_rate", 0),
        signals.get("avg_response_time_hours", 999)
    )

    reliability = score_reliability(
        signals.get("interview_completion_rate", 0),
        signals.get("offer_acceptance_rate", -1)
    )

    notice = score_notice_period(
        signals.get("notice_period_days", 90)
    )

    open_signals = score_open_signals(signals)

    # Weighted combination
    multiplier = (
        recency       * 0.25 +
        responsiveness * 0.25 +
        reliability   * 0.20 +
        notice        * 0.15 +
        open_signals  * 0.15
    )

    return {
        "candidate_id": candidate["candidate_id"],
        "behavioral_multiplier": round(multiplier, 4),
        "recency_score": round(recency, 4),
        "responsiveness_score": round(responsiveness, 4),
        "reliability_score": round(reliability, 4),
        "notice_score": round(notice, 4),
        "open_signals_score": round(open_signals, 4),
        # These go into the reasoning column later
        "days_since_active": (
            REFERENCE_DATE -
            datetime.strptime(
                signals.get("last_active_date", "2020-01-01"),
                "%Y-%m-%d"
            ).date()
        ).days,
        "response_rate": signals.get("recruiter_response_rate", 0),
        "notice_period_days": signals.get("notice_period_days", 90),
        "open_to_work": signals.get("open_to_work_flag", False)
    }