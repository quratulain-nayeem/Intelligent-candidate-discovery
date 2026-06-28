
# src/experience_scorer.py
#
# WHY THIS FILE EXISTS:
# Some signals in the JD are explicitly structured — not semantic,
# not behavioral, just hard facts about a candidate's background.
# Years of experience, education tier, consulting history, GitHub activity.
#
# These don't need embeddings or trust formulas.
# They need direct comparisons against what the JD literally states.
#
# WHY THIS IS SEPARATE FROM skill_scorer.py:
# Skill scorer measures WHAT they know.
# This file measures WHO they are structurally —
# how long they've been doing it, where they studied,
# whether their background matches the product company requirement.


# WHY THESE CONSULTING FIRMS (same list as role_gate.py):
# The JD explicitly names consulting-only backgrounds as a red flag.
# We use the same list for consistency — role_gate uses it to classify,
# this file uses it to penalize in the score.
CONSULTING_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "mindtree", "l&t infotech", "ltimindtree",
    "persistent systems", "niit technologies"
}


def score_years_of_experience(years: float) -> float:
    """
    WHY THIS RANGE:
    JD says 5-9 years explicitly. Peak score is 6-8 years —
    that's someone senior enough to have shipped production systems
    but not so senior they're purely managerial.

    Below 4: likely hasn't led a production system end-to-end
    Above 12: probably wants a staff/principal role, overqualified
    for what the JD describes

    We don't hard-cutoff — we penalize softly because a 4-year
    candidate who built production retrieval systems beats a
    7-year candidate who maintained legacy Java. The semantic
    and skill scores handle that nuance. This just adjusts.
    """
    if 6 <= years <= 8:
        return 1.0
    elif 5 <= years < 6 or 8 < years <= 9:
        return 0.90
    elif 4 <= years < 5 or 9 < years <= 12:
        return 0.75
    elif years < 4:
        return 0.60
    else:  # 12+
        return 0.65


def score_education(education: list) -> float:
    """
    WHY TIER MATTERS BUT ISN'T DECISIVE:
    The JD says education is a soft signal, not a hard filter.
    Tier 1 (IITs, IISc, NITs, top global universities) gets a boost.
    Tier 2 (state universities, decent private colleges) is neutral.
    Tier 3/4 is slightly below neutral but not disqualifying.

    WHY WE LOOK AT THE HIGHEST TIER ACROSS ALL DEGREES:
    Someone with a tier_3 undergrad but tier_1 masters is treated
    as tier_1. The masters matters more for this role.

    -1 means no education data — treat as neutral, not penalized.
    """
    if not education:
        return 0.75  # neutral, no data

    tier_scores = {
        "tier_1": 1.0,
        "tier_2": 0.85,
        "tier_3": 0.75,
        "tier_4": 0.70
    }

    best_score = 0.0
    for edu in education:
        tier = edu.get("tier", "tier_3")
        score = tier_scores.get(tier, 0.75)
        best_score = max(best_score, score)

    return best_score


def score_consulting_background(career_history: list) -> float:
    """
    WHY MULTIPLICATIVE PENALTY NOT DISQUALIFICATION:
    The JD says consulting-only is a red flag, not an automatic reject.
    "Currently at Infosys but prior product company work" is explicitly
    called out as acceptable.

    So we check what fraction of their career is consulting.
    Pure consulting (75%+) gets a meaningful penalty.
    Majority product company gets no penalty.
    Mixed background gets a small penalty.

    Returns a multiplier: 1.0 = no penalty, lower = penalized.
    """
    if not career_history:
        return 1.0

    companies = [role.get("company", "").lower() for role in career_history]
    total = len(companies)

    consulting_count = sum(
        1 for company in companies
        if any(flag in company for flag in CONSULTING_COMPANIES)
    )

    consulting_fraction = consulting_count / total

    if consulting_fraction >= 0.75:
        return 0.65  # significant penalty — consulting-only background
    elif consulting_fraction >= 0.50:
        return 0.80  # moderate penalty — majority consulting
    elif consulting_fraction >= 0.25:
        return 0.92  # small penalty — some consulting in otherwise product career
    else:
        return 1.0   # no penalty — product company background


def score_github_activity(github_score: int) -> float:
    """
    WHY GITHUB MATTERS FOR THIS ROLE:
    The JD is for someone who builds production AI systems.
    Active GitHub signals someone who codes beyond their job,
    contributes to open source, or maintains personal projects.

    -1 means no GitHub account — neutral, not penalized.
    We don't penalize absence because many strong engineers
    at product companies don't maintain public GitHub.

    0-10: has account, no meaningful activity
    10-30: some activity
    30+: actively coding
    """
    if github_score == -1:
        return 0.80  # no account — neutral

    if github_score >= 50:
        return 1.0
    elif github_score >= 30:
        return 0.90
    elif github_score >= 10:
        return 0.80
    else:
        return 0.70  # account exists but inactive


def score_location_fit(candidate_location: str) -> float:
    """
    WHY: JD says Pune/Noida hybrid. They'll consider remote
    but prefer local candidates. This is a soft signal.

    We check if the candidate's location string mentions
    target cities or states. Not a hard filter.
    """
    if not candidate_location:
        return 0.85  # unknown location — slightly below neutral

    location_lower = candidate_location.lower()

    # Direct city match
    target_cities = {"pune", "noida", "delhi", "gurgaon", "gurugram", "ncr"}
    if any(city in location_lower for city in target_cities):
        return 1.0

    # Same region but not exact city
    nearby = {"mumbai", "maharashtra", "up ", "uttar pradesh"}
    if any(place in location_lower for place in nearby):
        return 0.90

    # Rest of India — remote possible
    return 0.80


def compute_experience_score(candidate: dict) -> dict:
    """
    WHY THIS IS THE MAIN FUNCTION:
    Combines all structured signals into one experience score.

    Component weights:
    - Years of experience (0.40): JD is explicit about the range
    - Consulting penalty (0.30): JD explicitly flags this as a disqualifier
    - Education (0.15): soft signal per JD, real but not decisive
    - GitHub (0.10): nice signal, absent is neutral
    - Location (0.05): soft preference, not a hard filter

    WHY CONSULTING GETS 0.30:
    It's the only signal in this file the JD calls out as
    a specific red flag. Everything else is preference.
    Consulting-only background is closer to disqualification.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    years = profile.get("years_of_experience", 0)
    education = candidate.get("education", [])
    career_history = candidate.get("career_history", [])
    github_score = signals.get("github_activity_score", -1)
    location = profile.get("location", "")

    exp_score = score_years_of_experience(years)
    edu_score = score_education(education)
    consulting_multiplier = score_consulting_background(career_history)
    github_score_val = score_github_activity(github_score)
    location_score = score_location_fit(location)

    # Weighted combination
    raw_score = (
        exp_score      * 0.40 +
        edu_score      * 0.15 +
        github_score_val * 0.10 +
        location_score * 0.05
    )

    # Consulting penalty applied as multiplier on top
    # WHY MULTIPLICATIVE: a consulting-only candidate with otherwise
    # great scores should still be meaningfully penalized, not just
    # have their score slightly adjusted by adding a weighted component
    final_score = raw_score * consulting_multiplier * 0.30 + raw_score * 0.70

    return {
        "candidate_id": candidate["candidate_id"],
        "experience_score": round(final_score, 4),
        "years_of_experience": years,
        "exp_score": round(exp_score, 4),
        "edu_score": round(edu_score, 4),
        "consulting_multiplier": round(consulting_multiplier, 4),
        "github_score": round(github_score_val, 4),
        "location_score": round(location_score, 4),
        "is_consulting_heavy": consulting_multiplier < 0.85
    }