# src/honeypot_filter.py
#
# WHY THIS FILE EXISTS:
# Redrob planted ~80 fake "honeypot" candidates in the dataset.
# These profiles look great on paper (lots of AI skills) but have
# internally contradictory data. If your ranker puts 10+ honeypots
# in your top 100, your submission is automatically disqualified.
#
# HOW WE CATCH THEM:
# We look for two types of impossibilities:
#   1. Career duration doesn't match stated experience
#   2. Expert-level skills with zero evidence (0 months, 0 endorsements)


def check_experience_mismatch(candidate: dict) -> bool:
    """
    WHY: A candidate's career history has actual start/end dates.
    If the total months across all their jobs is way more than their
    stated years_of_experience, someone fabricated the numbers.
    
    We allow a 36-month buffer because people have gaps between jobs,
    freelance periods, etc. But 4+ years of unexplained extra experience
    is a red flag we can't ignore.
    """
    stated_years = candidate["profile"]["years_of_experience"]
    stated_months = stated_years * 12

    total_career_months = sum(
        role["duration_months"]
        for role in candidate["career_history"]
    )

    # If career months exceed stated experience by more than 3 years,
    # the profile is internally inconsistent
    return total_career_months > stated_months + 36


def check_expert_zero_evidence(candidate: dict) -> bool:
    """
    WHY: "Expert" proficiency means you've used something deeply for years.
    If someone claims expert in 3+ skills but every single one has
    0 months of duration AND 0 endorsements, they're keyword stuffing.
    
    Real experts have at least one of: time spent, or peer validation.
    Having neither on multiple expert skills is a honeypot signal.
    """
    expert_skills = [
        skill for skill in candidate["skills"]
        if skill["proficiency"] == "expert"
    ]

    if len(expert_skills) < 3:
        # Not enough expert claims to be suspicious
        return False

    zero_evidence_count = sum(
        1 for skill in expert_skills
        if skill.get("duration_months", 0) == 0
        and skill.get("endorsements", 0) == 0
    )

    # If 3+ expert skills all have zero evidence, it's a flag
    return zero_evidence_count >= 3


def is_honeypot(candidate: dict) -> bool:
    """
    WHY: We combine both checks. A candidate needs to trigger
    at least 2 flags to be marked a honeypot. One flag alone
    could be a data quality issue. Two flags together means
    the profile is almost certainly fabricated.
    
    We're conservative on purpose — a false positive (marking a
    real candidate as honeypot) costs us points. A false negative
    (missing a honeypot) only costs us if 10+ sneak into top 100.
    """
    flags = 0

    if check_experience_mismatch(candidate):
        flags += 1

    if check_expert_zero_evidence(candidate):
        flags += 1

    return flags >= 2


def score_honeypot_risk(candidate: dict) -> float:
    """
    WHY: Instead of binary honeypot/not-honeypot, we return
    a risk score from 0.0 to 1.0. This lets us use it as a
    penalty multiplier later rather than a hard cutoff.
    
    0.0 = clean profile
    0.5 = one suspicious signal  
    1.0 = confirmed honeypot (2+ flags)
    """
    flags = 0

    if check_experience_mismatch(candidate):
        flags += 1

    if check_expert_zero_evidence(candidate):
        flags += 1

    return min(flags / 2.0, 1.0)