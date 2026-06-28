# src/skill_scorer.py
#
# WHY THIS FILE EXISTS:
# The dataset gives us three pieces of evidence per skill:
#   - proficiency (beginner/intermediate/advanced/expert)
#   - endorsements (how many people validated this skill)
#   - duration_months (how long they've actually used it)
#
# Most submissions will just count how many AI skills a candidate has.
# That's easily gamed â€” anyone can list "expert Python" with 0 months
# and 0 endorsements. We weight skills by the evidence behind them.
#
# We also cross-reference against skill_assessment_scores in
# redrob_signals â€” if they took a platform test, that's independent
# third-party validation. We boost those skills.


import math

# WHY THESE WEIGHTS:
# Proficiency is self-reported. We use it but discount it compared
# to evidence. Expert = 4x beginner, but evidence multiplies it further.
PROFICIENCY_WEIGHTS = {
    "beginner": 1.0,
    "intermediate": 2.0,
    "advanced": 3.0,
    "expert": 4.0
}

# WHY THESE SPECIFIC SKILLS:
# Pulled directly from the JD. Split into must-have and nice-to-have
# exactly as the JD phrases it. Must-haves get 2x weight in the
# final skill score. Nice-to-haves get 1x.
MUST_HAVE_SKILLS = {
    # Production retrieval systems
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "elasticsearch", "opensearch", "vector search",
    "sentence transformers", "embeddings",
    # Evaluation frameworks
    "ndcg", "mrr", "map", "ranking evaluation", "a/b testing",
    "information retrieval",
    # Core engineering
    "python", "pytorch", "tensorflow",
    # LLM/NLP work
    "nlp", "llm", "transformers", "bert", "rag",
    "retrieval augmented generation", "semantic search",
    "recommendation systems"
}

NICE_TO_HAVE_SKILLS = {
    # Fine-tuning experience
    "lora", "qlora", "peft", "fine-tuning llms", "fine-tune",
    # Learning to rank
    "xgboost", "lightgbm", "learning to rank",
    # Infrastructure
    "docker", "kubernetes", "mlops", "fastapi", "redis",
    "kafka", "airflow", "spark",
    # Adjacent ML
    "deep learning", "machine learning", "scikit-learn",
    "hugging face", "langchain", "weights & biases"
}


def compute_skill_trust_score(skill: dict) -> float:
    """
    WHY THIS FORMULA:
    trust = proficiency_weight Ã— log(1 + endorsements) Ã— log(1 + duration)

    We use logarithms for endorsements and duration because the
    difference between 0 and 1 endorsement is huge (zero vs any
    validation). But the difference between 50 and 51 endorsements
    is almost nothing. Log captures this diminishing returns effect.

    A skill with:
    - expert proficiency, 50 endorsements, 36 months â†’ high trust
    - expert proficiency, 0 endorsements, 0 months â†’ near zero trust
    - beginner proficiency, 20 endorsements, 24 months â†’ moderate trust

    This is why keyword stuffers get caught. Listing 10 expert skills
    with 0 evidence collapses to near-zero trust score.
    """
    proficiency = skill.get("proficiency", "beginner")
    endorsements = skill.get("endorsements", 0)
    duration = skill.get("duration_months", 0)

    weight = PROFICIENCY_WEIGHTS.get(proficiency, 1.0)

    # log(1 + x) so that 0 endorsements gives log(1) = 0, not log(0) = error
    endorsement_factor = math.log(1 + endorsements)
    duration_factor = math.log(1 + duration)

    return weight * endorsement_factor * duration_factor


def get_assessment_boost(skill_name: str, assessment_scores: dict) -> float:
    """
    WHY: If a candidate took a Redrob platform assessment for this skill
    and scored well, that's independent third-party validation.
    We boost the trust score for assessed skills.

    Score 80+ â†’ 1.5x boost (strong independent validation)
    Score 60-79 â†’ 1.2x boost (moderate validation)
    Score below 60 â†’ 0.9x slight penalty (claimed skill, weak assessment)
    No assessment â†’ 1.0x neutral (no data either way)
    """
    # Normalize skill name to match assessment keys
    normalized = skill_name.lower().strip()

    for assessed_skill, score in assessment_scores.items():
        if assessed_skill.lower().strip() == normalized:
            if score >= 80:
                return 1.5
            elif score >= 60:
                return 1.2
            else:
                return 0.9

    return 1.0


def score_candidate_skills(candidate: dict) -> dict:
    """
    WHY THIS IS THE MAIN FUNCTION:
    Computes three things:
    1. must_have_score â€” weighted trust score for JD-required skills
    2. nice_to_have_score â€” weighted trust score for bonus skills
    3. final_skill_score â€” normalized combination of both

    We normalize at the end so scores are always 0.0 to 1.0,
    which makes it easy to combine with other scoring components later.

    Must-have skills get 2x weight because the JD says these are
    the actual disqualifiers â€” if you don't have them, you're out.
    Nice-to-haves are bonuses, not requirements.
    """
    skills = candidate.get("skills", [])
    assessment_scores = candidate.get(
        "redrob_signals", {}
    ).get("skill_assessment_scores", {})

    must_have_total = 0.0
    nice_to_have_total = 0.0
    matched_must_have = []
    matched_nice_to_have = []

    for skill in skills:
        skill_name = skill.get("name", "").lower().strip()
        trust = compute_skill_trust_score(skill)
        boost = get_assessment_boost(skill["name"], assessment_scores)
        adjusted_trust = trust * boost

        if skill_name in MUST_HAVE_SKILLS:
            must_have_total += adjusted_trust * 2.0  # 2x weight for must-haves
            matched_must_have.append(skill["name"])

        elif skill_name in NICE_TO_HAVE_SKILLS:
            nice_to_have_total += adjusted_trust * 1.0
            matched_nice_to_have.append(skill["name"])

    # WHY WE NORMALIZE THIS WAY:
    # A perfect must-have score would be: all must-have skills present,
    # each with expert proficiency, ~50 endorsements, ~36 months.
    # That gives roughly: 4 Ã— log(51) Ã— log(37) Ã— 2 â‰ˆ 4 Ã— 3.93 Ã— 3.61 Ã— 2 â‰ˆ 113
    # per skill. With ~10 must-have skills matched, max â‰ˆ 1130.
    # We cap normalization at 200 to avoid extreme outliers dominating.
    # This is a design choice â€” adjust if scores cluster at extremes.
    MUST_HAVE_MAX = 200.0
    NICE_TO_HAVE_MAX = 100.0

    norm_must = min(must_have_total / MUST_HAVE_MAX, 1.0)
    norm_nice = min(nice_to_have_total / NICE_TO_HAVE_MAX, 1.0)

    # Must-haves are 70% of skill score, nice-to-haves are 30%
    final_skill_score = (norm_must * 0.7) + (norm_nice * 0.3)
    evidenced_depth_count = sum(
        1 for skill in skills
        if skill.get("endorsements", 0) > 5 and skill.get("duration_months", 0) > 12
    )
    depth_bonus = min(evidenced_depth_count / 5.0, 1.0)
    final_skill_score *= 0.85 + 0.15 * depth_bonus

    return {
        "candidate_id": candidate["candidate_id"],
        "final_skill_score": round(final_skill_score, 4),
        "must_have_score_raw": round(must_have_total, 4),
        "nice_to_have_score_raw": round(nice_to_have_total, 4),
        "matched_must_have": matched_must_have,
        "matched_nice_to_have": matched_nice_to_have,
        "skills_assessed_count": len(assessment_scores),
        "deep_evidenced_skill_count": evidenced_depth_count
    }
