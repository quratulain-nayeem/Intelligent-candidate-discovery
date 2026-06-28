# rank.py
#
# Single entry point for producing the Redrob submission CSV.
# Command:
#   python rank.py --candidates ./data/candidates.jsonl --out ./outputs/submission.csv
#
# The ranking philosophy follows the JD closely: do not reward generic AI
# keyword stuffing. Prefer candidates whose career history proves they shipped
# retrieval, ranking, search, or recommendation systems in production.

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from src.behavioral_scorer import compute_behavioral_multiplier
from src.experience_scorer import compute_experience_score
from src.honeypot_filter import score_honeypot_risk
from src.production_evidence_scorer import score_production_evidence
from src.role_gate import classify_candidate
from src.semantic_scorer import (
    build_faiss_index,
    build_jd_text,
    embed_candidates,
    get_semantic_scores,
    load_embeddings,
    load_model,
    save_embeddings,
)
from src.skill_scorer import score_candidate_skills

WEIGHT_SEMANTIC = 0.15
WEIGHT_PRODUCTION = 0.15
WEIGHT_SKILL = 0.25
WEIGHT_EXPERIENCE = 0.10
WEIGHT_PRODUCTION_DEPTH = 0.35

TARGET_CITIES = {"pune", "noida"}
GOOD_INDIA_CITIES = {
    "delhi", "gurgaon", "gurugram", "ncr", "mumbai", "hyderabad",
    "bangalore", "bengaluru",
}


def load_candidates(path: str) -> list:
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates)} candidates")
    return candidates


def get_or_build_embeddings(candidates: list, model, cache_path: str) -> np.ndarray:
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        embeddings = load_embeddings(cache_path)
        if embeddings.shape[0] == len(candidates):
            return embeddings
        print(
            "Cached embeddings do not match candidate count; rebuilding "
            f"({embeddings.shape[0]} cached vs {len(candidates)} candidates)."
        )

    print("Building embeddings (first run, this may take a while)...")
    embeddings = embed_candidates(candidates, model)
    save_embeddings(embeddings, cache_path)
    return embeddings


def _first_or_none(values: list) -> str:
    return values[0] if values else "none"


def compute_logistics_multiplier(candidate: dict) -> dict:
    """
    Small top-rank polish for explicit JD logistics.

    This should never rescue a weak candidate, but among close candidates it
    prefers India/Pune-Noida fit, sub-30 notice, and the 6-8 year sweet spot.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    location = str(profile.get("location", "")).lower()
    country = str(profile.get("country", "")).lower()
    years = float(profile.get("years_of_experience", 0) or 0)
    notice = int(signals.get("notice_period_days", 90) or 90)
    willing_to_relocate = bool(signals.get("willing_to_relocate", False))

    multiplier = 1.0
    reasons = []

    if "india" not in country:
        multiplier *= 0.85
        reasons.append("outside India")
    elif any(city in location for city in TARGET_CITIES):
        multiplier *= 1.02
        reasons.append("target city")
    elif any(city in location for city in GOOD_INDIA_CITIES):
        multiplier *= 1.01
        reasons.append("JD-flexible India metro")
    elif "india" in country and willing_to_relocate:
        multiplier *= 1.005
        reasons.append("willing to relocate")

    if notice <= 30:
        multiplier *= 1.015
        reasons.append("notice <=30d")
    elif notice <= 60:
        multiplier *= 0.99
        reasons.append("notice 31-60d")
    else:
        multiplier *= 0.95
        reasons.append("notice >60d")

    if 6 <= years <= 8:
        multiplier *= 1.01
        reasons.append("6-8yr sweet spot")
    elif 5 <= years <= 9:
        multiplier *= 1.005
        reasons.append("5-9yr fit")
    elif 4.5 <= years < 5:
        multiplier *= 0.99
        reasons.append("slightly below exp range")
    elif years < 4.5:
        multiplier *= 0.95
        reasons.append("below exp range")
    elif years > 12:
        multiplier *= 0.88
        reasons.append("above target seniority")

    return {
        "logistics_multiplier": round(multiplier, 4),
        "reasons": reasons,
    }


def build_reasoning(candidate: dict, scores: dict, rank: int) -> str:
    profile = candidate.get("profile", {})

    title = profile.get("current_title", "Unknown")
    years = profile.get("years_of_experience", 0)
    location = profile.get("location", "Unknown")
    country = profile.get("country", "Unknown")

    production = scores["production"]
    prod_score = production["production_evidence_score"]
    domain = _first_or_none(production["matched_domains"])
    eval_term = _first_or_none(production["matched_eval"])
    infra = _first_or_none(production["matched_infra"])
    career_note = _first_or_none(production.get("evidence_phrases", []))

    matched = scores["skill"]["matched_must_have"][:2]
    skill_parts = []
    skill_lookup = {s.get("name", "").lower(): s for s in candidate.get("skills", [])}
    for skill_name in matched:
        s = skill_lookup.get(skill_name.lower(), {})
        prof = s.get("proficiency", "")[:3]
        end = s.get("endorsements", 0)
        mo = s.get("duration_months", 0)
        skill_parts.append(f"{skill_name}({prof}, {end} end, {mo} mo)")
    skill_str = "; ".join(skill_parts) if skill_parts else "no must-have skill evidence"

    days_active = scores["behavioral"]["days_since_active"]
    active_str = f"{days_active}d ago" if days_active < 999 else "unknown"
    notice = scores["behavioral"]["notice_period_days"]
    response = scores["behavioral"]["response_rate"]
    consulting = "consulting-heavy" if scores["experience"]["is_consulting_heavy"] else "not consulting-heavy"

    if prod_score >= 0.70:
        fit = f"strong shipped systems signal around {domain}"
    elif prod_score >= 0.45:
        fit = f"credible production {domain} evidence"
    elif prod_score >= 0.25:
        fit = f"adjacent {domain} evidence but less operational depth"
    else:
        fit = "limited explicit production retrieval/ranking evidence"

    detail_bits = []
    if career_note != "none":
        detail_bits.append(career_note)
    if infra != "none":
        detail_bits.append(f"infra {infra}")
    if eval_term != "none":
        detail_bits.append(f"eval {eval_term}")
    detail = "; ".join(detail_bits[:3]) if detail_bits else "career proof is the main uncertainty"

    concern = []
    if "india" not in str(country).lower():
        concern.append(f"outside India ({country})")
    if notice > 60:
        concern.append(f"notice {notice}d")
    elif notice > 30 and rank <= 25:
        concern.append(f"notice {notice}d is above ideal")
    if response < 0.30:
        concern.append(f"low response {response:.2f}")
    if scores["experience"]["is_consulting_heavy"]:
        concern.append("consulting-heavy background")
    if prod_score < 0.35 and rank <= 50:
        concern.append("production proof weaker than ideal")
    concern_str = f" Concern: {', '.join(concern)}." if concern else ""

    return (
        f"{title}, {years} yrs, {location}; {fit} ({detail}). "
        f"Skills: {skill_str}. Active {active_str}, response {response:.2f}, "
        f"notice {notice}d, {consulting}.{concern_str}"
    )



def normalize_semantic_score(score: float) -> float:
    """Decompress MiniLM cosine scores for the relevant-candidate band."""
    return max(0.0, min((score - 0.50) / 0.12, 1.0))

SENIOR_TITLE_MARKERS = {
    "senior", "lead", "staff", "principal", "head", "founding",
    "distinguished", "applied",
}


def title_seniority_multiplier(candidate: dict, production: dict, experience: dict) -> float:
    """Keep keyword-rich non-senior titles from sneaking into the top 10."""
    profile = candidate.get("profile", {})
    title = str(profile.get("current_title", "")).lower()
    years = float(experience.get("years_of_experience", 0) or 0)
    production_depth = production.get("production_depth_score", 0.0)

    if any(marker in title for marker in SENIOR_TITLE_MARKERS):
        return 1.0

    # Plain AI Engineer can be senior-enough only with strong scope evidence.
    if title.strip() == "ai engineer" and years >= 6.5 and production_depth >= 0.80:
        return 1.0

    # Specialist IC titles are useful, but without seniority markers they should
    # not beat clearly senior/lead/staff candidates for a senior role.
    return 0.88


def seniority_penalty(years: float) -> float:
    """The JD wants 5-9 years; 12+/15+ years should not sit in the top 20."""
    if years >= 15:
        return 0.70
    if years >= 12:
        return 0.80
    if years > 10:
        return 0.92
    return 1.0

def compute_final_score(
    semantic: float,
    production: dict,
    skill: dict,
    experience: dict,
    behavioral: dict,
    logistics: dict,
    honeypot_risk: float,
) -> float:
    production_score = production["production_evidence_score"]
    production_depth = production.get("production_depth_score", 0.0)
    semantic_score = normalize_semantic_score(semantic)
    years = float(experience.get("years_of_experience", 0) or 0)
    composite = (
        semantic_score * WEIGHT_SEMANTIC
        + production_score * WEIGHT_PRODUCTION
        + production_depth * WEIGHT_PRODUCTION_DEPTH
        + skill["final_skill_score"] * WEIGHT_SKILL
        + experience["experience_score"] * WEIGHT_EXPERIENCE
    )

    if production_score < 0.50:
        composite *= 0.92
    if production_depth < 0.35:
        composite *= 0.93
    elif production_depth >= 0.80:
        composite *= 1.015

    composite *= seniority_penalty(years)
    composite *= logistics["logistics_multiplier"]
    behavioral_tiebreaker = 0.85 + behavioral["behavioral_multiplier"] * 0.15
    composite *= behavioral_tiebreaker
    days_inactive = behavioral.get("days_since_active", 0)
    response_rate = behavioral.get("response_rate", 1.0)
    if days_inactive > 120 and response_rate < 0.20:
        composite *= 0.90
    elif days_inactive > 90 and response_rate < 0.30:
        composite *= 0.95
    composite *= 1.0 - honeypot_risk * 0.95

    return round(composite, 6)


def run_pipeline(candidates_path: str, jd_path: str, out_path: str):
    candidates = load_candidates(candidates_path)

    model = load_model()
    jd_text = build_jd_text(jd_path)
    jd_embedding = model.encode(
        [jd_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    embeddings = get_or_build_embeddings(
        candidates,
        model,
        cache_path="outputs/candidate_embeddings.npy",
    )

    print("Running FAISS search...")
    index = build_faiss_index(embeddings)
    top_k = min(15000, len(candidates))
    top_indices, top_scores = get_semantic_scores(jd_embedding, index, top_k=top_k)
    print(f"Top {top_k} candidates retrieved by semantic similarity")

    print("Scoring candidates...")
    scored = []

    for cand_idx, sem_score in zip(top_indices, top_scores):
        if cand_idx < 0:
            continue

        candidate = candidates[cand_idx]

        honeypot_risk = score_honeypot_risk(candidate)
        role_result = classify_candidate(candidate)
        production_result = score_production_evidence(candidate)
        skill_result = score_candidate_skills(candidate)
        experience_result = compute_experience_score(candidate)
        behavioral_result = compute_behavioral_multiplier(candidate)
        logistics_result = compute_logistics_multiplier(candidate)

        final = compute_final_score(
            semantic=float(sem_score),
            production=production_result,
            skill=skill_result,
            experience=experience_result,
            behavioral=behavioral_result,
            logistics=logistics_result,
            honeypot_risk=honeypot_risk,
        )

        final *= title_seniority_multiplier(candidate, production_result, experience_result)
        if role_result["classification"] == "NOISE":
            final *= 0.05

        scored.append(
            {
                "candidate_id": candidate["candidate_id"],
                "final_score": round(final, 6),
                "semantic_score": float(sem_score),
                "classification": role_result["classification"],
                "honeypot_risk": honeypot_risk,
                "production": production_result,
                "skill": skill_result,
                "experience": experience_result,
                "behavioral": behavioral_result,
                "logistics": logistics_result,
                "_candidate": candidate,
            }
        )

    scored.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    # Hard availability quarantine: very low-response candidates are not top-50 hires.
    # They may be technically strong, but response_rate < 0.30 means recruiter risk is too high.
    reachable = [
        entry for entry in scored
        if entry["behavioral"].get("response_rate", 1.0) >= 0.30
    ]
    unreachable = [
        entry for entry in scored
        if entry["behavioral"].get("response_rate", 1.0) < 0.30
    ]
    scored = reachable[:50] + unreachable + reachable[50:]
    top_100 = scored[:100]
    # Submission validator requires scores to be non-increasing after rank-floor reranking.
    previous_score = None
    for entry in top_100:
        if previous_score is not None and entry["final_score"] >= previous_score:
            entry["final_score"] = round(previous_score - 0.000001, 6)
        previous_score = entry["final_score"]

    print("Building submission CSV...")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank, entry in enumerate(top_100, start=1):
            reasoning = build_reasoning(
                entry["_candidate"],
                {
                    "production": entry["production"],
                    "skill": entry["skill"],
                    "experience": entry["experience"],
                    "behavioral": entry["behavioral"],
                    "logistics": entry["logistics"],
                },
                rank,
            )
            writer.writerow(
                [
                    entry["candidate_id"],
                    rank,
                    f"{entry['final_score']:.6f}",
                    reasoning,
                ]
            )


    diagnostics_path = Path(out_path).parent / "diagnostics_top50.csv"
    with open(diagnostics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "candidate_id", "title", "location", "classification",
            "semantic", "production", "production_depth", "skill", "experience",
            "behavioral", "logistics", "honeypot", "final_score",
        ])
        for rank, entry in enumerate(top_100[:50], start=1):
            profile = entry["_candidate"].get("profile", {})
            writer.writerow([
                rank,
                entry["candidate_id"],
                profile.get("current_title", ""),
                profile.get("location", ""),
                entry["classification"],
                f"{entry['semantic_score']:.6f}",
                entry["production"]["production_evidence_score"],
                entry["production"].get("production_depth_score", 0.0),
                entry["skill"]["final_skill_score"],
                entry["experience"]["experience_score"],
                entry["behavioral"]["behavioral_multiplier"],
                entry["logistics"]["logistics_multiplier"],
                entry["honeypot_risk"],
                f"{entry['final_score']:.6f}",
            ])
    print(f"\nDone. Submission written to {out_path}")
    print("Top 5 candidates:")
    for i, entry in enumerate(top_100[:5], 1):
        prod = entry["production"]["production_evidence_score"]
        logi = entry["logistics"]["logistics_multiplier"]
        print(
            f"  {i}. {entry['candidate_id']} | score: {entry['final_score']:.4f} | "
            f"class: {entry['classification']} | prod: {prod:.2f} | "
            f"logistics: {logi:.3f} | honeypot: {entry['honeypot_risk']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="./data/candidates.jsonl")
    parser.add_argument("--jd", default="./data/job_description.docx")
    parser.add_argument("--out", default="./outputs/submission.csv")
    args = parser.parse_args()

    run_pipeline(args.candidates, args.jd, args.out)




