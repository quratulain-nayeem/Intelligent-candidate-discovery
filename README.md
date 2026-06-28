# Redrob AI Candidate Ranker


```md
# Intelligent Candidate Discovery

AI-powered candidate ranking pipeline for the Redrob x Hack2Skill India Runs Data & AI Challenge.

## Overview

This project ranks 100,000 candidates for a Senior AI Engineer role and generates a top-100 submission CSV.

The system goes beyond keyword matching by combining semantic retrieval, production evidence, skill trust, experience fit, seniority, behavioral availability, logistics, and risk checks.

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce Submission

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./outputs/submission.csv
```

Validate output:

```bash
python data/validate_submission.py outputs/submission.csv
```

## Pipeline

5-layer ranking system:

1. **JD Understanding**  
   Converts the JD into a focused Senior AI Engineer search query.

2. **Semantic Retrieval**  
   Uses `SentenceTransformer + FAISS` to retrieve top candidates from 100,000 profiles.

3. **Multi-Signal Scoring**  
   Scores semantic fit, production proof, production depth, skill trust, experience, seniority, and availability.

4. **Risk & Quality Layer**  
   Handles honeypots, ghost candidates, suspicious profiles, and non-senior keyword matches.

5. **Ranked Output**  
   Produces a validated top-100 CSV with score and explanation.

## Key Differentiators

- Focuses on **production evidence**, not keyword stuffing
- Detects shipped retrieval/ranking/search systems
- Uses FAISS-based semantic retrieval
- Penalizes inactive or low-response candidates
- Adds explainable reasoning for every ranked candidate
- Runs locally with no API calls during ranking

## Outputs

```text
outputs/submission.csv
```

Final top-100 ranked candidates.

```text
outputs/diagnostics_top50.csv
```

Score breakdown for debugging and inspection.

## Notes

The full `data/candidates.jsonl` file is not committed to GitHub because it exceeds GitHub file size limits. Place it inside the `data/` folder before running the pipeline.
```
