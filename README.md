# Redrob AI Candidate Ranker

Ranks 100K candidates for a Senior AI Engineer JD using a 
multi-signal pipeline: semantic similarity, skill trust scoring, 
career trajectory classification, and behavioral availability signals.

## Run

```bash
pip install -r requirements.txt
python rank.py --candidates ./data/candidates.jsonl --out ./outputs/submission.csv
```

Runs in under 5 minutes on CPU after first-run embedding cache is built.