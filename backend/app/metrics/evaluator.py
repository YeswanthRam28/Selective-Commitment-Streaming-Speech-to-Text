import re
import string
import numpy as np
from typing import List, Dict, Any

def normalize_text(text: str) -> str:
    """
    Standard text normalization for speech evaluation:
    - Lowercase
    - Remove punctuation
    - Normalize whitespace
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    tokens = text.strip().split()
    return " ".join(tokens)

def compute_wer(reference: str, hypothesis: str) -> Dict[str, Any]:
    """
    Computes Word Error Rate using Levenshtein distance on normalized tokens.
    """
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    r_len = len(ref_words)
    h_len = len(hyp_words)

    if r_len == 0:
        return {
            "wer": 1.0 if h_len > 0 else 0.0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": h_len,
            "ref_len": 0,
            "hyp_len": h_len
        }

    # DP table
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub = dp[i - 1][j - 1] + 1
                ins = dp[i][j - 1] + 1
                dlt = dp[i - 1][j] + 1
                dp[i][j] = min(sub, ins, dlt)

    total_edits = dp[r_len][h_len]
    wer = total_edits / r_len

    return {
        "wer": round(wer, 4),
        "total_edits": total_edits,
        "ref_len": r_len,
        "hyp_len": h_len
    }

def summarize_latencies(latencies: List[float]) -> Dict[str, float]:
    if not latencies:
        return {"median": 0.0, "p90": 0.0, "p99": 0.0, "mean": 0.0}
    arr = np.array(latencies)
    return {
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(np.mean(arr))
    }
