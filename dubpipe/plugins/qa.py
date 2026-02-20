from __future__ import annotations
from collections import Counter
from typing import Dict, List
import re

from ..core.models import Segment


def build_qa_summary(segments: List[Segment]) -> Dict:
    flags = Counter()
    dur_off = []
    for s in segments:
        for f in s.qa_flags:
            flags[f] += 1
        if s.audio_duration is not None:
            dur_off.append(abs(s.audio_duration - s.target_duration))

    return {
        "flags": dict(flags),
        "segments_with_flags": sum(1 for s in segments if s.qa_flags),
        "avg_abs_duration_diff_sec": (sum(dur_off) / len(dur_off)) if dur_off else None,
        "max_abs_duration_diff_sec": max(dur_off) if dur_off else None,
    }


def wer(ref: str, hyp: str) -> float:
    # Simple word error rate (Levenshtein distance on words)
    r = _tokenize(ref)
    h = _tokenize(hyp)
    if not r:
        return 0.0 if not h else 1.0

    # DP edit distance
    dp = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1):
        dp[i][0] = i
    for j in range(len(h)+1):
        dp[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            cost = 0 if r[i-1] == h[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    return dp[len(r)][len(h)] / len(r)


def _tokenize(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split() if s else []
