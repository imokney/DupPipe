from __future__ import annotations
import re
from typing import List

from ..core.config import TimingFitConfig
from ..core.models import Segment

# Heuristic: average English speech ~ 150 wpm ≈ 2.5 w/s.
# average word length incl. space ~ 6 chars => ~15 chars/sec.
DEFAULT_CHARS_PER_SEC = 15.0

_FILLERS = re.compile(r"\b(very|really|just|actually|basically|literally)\b", re.IGNORECASE)


def adapt_segments_for_timing(segments: List[Segment], cfg: TimingFitConfig) -> List[Segment]:
    for s in segments:
        t = (s.text_en_raw or "").strip()
        # Basic cleanup
        t = re.sub(r"\s+", " ", t).strip()
        t = _FILLERS.sub("", t)
        t = re.sub(r"\s+", " ", t).strip()

        # Fit estimate: max chars we can say within target duration at max speedup
        if s.target_duration <= 0.2:
            s.text_en_speak = t
            s.qa_flags.append("tiny_segment")
            continue

        max_chars = int(DEFAULT_CHARS_PER_SEC * s.target_duration * cfg.max_speedup)
        if len(t) > max_chars:
            # Mark for compress; keep text as-is for higher quality. A later optional step may compress it.
            s.text_en_speak = t
            s.qa_flags.append("needs_text_compress")
        else:
            s.text_en_speak = t

        # Store desired speed (heuristic)
        est_dur = max(0.3, len(s.text_en_speak) / DEFAULT_CHARS_PER_SEC)
        desired_speed = est_dur / s.target_duration  # >1 means faster speech
        s.tts_params["desired_speed"] = float(desired_speed)

    return segments
