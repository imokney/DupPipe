from __future__ import annotations
from typing import List
import os
import re

from ..core.models import Segment

def compress_segments_llm_if_enabled(segments: List[Segment], enabled: bool) -> List[Segment]:
    """Optional: use OpenAI to compress EN text to fit segment timing.
    Only runs when enabled AND OPENAI_API_KEY is present.
    Falls back to heuristic compression otherwise.
    """
    if not enabled:
        return segments

    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        # No key -> heuristic-only pass (better than hard trim)
        return [ _heuristic_compress(s) for s in segments ]

    try:
        from openai import OpenAI
    except Exception:
        return [ _heuristic_compress(s) for s in segments ]

    client = OpenAI(api_key=key)

    for s in segments:
        if "needs_text_compress" not in (s.qa_flags or []):
            continue
        raw = (s.text_en_raw or "").strip()
        speak = (s.text_en_speak or raw).strip()
        if not speak:
            continue
        # Target words estimate: ~2.5 w/s, add 5% cushion
        target_words = max(3, int(2.5 * max(0.3, s.target_duration) * 0.95))
        prompt = (
            "You are adapting a translation for dubbing. "
            "Rewrite the English line to fit the same meaning and tone, "
            f"but shorter: target <= {target_words} words. "
            "Keep it natural spoken English. No brackets.\n\n"
            f"Original RU (context): {s.text_ru}\n"
            f"Current EN: {speak}\n"
            "Shortened EN:"
        )
        try:
            resp = client.responses.create(
                model="gpt-4.1-mini",
                input=prompt,
                max_output_tokens=80,
            )
            out = (resp.output_text or "").strip()
            out = re.sub(r"\s+", " ", out).strip()
            if out:
                s.text_en_speak = out
                # remove flag if we made it shorter
                if len(out) < len(speak):
                    try:
                        s.qa_flags.remove("needs_text_compress")
                    except Exception:
                        pass
        except Exception:
            # fallback heuristic on failure
            _heuristic_compress(s)

    return segments

_FILLERS = re.compile(r"\b(very|really|just|actually|basically|literally|kind of|sort of)\b", re.IGNORECASE)

def _heuristic_compress(s: Segment) -> Segment:
    t = (s.text_en_speak or s.text_en_raw or "").strip()
    if not t:
        return s
    t = re.sub(r"\s+", " ", t)
    t = _FILLERS.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    # drop leading discourse markers
    t = re.sub(r"^(Well|So|You know|I mean),\s+", "", t, flags=re.IGNORECASE)
    s.text_en_speak = t
    return s
