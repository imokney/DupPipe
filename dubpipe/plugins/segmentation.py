from __future__ import annotations

import os
import json
import re
from typing import List, Tuple

from ..core.config import SegmentationConfig
from ..core.models import Segment, Word

_PUNCT_SPLIT = re.compile(r"(?<=[\.!\?…])\s+")
_END_PUNCT = re.compile(r"[\.!\?…]\s*$")


def normalize_segments(segments: List[Segment], cfg: SegmentationConfig) -> List[Segment]:
    """Shape the segment list so TTS sounds natural.

    Stages:
      1) Optional punctuation split for overly-long segments (word-aware if available)
      2) Smart merge to avoid micro-chunks
      3) Optional semantic chunking (word-aware; optionally LLM)
    """
    out: List[Segment] = []
    for seg in segments:
        dur = seg.target_duration
        if cfg.split_on_punct and dur > cfg.max_dur_sec and (seg.text_ru or ""):
            out.extend(_split_segment_on_punct(seg))
        else:
            out.append(seg)

    if cfg.smart_merge:
        out = _smart_merge(out, cfg)

    if cfg.semantic_chunking:
        out = _semantic_chunk(out, cfg)

    # Final: ensure no empties
    out = [s for s in out if (s.text_ru or "").strip() and s.end > s.start]
    # Re-id sequentially
    for i, s in enumerate(out, start=1):
        s.seg_id = f"{i:05d}"
    return out


def _split_segment_on_punct(seg: Segment) -> List[Segment]:
    parts = [p.strip() for p in _PUNCT_SPLIT.split(seg.text_ru or "") if p.strip()]
    if len(parts) <= 1:
        return [seg]

    # If we have word timestamps, align splits to nearest word boundaries.
    if seg.words:
        return _split_with_words(seg, parts)

    # Fallback: split time evenly
    t0 = seg.start
    total = max(0.001, seg.end - seg.start)
    step = total / len(parts)
    out: List[Segment] = []
    for j, part in enumerate(parts):
        out.append(Segment(
            seg_id=f"{seg.seg_id}_{j}",
            start=t0 + j * step,
            end=t0 + (j + 1) * step,
            speaker=seg.speaker,
            text_ru=part,
            words=[],
        ))
    return out


def _split_with_words(seg: Segment, parts: List[str]) -> List[Segment]:
    # Build a running text from words to find approximate boundaries.
    # We'll split by distributing words proportionally to part lengths.
    words = seg.words
    if not words:
        return [seg]

    # Simple proportional allocation by character length.
    total_chars = sum(len(p) for p in parts)
    if total_chars <= 0:
        return [seg]

    # Precompute cumulative char targets
    targets = []
    cum = 0
    for p in parts:
        cum += len(p)
        targets.append(cum / total_chars)

    out: List[Segment] = []
    n = len(words)
    last_idx = 0
    for j, frac in enumerate(targets):
        idx = int(round(frac * n))
        idx = max(last_idx + 1, min(n, idx))
        chunk_words = words[last_idx:idx]
        if not chunk_words:
            continue
        text = _words_to_text(chunk_words)
        out.append(Segment(
            seg_id=f"{seg.seg_id}_{j}",
            start=float(chunk_words[0].s),
            end=float(chunk_words[-1].e),
            speaker=seg.speaker,
            text_ru=text,
            words=chunk_words,
        ))
        last_idx = idx
    return out or [seg]


def _smart_merge(segments: List[Segment], cfg: SegmentationConfig) -> List[Segment]:
    """Merge adjacent segments to reduce choppiness while keeping timing sane."""
    if not segments:
        return []

    max_gap = float(cfg.max_merge_gap_ms) / 1000.0
    merged: List[Segment] = []
    buf = segments[0]

    def buf_ends_sentence(s: Segment) -> bool:
        return bool(_END_PUNCT.search((s.text_ru or "")))

    for seg in segments[1:]:
        gap = max(0.0, float(seg.start) - float(buf.end))
        cand_dur = float(seg.end) - float(buf.start)

        should_merge = False

        # Merge if buffer is too short
        if buf.target_duration < cfg.min_dur_sec:
            should_merge = True

        # Merge if small gap and we're under target duration
        if gap <= max_gap and cand_dur <= max(cfg.max_dur_sec, cfg.max_chunk_sec):
            if buf.target_duration < cfg.target_dur_sec:
                should_merge = True
            # Optionally avoid merging across strong sentence boundary unless very short
            if cfg.prefer_sentence_boundary and buf_ends_sentence(buf) and buf.target_duration >= cfg.min_dur_sec:
                # keep boundary unless the next piece is tiny
                if seg.target_duration < cfg.min_dur_sec:
                    should_merge = True
                else:
                    should_merge = False

        if should_merge:
            buf = _merge_two(buf, seg)
        else:
            merged.append(buf)
            buf = seg

    merged.append(buf)
    return merged


def _merge_two(a: Segment, b: Segment) -> Segment:
    a.end = max(float(a.end), float(b.end))
    a.text_ru = (str(a.text_ru or "") + " " + str(b.text_ru or "")).strip()
    # Merge words if present
    if a.words or b.words:
        a.words = (a.words or []) + (b.words or [])
    return a


def _semantic_chunk(segments: List[Segment], cfg: SegmentationConfig) -> List[Segment]:
    """Re-chunk using word timestamps so chunks align to sentence/meaning boundaries.

    If OPENAI_API_KEY is available and cfg.semantic_mode == "meaning",
    we ask the model for boundary indices; otherwise we do a punctuation heuristic.
    """
    # Flatten words; if we don't have words, fall back to heuristic on segment texts.
    words: List[Word] = []
    speaker = segments[0].speaker if segments else "spk_0"
    for s in segments:
        speaker = s.speaker or speaker
        if s.words:
            words.extend(s.words)

    if not words:
        return _semantic_chunk_no_words(segments, cfg)

    boundaries = None
    if cfg.semantic_mode.lower() == "meaning" and os.getenv("OPENAI_API_KEY", "").strip():
        try:
            boundaries = _llm_boundaries(words, cfg)
        except Exception:
            boundaries = None

    if not boundaries:
        boundaries = _heuristic_boundaries(words, cfg)

    out: List[Segment] = []
    for i, (lo, hi) in enumerate(boundaries, start=1):
        chunk = words[lo:hi]
        if not chunk:
            continue
        out.append(Segment(
            seg_id=f"sc_{i:03d}",
            start=float(chunk[0].s),
            end=float(chunk[-1].e),
            speaker=speaker,
            text_ru=_words_to_text(chunk),
            words=chunk,
        ))
    return out or segments


def _semantic_chunk_no_words(segments: List[Segment], cfg: SegmentationConfig) -> List[Segment]:
    # Fallback: merge segments until punctuation / duration thresholds.
    out: List[Segment] = []
    buf = None
    for s in segments:
        if buf is None:
            buf = s
            continue
        cand = Segment(seg_id=buf.seg_id, start=buf.start, end=s.end, speaker=buf.speaker,
                       text_ru=(buf.text_ru + " " + s.text_ru).strip(), words=[])
        if cand.target_duration <= cfg.max_chunk_sec and (buf.target_duration < cfg.target_dur_sec):
            buf = cand
            # If we reached a sentence boundary, flush
            if cfg.prefer_sentence_boundary and _END_PUNCT.search(buf.text_ru or ""):
                out.append(buf); buf = None
        else:
            out.append(buf); buf = s
    if buf is not None:
        out.append(buf)
    return out


def _heuristic_boundaries(words: List[Word], cfg: SegmentationConfig) -> List[Tuple[int, int]]:
    # Create boundaries mostly at sentence punctuation, respecting duration and max_words.
    max_sec = float(cfg.max_chunk_sec)
    min_sec = float(cfg.min_dur_sec)
    max_words = int(cfg.max_words)

    boundaries: List[Tuple[int, int]] = []
    lo = 0
    while lo < len(words):
        hi = lo
        start_t = words[lo].s
        last_good = None

        while hi < len(words):
            w = words[hi].w or ""
            end_t = words[hi].e
            dur = float(end_t - start_t)

            # Candidate end if sentence punctuation
            if cfg.prefer_sentence_boundary and re.search(r"[\.!\?…]$", w):
                if dur >= min_sec:
                    last_good = hi + 1

            # Force cut if exceeds constraints
            if dur >= max_sec or (hi - lo + 1) >= max_words:
                break
            hi += 1

        # Prefer last_good if exists and not too small
        if last_good and last_good > lo:
            cut = last_good
        else:
            cut = min(len(words), max(lo + 1, hi + 1))
        boundaries.append((lo, cut))
        lo = cut

    return boundaries


def _llm_boundaries(words: List[Word], cfg: SegmentationConfig) -> List[Tuple[int, int]]:
    """Ask an LLM to propose chunk boundaries by word indices.

    Output is a list of [start_idx, end_idx) pairs.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # To control cost, process in windows.
    window_words = 260
    boundaries: List[Tuple[int, int]] = []

    offset = 0
    while offset < len(words):
        chunk = words[offset: offset + window_words]
        # Build compact text with indices
        # e.g. 0:Hello 1:world 2:...
        indexed = " ".join([f"{i}:{w.w}" for i, w in enumerate(chunk)])
        prompt = (
            "You are segmenting a transcript for dubbing.\n"
            "Return JSON with key boundaries: a list of [start,end] word-index pairs (end exclusive).\n"
            "Rules: keep sentences/meaningful phrases intact; each chunk 2-10 seconds; max 32 words; avoid splitting questions; avoid 1-word chunks.\n"
            "Text with word indices:\n" + indexed
        )
        r = client.responses.create(
            model=os.getenv("DUBPIPE_OPENAI_MODEL", "gpt-4o-mini"),
            input=prompt,
        )
        txt = (r.output_text or "").strip()
        # Extract JSON
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            # fallback heuristic for this window
            local = _heuristic_boundaries(chunk, cfg)
            boundaries.extend([(a+offset, b+offset) for a,b in local])
            offset += len(chunk)
            continue
        obj = json.loads(m.group(0))
        pairs = obj.get("boundaries", [])
        local_pairs = []
        for p in pairs:
            if not (isinstance(p, list) and len(p)==2):
                continue
            a,b = int(p[0]), int(p[1])
            if b<=a: 
                continue
            a = max(0, min(len(chunk), a))
            b = max(0, min(len(chunk), b))
            if b-a < 2:
                continue
            local_pairs.append((a+offset, b+offset))
        if not local_pairs:
            local_pairs = [(a+offset, b+offset) for a,b in _heuristic_boundaries(chunk, cfg)]
        boundaries.extend(local_pairs)
        offset += len(chunk)

    # Coalesce gaps/overlaps conservatively
    boundaries2: List[Tuple[int,int]] = []
    cur = 0
    for a,b in boundaries:
        if a > cur:
            a = cur
        if a < cur:
            a = cur
        if b <= a:
            continue
        boundaries2.append((a,b))
        cur = b
    if boundaries2 and boundaries2[-1][1] < len(words):
        boundaries2.append((boundaries2[-1][1], len(words)))
    if not boundaries2:
        return _heuristic_boundaries(words, cfg)
    return boundaries2


def _words_to_text(words: List[Word]) -> str:
    # Join words and clean spacing before punctuation.
    text = " ".join([w.w for w in words if (w.w or "").strip()]).strip()
    # Fix spaces before punctuation
    text = re.sub(r"\s+([,.;:!?…])", r"\1", text)
    text = re.sub(r"\s+([\)\]\}])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    return text
