from __future__ import annotations
from pathlib import Path
from typing import List, Optional

from ..core.config import ASRConfig
from ..core.models import Segment, Word


def segments_from_srt_or_asr(
    ru_srt: Optional[Path],
    audio_for_asr: Path,
    out_dir: Path,
    asr_cfg: ASRConfig,
) -> List[Segment]:
    if ru_srt is not None:
        return _segments_from_srt(ru_srt)

    # Otherwise run ASR
    return _segments_from_asr_faster_whisper(audio_for_asr, out_dir, asr_cfg)


def _segments_from_srt(srt_path: Path) -> List[Segment]:
    import pysrt

    subs = pysrt.open(str(srt_path), encoding="utf-8")
    out: List[Segment] = []
    for i, s in enumerate(subs, start=1):
        start = s.start.ordinal / 1000.0
        end = s.end.ordinal / 1000.0
        text = (s.text or "").replace("\n", " ").strip()
        out.append(Segment(seg_id=f"{i:05d}", start=start, end=end, text_ru=text))
    return out


def _segments_from_asr_faster_whisper(audio_path: Path, out_dir: Path, cfg: ASRConfig) -> List[Segment]:
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise RuntimeError(
            "ASR requested but faster-whisper is not installed. "
            "Install it or provide --ru-srt to skip ASR."
        ) from e

    # You can tune device="cuda" if you have GPU.
    model = WhisperModel(cfg.model_size, device="auto", compute_type="auto")

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=cfg.language,
        vad_filter=True,
        beam_size=5,
        word_timestamps=True,
    )

    out: List[Segment] = []
    for i, seg in enumerate(segments_iter, start=1):
        text = (seg.text or "").strip()
        out.append(Segment(
            seg_id=f"{i:05d}",
            start=float(seg.start),
            end=float(seg.end),
            text_ru=text,
            words=[Word(w=(w.word or '').strip(), s=float(w.start), e=float(w.end), p=getattr(w, 'probability', None)) for w in (getattr(seg, 'words', None) or [])],
        ))

    # Save debug transcript
    dbg = out_dir / "ru_transcript.txt"
    dbg.write_text("\n".join([f"[{s.start:.2f}-{s.end:.2f}] {s.text_ru}" for s in out]), encoding="utf-8")
    return out
