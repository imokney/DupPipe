from __future__ import annotations
from pathlib import Path
from typing import List

from pydub import AudioSegment

from ..core.models import Segment


def assemble_voice_timeline(segments: List[Segment], out_wav: Path, total_duration_sec: float) -> None:
    total_ms = int(total_duration_sec * 1000)
    timeline = AudioSegment.silent(duration=total_ms)

    for s in segments:
        if not s.audio_path:
            continue
        seg_audio = AudioSegment.from_file(s.audio_path)
        start_ms = int(s.start * 1000)
        timeline = timeline.overlay(seg_audio, position=start_ms)

    timeline.export(out_wav, format="wav")
