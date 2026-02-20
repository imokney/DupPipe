from __future__ import annotations
from pathlib import Path
from typing import Optional

from pydub import AudioSegment
from ..core.config import MixConfig


def mix_and_export(voice_wav: Path, bg_wav: Optional[Path], out_dir: Path, mix_cfg: MixConfig) -> Path:
    voice = AudioSegment.from_file(voice_wav)
    voice = voice + float(mix_cfg.voice_gain_db)

    if bg_wav and bg_wav.exists():
        bg = AudioSegment.from_file(bg_wav)
        bg = bg + float(mix_cfg.bg_gain_db)

        # Ensure same length
        if len(bg) < len(voice):
            bg = bg + AudioSegment.silent(duration=(len(voice) - len(bg)))
        if len(voice) < len(bg):
            voice = voice + AudioSegment.silent(duration=(len(bg) - len(voice)))

        mixed = bg.overlay(voice)
    else:
        mixed = voice

    if mix_cfg.export_format.lower() == "wav":
        out = out_dir / "en_dub_audio.wav"
        mixed.export(out, format="wav")
        return out

    # Default: m4a (AAC) – requires ffmpeg
    out = out_dir / "en_dub_audio.m4a"
    mixed.export(out, format="ipod")  # pydub uses ffmpeg; "ipod" usually maps to AAC/m4a
    return out
