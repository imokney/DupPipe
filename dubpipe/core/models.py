from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Word:
    w: str
    s: float
    e: float
    p: Optional[float] = None


@dataclass
class Segment:
    seg_id: str
    start: float
    end: float
    speaker: str = "spk_0"
    text_ru: str = ""
    words: List[Word] = field(default_factory=list)

    # Added by translation/adaptation
    text_en_raw: str = ""
    text_en_speak: str = ""

    # Voice + rendering
    voice_id: Optional[str] = None
    tts_params: Dict[str, Any] = field(default_factory=dict)

    # Output paths
    audio_path: Optional[str] = None
    audio_duration: Optional[float] = None

    qa_flags: List[str] = field(default_factory=list)

    @property
    def target_duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class JobReport:
    input_video: str
    profile: str
    segments_total: int
    estimated_credits: float
    estimated_chars: int
    warnings: List[str] = field(default_factory=list)
    qa_summary: Dict[str, Any] = field(default_factory=dict)
