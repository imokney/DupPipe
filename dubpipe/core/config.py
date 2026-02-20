from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ElevenLabsConfig:
    api_key_env: str = "ELEVENLABS_API_KEY"
    model_id: str = "eleven_turbo_v2_5"
    output_format: str = "mp3_44100_128"
    language_code: Optional[str] = "en"
    voice_settings: Dict[str, Any] = field(default_factory=lambda: {
        "stability": 0.3,
        "similarity_boost": 0.9,
        "style": 0.1,
        "use_speaker_boost": True,
        "speed": 1.0,
    })
    # Cost estimator (editable): credits per character for your plan/model.
    credits_per_character: float = 0.5


@dataclass
class VoiceRouterConfig:
    # Map speaker -> preferred and fallback voice IDs
    # You can put your cloned voice id into primary, and an actor voice id into fallback.
    primary_voice_id: str
    fallback_voice_id: Optional[str] = None

    # If enabled, we do a tiny calibration that spends a small amount of credits:
    # generate ~2 segments with primary voice and run EN ASR to estimate intelligibility.
    accent_test_enabled: bool = False
    accent_test_segments: int = 2
    accent_test_wer_threshold: float = 0.20


@dataclass
class ASRConfig:
    backend: str = "faster_whisper"
    model_size: str = "medium"  # faster-whisper sizes: tiny, base, small, medium, large-v3, ...
    language: str = "ru"


@dataclass
class SeparationConfig:
    enabled: bool = True
    backend: str = "demucs"
    two_stems: str = "vocals"  # demucs option: vocals
    model: str = "htdemucs"
    # demucs can be slow on CPU; you may disable separation to speed up (but mixing quality will drop)


@dataclass
class TranslationConfig:
    backend: str = "argos"
    # you can later add: deepl, openai, etc.
    src_lang: str = "ru"
    tgt_lang: str = "en"


@dataclass
class SegmentationConfig:
    # Base guardrails
    min_dur_sec: float = 1.5
    max_dur_sec: float = 10.0
    # If ASR produces long segments, we will split on punctuation where possible.
    split_on_punct: bool = True

    # Smart merge (reduces "choppy" micro-segments)
    smart_merge: bool = True
    target_dur_sec: float = 4.5
    max_merge_gap_ms: int = 350
    prefer_sentence_boundary: bool = True

    # Semantic chunking (optional; uses word timestamps and optionally LLM)
    semantic_chunking: bool = False
    semantic_mode: str = "meaning"  # "sentences" | "meaning"
    max_chunk_sec: float = 10.0
    max_words: int = 32


@dataclass
class TimingFitConfig:
    max_speedup: float = 1.20
    max_slowdown: float = 0.85
    max_time_stretch: float = 0.10  # if stretch ratio exceeds this, we flag QA


@dataclass
class MixConfig:
    enabled: bool = True
    # If separation is enabled, we mix voice onto background stem.
    # If separation is disabled, we will output voice-only by default.
    bg_gain_db: float = -3.0
    voice_gain_db: float = 0.0
    export_format: str = "m4a"  # wav or m4a


@dataclass
class CacheConfig:
    root: Path = Path("./cache")
    tts_subdir: str = "tts"


@dataclass
class CinemaConfig:
    # "Cinema mode" knobs (all optional via GUI checkboxes)
    text_compress: bool = False
    speed_matching: bool = True
    pause_matching: bool = True
    loudness_normalize: bool = False
    retry_bad_segments: bool = False

    # Video export
    render_video: bool = True
    burn_subtitles: bool = False

    # Limits
    max_speedup: float = 1.08   # cap atempo speedup
    max_slowdown: float = 1.00 # 1.00 means no slowdown; keep pauses instead
    target_lufs: float = -16.0

@dataclass
class Profile:
    name: str = "standard"
    asr: ASRConfig = field(default_factory=ASRConfig)
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    elevenlabs: ElevenLabsConfig = field(default_factory=ElevenLabsConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    timing_fit: TimingFitConfig = field(default_factory=TimingFitConfig)
    mix: MixConfig = field(default_factory=MixConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    cinema: CinemaConfig = field(default_factory=CinemaConfig)
    voice_router: VoiceRouterConfig = None  # must be provided in YAML


def load_profile(profile_name: str) -> Profile:
    path = Path(__file__).resolve().parents[2] / "profiles" / f"{profile_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "voice_router" not in data:
        raise ValueError("Profile must define voice_router.primary_voice_id (and optionally fallback_voice_id).")

    p = Profile(name=profile_name)

    # Simple shallow mapping; keep starter minimal.
    def update_dataclass(dc, upd: dict):
        for k, v in (upd or {}).items():
            if hasattr(dc, k):
                setattr(dc, k, v)

    update_dataclass(p.asr, data.get("asr"))
    update_dataclass(p.separation, data.get("separation"))
    update_dataclass(p.translation, data.get("translation"))
    update_dataclass(p.elevenlabs, data.get("elevenlabs"))
    update_dataclass(p.segmentation, data.get("segmentation"))
    update_dataclass(p.timing_fit, data.get("timing_fit"))
    update_dataclass(p.mix, data.get("mix"))
    update_dataclass(p.cache, data.get("cache"))
    update_dataclass(p.cinema, data.get("cinema"))

    p.voice_router = VoiceRouterConfig(**data["voice_router"])
    # Expand cache root relative to project
    if isinstance(p.cache.root, str):
        p.cache.root = Path(p.cache.root)
    return p
