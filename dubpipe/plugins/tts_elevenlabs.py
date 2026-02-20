from __future__ import annotations
import hashlib
import os
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.config import ElevenLabsConfig, CacheConfig, Profile
from ..core.models import Segment
from .ffmpeg_tools import time_stretch_wav


def _get_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return float(default)



def estimate_tts_cost(segments: List[Segment], cfg: ElevenLabsConfig) -> Tuple[float, int]:
    chars = sum(len((s.text_en_speak or "").strip()) for s in segments if (s.text_en_speak or "").strip())
    credits = chars * float(cfg.credits_per_character)
    return float(credits), int(chars)


def synthesize_segments_with_cache(
    segments: List[Segment],
    tts_cfg: ElevenLabsConfig,
    cache_cfg: CacheConfig,
    out_dir: Path,
) -> List[Segment]:
    load_dotenv()
    api_key = os.getenv(tts_cfg.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing {tts_cfg.api_key_env}. Put it in your environment or .env file.")

    # Lazy import to keep startup fast
    from elevenlabs.client import ElevenLabs
    from elevenlabs import VoiceSettings

    client = ElevenLabs(api_key=api_key)

    cache_root = Path(cache_cfg.root) / cache_cfg.tts_subdir
    cache_root.mkdir(parents=True, exist_ok=True)

    seg_audio_dir = out_dir / "segments_audio"
    seg_audio_dir.mkdir(parents=True, exist_ok=True)

    for s in segments:
        text = (s.text_en_speak or "").strip()
        if not text:
            s.qa_flags.append("empty_text")
            continue

        # Build per-segment voice settings
        voice_settings = dict(tts_cfg.voice_settings or {})
        # Override speed per segment if provided
        if "speed" in s.tts_params:
            voice_settings["speed"] = float(s.tts_params["speed"])

        cache_key = _hash_key(
            voice_id=s.voice_id or "",
            model_id=tts_cfg.model_id,
            output_format=tts_cfg.output_format,
            language_code=tts_cfg.language_code or "",
            voice_settings=voice_settings,
            text=text,
        )
        cache_mp3 = cache_root / f"{cache_key}.mp3"
        out_mp3 = seg_audio_dir / f"{s.seg_id}.mp3"
        out_wav = seg_audio_dir / f"{s.seg_id}.wav"

        if cache_mp3.exists():
            out_mp3.write_bytes(cache_mp3.read_bytes())
        else:
            audio_iter = _tts_convert_with_retry(
                client=client,
                voice_id=s.voice_id,
                text=text,
                model_id=tts_cfg.model_id,
                output_format=tts_cfg.output_format,
                voice_settings=VoiceSettings(**voice_settings),
                language_code=tts_cfg.language_code,
            )
            with open(cache_mp3, "wb") as f:
                for chunk in audio_iter:
                    if chunk:
                        f.write(chunk)
            out_mp3.write_bytes(cache_mp3.read_bytes())

        # Convert to wav for timeline assembly / mixing
        _mp3_to_wav(out_mp3, out_wav)

        # Optional timing fit by time-stretch, if audio duration differs too much
        from pydub import AudioSegment
        a = AudioSegment.from_file(out_wav)
        dur = len(a) / 1000.0
        s.audio_duration = dur

        target = s.target_duration
        if target > 0.2 and dur > 0.2:
            ratio = dur / target  # >1 means we need to speed up (tempo>1)
            # Controls from runner (set per-job):
            enable = os.getenv('DUBPIPE_SPEED_MATCHING', '1').strip() not in ('0','false','False')
            max_speedup = _get_float_env('DUBPIPE_MAX_SPEEDUP', 1.08)
            min_tempo = _get_float_env('DUBPIPE_MIN_TEMPO', 1.00)  # 1.00 => no slowdown (keep pauses)
            if enable:
                tempo = max(min_tempo, min(max_speedup, ratio))
            else:
                tempo = 1.0
            if abs(1.0 - tempo) > 0.02:
                stretched = seg_audio_dir / f"{s.seg_id}.stretched.wav"
                time_stretch_wav(out_wav, stretched, tempo=tempo)
                out_wav.unlink(missing_ok=True)
                stretched.rename(out_wav)
                a2 = AudioSegment.from_file(out_wav)
                s.audio_duration = len(a2) / 1000.0

        s.audio_path = str(out_wav)

    return segments


def synthesize_text_preview(text: str, voice_id: str, profile: Profile, out_wav: Path) -> None:
    # Minimal helper for accent test: do a short convert and export as wav.
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv(profile.elevenlabs.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing {profile.elevenlabs.api_key_env}. Put it in your environment or .env file.")

    from elevenlabs.client import ElevenLabs
    from elevenlabs import VoiceSettings

    client = ElevenLabs(api_key=api_key)
    # We write mp3 then convert to wav
    tmp_mp3 = out_wav.with_suffix(".mp3")

    audio_iter = _tts_convert_with_retry(
        client=client,
        voice_id=voice_id,
        text=text,
        model_id=profile.elevenlabs.model_id,
        output_format=profile.elevenlabs.output_format,
        voice_settings=VoiceSettings(**profile.elevenlabs.voice_settings),
        language_code=profile.elevenlabs.language_code,
    )
    with open(tmp_mp3, "wb") as f:
        for chunk in audio_iter:
            if chunk:
                f.write(chunk)
    _mp3_to_wav(tmp_mp3, out_wav)
    tmp_mp3.unlink(missing_ok=True)


def _mp3_to_wav(mp3_path: Path, wav_path: Path) -> None:
    from pydub import AudioSegment
    a = AudioSegment.from_file(mp3_path)
    a.export(wav_path, format="wav")


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=20))
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=20))
def _tts_convert_with_retry(
    client,
    voice_id: str,
    text: str,
    model_id: str,
    output_format: str,
    voice_settings,
    language_code: str | None,
):
    # ElevenLabs SDK returns an iterator of bytes for convert()
    kwargs = dict(
        voice_id=voice_id,
        text=text,
        model_id=model_id,
        output_format=output_format,
        voice_settings=voice_settings,
    )
    # language_code is supported by the HTTP API; some SDK versions may not expose it.
    if language_code:
        kwargs["language_code"] = language_code

    try:
        return client.text_to_speech.convert(**kwargs)
    except TypeError:
        # Fallback for SDK versions that don't accept language_code
        kwargs.pop("language_code", None)
        return client.text_to_speech.convert(**kwargs)


def _hash_key(
    voice_id: str,
    model_id: str,
    output_format: str,
    language_code: str,
    voice_settings: dict,
    text: str,
) -> str:
    import json
    payload = {
        "voice_id": voice_id,
        "model_id": model_id,
        "output_format": output_format,
        "language_code": language_code,
        "voice_settings": voice_settings,
        "text": text,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]
