from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import math

from ..core.models import Segment
from ..core.config import Profile
from .tts_elevenlabs import synthesize_text_preview
from .qa import wer
from .transcript_sources import _segments_from_asr_faster_whisper
from ..core.config import ASRConfig


def assign_voices(
    segments: List[Segment],
    profile: Profile,
    out_dir: Path,
    dry_run: bool = False,
) -> Tuple[List[Segment], Optional[str]]:
    vr = profile.voice_router
    primary = vr.primary_voice_id
    fallback = vr.fallback_voice_id

    # Default: assign primary to all
    for s in segments:
        s.voice_id = primary

    decision_note = None

    if vr.accent_test_enabled and fallback and not dry_run:
        # Take first N segments with text, generate with primary voice, run EN ASR, compute WER
        sample_texts = [s.text_en_speak for s in segments if s.text_en_speak.strip()]
        sample_texts = sample_texts[: vr.accent_test_segments]
        if sample_texts:
            print("[info] Running accent/intelligibility test (small credit spend)...")
            preview_wavs = []
            combined_ref = " ".join(sample_texts)

            for i, txt in enumerate(sample_texts, start=1):
                wav_path = out_dir / f"_accent_test_{i}.wav"
                synthesize_text_preview(txt, voice_id=primary, profile=profile, out_wav=wav_path)
                preview_wavs.append(wav_path)

            # Concatenate previews, then ASR in English
            combined_wav = out_dir / "_accent_test_combined.wav"
            _concat_wavs(preview_wavs, combined_wav)

            # Run ASR in English on the synthetic audio
            en_asr_cfg = ASRConfig(model_size="small", language="en")
            hyp_segments = _segments_from_asr_faster_whisper(combined_wav, out_dir, en_asr_cfg)
            hyp = " ".join([s.text_ru for s in hyp_segments])  # text_ru field is used for transcript in starter
            score = wer(combined_ref, hyp)

            print(f"[info] Accent test WER={score:.3f} (threshold {vr.accent_test_wer_threshold})")
            if score >= vr.accent_test_wer_threshold:
                for s in segments:
                    s.voice_id = fallback
                    s.qa_flags.append("voice_fallback_due_to_accent")
                decision_note = f"VoiceRouter: accent test failed (WER={score:.3f}); used fallback voice_id."
            else:
                decision_note = f"VoiceRouter: accent test passed (WER={score:.3f}); kept primary voice_id."

    # Compute per-segment speed setting from heuristic desired_speed
    for s in segments:
        desired = float(s.tts_params.get("desired_speed", 1.0))
        # Clamp into ElevenLabs speed supported range (the SDK allows speed in voice settings; keep safe range here)
        speed = max(profile.timing_fit.max_slowdown, min(profile.timing_fit.max_speedup, desired))
        s.tts_params["speed"] = speed

    return segments, decision_note


def _concat_wavs(wavs: List[Path], out_wav: Path) -> None:
    from pydub import AudioSegment
    combined = AudioSegment.empty()
    for w in wavs:
        combined += AudioSegment.from_file(w)
    combined.export(out_wav, format="wav")
