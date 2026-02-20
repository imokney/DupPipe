from __future__ import annotations
import json
import os
from pathlib import Path

from .config import Profile
from .models import JobReport
from ..plugins.ffmpeg_tools import extract_audio
from ..plugins.separation_demucs import separate_if_enabled
from ..plugins.transcript_sources import segments_from_srt_or_asr
from ..plugins.segmentation import normalize_segments
from ..plugins.translate_argos import translate_segments
from ..plugins.speech_adapt import adapt_segments_for_timing
from ..plugins.voice_router import assign_voices
from ..plugins.text_compress import compress_segments_llm_if_enabled
from ..plugins.audio_post import loudness_normalize
from ..plugins.video_render import mux_audio_into_video, burn_subtitles
from ..plugins.tts_elevenlabs import synthesize_segments_with_cache, estimate_tts_cost
from ..plugins.assemble_timeline import assemble_voice_timeline
from ..plugins.mixer import mix_and_export
from ..plugins.qa import build_qa_summary


def run_job(
    input_video: Path,
    ru_srt: Path | None,
    profile: Profile,
    out_dir: Path,
    dry_run: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"

    # 1) Extract audio
    audio_wav = out_dir / "audio.wav"
    extract_audio(input_video, audio_wav)

    # 2) Separation (optional)
    vocals_wav, bg_wav = separate_if_enabled(audio_wav, out_dir, profile.separation)

    # 3) Transcript segments (from SRT or ASR)
    base_segments = segments_from_srt_or_asr(
        ru_srt=ru_srt,
        audio_for_asr=vocals_wav or audio_wav,
        out_dir=out_dir,
        asr_cfg=profile.asr
    )

    # 4) Normalize/segment shape
    segments = normalize_segments(base_segments, profile.segmentation)

    # 5) Translation RU->EN (offline Argos in starter)
    segments = translate_segments(segments, profile.translation)

    # 6) Speech adaptation (fit to timing)
    segments = adapt_segments_for_timing(segments, profile.timing_fit)

    # 6b) Optional LLM-based text compress (only touches flagged segments)
    segments = compress_segments_llm_if_enabled(segments, enabled=getattr(profile, 'cinema', None).text_compress if getattr(profile, 'cinema', None) else False)

    # 7) Voice routing (primary/fallback; optional accent test)
    segments, voice_decision = assign_voices(segments, profile, out_dir, dry_run=dry_run)

    # 8) Estimate cost and optionally stop
    est_credits, est_chars = estimate_tts_cost(segments, profile.elevenlabs)
    if dry_run:
        rep = JobReport(
            input_video=str(input_video),
            profile=profile.name,
            segments_total=len(segments),
            estimated_credits=est_credits,
            estimated_chars=est_chars,
            warnings=["DRY RUN: no ElevenLabs calls were made."] + ([] if voice_decision is None else [voice_decision]),
        )
        rep.qa_summary = build_qa_summary(segments)
        report_path.write_text(json.dumps(rep.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[dry-run] Wrote report: {report_path}")
        return

    # Set per-job controls for speed/pause matching (read by TTS module)
    if getattr(profile, 'cinema', None):
        os.environ['DUBPIPE_SPEED_MATCHING'] = '1' if profile.cinema.speed_matching else '0'
        os.environ['DUBPIPE_MAX_SPEEDUP'] = str(profile.cinema.max_speedup)
        os.environ['DUBPIPE_MIN_TEMPO'] = '1.0' if profile.cinema.pause_matching else str(profile.cinema.max_slowdown)
    
    # 9) TTS per segment with cache
    segments = synthesize_segments_with_cache(segments, profile.elevenlabs, profile.cache, out_dir)

    # 10) Assemble full EN voice track aligned to timeline
    voice_track_wav = out_dir / "en_voice_timeline.wav"
    assemble_voice_timeline(segments, voice_track_wav, total_duration_sec=get_audio_duration(audio_wav))

    # 11) Mix with background (if available) and export
    final_audio = mix_and_export(
        voice_wav=voice_track_wav,
        bg_wav=bg_wav if profile.mix.enabled else None,
        out_dir=out_dir,
        mix_cfg=profile.mix
    )

    # 11b) Optional loudness normalization
    if getattr(profile, 'cinema', None) and profile.cinema.loudness_normalize:
        norm_path = out_dir / (final_audio.stem + "_norm" + final_audio.suffix)
        try:
            final_audio = loudness_normalize(final_audio, norm_path, target_lufs=profile.cinema.target_lufs)
        except Exception as e:
            print(f"[warn] Loudness normalize failed: {e}")

    # 12) Export EN SRT
    srt_path = out_dir / "en.srt"
    export_en_srt(segments, srt_path)

    # 12b) Optional render final video
    if getattr(profile, 'cinema', None) and profile.cinema.render_video:
        try:
            out_mp4 = out_dir / "final_dubbed.mp4"
            mux_audio_into_video(input_video, final_audio, out_mp4)
            if profile.cinema.burn_subtitles:
                out_mp4s = out_dir / "final_dubbed_subs.mp4"
                burn_subtitles(input_video, final_audio, srt_path, out_mp4s)
        except Exception as e:
            print(f"[warn] Video render failed: {e}")

    # 13) Report
    qa_summary = build_qa_summary(segments)
    rep = JobReport(
        input_video=str(input_video),
        profile=profile.name,
        segments_total=len(segments),
        estimated_credits=est_credits,
        estimated_chars=est_chars,
        warnings=[] if voice_decision is None else [voice_decision],
        qa_summary=qa_summary,
    )
    report_path.write_text(json.dumps(rep.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Final audio: {final_audio}")
    print(f"Subtitles: {srt_path}")
    print(f"Report: {report_path}")


# ---- helpers kept here for starter simplicity ----

def get_audio_duration(wav_path: Path) -> float:
    from pydub import AudioSegment
    a = AudioSegment.from_file(wav_path)
    return len(a) / 1000.0


def export_en_srt(segments, srt_path: Path) -> None:
    import pysrt

    subs = pysrt.SubRipFile()
    for i, seg in enumerate(segments, start=1):
        start_ms = int(seg.start * 1000)
        end_ms = int(seg.end * 1000)
        sub = pysrt.SubRipItem(
            index=i,
            start=pysrt.SubRipTime(milliseconds=start_ms),
            end=pysrt.SubRipTime(milliseconds=end_ms),
            text=(seg.text_en_speak or seg.text_en_raw or "").strip(),
        )
        subs.append(sub)
    subs.save(str(srt_path), encoding="utf-8")
