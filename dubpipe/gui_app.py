from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
from dotenv import load_dotenv

from .core.config import load_profile
from .core.runner import run_job


def _coerce_file_path(file_obj) -> Optional[Path]:
    if file_obj is None:
        return None
    # Gradio may return a string path or an object with .name/.path
    if isinstance(file_obj, (str, Path)):
        return Path(file_obj)

    for attr in ("name", "path"):
        if hasattr(file_obj, attr):
            v = getattr(file_obj, attr)
            if v:
                return Path(v)

    if isinstance(file_obj, dict):
        for k in ("name", "path"):
            v = file_obj.get(k)
            if v:
                return Path(v)

    return None


def _safe_job_name(default: str) -> str:
    keep = []
    for ch in default:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch in (" ", "."):
            keep.append("_")
    s = "".join(keep).strip("_")
    return s or "job"


def _prepare_work_dir(out_base: Path, job_name: str) -> Path:
    out_base.mkdir(parents=True, exist_ok=True)
    out_dir = out_base / job_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _copy_into_job(src: Path, out_dir: Path, new_name: str) -> Path:
    dst = out_dir / new_name
    shutil.copy2(src, dst)
    return dst


def _maybe_set_api_key(api_key: str, env_name: str = "ELEVENLABS_API_KEY") -> None:
    api_key = (api_key or "").strip()
    if api_key:
        os.environ[env_name] = api_key


def _load_voices(api_key: str) -> str:
    """Fetch available ElevenLabs voices (for GUI dropdown/help)."""
    load_dotenv()
    _maybe_set_api_key(api_key)
    key = os.getenv("ELEVENLABS_API_KEY")
    if not key:
        return "No API key found. Put ELEVENLABS_API_KEY in your .env or paste it into the field."

    try:
        import requests
    except Exception:
        return "Missing dependency: requests. Install it with: pip install requests"

    url = "https://api.elevenlabs.io/v1/voices"
    headers = {
        "xi-api-key": key,
        "accept": "application/json",
        "user-agent": "dubpipe-starter/0.1 (voice-list)",
    }

    try:
        r = requests.get(url, headers=headers, timeout=20)
    except Exception as e:
        return f"Failed to call ElevenLabs API: {e}"

    if r.status_code != 200:
        body_preview = (r.text or "")[:600]
        hint = ""
        if "cf-mitigated" in (r.headers or {}) or "Just a moment" in body_preview:
            hint = (
                "\n\nHint: It looks like your request was blocked by Cloudflare / regional restrictions. "
                "Try a different network/VPN/egress IP, or check if ElevenLabs is accessible from your region."
            )
        return f"HTTP {r.status_code} from {url}. Body (first 600 chars):\n{body_preview}{hint}"

    try:
        data = r.json()
    except Exception:
        return "Could not parse JSON from ElevenLabs response."

    items = data.get("voices", []) if isinstance(data, dict) else []
    if not items:
        return "No voices returned."

    out_lines = []
    for v in items:
        vid = v.get("voice_id")
        name = v.get("name")
        category = v.get("category")
        out_lines.append(f"- {name} | id={vid} | category={category}")

    return "\n".join(out_lines)

def _run(
    video_file,
    ru_srt_file,
    profile_name: str,
    out_base: str,
    job_name: str,
    dry_run: bool,
    eleven_api_key: str,
    openai_api_key: str,
    primary_voice_id: str,
    fallback_voice_id: str,
    accent_test_enabled: bool,
    separation_enabled: bool,
    model_id: str,
    output_format: str,
    credits_per_character: float,
    seg_min_dur: float,
    seg_max_dur: float,
    split_on_punct: bool,
    smart_merge: bool,
    semantic_chunking: bool,
    semantic_mode: str,
    max_merge_gap: float,
    text_compress: bool,
    speed_matching: bool,
    pause_matching: bool,
    loudness_normalize: bool,
    render_video: bool,
    burn_subtitles: bool,
    max_speedup: float,
    max_slowdown: float,
    target_lufs: float,
) -> Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
    load_dotenv()
    _maybe_set_api_key(eleven_api_key)
    _maybe_set_api_key(openai_api_key, "OPENAI_API_KEY")

    video_path = _coerce_file_path(video_file)
    srt_path = _coerce_file_path(ru_srt_file)

    if video_path is None or not video_path.exists():
        return ("Please upload a video file.", None, None, None, None)

    if not (job_name or "").strip():
        job_name = _safe_job_name(video_path.stem)

    out_dir = _prepare_work_dir(Path(out_base), job_name)

    # Copy inputs into job dir for reproducibility
    job_video = _copy_into_job(video_path, out_dir, "input_video" + video_path.suffix)
    job_srt = None
    if srt_path and srt_path.exists():
        job_srt = _copy_into_job(srt_path, out_dir, "ru_subs.srt")

    profile = load_profile(profile_name)


        # Segmentation controls (GUI)
    try:
        profile.segmentation.min_dur_sec = float(seg_min_dur)
        profile.segmentation.max_dur_sec = float(seg_max_dur)
        profile.segmentation.split_on_punct = bool(split_on_punct)
        # Optional knobs (exist depending on profile schema)
        if hasattr(profile.segmentation, "smart_merge"):
            profile.segmentation.smart_merge = bool(smart_merge)
        if hasattr(profile.segmentation, "semantic_chunking"):
            profile.segmentation.semantic_chunking = bool(semantic_chunking)
        elif hasattr(profile.segmentation, "semantic_chunking_enabled"):
            profile.segmentation.semantic_chunking_enabled = bool(semantic_chunking)
        if hasattr(profile.segmentation, "semantic_mode"):
            profile.segmentation.semantic_mode = str(semantic_mode)
        if hasattr(profile.segmentation, "max_merge_gap_sec"):
            profile.segmentation.max_merge_gap_sec = float(max_merge_gap)
        elif hasattr(profile.segmentation, "max_merge_gap"):
            profile.segmentation.max_merge_gap = float(max_merge_gap)
    except Exception:
        pass


    # Apply overrides
    if (primary_voice_id or "").strip():
        profile.voice_router.primary_voice_id = primary_voice_id.strip()
    if (fallback_voice_id or "").strip():
        profile.voice_router.fallback_voice_id = fallback_voice_id.strip()
    else:
        profile.voice_router.fallback_voice_id = None

    profile.voice_router.accent_test_enabled = bool(accent_test_enabled)
    profile.separation.enabled = bool(separation_enabled)

    if (model_id or "").strip():
        profile.elevenlabs.model_id = model_id.strip()
    if (output_format or "").strip():
        profile.elevenlabs.output_format = output_format.strip()
    try:
        profile.elevenlabs.credits_per_character = float(credits_per_character)
    except Exception:
        pass


    # Cinema mode toggles (GUI checkboxes)
    try:
        profile.cinema.text_compress = bool(text_compress)
        profile.cinema.speed_matching = bool(speed_matching)
        profile.cinema.pause_matching = bool(pause_matching)
        profile.cinema.loudness_normalize = bool(loudness_normalize)
        profile.cinema.render_video = bool(render_video)
        profile.cinema.burn_subtitles = bool(burn_subtitles)
        profile.cinema.max_speedup = float(max_speedup)
        profile.cinema.max_slowdown = float(max_slowdown)
        profile.cinema.target_lufs = float(target_lufs)
    except Exception:
        pass

    try:
        run_job(
            input_video=job_video,
            ru_srt=job_srt,
            profile=profile,
            out_dir=out_dir,
            dry_run=bool(dry_run),
        )
    except Exception as e:
        return (f"Run failed: {e}", None, None, None, None)

    report_path = out_dir / "report.json"
    report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else "{}"

    final_audio = None
    for cand in ("en_dub_audio.m4a", "en_dub_audio.wav"):
        p = out_dir / cand
        if p.exists():
            final_audio = str(p)
            break

    en_srt = str(out_dir / "en.srt") if (out_dir / "en.srt").exists() else None
    final_video = str(out_dir / 'final_dubbed.mp4') if (out_dir / 'final_dubbed.mp4').exists() else None
    return (report_text, final_audio, en_srt, str(report_path) if report_path.exists() else None, final_video)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="DubPipe — RU→EN TTS Dubbing") as demo:
        gr.Markdown(
            """# DubPipe (GUI) — RU→EN dubbing (TTS-first)

This interface wraps the pipeline in a user-friendly way:
1) Upload video (+ optional RU SRT)
2) Choose a profile (economy/standard/premium)
3) Set voice IDs (clone + optional fallback actor voice)
4) Run **Dry-run** first (no ElevenLabs calls)
5) Run full generation (audio + EN subtitles)
"""
        )

        with gr.Row():
            video_file = gr.File(label="Input video (mp4/mkv/...)")
            ru_srt_file = gr.File(label="Optional RU subtitles (SRT)", file_types=[".srt"])

        with gr.Row():
            profile_name = gr.Dropdown(
                choices=["economy", "standard", "premium"],
                value="standard",
                label="Profile",
            )
            dry_run = gr.Checkbox(value=True, label="Dry-run (no ElevenLabs calls)")

        with gr.Accordion("API keys & Voice routing", open=True):
            eleven_api_key = gr.Textbox(
                label="ELEVENLABS_API_KEY (optional override)",
                type="password",
                placeholder="Leave empty to use .env / environment variable",
            )
            openai_api_key = gr.Textbox(
                label="OPENAI_API_KEY (optional override)",
                type="password",
                placeholder="Leave empty to use .env / environment variable",
            )

            with gr.Row():
                primary_voice_id = gr.Textbox(label="Primary voice_id (clone or actor)")
                fallback_voice_id = gr.Textbox(label="Fallback voice_id (optional)")

            with gr.Row():
                accent_test_enabled = gr.Checkbox(
                    value=False,
                    label="Enable accent/intelligibility test (small credit spend)",
                )
                separation_enabled = gr.Checkbox(value=True, label="Enable background separation (Demucs)")

            with gr.Row():
                model_id = gr.Textbox(value="eleven_multilingual_v2", label="model_id")
                output_format = gr.Textbox(value="mp3_44100_128", label="output_format")
                credits_per_character = gr.Number(value=0.5, label="Credits per character (for cost estimate)")

            load_voices_btn = gr.Button("List my ElevenLabs voices")
            voices_box = gr.Textbox(lines=10, label="Voices (name | id | category)", interactive=False)



        with gr.Accordion("Segmentation & timing (reduce tiny chunks)", open=False):
            with gr.Row():
                seg_min_dur = gr.Slider(minimum=0.5, maximum=3.0, value=1.2, step=0.1, label="Min segment duration (sec)")
                seg_max_dur = gr.Slider(minimum=4.0, maximum=20.0, value=8.0, step=0.5, label="Max segment duration (sec)")
            split_on_punct = gr.Checkbox(value=True, label="Split long segments on punctuation (sentences)")
            with gr.Row():
                smart_merge = gr.Checkbox(value=True, label="Smart merge tiny segments (merge short neighbors)")
                semantic_chunking = gr.Checkbox(value=False, label="Semantic chunking (LLM, needs OpenAI key)")
            semantic_mode = gr.Dropdown(
                choices=["sentences", "meaning_phrases"],
                value="sentences",
                label="Semantic mode",
            )
            max_merge_gap = gr.Slider(
                minimum=0.05,
                maximum=1.00,
                value=0.35,
                step=0.05,
                label="Max merge gap (sec)",
            )

        with gr.Accordion("Cinema mode (quality knobs)", open=False):
            with gr.Row():
                text_compress = gr.Checkbox(value=False, label="Text compress (fit timing)")
                speed_matching = gr.Checkbox(value=True, label="Speed matching (auto)")
                pause_matching = gr.Checkbox(value=True, label="Pause matching (prefer silences)")

            with gr.Row():
                loudness_normalize = gr.Checkbox(value=False, label="Loudness normalize (LUFS)")
                render_video = gr.Checkbox(value=True, label="Render final MP4")
                burn_subtitles = gr.Checkbox(value=False, label="Burn subtitles into video")

            with gr.Row():
                max_speedup = gr.Slider(minimum=1.00, maximum=1.15, value=1.08, step=0.01, label="Max speedup")
                max_slowdown = gr.Slider(minimum=0.85, maximum=1.00, value=0.97, step=0.01, label="Max slowdown")
                target_lufs = gr.Number(value=-16.0, label="Target LUFS (e.g., -16)")

        with gr.Accordion("Output", open=True):
            out_base = gr.Textbox(value="./out", label="Output base directory")
            job_name = gr.Textbox(value="", label="Job name (optional)")

        with gr.Row():
            run_btn = gr.Button("Run pipeline", variant="primary")

        report = gr.Textbox(lines=16, label="Report (stdout/stderr)", interactive=False)
        with gr.Row():
            out_audio = gr.File(label="Final dubbed audio")
            out_srt = gr.File(label="Output EN subtitles (SRT)")
        with gr.Row():
            out_report = gr.File(label="Full report.json")
            out_video = gr.File(label="Final dubbed video (MP4)")

        load_voices_btn.click(fn=_load_voices, inputs=[eleven_api_key], outputs=[voices_box])

        run_btn.click(
            fn=_run,
            inputs=[
                video_file,
                ru_srt_file,
                profile_name,
                out_base,
                job_name,
                dry_run,
                eleven_api_key,
                openai_api_key,
                primary_voice_id,
                fallback_voice_id,
                accent_test_enabled,
                separation_enabled,
                model_id,
                output_format,
                credits_per_character,
                seg_min_dur,
                seg_max_dur,
                split_on_punct,
                smart_merge,
                semantic_chunking,
                semantic_mode,
                max_merge_gap,
                text_compress,
                speed_matching,
                pause_matching,
                loudness_normalize,
                render_video,
                burn_subtitles,
                max_speedup,
                max_slowdown,
                target_lufs,
            ],
            outputs=[report, out_audio, out_srt, out_report, out_video],
        )

        gr.Markdown(
            """### Tips
- Provide RU SRT to skip ASR and speed up.
- Start with **Dry-run** to see the estimated cost without spending credits.
- If your clone sounds too accented, set a fallback actor voice and enable the accent test.
"""
        )

    return demo

def main() -> None:
    app = build_app()
    app.queue().launch()


if __name__ == "__main__":
    main()