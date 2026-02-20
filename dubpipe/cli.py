import argparse
from pathlib import Path

from .core.runner import run_job
from .core.config import load_profile


def main() -> int:
    p = argparse.ArgumentParser(prog="dubpipe", description="TTS-first RU→EN dubbing pipeline (starter).")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run pipeline on a video")
    run.add_argument("--input", required=True, type=Path, help="Path to input video (mp4/mkv/...)")
    run.add_argument("--ru-srt", type=Path, default=None, help="Optional RU subtitles (SRT). If provided, ASR is skipped.")
    run.add_argument("--profile", type=str, default="standard", help="Profile name (YAML in profiles/)")
    run.add_argument("--out", type=Path, default=Path("./out"), help="Output directory")
    run.add_argument("--job-name", type=str, default=None, help="Optional job name (folder name). Default derived from input file name.")
    run.add_argument("--dry-run", action="store_true", help="Plan only: do not call ElevenLabs (no credit spending).")

    args = p.parse_args()

    profile = load_profile(args.profile)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    job_name = args.job_name or args.input.stem
    run_job(
        input_video=args.input,
        ru_srt=args.ru_srt,
        profile=profile,
        out_dir=out_dir / job_name,
        dry_run=args.dry_run,
    )
    return 0
