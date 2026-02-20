from __future__ import annotations
from pathlib import Path
import subprocess
import shlex

def mux_audio_into_video(input_video: Path, audio_in: Path, out_mp4: Path) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg","-y",
        "-i", str(input_video),
        "-i", str(audio_in),
        "-map","0:v:0","-map","1:a:0",
        "-c:v","copy",
        "-c:a","aac","-b:a","192k",
        "-shortest",
        str(out_mp4)
    ]
    _run(cmd)

def burn_subtitles(input_video: Path, audio_in: Path, srt_path: Path, out_mp4: Path) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    # Burn-in requires re-encode of video stream.
    # Use a reasonable default codec; user can change later.
    # Need to escape path for subtitles filter on Windows
    subf = str(srt_path).replace('\\','/').replace(':','\\:')
    vf = f"subtitles='{subf}'"
    cmd = [
        "ffmpeg","-y",
        "-i", str(input_video),
        "-i", str(audio_in),
        "-vf", vf,
        "-map","0:v:0","-map","1:a:0",
        "-c:v","libx264","-preset","veryfast","-crf","20",
        "-c:a","aac","-b:a","192k",
        "-shortest",
        str(out_mp4)
    ]
    _run(cmd)

def _run(cmd):
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{(e.stderr or '')[-2000:]}") from e
