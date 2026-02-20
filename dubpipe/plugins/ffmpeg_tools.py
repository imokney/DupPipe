from __future__ import annotations
import subprocess
from pathlib import Path


def extract_audio(input_video: Path, out_wav: Path) -> None:
    # 48kHz stereo wav; you can change to mono if you prefer
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "48000",
        "-ac", "2",
        str(out_wav),
    ]
    _run(cmd, "ffmpeg extract_audio")


def time_stretch_wav(in_wav: Path, out_wav: Path, tempo: float) -> None:
    # ffmpeg atempo supports 0.5..2.0 per filter; chain if needed.
    if tempo <= 0:
        raise ValueError("tempo must be > 0")

    filters = []
    t = tempo
    while t > 2.0:
        filters.append("atempo=2.0")
        t /= 2.0
    while t < 0.5:
        filters.append("atempo=0.5")
        t /= 0.5
    filters.append(f"atempo={t:.6f}")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_wav),
        "-filter:a", ",".join(filters),
        str(out_wav),
    ]
    _run(cmd, "ffmpeg time_stretch")


def _run(cmd: list[str], label: str) -> None:
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as e:
        raise RuntimeError(f"{label}: ffmpeg not found. Install ffmpeg and ensure it's in PATH.") from e
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"{label} failed:\n{msg}") from e
