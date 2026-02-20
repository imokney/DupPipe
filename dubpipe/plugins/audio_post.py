from __future__ import annotations
from pathlib import Path
import subprocess

def loudness_normalize(in_audio: Path, out_audio: Path, target_lufs: float = -16.0) -> Path:
    out_audio.parent.mkdir(parents=True, exist_ok=True)
    # Single-pass EBU R128 loudnorm. Good enough for starter.
    flt = f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11"
    cmd = ["ffmpeg","-y","-i",str(in_audio),"-af",flt,"-c:a","aac","-b:a","192k",str(out_audio)]
    _run(cmd)
    return out_audio

def _run(cmd):
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{(e.stderr or '')[-2000:]}") from e
