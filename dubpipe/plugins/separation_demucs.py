from __future__ import annotations

import shutil
import subprocess
import os
from pathlib import Path
from typing import Optional, Tuple

from ..core.config import SeparationConfig


def _run_demucs_cli(audio_wav: Path, sep_dir: Path, cfg: SeparationConfig) -> Tuple[Optional[Path], Optional[Path], str]:
    """Run demucs CLI if available. Returns (vocals, bg, log_text)."""
    cmd = [
        "demucs",
        "-n", cfg.model,
        "--two-stems", cfg.two_stems,
        "-o", str(sep_dir),
        str(audio_wav),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    log_text = (p.stdout or "") + "\n" + (p.stderr or "")
    if p.returncode != 0:
        return None, None, log_text

    track_dir = sep_dir / cfg.model / audio_wav.stem
    vocals = track_dir / "vocals.wav"
    bg = track_dir / "no_vocals.wav"  # when using --two-stems vocals
    if not vocals.exists() or not bg.exists():
        return None, None, log_text
    return vocals, bg, log_text


def _run_demucs_python(audio_wav: Path, sep_dir: Path, cfg: SeparationConfig) -> Tuple[Optional[Path], Optional[Path], str]:
    """Python fallback that avoids torchaudio.save/torchcodec by saving with soundfile."""
    try:
        import torch
        import torchaudio
        import soundfile as sf
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
    except Exception as e:
        return None, None, f"Python demucs fallback missing deps: {e}"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = get_model(cfg.model)
        model.to(device)
        model.eval()
    except Exception as e:
        return None, None, f"Failed to load demucs model '{cfg.model}': {e}"

    # Read wav
    try:
        wav, sr = torchaudio.load(str(audio_wav))  # shape [C, T], float32
    except Exception as e:
        return None, None, f"torchaudio.load failed: {e}"

    # Demucs models typically expect 44.1k; resample if needed.
    target_sr = getattr(model, "samplerate", 44100)
    if sr != target_sr:
        try:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
            sr = target_sr
        except Exception as e:
            return None, None, f"Resample {sr}->{target_sr} failed: {e}"

    # Ensure expected channel count (2 for most models). If mono, duplicate.
    target_ch = getattr(model, "audio_channels", 2)
    if wav.shape[0] == 1 and target_ch == 2:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > target_ch:
        wav = wav[:target_ch, :]
    elif wav.shape[0] < target_ch:
        # pad by repeating last channel
        wav = torch.cat([wav] + [wav[-1:, :]] * (target_ch - wav.shape[0]), dim=0)

    with torch.no_grad():
        try:
            sources = apply_model(model, wav[None].to(device), device=device, split=True, progress=False)[0]
            # sources: [S, C, T]
        except Exception as e:
            return None, None, f"apply_model failed: {e}"

    src_names = list(getattr(model, "sources", []))
    if "vocals" not in src_names:
        return None, None, f"Unexpected demucs sources: {src_names}"

    v_idx = src_names.index("vocals")
    vocals = sources[v_idx].detach().cpu()

    # Background = sum of all non-vocals sources to match --two-stems vocals behavior
    bg = torch.zeros_like(vocals)
    for i, name in enumerate(src_names):
        if i == v_idx:
            continue
        bg += sources[i].detach().cpu()

    # Save as 16-bit PCM WAV using soundfile (no torchcodec needed)
    track_dir = sep_dir / cfg.model / audio_wav.stem
    track_dir.mkdir(parents=True, exist_ok=True)
    vocals_path = track_dir / "vocals.wav"
    bg_path = track_dir / "no_vocals.wav"

    def _write_wav(path: Path, tensor: "torch.Tensor") -> None:
        # tensor: [C, T], float32 in [-1, 1] typically
        audio = tensor.transpose(0, 1).numpy()  # [T, C]
        sf.write(str(path), audio, sr, subtype="PCM_16")

    try:
        _write_wav(vocals_path, vocals)
        _write_wav(bg_path, bg)
    except Exception as e:
        return None, None, f"soundfile.write failed: {e}"

    return vocals_path, bg_path, f"Python demucs fallback OK (device={device}, sr={sr}, sources={src_names})"


def separate_if_enabled(audio_wav: Path, out_dir: Path, cfg: SeparationConfig) -> Tuple[Optional[Path], Optional[Path]]:
    """Run separation if enabled.

    Windows note: demucs CLI may crash while saving audio due to torchaudio->torchcodec issues.
    To keep the pipeline robust and quality-preserving, we default to the Python demucs path on Windows,
    which saves stems via soundfile (no torchcodec). You can force CLI by setting env:
        DUBPIPE_DEMUCS_MODE=cli
    or force Python with:
        DUBPIPE_DEMUCS_MODE=python
    """
    if not cfg.enabled:
        return None, None

    sep_dir = out_dir / "separation"
    sep_dir.mkdir(parents=True, exist_ok=True)

    mode = (os.getenv("DUBPIPE_DEMUCS_MODE") or "").strip().lower()
    if mode not in ("cli", "python", ""):
        mode = ""

    is_windows = (os.name == "nt")

    # Default: python on Windows, CLI elsewhere (unless overridden)
    prefer_python = (mode == "python") or (is_windows and mode != "cli")

    if prefer_python:
        vocals, bg, msg = _run_demucs_python(audio_wav, sep_dir, cfg)
        if vocals and bg:
            print(f"[info] {msg}")
            return vocals, bg
        print(f"[warn] Separation failed (python): {msg}")
        # fall through to CLI as last resort

    # CLI path (optional)
    if shutil.which("demucs") is not None:
        print("[info] Running demucs separation (CLI)...")
        vocals, bg, log_text = _run_demucs_cli(audio_wav, sep_dir, cfg)
        if vocals and bg:
            return vocals, bg
        # Don't spam full traceback; summarize and hint
        hint = ""
        if any(k in log_text.lower() for k in ["torchcodec", "libtorchcodec", "save_with_torchcodec"]):
            hint = " (torchcodec/torchaudio save issue detected; set DUBPIPE_DEMUCS_MODE=python)"
        print(f"[warn] demucs CLI failed{hint}.")
    else:
        print("[warn] Separation enabled but demucs CLI is not in PATH.")

    return None, None
