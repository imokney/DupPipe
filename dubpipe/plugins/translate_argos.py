from __future__ import annotations
from typing import List

from ..core.config import TranslationConfig
from ..core.models import Segment


def translate_segments(segments: List[Segment], cfg: TranslationConfig) -> List[Segment]:
    if cfg.backend != "argos":
        raise NotImplementedError("Starter project supports only Argos translation backend. Add more backends later.")

    translator = _get_argos_translator(cfg.src_lang, cfg.tgt_lang)
    for s in segments:
        if not s.text_ru.strip():
            s.text_en_raw = ""
            continue
        s.text_en_raw = translator(s.text_ru)
    return segments


def _get_argos_translator(src: str, tgt: str):
    try:
        from argostranslate import package, translate
    except Exception as e:
        raise RuntimeError(
            "Argos Translate is not installed. Install it with: pip install argostranslate"
        ) from e

    # Ensure language package installed
    installed = translate.get_installed_languages()
    if not any(l.code == src for l in installed) or not any(l.code == tgt for l in installed):
        print("[info] Argos language pack missing. Downloading ru->en package (one-time)...")
        package.update_package_index()
        avail = package.get_available_packages()
        pkg = next((p for p in avail if p.from_code == src and p.to_code == tgt), None)
        if pkg is None:
            raise RuntimeError(f"Argos package {src}->{tgt} not found in index.")
        path = pkg.download()
        package.install_from_path(path)

    installed = translate.get_installed_languages()
    from_lang = next(l for l in installed if l.code == src)
    to_lang = next(l for l in installed if l.code == tgt)
    tr = from_lang.get_translation(to_lang)

    def _t(text: str) -> str:
        return tr.translate(text)

    return _t
