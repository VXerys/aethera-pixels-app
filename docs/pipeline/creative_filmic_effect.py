"""Creative filmic grading tuned for natural but perceptible impact."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np


def _ensure_three_channels(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("CreativeFilmicEffect expects an HxWx3 image")


def _to_float(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def _luminance(bgr: np.ndarray) -> np.ndarray:
    return 0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1] + 0.299 * bgr[:, :, 2]


def _skin_mask(bgr: np.ndarray) -> np.ndarray:
    bgr_u8 = (bgr * 255.0).astype(np.uint8)
    hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 30, 60])
    upper1 = np.array([25, 200, 255])
    lower2 = np.array([160, 30, 60])
    upper2 = np.array([180, 200, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 2.0)


def _filmic_curve(luma: np.ndarray, strength: float) -> np.ndarray:
    strength = np.clip(strength, 0.0, 1.0)
    midpoint = 0.52
    contrast = 6.0 + strength * 7.0
    shaped = 1.0 / (1.0 + np.exp(-(luma - midpoint) * contrast))
    shaped = (shaped - shaped.min()) / (shaped.max() - shaped.min() + 1e-6)
    return np.clip(shaped, 0.0, 1.0)


def _apply_split_tone(lab: np.ndarray, luma: np.ndarray, shadow_warmth: float,
                      highlight_cool: float, saturation_boost: float) -> np.ndarray:
    shadow_mask = np.clip((0.55 - luma) * 2.2, 0.0, 1.0)
    highlight_mask = np.clip((luma - 0.45) * 2.2, 0.0, 1.0)

    lab[:, :, 1] += shadow_mask * shadow_warmth * 14.0
    lab[:, :, 2] += shadow_mask * shadow_warmth * 24.0
    lab[:, :, 1] -= highlight_mask * highlight_cool * 12.0
    lab[:, :, 2] -= highlight_mask * highlight_cool * 26.0

    chroma_scale = 1.0 + saturation_boost * 0.1
    lab[:, :, 1] *= chroma_scale
    lab[:, :, 2] *= chroma_scale
    return lab


def _micro_contrast(bgr: np.ndarray, amount: float) -> np.ndarray:
    amount = np.clip(amount, 0.0, 0.4)
    if amount <= 1e-4:
        return bgr
    blur = cv2.GaussianBlur(bgr, (0, 0), 1.6)
    detail = bgr - blur
    return np.clip(bgr + detail * (amount * 1.6), 0.0, 1.0)


def _midtone_compress(luma: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 1e-4:
        return luma
    low = cv2.GaussianBlur(luma, (0, 0), 3.0)
    high = cv2.GaussianBlur(luma, (0, 0), 0.7)
    mask = np.clip((high - low) * 2.5 + 0.5, 0.0, 1.0)
    return np.clip(luma * (1.0 - strength * 0.15) + mask * strength * 0.15, 0.0, 1.0)


def _film_grain(bgr: np.ndarray, strength: float) -> np.ndarray:
    strength = np.clip(strength, 0.0, 0.08)
    if strength <= 1e-4:
        return bgr
    noise = np.random.normal(0, strength, bgr.shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.7)
    luma = _luminance(bgr)
    scale = (0.55 + 0.45 * luma)[:, :, None]
    return np.clip(bgr + noise * scale, 0.0, 1.0)


def _bloom(bgr: np.ndarray, strength: float) -> np.ndarray:
    strength = np.clip(strength, 0.0, 0.2)
    if strength <= 1e-4:
        return bgr
    luma = _luminance(bgr)
    mask = np.clip((luma - 0.7) * 3.0, 0.0, 1.0)
    blurred = cv2.GaussianBlur(bgr, (0, 0), 6.0)
    return np.clip(bgr + blurred * (mask[:, :, None] * strength), 0.0, 1.0)


@dataclass(frozen=True)
class FilmicParams:
    tone_strength: float = 0.35
    shadow_warmth: float = 0.18
    highlight_cool: float = 0.15
    micro_contrast: float = 0.12
    grain: float = 0.02
    bloom: float = 0.06
    mix: float = 0.75
    skin_preserve: float = 0.4

    @staticmethod
    def from_dict(data: Dict) -> "FilmicParams":
        return FilmicParams(
            tone_strength=float(data.get("tone_strength", 0.35)),
            shadow_warmth=float(data.get("shadow_warmth", 0.18)),
            highlight_cool=float(data.get("highlight_cool", 0.15)),
            micro_contrast=float(data.get("micro_contrast", 0.12)),
            grain=float(data.get("grain", 0.02)),
            bloom=float(data.get("bloom", 0.06)),
            mix=float(data.get("mix", 0.75)),
            skin_preserve=float(data.get("skin_preserve", 0.4)),
        )


class CreativeFilmicEffect:
    """Cinematic colour styling with adaptive contrast and toning."""

    @staticmethod
    def process(image: np.ndarray, params: Dict) -> np.ndarray:
        _ensure_three_channels(image)
        bgr = _to_float(image)
        cfg = FilmicParams.from_dict(params)

        luma = _luminance(bgr)
        filmic = _filmic_curve(luma, cfg.tone_strength)
        tone_mixed = (1.0 - cfg.tone_strength) * luma + cfg.tone_strength * filmic
        tone_mixed = _midtone_compress(tone_mixed, cfg.tone_strength)

        lab = cv2.cvtColor((bgr * 255.0).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] = np.clip(tone_mixed * 255.0, 0.0, 255.0)
        lab = _apply_split_tone(lab, tone_mixed, cfg.shadow_warmth, cfg.highlight_cool, cfg.mix)
        styled = cv2.cvtColor(np.clip(lab, 0.0, 255.0).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0

        styled = _micro_contrast(styled, cfg.micro_contrast)
        styled = _film_grain(styled, cfg.grain)
        styled = _bloom(styled, cfg.bloom)

        styled = np.clip(styled, 0.0, 1.0)
        mix = np.clip(cfg.mix, 0.0, 1.0)
        blended = (1.0 - mix) * bgr + mix * styled

        skin = _skin_mask(bgr)
        if skin.max() > 0:
            preserve = np.clip(cfg.skin_preserve, 0.0, 1.0)
            blended = blended * (1.0 - skin[:, :, None] * preserve) + bgr * (skin[:, :, None] * preserve)

        return (np.clip(blended, 0.0, 1.0) * 255.0).astype(np.uint8)
