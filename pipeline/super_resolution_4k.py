"""4K super-resolution pipeline focused on natural clarity restoration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np


def _ensure_bgr(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("SuperResolution4K expects an HxWx3 image")


def _to_float(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def _estimate_blur(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur = 1.0 - np.tanh(lap_var / 180.0)
    return float(np.clip(blur, 0.0, 1.0))


def _estimate_noise(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    mad = np.median(np.abs(lap - np.median(lap)))
    sigma = mad / 0.6745
    noise = sigma / 18.0
    return float(np.clip(noise, 0.0, 1.0))


def _fast_denoise(image: np.ndarray, strength: float, noise_est: float) -> np.ndarray:
    h_luma = 6.0 + 12.0 * max(strength, noise_est)
    h_color = h_luma * 0.7
    denoised = cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h=float(h_luma),
        hColor=float(h_color),
        templateWindowSize=7,
        searchWindowSize=21,
    )
    return denoised.astype(np.float32) / 255.0


def _lab_channels(bgr_float: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor((bgr_float * 255.0).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0] / 255.0
    return l, lab


def _structure_map(l_channel: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(l_channel, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l_channel, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = cv2.GaussianBlur(mag, (0, 0), 1.2)
    mag /= (mag.max() + 1e-6)
    return np.clip(mag, 0.0, 1.0)


def _multi_scale_detail(l_channel: np.ndarray, blur_level: float) -> np.ndarray:
    small = l_channel - cv2.GaussianBlur(l_channel, (0, 0), 0.8 + blur_level * 0.4)
    mid = l_channel - cv2.GaussianBlur(l_channel, (0, 0), 1.6 + blur_level * 0.9)
    large = l_channel - cv2.GaussianBlur(l_channel, (0, 0), 2.8 + blur_level * 1.4)
    return (0.55 * small) + (0.3 * mid) + (0.15 * large)


def _halo_suppression(structure: np.ndarray, halo_guard: float) -> np.ndarray:
    soft = cv2.GaussianBlur(1.0 - structure, (0, 0), 2.0)
    return halo_guard + (1.0 - halo_guard) * (1.0 - soft)


def _local_contrast(l_channel: np.ndarray, amount: float) -> np.ndarray:
    amount = np.clip(amount, 0.0, 0.4)
    if amount <= 1e-4:
        return l_channel
    clip_limit = 1.0 + amount * 2.2
    grid = int(14 - amount * 16)
    grid = max(8, min(16, grid))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    l8 = np.clip(l_channel * 255.0, 0, 255).astype(np.uint8)
    enhanced = clahe.apply(l8).astype(np.float32) / 255.0
    return np.clip(enhanced, 0.0, 1.0)


def _final_refine(bgr_float: np.ndarray) -> np.ndarray:
    refined = cv2.bilateralFilter(bgr_float, d=5, sigmaColor=0.04, sigmaSpace=4)
    return np.clip(refined, 0.0, 1.0)


@dataclass(frozen=True)
class SR4KParams:
    scale_factor: int = 4
    denoise_strength: float = 0.6
    detail_boost: float = 0.7
    local_contrast: float = 0.18
    halo_guard: float = 0.55
    edge_mix: float = 0.7

    @staticmethod
    def from_dict(data: Dict) -> "SR4KParams":
        return SR4KParams(
            scale_factor=int(data.get("scale_factor", 4)),
            denoise_strength=float(data.get("denoise_strength", 0.6)),
            detail_boost=float(data.get("detail_boost", 0.7)),
            local_contrast=float(data.get("local_contrast", 0.18)),
            halo_guard=float(data.get("halo_guard", 0.55)),
            edge_mix=float(data.get("edge_mix", 0.7)),
        )


class SuperResolution4K:
    """Adaptive 4K upscaling with structure-aware detail recovery."""

    @staticmethod
    def process(image: np.ndarray, params: Dict) -> np.ndarray:
        _ensure_bgr(image)
        original_u8 = image if image.dtype == np.uint8 else _to_uint8(image)

        cfg = SR4KParams.from_dict(params)
        cfg = SR4KParams(
            scale_factor=max(2, min(4, cfg.scale_factor)),
            denoise_strength=np.clip(cfg.denoise_strength, 0.2, 1.2),
            detail_boost=np.clip(cfg.detail_boost, 0.3, 1.2),
            local_contrast=np.clip(cfg.local_contrast, 0.0, 0.4),
            halo_guard=np.clip(cfg.halo_guard, 0.3, 0.9),
            edge_mix=np.clip(cfg.edge_mix, 0.4, 1.0),
        )

        blur_level = _estimate_blur(original_u8)
        noise_level = _estimate_noise(original_u8)

        denoised = _fast_denoise(original_u8, cfg.denoise_strength, noise_level)

        h, w = denoised.shape[:2]
        upscaled = cv2.resize(
            denoised,
            (w * cfg.scale_factor, h * cfg.scale_factor),
            interpolation=cv2.INTER_LANCZOS4,
        )

        l_channel, lab = _lab_channels(upscaled)
        structure = _structure_map(l_channel)

        detail_gain = cfg.detail_boost * (0.55 + 0.45 * blur_level)
        detail = _multi_scale_detail(l_channel, blur_level)
        detail *= detail_gain * (0.5 + 0.5 * structure)
        detail = cv2.GaussianBlur(detail, (0, 0), 0.6)

        halo_blend = _halo_suppression(structure, cfg.halo_guard)
        enhanced_l = l_channel + detail * cfg.edge_mix * halo_blend
        enhanced_l = np.clip(enhanced_l, 0.0, 1.0)

        enhanced_l = _local_contrast(enhanced_l, cfg.local_contrast)

        lab[:, :, 0] = np.clip(enhanced_l * 255.0, 0.0, 255.0)
        restored = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0

        refined = _final_refine(restored)
        return _to_uint8(refined)
