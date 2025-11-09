"""Selective texture enhancement tuned for controlled detail recovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np


def _ensure_rgb(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("SelectiveTextureEnhancement expects an HxWx3 image")


def _to_float(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def _to_lab(rgb: np.ndarray) -> np.ndarray:
    rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    lab_u8 = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_u8, cv2.COLOR_LAB2RGB)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def _gradient_map(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = cv2.GaussianBlur(mag, (0, 0), 1.1)
    mag /= (mag.max() + 1e-6)
    return np.clip(mag, 0.0, 1.0)


def _variance_map(gray: np.ndarray) -> np.ndarray:
    mean = cv2.GaussianBlur(gray, (0, 0), 2.0)
    sq_mean = cv2.GaussianBlur(gray * gray, (0, 0), 2.0)
    return np.maximum(sq_mean - mean * mean, 0.0)


def _texture_strength(gray: np.ndarray, threshold: float, softness: float) -> np.ndarray:
    var_map = _variance_map(gray)
    low, high = np.percentile(var_map, [15, 90])
    norm = np.clip((var_map - low) / (high - low + 1e-6), 0.0, 1.0)
    mask = np.clip((norm - threshold) / (1.0 - threshold + 1e-6), 0.0, 1.0)
    mask = np.power(mask, np.clip(softness, 0.5, 2.0))
    return cv2.GaussianBlur(mask, (0, 0), 1.0)


def _skin_mask(rgb: np.ndarray) -> np.ndarray:
    rgb_u8 = (rgb * 255.0).astype(np.uint8)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    lower1 = np.array([0, 30, 60])
    upper1 = np.array([25, 200, 255])
    lower2 = np.array([160, 30, 60])
    upper2 = np.array([180, 200, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 2.0)


def _detail_component(gray: np.ndarray, sigma: float) -> np.ndarray:
    return gray - cv2.GaussianBlur(gray, (0, 0), sigma)


def _multi_scale_detail(gray: np.ndarray, blur_factor: float) -> np.ndarray:
    fine = _detail_component(gray, 0.7 + blur_factor * 0.4)
    medium = _detail_component(gray, 1.5 + blur_factor * 0.8)
    broad = _detail_component(gray, 2.8 + blur_factor * 1.3)
    return 0.6 * fine + 0.3 * medium + 0.1 * broad


def _micro_contrast(gray: np.ndarray, amount: float) -> np.ndarray:
    amount = np.clip(amount, 0.0, 0.3)
    if amount <= 1e-4:
        return gray
    clip_limit = 1.0 + amount * 2.0
    grid = max(6, min(12, int(12 - amount * 10)))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    enhanced = clahe.apply(gray_u8).astype(np.float32) / 255.0
    return np.clip(enhanced, 0.0, 1.0)


@dataclass(frozen=True)
class TextureParams:
    detail_gain: float = 0.55
    micro_contrast: float = 0.12
    blend: float = 0.7
    texture_threshold: float = 0.25
    texture_softness: float = 1.0
    skin_protect: float = 0.65
    edge_boost: float = 0.3

    @staticmethod
    def from_dict(data: Dict) -> "TextureParams":
        return TextureParams(
            detail_gain=float(data.get("detail_gain", 0.55)),
            micro_contrast=float(data.get("micro_contrast", 0.12)),
            blend=float(data.get("blend", 0.7)),
            texture_threshold=float(data.get("texture_threshold", 0.25)),
            texture_softness=float(data.get("texture_softness", 1.0)),
            skin_protect=float(data.get("skin_protect", 0.65)),
            edge_boost=float(data.get("edge_boost", 0.3)),
        )


class SelectiveTextureEnhancement:
    """Multi-scale detail refinement with adaptive masks and skin safety."""

    @staticmethod
    def process(image: np.ndarray, params: Dict) -> np.ndarray:
        _ensure_rgb(image)
        rgb = _to_float(image)
        lab = _to_lab(rgb)
        cfg = TextureParams.from_dict(params)

        luminance = lab[:, :, 0] / 255.0
        blur_indicator = 1.0 - np.tanh(cv2.Laplacian(luminance, cv2.CV_32F).var() * 60.0)

        texture = _texture_strength(luminance, np.clip(cfg.texture_threshold, 0.05, 0.6), cfg.texture_softness)
        structure = _gradient_map(luminance)
        edge_emphasis = np.clip(structure * cfg.edge_boost, 0.0, 0.6)

        mask = np.clip(texture * (0.5 + structure * 0.5) + edge_emphasis, 0.0, 1.0)
        skin = _skin_mask(rgb)
        mask *= 1.0 - skin * np.clip(cfg.skin_protect, 0.0, 1.0)

        detail = _multi_scale_detail(luminance, blur_indicator)
        detail *= cfg.detail_gain * (0.6 + 0.4 * structure)
        detail = cv2.GaussianBlur(detail, (0, 0), 0.5)

        enhanced_l = np.clip(luminance + detail * mask, 0.0, 1.0)
        blend = np.clip(cfg.blend, 0.3, 0.9)
        enhanced_l = (1.0 - blend) * luminance + blend * enhanced_l
        enhanced_l = _micro_contrast(enhanced_l, cfg.micro_contrast)

        lab[:, :, 0] = np.clip(enhanced_l * 255.0, 0.0, 255.0)
        return _lab_to_rgb(lab)
