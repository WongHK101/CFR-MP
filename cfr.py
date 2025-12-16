# -*- coding: utf-8 -*-
"""
align_rgb_thermal_global.py

Usage:
python align_rgb_thermal_global.py ^
  --rgb_dir "...\vis" ^
  --th_dir  "...\ir"  ^
  --out_dir "...\gpt" ^
  [--samples 200]

Deps:
  pip install pillow opencv-python numpy piexif
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:
    raise RuntimeError("opencv-python is required. Please install: pip install opencv-python") from e

try:
    from PIL import Image, ImageOps, ExifTags
except Exception as e:
    raise RuntimeError("Pillow is required. Please install: pip install pillow") from e

try:
    import piexif  # type: ignore
    import piexif.helper  # type: ignore
    HAS_PIEXIF = True
except Exception:
    HAS_PIEXIF = False


# ----------------------------
# Logging
# ----------------------------

def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Logger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = open(self.log_path, "w", encoding="utf-8", errors="ignore")

    def close(self) -> None:
        try:
            self.fp.close()
        except Exception:
            pass

    def log(self, level: str, msg: str) -> None:
        line = f"{now_ts()} [{level}] {msg}"
        print(line)
        self.fp.write(line + "\n")
        self.fp.flush()


# ----------------------------
# FS utils
# ----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def stem_key(path: Path) -> str:
    return path.stem.strip()


def safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, tuple) and len(x) == 2:
            num, den = x
            if float(den) == 0:
                return None
            return float(num) / float(den)
        return float(x)
    except Exception:
        return None


def _float_to_rational(x: float, max_den: int = 1000000) -> Tuple[int, int]:
    # best-effort rational
    if x <= 0:
        return (0, 1)
    # simple continued fraction-ish via rounding denominator
    den = max_den
    num = int(round(x * den))
    # reduce gcd
    g = int(np.gcd(num, den))
    num //= g
    den //= g
    return (num, den)


# ----------------------------
# Image conversions
# ----------------------------

def exif_transposed_pil(img_path: Path) -> Image.Image:
    im = Image.open(img_path)
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    return im


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    if img.mode == "I;16" or img.mode == "I":
        arr = np.array(img)
        if arr.dtype != np.uint16:
            arr = arr.astype(np.uint16, copy=False)
        if arr.max() > 0:
            arr8 = (arr.astype(np.float32) / float(arr.max()) * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr8 = np.zeros_like(arr, dtype=np.uint8)
        return cv2.cvtColor(arr8, cv2.COLOR_GRAY2BGR)

    if img.mode == "L":
        arr = np.array(img, dtype=np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_pil_rgb(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ----------------------------
# EXIF reading (robust)
# ----------------------------

@dataclass
class ExifLensInfo:
    focal_35mm: Optional[float] = None
    focal_mm: Optional[float] = None
    digital_zoom: Optional[float] = None
    orientation: Optional[int] = None

    def effective_focal_35mm(self) -> Optional[float]:
        if self.focal_35mm is None:
            return None
        dz = self.digital_zoom if (self.digital_zoom is not None and self.digital_zoom > 0) else 1.0
        return float(self.focal_35mm) * float(dz)


def _get_exif_ifd(img: Image.Image) -> Tuple[dict, dict]:
    ex = img.getexif()
    ifd0 = dict(ex) if ex is not None else {}
    exif_ifd = {}
    try:
        if hasattr(ex, "get_ifd") and hasattr(ExifTags, "IFD"):
            exif_ifd = dict(ex.get_ifd(ExifTags.IFD.Exif))  # type: ignore
    except Exception:
        exif_ifd = {}
    return ifd0, exif_ifd


def read_exif_lens_info(img_path: Path) -> Tuple[ExifLensInfo, Dict[str, Optional[float]]]:
    info = ExifLensInfo()
    raw: Dict[str, Optional[float]] = {
        "FocalLengthIn35mmFilm": None,
        "FocalLength": None,
        "DigitalZoomRatio": None,
        "Orientation": None,
    }
    try:
        with Image.open(img_path) as im:
            exif0, exif_sub = _get_exif_ifd(im)

            TAG_F35 = 41989
            TAG_FOCAL = 37386
            TAG_DZ = 41988
            TAG_ORI = 274

            ori = exif0.get(TAG_ORI, None)
            info.orientation = int(ori) if isinstance(ori, (int, np.integer)) else None
            raw["Orientation"] = safe_float(ori)

            f35 = exif_sub.get(TAG_F35, exif0.get(TAG_F35, None))
            info.focal_35mm = safe_float(f35)
            raw["FocalLengthIn35mmFilm"] = safe_float(f35)

            fmm = exif_sub.get(TAG_FOCAL, exif0.get(TAG_FOCAL, None))
            info.focal_mm = safe_float(fmm)
            raw["FocalLength"] = safe_float(fmm)

            dz = exif_sub.get(TAG_DZ, exif0.get(TAG_DZ, None))
            info.digital_zoom = safe_float(dz)
            raw["DigitalZoomRatio"] = safe_float(dz)
    except Exception:
        pass
    return info, raw


def compute_zoom_ratio(rgb_exif: ExifLensInfo, th_exif: ExifLensInfo) -> Optional[float]:
    a = th_exif.effective_focal_35mm()
    b = rgb_exif.effective_focal_35mm()
    if a is not None and b is not None and b > 0:
        return float(a) / float(b)

    a2 = th_exif.focal_mm
    b2 = rgb_exif.focal_mm
    if a2 is not None and b2 is not None and b2 > 0:
        return float(a2) / float(b2)
    return None


# ----------------------------
# EXIF preservation (piexif-based, keep GPS/ExifIFD)
# ----------------------------

def _get_src_exif_bytes(src_path: Path) -> Optional[bytes]:
    try:
        with Image.open(src_path) as im:
            b = im.info.get("exif", None)
            if b:
                return b
            ex = im.getexif()
            if ex and len(ex) > 0:
                return ex.tobytes()
    except Exception:
        pass
    return None


def _update_exif_bytes_piexif(
    src_exif_bytes: bytes,
    out_w: int,
    out_h: int,
    zoom_factor: Optional[float],
    drop_makernote: bool = False,
    drop_thumbnail: bool = False,
) -> Optional[bytes]:
    if not HAS_PIEXIF:
        return None

    try:
        ex = piexif.load(src_exif_bytes)
    except Exception:
        return None

    # Update orientation & sizes (IFD0 + ExifIFD)
    try:
        ex["0th"][piexif.ImageIFD.Orientation] = 1
    except Exception:
        pass
    try:
        ex["0th"][piexif.ImageIFD.ImageWidth] = int(out_w)
        ex["0th"][piexif.ImageIFD.ImageLength] = int(out_h)
    except Exception:
        pass

    try:
        ex["Exif"][piexif.ExifIFD.PixelXDimension] = int(out_w)
        ex["Exif"][piexif.ExifIFD.PixelYDimension] = int(out_h)
    except Exception:
        pass

    # Update focal/zoom tags if we can (crop => acts like zoom-in)
    if zoom_factor is not None and zoom_factor > 0:
        try:
            # DigitalZoomRatio (rational)
            old = ex["Exif"].get(piexif.ExifIFD.DigitalZoomRatio, None)
            old_f = safe_float(old)
            if old_f is None or old_f <= 0:
                new_f = float(zoom_factor)
            else:
                new_f = float(old_f) * float(zoom_factor)
            ex["Exif"][piexif.ExifIFD.DigitalZoomRatio] = _float_to_rational(new_f)
        except Exception:
            pass

        try:
            # FocalLength (rational)
            old = ex["Exif"].get(piexif.ExifIFD.FocalLength, None)
            old_f = safe_float(old)
            if old_f is not None and old_f > 0:
                ex["Exif"][piexif.ExifIFD.FocalLength] = _float_to_rational(float(old_f) * float(zoom_factor))
        except Exception:
            pass

        try:
            # FocalLengthIn35mmFilm (short)
            old = ex["Exif"].get(piexif.ExifIFD.FocalLengthIn35mmFilm, None)
            if old is not None:
                old_i = int(old)
                if old_i > 0:
                    ex["Exif"][piexif.ExifIFD.FocalLengthIn35mmFilm] = int(round(old_i * float(zoom_factor)))
        except Exception:
            pass

    if drop_makernote:
        try:
            if piexif.ExifIFD.MakerNote in ex["Exif"]:
                del ex["Exif"][piexif.ExifIFD.MakerNote]
        except Exception:
            pass

    if drop_thumbnail:
        try:
            ex["thumbnail"] = None
        except Exception:
            pass

    try:
        return piexif.dump(ex)
    except Exception:
        return None


def save_rgb_with_updated_exif(
    src_rgb_path: Path,
    out_bgr: np.ndarray,
    dst_path: Path,
    zoom_factor: Optional[float],
    logger: Optional[Logger] = None,
) -> None:
    ensure_dir(dst_path.parent)
    suffix = dst_path.suffix.lower()

    if suffix not in {".jpg", ".jpeg"}:
        cv2.imwrite(str(dst_path), out_bgr)
        return

    out_h, out_w = out_bgr.shape[:2]
    pil_img = bgr_to_pil_rgb(out_bgr)
    save_kwargs = dict(format="JPEG", quality=95, optimize=True)
    try:
        save_kwargs["subsampling"] = 0
    except Exception:
        pass

    src_exif_bytes = _get_src_exif_bytes(src_rgb_path)

    # If no exif in source, save without
    if not src_exif_bytes:
        pil_img.save(str(dst_path), **save_kwargs)
        return

    if not HAS_PIEXIF:
        if logger:
            logger.log("WARN", "piexif not installed -> EXIF may lose GPS/ExifIFD. Install: pip install piexif")
        # fallback: write raw bytes directly (may still be OK sometimes)
        try:
            pil_img.save(str(dst_path), exif=src_exif_bytes, **save_kwargs)
            return
        except Exception:
            pil_img.save(str(dst_path), **save_kwargs)
            return

    # Try 1: keep everything
    exif_bytes = _update_exif_bytes_piexif(src_exif_bytes, out_w, out_h, zoom_factor,
                                          drop_makernote=False, drop_thumbnail=False)
    if exif_bytes:
        try:
            pil_img.save(str(dst_path), exif=exif_bytes, **save_kwargs)
            return
        except Exception:
            pass

    # Try 2: drop MakerNote (often problematic)
    exif_bytes2 = _update_exif_bytes_piexif(src_exif_bytes, out_w, out_h, zoom_factor,
                                           drop_makernote=True, drop_thumbnail=False)
    if exif_bytes2:
        try:
            pil_img.save(str(dst_path), exif=exif_bytes2, **save_kwargs)
            return
        except Exception:
            pass

    # Try 3: drop MakerNote + thumbnail
    exif_bytes3 = _update_exif_bytes_piexif(src_exif_bytes, out_w, out_h, zoom_factor,
                                           drop_makernote=True, drop_thumbnail=True)
    if exif_bytes3:
        try:
            pil_img.save(str(dst_path), exif=exif_bytes3, **save_kwargs)
            return
        except Exception:
            pass

    # Last: no EXIF
    pil_img.save(str(dst_path), **save_kwargs)


# ----------------------------
# Matching / crop logic (unchanged)
# ----------------------------

def downscale_max(bgr: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    h, w = bgr.shape[:2]
    if max(h, w) <= max_side:
        return bgr, 1.0
    s = max_side / float(max(h, w))
    nw, nh = int(round(w * s)), int(round(h * s))
    ds = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return ds, s


def to_gray_for_match(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag.astype(np.uint8)


def crop_box_from_fov(
    rgb_w: int, rgb_h: int, th_w: int, th_h: int,
    fov_frac_w: float,
    cx: float, cy: float
) -> Tuple[int, int, int, int]:
    aspect_th = th_w / float(th_h)
    crop_w = rgb_w * float(fov_frac_w)
    crop_h = crop_w / aspect_th

    if crop_h > rgb_h:
        crop_h = rgb_h * float(fov_frac_w)
        crop_w = crop_h * aspect_th

    crop_w = max(2.0, min(float(rgb_w), crop_w))
    crop_h = max(2.0, min(float(rgb_h), crop_h))

    x0 = int(round(cx - crop_w / 2.0))
    y0 = int(round(cy - crop_h / 2.0))
    x1 = int(round(x0 + crop_w))
    y1 = int(round(y0 + crop_h))
    return x0, y0, x1, y1


def crop_with_pad(bgr: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    h, w = bgr.shape[:2]
    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - w)
    pad_b = max(0, y1 - h)
    if pad_l or pad_t or pad_r or pad_b:
        bgr = cv2.copyMakeBorder(bgr, pad_t, pad_b, pad_l, pad_r,
                                 borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        x0 += pad_l
        x1 += pad_l
        y0 += pad_t
        y1 += pad_t
    x0 = max(0, min(x0, bgr.shape[1] - 1))
    y0 = max(0, min(y0, bgr.shape[0] - 1))
    x1 = max(x0 + 1, min(x1, bgr.shape[1]))
    y1 = max(y0 + 1, min(y1, bgr.shape[0]))
    return bgr[y0:y1, x0:x1]


def ncc_score(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("ncc_score requires same shape")
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    af -= af.mean()
    bf -= bf.mean()
    denom = (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-6)
    return float((af * bf).sum() / denom)


@dataclass
class FitResult:
    ok: bool
    reason: str
    fov_frac: Optional[float] = None
    match_score: Optional[float] = None
    ncc: Optional[float] = None
    cx_off_frac: Optional[float] = None
    cy_off_frac: Optional[float] = None


def estimate_fit_on_pair(
    rgb_bgr: np.ndarray,
    th_bgr: np.ndarray,
    th_size: Tuple[int, int],
    fov_candidates: List[float],
    rgb_max_side: int = 900,
    th_max_side: int = 600,
) -> FitResult:
    th_w, th_h = th_size
    rgb_h, rgb_w = rgb_bgr.shape[:2]

    rgb_ds, s_rgb = downscale_max(rgb_bgr, rgb_max_side)
    th_ds, _ = downscale_max(th_bgr, th_max_side)

    rgb_edge = to_gray_for_match(rgb_ds)
    th_edge = to_gray_for_match(th_ds)

    rgb_ds_h, rgb_ds_w = rgb_ds.shape[:2]
    th_ds_h, th_ds_w = th_ds.shape[:2]
    aspect_th = th_ds_w / float(th_ds_h)

    best = None  # (score, fov, x0, y0, tw, thh)

    for f in fov_candidates:
        tw = int(round(rgb_ds_w * float(f)))
        thh = int(round(tw / aspect_th))
        if thh > rgb_ds_h:
            thh = int(round(rgb_ds_h * float(f)))
            tw = int(round(thh * aspect_th))

        if tw < 40 or thh < 40:
            continue
        if tw >= rgb_ds_w or thh >= rgb_ds_h:
            continue

        tpl = cv2.resize(th_edge, (tw, thh), interpolation=cv2.INTER_AREA)
        try:
            res = cv2.matchTemplate(rgb_edge, tpl, cv2.TM_CCOEFF_NORMED)
            _, maxv, _, maxloc = cv2.minMaxLoc(res)
        except Exception:
            continue

        if best is None or float(maxv) > best[0]:
            best = (float(maxv), float(f), int(maxloc[0]), int(maxloc[1]), int(tw), int(thh))

    if best is None:
        return FitResult(ok=False, reason="no_valid_candidate")

    match_score, f_best, x0_ds, y0_ds, tw, thh = best

    x0 = int(round(x0_ds / s_rgb))
    y0 = int(round(y0_ds / s_rgb))
    x1 = int(round((x0_ds + tw) / s_rgb))
    y1 = int(round((y0_ds + thh) / s_rgb))

    crop = crop_with_pad(rgb_bgr, (x0, y0, x1, y1))
    crop_rs = cv2.resize(crop, (th_w, th_h), interpolation=cv2.INTER_AREA)
    crop_edge = to_gray_for_match(crop_rs)

    th_rs = cv2.resize(th_bgr, (th_w, th_h), interpolation=cv2.INTER_AREA)
    th_edge_rs = to_gray_for_match(th_rs)

    ncc = ncc_score(crop_edge, th_edge_rs)

    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    cx_off_frac = (cx - rgb_w / 2.0) / float(rgb_w)
    cy_off_frac = (cy - rgb_h / 2.0) / float(rgb_h)

    if not (match_score >= 0.02 and ncc >= 0.01):
        return FitResult(
            ok=False,
            reason=f"low_score(match={match_score:.3f},ncc={ncc:.3f})",
            fov_frac=f_best,
            match_score=match_score,
            ncc=ncc,
            cx_off_frac=cx_off_frac,
            cy_off_frac=cy_off_frac,
        )

    return FitResult(
        ok=True,
        reason="ok",
        fov_frac=f_best,
        match_score=match_score,
        ncc=ncc,
        cx_off_frac=cx_off_frac,
        cy_off_frac=cy_off_frac,
    )


# ----------------------------
# Visualization / montage
# ----------------------------

def resize_keep_aspect_pad(bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if w <= 0 or h <= 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    scale = min(target_w / w, target_h / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0 = (target_w - nw) // 2
    y0 = (target_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def put_text(bgr: np.ndarray, text: str, org: Tuple[int, int], scale: float = 0.9,
             color: Tuple[int, int, int] = (255, 255, 255), thickness: int = 2) -> None:
    cv2.putText(bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def overlay_thermal_rgb(th_bgr: np.ndarray, rgb_bgr: np.ndarray, alpha_th: float = 0.60) -> np.ndarray:
    if th_bgr.shape[:2] != rgb_bgr.shape[:2]:
        rgb_bgr = cv2.resize(rgb_bgr, (th_bgr.shape[1], th_bgr.shape[0]), interpolation=cv2.INTER_AREA)
    beta = 1.0 - alpha_th
    return cv2.addWeighted(th_bgr, alpha_th, rgb_bgr, beta, 0.0)


def draw_boxes_on_vis(
    vis_bgr: np.ndarray,
    fit_box: Tuple[int, int, int, int],
    exif_box: Optional[Tuple[int, int, int, int]]
) -> np.ndarray:
    out = vis_bgr.copy()
    x0, y0, x1, y1 = fit_box
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 8)  # FIT: red
    if exif_box is not None:
        gx0, gy0, gx1, gy1 = exif_box
        cv2.rectangle(out, (gx0, gy0), (gx1, gy1), (0, 255, 0), 8)  # EXIF: green
    return out


def make_comparison_montage(
    vis_boxes_bgr: np.ndarray,
    th_bgr: np.ndarray,
    img_exif_bgr: Optional[np.ndarray],
    img_fit_bgr: np.ndarray,
    cell_w: int,
    cell_h: int
) -> np.ndarray:
    """
    If EXIF exists: 3x2
      [vis]        [ir]
      [image-exif] [image-fit]
      [ov-exif]    [ov-fit]

    Else: 3 rows
      top: vis spans 2 cells
      mid: [ir][image-fit]
      bot: overlay-fit spans 2 cells
    """
    ov_fit = overlay_thermal_rgb(th_bgr, img_fit_bgr, alpha_th=0.60)
    ov_exif = overlay_thermal_rgb(th_bgr, img_exif_bgr, alpha_th=0.60) if img_exif_bgr is not None else None

    if img_exif_bgr is None or ov_exif is None:
        top = resize_keep_aspect_pad(vis_boxes_bgr, cell_w * 2, cell_h)
        put_text(top, "VIS (red=FIT)", (20, 50), 1.0)

        ir_cell = resize_keep_aspect_pad(th_bgr, cell_w, cell_h)
        put_text(ir_cell, "ir", (20, 50), 1.0)

        fit_cell = resize_keep_aspect_pad(img_fit_bgr, cell_w, cell_h)
        put_text(fit_cell, "image-fit", (20, 50), 1.0)

        mid = np.concatenate([ir_cell, fit_cell], axis=1)

        ov_fit_big = resize_keep_aspect_pad(ov_fit, cell_w * 2, cell_h)
        put_text(ov_fit_big, "overlay-fit (ir + image-fit)", (20, 50), 1.0)

        return np.concatenate([top, mid, ov_fit_big], axis=0)

    A = resize_keep_aspect_pad(vis_boxes_bgr, cell_w, cell_h)
    B = resize_keep_aspect_pad(th_bgr, cell_w, cell_h)
    C = resize_keep_aspect_pad(img_exif_bgr, cell_w, cell_h)
    D = resize_keep_aspect_pad(img_fit_bgr, cell_w, cell_h)
    E = resize_keep_aspect_pad(ov_exif, cell_w, cell_h)
    F = resize_keep_aspect_pad(ov_fit, cell_w, cell_h)

    put_text(A, "vis (red=FIT, green=EXIF)", (20, 50), 0.85)
    put_text(B, "ir", (20, 50), 1.0)
    put_text(C, "image-exif", (20, 50), 1.0)
    put_text(D, "image-fit", (20, 50), 1.0)
    put_text(E, "overlay-exif", (20, 50), 1.0)
    put_text(F, "overlay-fit", (20, 50), 1.0)

    row1 = np.concatenate([A, B], axis=1)
    row2 = np.concatenate([C, D], axis=1)
    row3 = np.concatenate([E, F], axis=1)
    return np.concatenate([row1, row2, row3], axis=0)


# ----------------------------
# Pairing / IO
# ----------------------------

@dataclass
class PairItem:
    stem: str
    rgb_path: Path
    th_path: Path


def find_pairs(rgb_dir: Path, th_dir: Path, logger: Logger) -> List[PairItem]:
    rgb_files = [p for p in rgb_dir.glob("*") if p.is_file() and is_image_file(p)]
    th_files = [p for p in th_dir.glob("*") if p.is_file() and is_image_file(p)]
    rgb_map: Dict[str, Path] = {stem_key(p): p for p in rgb_files}

    pairs: List[PairItem] = []
    for p in th_files:
        k = stem_key(p)
        if k in rgb_map:
            pairs.append(PairItem(stem=k, rgb_path=rgb_map[k], th_path=p))

    pairs.sort(key=lambda x: x.stem)
    logger.log("INFO", f"Found matched pairs={len(pairs)}")
    return pairs


def load_pair_images(pair: PairItem) -> Tuple[np.ndarray, np.ndarray]:
    rgb_pil = exif_transposed_pil(pair.rgb_path)
    th_pil = exif_transposed_pil(pair.th_path)
    return pil_to_bgr(rgb_pil), pil_to_bgr(th_pil)


def save_png(path: Path, bgr: np.ndarray) -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), bgr)


def robust_median(vals: List[float]) -> float:
    return float(np.median(np.array(vals, dtype=np.float32)))


def output_rgb_filename(pair: PairItem) -> str:
    suf = pair.rgb_path.suffix.lower()
    if suf in {".jpg", ".jpeg"}:
        return pair.rgb_path.name
    return f"{pair.stem}.jpg"


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb_dir", type=str, required=True)
    ap.add_argument("--th_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--samples", type=int, default=None)
    args = ap.parse_args()

    rgb_dir = Path(args.rgb_dir)
    th_dir = Path(args.th_dir)
    out_dir = Path(args.out_dir)

    debug_dir = out_dir / "debug"
    model_dir = out_dir / "model"
    img_fit_dir = out_dir / "image" / "image-fit"
    comp_dir = out_dir / "image" / "comparison"

    ensure_dir(debug_dir)
    ensure_dir(model_dir)
    ensure_dir(img_fit_dir)
    ensure_dir(comp_dir)

    logger = Logger(debug_dir / "run.log")
    per_image_path = debug_dir / "per_image.jsonl"
    summary_path = debug_dir / "summary.json"

    logger.log("INFO", "RUN START")
    logger.log("INFO", f"rgb_dir={rgb_dir}")
    logger.log("INFO", f"th_dir={th_dir}")
    logger.log("INFO", f"out_dir={out_dir}")
    if not HAS_PIEXIF:
        logger.log("WARN", "piexif not installed. Install to preserve GPS/ExifIFD: pip install piexif")

    pairs = find_pairs(rgb_dir, th_dir, logger)
    if not pairs:
        logger.log("ERROR", "No matched pairs found. Check filenames by stem.")
        logger.close()
        return 2

    if args.samples is not None and args.samples > 0 and args.samples < len(pairs):
        rnd = random.Random(0)
        pairs = rnd.sample(pairs, args.samples)
        pairs.sort(key=lambda x: x.stem)
        logger.log("INFO", f"Processing SAMPLED pairs={len(pairs)} (--samples={args.samples})")
    else:
        logger.log("INFO", f"Processing ALL pairs={len(pairs)} (no --samples)")

    rgb0, th0 = load_pair_images(pairs[0])
    th_h0, th_w0 = th0.shape[:2]
    rgb_h0, rgb_w0 = rgb0.shape[:2]
    th_size = (th_w0, th_h0)
    logger.log("INFO", f"Thermal size={th_w0}x{th_h0}, RGB(example)={rgb_w0}x{rgb_h0}")

    # EXIF probe
    exif_ok_count = 0
    exif_zoom_ratios: List[float] = []
    exif_fov_fracs_w: List[float] = []
    exif_raw_debug: List[dict] = []

    probe_n = min(30, len(pairs))
    for i in range(probe_n):
        p = pairs[i]
        rgb_exif, rgb_raw = read_exif_lens_info(p.rgb_path)
        th_exif, th_raw = read_exif_lens_info(p.th_path)
        zr = compute_zoom_ratio(rgb_exif, th_exif)
        exif_raw_debug.append({
            "stem": p.stem,
            "rgb_raw": rgb_raw,
            "th_raw": th_raw,
            "rgb_eff_f35": rgb_exif.effective_focal_35mm(),
            "th_eff_f35": th_exif.effective_focal_35mm(),
            "zoom_ratio": zr,
        })
        if zr is None or zr <= 0:
            continue
        exif_ok_count += 1
        exif_zoom_ratios.append(zr)
        exif_fov_fracs_w.append(1.0 / zr)

    exif_usable = exif_ok_count >= max(3, int(0.2 * probe_n))
    if exif_usable:
        zr_med = robust_median(exif_zoom_ratios)
        fov_exif_med = robust_median(exif_fov_fracs_w)
        logger.log("INFO", f"[EXIF] usable=True probe_ok={exif_ok_count}/{probe_n} "
                           f"zoom_ratio(med)={zr_med:.4f} fov_frac_w(med)={fov_exif_med:.4f}")
        img_exif_dir = out_dir / "image" / "image-exif"
        ensure_dir(img_exif_dir)
    else:
        zr_med = None
        fov_exif_med = None
        img_exif_dir = None
        logger.log("WARN", f"[EXIF] usable=False probe_ok={exif_ok_count}/{probe_n}. No image-exif/ model_exif.json.")

    # FIT estimation
    logger.log("INFO", "[FIT] Estimating global FIT model by template matching...")
    ok_fovs: List[float] = []
    ok_cxoffs: List[float] = []
    ok_cyoffs: List[float] = []

    if exif_usable and fov_exif_med is not None:
        base = float(fov_exif_med)
        fov_candidates = sorted({max(0.15, min(0.95, base * k)) for k in [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15]})
    else:
        fov_candidates = [round(x, 3) for x in np.linspace(0.20, 0.95, 16).tolist()]

    logger.log("INFO", f"[FIT] fov_candidates={fov_candidates}")

    t_fit0 = time.perf_counter()
    for idx, pair in enumerate(pairs, start=1):
        t0 = time.perf_counter()
        rgb_bgr, th_bgr = load_pair_images(pair)
        r = estimate_fit_on_pair(rgb_bgr, th_bgr, th_size=th_size, fov_candidates=fov_candidates)
        dt = time.perf_counter() - t0
        if r.ok and r.fov_frac is not None and r.cx_off_frac is not None and r.cy_off_frac is not None:
            ok_fovs.append(float(r.fov_frac))
            ok_cxoffs.append(float(r.cx_off_frac))
            ok_cyoffs.append(float(r.cy_off_frac))
            logger.log("INFO", f"[FIT] {pair.stem} ({idx}/{len(pairs)}) OK "
                               f"fov={r.fov_frac:.3f} match={r.match_score:.3f} ncc={r.ncc:.3f} "
                               f"cx_off={r.cx_off_frac:.5f} cy_off={r.cy_off_frac:.5f} ({dt:.2f}s)")
        else:
            logger.log("INFO", f"[FIT] {pair.stem} ({idx}/{len(pairs)}) FAIL reason={r.reason} ({dt:.2f}s)")

    t_fit = time.perf_counter() - t_fit0

    if len(ok_fovs) < max(3, int(0.1 * len(pairs))):
        if exif_usable and fov_exif_med is not None:
            fit_fov_med = float(fov_exif_med)
            fit_cx_med = 0.0
            fit_cy_med = 0.0
            logger.log("WARN", f"[FIT] Too few OK fits. Fallback to EXIF fov={fit_fov_med:.4f}, offsets=0.")
        else:
            fit_fov_med = 1.0
            fit_cx_med = 0.0
            fit_cy_med = 0.0
            logger.log("ERROR", "[FIT] Too few OK fits and no EXIF. Fallback to fov=1.0.")
    else:
        fit_fov_med = robust_median(ok_fovs)
        fit_cx_med = robust_median(ok_cxoffs)
        fit_cy_med = robust_median(ok_cyoffs)

    logger.log("INFO", f"[FIT] Done. time={t_fit:.1f}s ok={len(ok_fovs)}/{len(pairs)} "
                       f"fit_fov_med={fit_fov_med:.4f} fit_cx_off_med={fit_cx_med:.5f} fit_cy_off_med={fit_cy_med:.5f}")

    # Save models
    model_fit = {
        "version": 1,
        "thermal_size": {"w": th_w0, "h": th_h0},
        "fit": {
            "fov_frac_w": float(fit_fov_med),
            "cx_off_frac": float(fit_cx_med),
            "cy_off_frac": float(fit_cy_med),
            "ok_count": int(len(ok_fovs)),
            "total_count": int(len(pairs)),
            "candidates": fov_candidates,
        },
    }
    model_fit_path = model_dir / "model_fit.json"
    with open(model_fit_path, "w", encoding="utf-8") as f:
        json.dump(model_fit, f, ensure_ascii=False, indent=2)

    model_exif_path = None
    if exif_usable and fov_exif_med is not None and zr_med is not None:
        model_exif = {
            "version": 1,
            "thermal_size": {"w": th_w0, "h": th_h0},
            "exif": {
                "zoom_ratio_med": float(zr_med),
                "fov_frac_w_med": float(fov_exif_med),
                "probe_ok": int(exif_ok_count),
                "probe_n": int(probe_n),
            },
            "probe_samples": exif_raw_debug,
        }
        model_exif_path = model_dir / "model_exif.json"
        with open(model_exif_path, "w", encoding="utf-8") as f:
            json.dump(model_exif, f, ensure_ascii=False, indent=2)

    # Apply
    logger.log("INFO", "[APPLY] Writing image-fit / (optional) image-exif / comparison ...")
    per_image_fp = open(per_image_path, "w", encoding="utf-8", errors="ignore")

    cell_w, cell_h = th_w0, th_h0
    fit_written = 0
    exif_written = 0

    t_apply0 = time.perf_counter()
    for idx, pair in enumerate(pairs, start=1):
        t0 = time.perf_counter()
        rgb_bgr, th_bgr = load_pair_images(pair)
        rgb_h, rgb_w = rgb_bgr.shape[:2]

        # FIT
        cx_fit = rgb_w / 2.0 + fit_cx_med * rgb_w
        cy_fit = rgb_h / 2.0 + fit_cy_med * rgb_h
        fit_box = crop_box_from_fov(rgb_w, rgb_h, th_w0, th_h0, float(fit_fov_med), cx_fit, cy_fit)
        fit_crop_w = max(1, int(fit_box[2] - fit_box[0]))
        zoom_fit = float(rgb_w) / float(fit_crop_w)

        crop_fit = crop_with_pad(rgb_bgr, fit_box)
        img_fit = cv2.resize(crop_fit, (th_w0, th_h0), interpolation=cv2.INTER_AREA)

        # EXIF per-image
        exif_box = None
        img_exif = None
        zoom_exif = None
        if exif_usable:
            rgb_exif, _ = read_exif_lens_info(pair.rgb_path)
            th_exif, _ = read_exif_lens_info(pair.th_path)
            zr = compute_zoom_ratio(rgb_exif, th_exif)
            if zr is not None and zr > 0:
                fov_exif_w = 1.0 / zr
                exif_box = crop_box_from_fov(rgb_w, rgb_h, th_w0, th_h0, float(fov_exif_w), rgb_w / 2.0, rgb_h / 2.0)
                exif_crop_w = max(1, int(exif_box[2] - exif_box[0]))
                zoom_exif = float(rgb_w) / float(exif_crop_w)
                crop_exif = crop_with_pad(rgb_bgr, exif_box)
                img_exif = cv2.resize(crop_exif, (th_w0, th_h0), interpolation=cv2.INTER_AREA)

        # vis with boxes
        vis_boxes = draw_boxes_on_vis(rgb_bgr, fit_box, exif_box)

        # Save outputs with EXIF
        out_name = output_rgb_filename(pair)

        out_fit_path = img_fit_dir / out_name
        save_rgb_with_updated_exif(pair.rgb_path, img_fit, out_fit_path, zoom_fit, logger=logger)
        fit_written += 1

        out_exif_path = None
        if img_exif is not None and img_exif_dir is not None:
            out_exif_path = img_exif_dir / out_name
            save_rgb_with_updated_exif(pair.rgb_path, img_exif, out_exif_path, zoom_exif, logger=logger)
            exif_written += 1

        # comparison
        cmp_img = make_comparison_montage(
            vis_boxes_bgr=vis_boxes,
            th_bgr=th_bgr,
            img_exif_bgr=img_exif,
            img_fit_bgr=img_fit,
            cell_w=cell_w,
            cell_h=cell_h,
        )
        cmp_path = comp_dir / f"{pair.stem}.png"
        save_png(cmp_path, cmp_img)

        dt = time.perf_counter() - t0
        per_image_fp.write(json.dumps({
            "stem": pair.stem,
            "idx": idx,
            "total": len(pairs),
            "time_sec": round(dt, 4),
            "fit_out": str(out_fit_path),
            "exif_out": str(out_exif_path) if out_exif_path else None,
            "comparison": str(cmp_path),
        }, ensure_ascii=False) + "\n")
        per_image_fp.flush()

        logger.log("INFO", f"[IMG] {pair.stem} ({idx}/{len(pairs)}) "
                           f"fit_out={out_fit_path.name} exif_out={'YES' if out_exif_path else 'NO'} ({dt:.2f}s)")

    t_apply = time.perf_counter() - t_apply0
    per_image_fp.close()

    summary = {
        "version": 1,
        "paths": {
            "out_dir": str(out_dir),
            "debug_dir": str(debug_dir),
            "model_fit": str(model_fit_path),
            "model_exif": str(model_exif_path) if model_exif_path else None,
            "image_fit_dir": str(img_fit_dir),
            "image_exif_dir": str(img_exif_dir) if img_exif_dir else None,
            "comparison_dir": str(comp_dir),
            "run_log": str(debug_dir / "run.log"),
            "per_image_jsonl": str(per_image_path),
        },
        "counts": {"pairs": int(len(pairs)), "fit_written": int(fit_written), "exif_written": int(exif_written)},
        "timing": {"fit_sec": round(t_fit, 3), "apply_sec": round(t_apply, 3)},
        "exif": {"usable": bool(exif_usable), "probe_ok": int(exif_ok_count), "probe_n": int(probe_n)},
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.log("INFO", "[SUMMARY] ------------------------------")
    logger.log("INFO", f"[SUMMARY] pairs={len(pairs)} fit_written={fit_written} exif_written={exif_written}")
    logger.log("INFO", f"[SUMMARY] model_fit={model_fit_path}")
    if model_exif_path:
        logger.log("INFO", f"[SUMMARY] model_exif={model_exif_path}")
    logger.log("INFO", f"[SUMMARY] summary={summary_path}")
    logger.log("INFO", "RUN END")
    logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
