# -*- coding: utf-8 -*-
"""
cfrv10.py

在 cfrv9 “匹配/裁剪效果良好 + 元数据尽量保留”基础上，按用户最新策略迭代：

EXIF / XMP 策略（面向 COLMAP 重建）
- 不变：尽量原样保留（Make/Model/时间/GPS/XMP-drone-dji 等）
- 变动：裁剪/resize 后必须更新（Orientation=1、ImageWidth/Height、ExifImageWidth/Height、
        DigitalZoomRatio、FocalLengthIn35mmFormat）
- 不可一致维护：删除 IFD1/MPF/Preview/Thumbnail 等预览块，避免缩略图/多图封装干扰

Debug（用于核对“新图 EXIF 是否正确”）
- debug/exif_audit-<suffix>.jsonl：逐图关键字段 + problems 列表（方向/尺寸等）
- debug/exif_src-<suffix>.json / exif_dst_fit-... / exif_dst_exif-... / exif_diff-...
- 修复 exiftool dump stdout 为空导致的 NoneType 异常

CLI 参数
1) --comparison：传入则输出对比图（默认不输出，提高速度）
2) --align {exif,fit,both}：默认 both
3) --stage {fit,apply,both}：默认 both
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
import tempfile

import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:
    raise RuntimeError("opencv-python is required. Install: pip install opencv-python") from e

try:
    from PIL import Image, ImageOps
except Exception as e:
    raise RuntimeError("Pillow is required. Install: pip install pillow") from e

try:
    import piexif  # type: ignore
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
            denf = float(den)
            if denf == 0:
                return None
            return float(num) / denf
        return float(x)
    except Exception:
        return None


# ----------------------------
# Image conversions
# ----------------------------

def exif_transposed_pil(img_path: Path) -> Image.Image:
    """
    读入图像并把 EXIF Orientation 应用到像素（得到“正向”像素）。
    后续保存时必须强制 Orientation=1，避免查看器二次旋转导致“倒图”。
    """
    im = Image.open(img_path)
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    return im


def pil_to_bgr(img: Image.Image) -> np.ndarray:
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
# EXIF read (robust)
# ----------------------------

@dataclass
class ExifLensInfo:
    focal_35mm: Optional[float] = None        # EXIF: FocalLengthIn35mmFilm (piexif field name)
    focal_mm: Optional[float] = None          # EXIF: FocalLength
    digital_zoom: Optional[float] = None      # EXIF: DigitalZoomRatio

    def effective_focal_35mm(self) -> Optional[float]:
        if self.focal_35mm is None:
            return None
        dz = self.digital_zoom if (self.digital_zoom is not None and self.digital_zoom > 0) else 1.0
        return float(self.focal_35mm) * float(dz)


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


def read_exif_lens_info(src_path: Path) -> ExifLensInfo:
    info = ExifLensInfo()
    b = _get_src_exif_bytes(src_path)
    if b is None:
        return info

    if HAS_PIEXIF:
        try:
            ex = piexif.load(b)
            f35 = ex.get("Exif", {}).get(piexif.ExifIFD.FocalLengthIn35mmFilm, None)
            fl = ex.get("Exif", {}).get(piexif.ExifIFD.FocalLength, None)
            dz = ex.get("Exif", {}).get(piexif.ExifIFD.DigitalZoomRatio, None)
            info.focal_35mm = safe_float(f35)
            info.focal_mm = safe_float(fl)
            info.digital_zoom = safe_float(dz)
            return info
        except Exception:
            pass

    # fallback (PIL tag IDs)
    try:
        with Image.open(src_path) as im:
            ex = im.getexif()
            info.focal_mm = safe_float(ex.get(37386, None))      # FocalLength
            info.digital_zoom = safe_float(ex.get(41988, None))  # DigitalZoomRatio
            info.focal_35mm = safe_float(ex.get(41989, None))    # FocalLengthIn35mmFilm
    except Exception:
        pass
    return info


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
# exiftool helpers
# ----------------------------

@lru_cache(maxsize=1)
def find_exiftool() -> Optional[str]:
    exe = shutil.which("exiftool") or shutil.which("exiftool.exe")
    return exe


def run_exiftool(cmd: List[str], logger: Logger) -> bool:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        out = (p.stdout or "").strip()
        err = (p.stderr or "").strip()
        if out:
            logger.log("INFO", f"[exiftool] {out}")
        if err:
            logger.log("WARN", f"[exiftool] {err}")
        return p.returncode == 0
    except Exception as e:
        logger.log("WARN", f"[exiftool] failed to run: {e}")
        return False


def _compute_zoom_updates_for_crop(
    src_rgb_path: Path,
    zoom_factor: Optional[float],
) -> Tuple[Optional[float], Optional[int]]:
    """
    把裁剪视场缩小解释成“数字变焦”：
      - DigitalZoomRatio = 原DZ * zoom_factor
      - FocalLengthIn35mmFormat = 原35mm等效 * zoom_factor （ExifTool 正确 tag：...Format）
    物理焦距 FocalLength 不改。
    """
    if zoom_factor is None or zoom_factor <= 0:
        return None, None

    src = read_exif_lens_info(src_rgb_path)
    base_dz = src.digital_zoom if (src.digital_zoom is not None and src.digital_zoom > 0) else 1.0
    upd_dz = float(base_dz) * float(zoom_factor)

    upd_f35 = None
    if src.focal_35mm is not None and src.focal_35mm > 0:
        upd_f35 = int(round(float(src.focal_35mm) * float(zoom_factor)))

    return upd_dz, upd_f35


# --- JPEG XMP packet helpers (v9) ---

_XMP_STD_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"
_XMP_EXT_HEADER = b"http://ns.adobe.com/xmp/extension/\x00"

def _extract_xmp_packet_from_jpeg(jpeg_path: Path) -> Optional[bytes]:
    """Extract the main XMP packet (RDF/XML) from a JPEG APP1 segment.

    Returns the raw XML bytes (without the APP1 header) or None if not present / not JPEG.
    """
    try:
        with open(jpeg_path, "rb") as fp:
            if fp.read(2) != b"\xFF\xD8":  # SOI
                return None
            while True:
                marker = fp.read(2)
                if len(marker) < 2:
                    return None
                # Scan to next 0xFF
                if marker[0] != 0xFF:
                    continue
                # Start of Scan or End of Image -> no more metadata segments
                if marker in (b"\xFF\xDA", b"\xFF\xD9"):
                    return None
                # Some markers have no length (RST, TEM, etc.) but APPn do.
                len_bytes = fp.read(2)
                if len(len_bytes) < 2:
                    return None
                seg_len = int.from_bytes(len_bytes, "big")
                if seg_len < 2:
                    return None
                seg = fp.read(seg_len - 2)
                if marker == b"\xFF\xE1":  # APP1
                    if seg.startswith(_XMP_STD_HEADER):
                        return seg[len(_XMP_STD_HEADER):]
                    # NOTE: Extended XMP is chunked; we don't reconstruct it here.
                    # Fall back to normal tag-copy if needed.
                    if seg.startswith(_XMP_EXT_HEADER):
                        # present but not reconstructed
                        return None
    except Exception:
        return None


def _write_temp_xmp(xmp_xml: bytes) -> Path:
    """Write XMP XML bytes to a temp file and return its path (caller deletes)."""
    tmp = tempfile.NamedTemporaryFile(prefix="cfrv9_xmp_", suffix=".xmp", delete=False)
    try:
        tmp.write(xmp_xml)
        tmp.flush()
    finally:
        tmp.close()
    return Path(tmp.name)


def _exiftool_copy_then_patch(
    exiftool: str,
    src_rgb_path: Path,
    dst_path: Path,
    out_w: int,
    out_h: int,
    upd_dz: Optional[float],
    upd_f35: Optional[float],
    logger: Logger,
) -> bool:
    """v9: One-pass copy+patch with raw XMP packet injection.

    - Copy all metadata with -all:all -unsafe
    - Strip preview/thumbnail/MPF/IFD1
    - Inject raw XMP packet from source JPEG (best effort)
    - Patch size + (DigitalZoomRatio, FocalLengthIn35mmFormat)
    - Force Orientation=1 in both IFD0 and XMP-tiff
    """
    xmp_tmp: Optional[Path] = None
    try:
        xmp_xml = _extract_xmp_packet_from_jpeg(src_rgb_path)
        if xmp_xml:
            xmp_tmp = _write_temp_xmp(xmp_xml)

        cmd: List[str] = [
            exiftool,
            "-overwrite_original",
            "-P",
            "-n",
            "-m",
            "-q",
            "-q",
            "-TagsFromFile", str(src_rgb_path),
            "-all:all",
            "-xmp:all",
            "-unsafe",
            "-icc_profile:all",

            # strip thumbnails / previews that may confuse downstream tools
            "-MPF:all=",
            "-IFD1:all=",
            "-PreviewImage=",
            "-ThumbnailImage=",
            "-JpgFromRaw=",

            # size fields (keep v8 policy: do NOT write PixelXDimension/YDimension)
            f"-IFD0:ImageWidth={int(out_w)}",
            f"-IFD0:ImageHeight={int(out_h)}",
            f"-ExifIFD:ExifImageWidth={int(out_w)}",
            f"-ExifIFD:ExifImageHeight={int(out_h)}",
        ]

        if upd_dz is not None:
            cmd.append(f"-ExifIFD:DigitalZoomRatio={float(upd_dz):.10f}")
        if upd_f35 is not None:
            cmd.append(f"-ExifIFD:FocalLengthIn35mmFormat={int(round(upd_f35))}")

        # raw XMP packet (best effort; placed before orientation patch)
        if xmp_tmp is not None:
            cmd.append(f"-XMP<={str(xmp_tmp)}")

        # final orientation normalization (must be last)
        cmd += [
            "-Orientation=1",
            "-IFD0:Orientation=1",
            "-XMP-tiff:Orientation=1",
            str(dst_path),
        ]

        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            logger.log("WARN", f"exiftool failed rc={r.returncode} dst={dst_path.name} err={r.stderr.strip()[:400]}")
            return False
        return True
    except Exception as e:
        logger.log("WARN", f"exiftool exception dst={dst_path.name}: {e}")
        return False
    finally:
        if xmp_tmp is not None:
            try:
                xmp_tmp.unlink(missing_ok=True)  # type: ignore[attr-defined]
            except Exception:
                try:
                    xmp_tmp.unlink()
                except Exception:
                    pass
def save_with_metadata(
    src_rgb_path: Path,
    dst_path: Path,
    out_bgr: np.ndarray,
    out_w: int,
    out_h: int,
    zoom_factor: Optional[float],
    logger: Logger,
) -> None:
    """
    保存像素 + 尽可能完整保留元数据（含 GPS / XMP-drone-dji 等）
    """
    ensure_dir(dst_path.parent)

    # 1) write pixels first
    pil = bgr_to_pil_rgb(out_bgr)
    save_kwargs = dict(format="JPEG", quality=95, optimize=True, subsampling=2)
    pil.save(str(dst_path), **save_kwargs)

    exiftool = find_exiftool()
    if exiftool:
        upd_dz, upd_f35 = _compute_zoom_updates_for_crop(src_rgb_path, zoom_factor)
        ok = _exiftool_copy_then_patch(
            exiftool=exiftool,
            src_rgb_path=src_rgb_path,
            dst_path=dst_path,
            out_w=out_w,
            out_h=out_h,
            upd_dz=upd_dz,
            upd_f35=upd_f35,
            logger=logger,
        )
        if ok:
            return
        logger.log("WARN", "exiftool copy/patch failed -> fallback to piexif (XMP/drone-dji may be lost).")

    # fallback: piexif (NOTE: will NOT preserve XMP blocks)
    b = _get_src_exif_bytes(src_rgb_path)
    if not (HAS_PIEXIF and b):
        return

    try:
        ex = piexif.load(b)

        # orientation + size
        try:
            ex["0th"][piexif.ImageIFD.Orientation] = 1
            ex["0th"][piexif.ImageIFD.ImageWidth] = int(out_w)
            ex["0th"][piexif.ImageIFD.ImageLength] = int(out_h)
        except Exception:
            pass
        try:
            ex["Exif"][piexif.ExifIFD.ExifImageWidth] = int(out_w)
            ex["Exif"][piexif.ExifIFD.ExifImageHeight] = int(out_h)
        except Exception:
            pass

        # remove thumbnails
        ex["thumbnail"] = None

        # update zoom + f35
        upd_dz, upd_f35 = _compute_zoom_updates_for_crop(src_rgb_path, zoom_factor)

        if upd_dz is not None:
            try:
                den = 1000000
                num = int(round(float(upd_dz) * den))
                g = int(np.gcd(num, den))
                ex["Exif"][piexif.ExifIFD.DigitalZoomRatio] = (num // g, den // g)
            except Exception:
                pass

        if upd_f35 is not None:
            try:
                ex["Exif"][piexif.ExifIFD.FocalLengthIn35mmFilm] = int(upd_f35)
            except Exception:
                pass

        exif_bytes = piexif.dump(ex)
        pil.save(str(dst_path), **save_kwargs, exif=exif_bytes)
    except Exception as e:
        logger.log("WARN", f"piexif fallback failed: {e}")

# ----------------------------
# Debug helpers (v9)
# ----------------------------

def _exiftool_dump_json(exiftool: str, img_path: Path, out_json: Path, logger: Logger) -> Optional[Dict]:
    """Dump full metadata as JSON (single image). Returns the first object dict.

    Robust to empty/None stdout (avoid NoneType write errors).
    """
    try:
        cmd = [exiftool, "-j", "-G1", "-a", "-u", "-s", "-n", str(img_path)]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        stdout = (r.stdout or "")
        stderr = (r.stderr or "").strip()
        if r.returncode != 0:
            logger.log("WARN", f"exiftool dump failed rc={r.returncode} file={img_path.name} err={stderr[:300]}")
            return None
        if not stdout.strip():
            logger.log("WARN", f"exiftool dump empty stdout file={img_path.name} err={stderr[:200]}")
            return None
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(stdout, encoding="utf-8", errors="ignore")
        arr = json.loads(stdout)
        if isinstance(arr, list) and arr:
            return arr[0]
        return None
    except Exception as e:
        logger.log("WARN", f"exiftool dump exception file={img_path.name}: {e}")
        return None


def _diff_meta_dict(src: Dict, dst: Dict) -> Dict:
    """Compute a lightweight diff between two exiftool JSON dicts."""
    ignore_prefix = ("File:", "System:", "ExifTool:", "Composite:")
    ignore_keys = {"SourceFile", "FileName", "Directory", "FileSize", "FileModifyDate", "FileAccessDate", "FileInodeChangeDate"}
    def norm(d: Dict) -> Dict:
        out = {}
        for k,v in d.items():
            if k in ignore_keys:
                continue
            if any(k.startswith(p) for p in ignore_prefix):
                continue
            out[k]=v
        return out

    a = norm(src)
    b = norm(dst)
    a_keys=set(a.keys()); b_keys=set(b.keys())
    missing = sorted(a_keys - b_keys)
    extra = sorted(b_keys - a_keys)
    changed = []
    for k in sorted(a_keys & b_keys):
        if a.get(k) != b.get(k):
            changed.append({"key": k, "src": a.get(k), "dst": b.get(k)})
    return {
        "missing_in_dst_count": len(missing),
        "extra_in_dst_count": len(extra),
        "changed_count": len(changed),
        "missing_in_dst": missing[:200],
        "extra_in_dst": extra[:200],
        "changed_sample": changed[:200],
    }


def _write_exif_audit_jsonl(
    exiftool: str,
    scan_dir: Path,
    out_jsonl: Path,
    kind: str,
    expected_w: int,
    expected_h: int,
    logger: Logger
) -> None:
    """Scan a directory with exiftool once and write per-file audit lines.

    Audit focuses on correctness for COLMAP:
    - Orientation normalized (IFD0 + optionally XMP-tiff)
    - Width/Height fields match expected (thermal size)
    - DigitalZoomRatio / FocalLengthIn35mmFormat present (recorded for review)
    """
    try:
        cmd = [
            exiftool,
            "-j", "-G1", "-a", "-u", "-s", "-n",
            "-r",
            "-Orientation",
            "-IFD0:Orientation",
            "-XMP-tiff:Orientation",
            "-IFD0:ImageWidth",
            "-IFD0:ImageHeight",
            "-ExifIFD:ExifImageWidth",
            "-ExifIFD:ExifImageHeight",
            "-ExifIFD:DigitalZoomRatio",
            "-ExifIFD:FocalLengthIn35mmFormat",
            "-GPSLatitude", "-GPSLongitude",
            "-XMP-drone-dji:UTCAtExposure",
            "-XMP-drone-dji:DroneModel",
            "-XMP-drone-dji:CameraSerialNumber",
            str(scan_dir),
        ]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            logger.log("WARN", f"exiftool audit failed rc={r.returncode} dir={scan_dir} err={(r.stderr or '').strip()[:300]}")
            return
        stdout = (r.stdout or "").strip()
        if not stdout:
            logger.log("WARN", f"exiftool audit empty stdout dir={scan_dir}")
            return
        arr = json.loads(stdout)
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(out_jsonl, "a", encoding="utf-8", errors="ignore") as fp:
            for obj in arr:
                if not isinstance(obj, dict):
                    continue

                problems: List[str] = []

                o1 = obj.get("Orientation") or obj.get("IFD0:Orientation")
                o2 = obj.get("XMP-tiff:Orientation")

                if o1 not in (None, 1, "1"):
                    problems.append("orientation_not_1")
                if o2 not in (None, 1, "1"):
                    problems.append("xmp_orientation_not_1")

                w0 = obj.get("IFD0:ImageWidth")
                h0 = obj.get("IFD0:ImageHeight")
                we = obj.get("ExifIFD:ExifImageWidth")
                he = obj.get("ExifIFD:ExifImageHeight")

                # size checks: expect both groups match thermal size
                if w0 is not None and int(w0) != int(expected_w):
                    problems.append("ifd0_width_mismatch")
                if h0 is not None and int(h0) != int(expected_h):
                    problems.append("ifd0_height_mismatch")
                if we is not None and int(we) != int(expected_w):
                    problems.append("exif_width_mismatch")
                if he is not None and int(he) != int(expected_h):
                    problems.append("exif_height_mismatch")

                fp.write(json.dumps({
                    "kind": kind,
                    "expected_size": {"w": int(expected_w), "h": int(expected_h)},
                    "SourceFile": obj.get("SourceFile"),
                    "Orientation": o1,
                    "XMP-tiff:Orientation": o2,
                    "IFD0:ImageWidth": w0,
                    "IFD0:ImageHeight": h0,
                    "ExifIFD:ExifImageWidth": we,
                    "ExifIFD:ExifImageHeight": he,
                    "ExifIFD:DigitalZoomRatio": obj.get("ExifIFD:DigitalZoomRatio"),
                    "ExifIFD:FocalLengthIn35mmFormat": obj.get("ExifIFD:FocalLengthIn35mmFormat"),
                    "GPSLatitude": obj.get("GPSLatitude"),
                    "GPSLongitude": obj.get("GPSLongitude"),
                    "XMP-drone-dji:UTCAtExposure": obj.get("XMP-drone-dji:UTCAtExposure"),
                    "XMP-drone-dji:DroneModel": obj.get("XMP-drone-dji:DroneModel"),
                    "XMP-drone-dji:CameraSerialNumber": obj.get("XMP-drone-dji:CameraSerialNumber"),
                    "problems": problems,
                }, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.log("WARN", f"exif audit exception dir={scan_dir}: {e}")

# ----------------------------
# Matching / crop logic (保持不变)
# ----------------------------

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
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 8)  # FIT red
    if exif_box is not None:
        gx0, gy0, gx1, gy1 = exif_box
        cv2.rectangle(out, (gx0, gy0), (gx1, gy1), (0, 255, 0), 8)  # EXIF green
    return out


def make_comparison_montage(
    vis_boxes_bgr: np.ndarray,
    th_bgr: np.ndarray,
    img_exif_bgr: Optional[np.ndarray],
    img_fit_bgr: np.ndarray,
    cell_w: int,
    cell_h: int
) -> np.ndarray:
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

    # performance / UX
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--comparison", action="store_true",
                    help="If set, output comparison montages (slower).")

    # user-requested defaults
    ap.add_argument("--align", type=str, default="both", choices=["exif", "fit", "both"],
                    help="Which aligned outputs to generate: exif/fit/both (default both).")
    ap.add_argument("--stage", type=str, default="both", choices=["fit", "apply", "both"],
                    help="Which stages to run: fit/apply/both (default both).")

    args = ap.parse_args()

    rgb_dir = Path(args.rgb_dir)
    th_dir = Path(args.th_dir)
    out_dir = Path(args.out_dir)
    suffix = out_dir.name

    # flags
    want_comp = bool(args.comparison)
    want_fit = args.align in ("fit", "both")
    want_exif = args.align in ("exif", "both")
    do_fit = args.stage in ("fit", "both")
    do_apply = args.stage in ("apply", "both")

    # comparison needs boxes; boxes need fit model
    need_fit_model = want_fit or want_comp
    need_exif_probe = want_exif or need_fit_model  # exif can help seed fov candidates

    # paths
    debug_dir = out_dir / "debug"
    model_dir = out_dir / "model"
    img_root = out_dir / "image"
    img_fit_dir = img_root / "image-fit"
    img_exif_dir = img_root / "image-exif"
    comp_dir = img_root / "comparison"

    ensure_dir(debug_dir)
    ensure_dir(model_dir)

    logger = Logger(debug_dir / f"run-{suffix}.log")
    exif_audit_path = debug_dir / f"exif_audit-{suffix}.jsonl"
    per_image_path = debug_dir / f"per_image-{suffix}.jsonl"
    summary_path = debug_dir / f"summary-{suffix}.json"

    logger.log("INFO", "RUN START")
    logger.log("INFO", f"rgb_dir={rgb_dir}")
    logger.log("INFO", f"th_dir={th_dir}")
    logger.log("INFO", f"out_dir={out_dir}")
    logger.log("INFO", f"stage={args.stage} align={args.align} comparison={want_comp}")
    logger.log("INFO", f"piexif_installed={HAS_PIEXIF}")
    logger.log("INFO", f"exiftool_found={bool(find_exiftool())}")

    pairs_all = find_pairs(rgb_dir, th_dir, logger)
    pairs_found = len(pairs_all)
    if not pairs_all:
        logger.log("ERROR", "No matched pairs found. Check filenames by stem.")
        logger.close()
        return 2

    pairs = pairs_all
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

    # ---------------- EXIF probe (for exif output & fit initialization) ----------------
    exif_ok_count = 0
    exif_zoom_ratios: List[float] = []
    exif_fov_fracs_w: List[float] = []

    exif_usable = False
    zr_med: Optional[float] = None
    fov_exif_med: Optional[float] = None
    probe_n = min(30, len(pairs)) if need_exif_probe else 0

    if need_exif_probe:
        for i in range(probe_n):
            p = pairs[i]
            rgb_exif = read_exif_lens_info(p.rgb_path)
            th_exif = read_exif_lens_info(p.th_path)
            zr = compute_zoom_ratio(rgb_exif, th_exif)
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
        else:
            logger.log("WARN", f"[EXIF] usable=False probe_ok={exif_ok_count}/{probe_n}. "
                               f"image-exif will be skipped unless --align exif is requested.")
    else:
        logger.log("INFO", "[EXIF] probe skipped (not needed by current flags).")

    if want_exif and not exif_usable:
        logger.log("ERROR", "[EXIF] align=exif/both requested but EXIF zoom ratio is not usable on this dataset.")
        logger.close()
        return 3

    # ---------------- FIT estimation / load model ----------------
    fit_fov_med: float = 1.0
    fit_cx_med: float = 0.0
    fit_cy_med: float = 0.0
    fov_candidates: List[float] = []
    ok_fovs: List[float] = []
    ok_cxoffs: List[float] = []
    ok_cyoffs: List[float] = []

    model_fit_path = model_dir / "model_fit.json"
    model_exif_path: Optional[Path] = None

    t_fit = 0.0
    if need_fit_model:
        if do_fit:
            logger.log("INFO", "[FIT] Estimating global FIT model...")
            if exif_usable and fov_exif_med is not None:
                base = float(fov_exif_med)
                fov_candidates = sorted({max(0.15, min(0.95, base * k))
                                         for k in [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15]})
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

            model_fit = {
                "version": 10,
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
            with open(model_fit_path, "w", encoding="utf-8") as f:
                json.dump(model_fit, f, ensure_ascii=False, indent=2)
        else:
            # load existing model_fit
            if not model_fit_path.exists():
                logger.log("ERROR", f"[FIT] stage=apply but model_fit.json not found at {model_fit_path}")
                logger.close()
                return 4
            try:
                model_fit = json.loads(model_fit_path.read_text(encoding="utf-8", errors="ignore"))
                fit_fov_med = float(model_fit["fit"]["fov_frac_w"])
                fit_cx_med = float(model_fit["fit"]["cx_off_frac"])
                fit_cy_med = float(model_fit["fit"]["cy_off_frac"])
                logger.log("INFO", f"[FIT] Loaded model_fit.json fov={fit_fov_med:.4f} cx_off={fit_cx_med:.5f} cy_off={fit_cy_med:.5f}")
            except Exception as e:
                logger.log("ERROR", f"[FIT] Failed to load model_fit.json: {e}")
                logger.close()
                return 5
    else:
        logger.log("INFO", "[FIT] Skipped (align=exif and comparison disabled).")

    # save model_exif when fitting and user wants exif/both
    if do_fit and want_exif and exif_usable and zr_med is not None and fov_exif_med is not None:
        model_exif = {
            "version": 10,
            "thermal_size": {"w": th_w0, "h": th_h0},
            "exif": {
                "zoom_ratio_med": float(zr_med),
                "fov_frac_w_med": float(fov_exif_med),
                "probe_ok": int(exif_ok_count),
                "probe_n": int(probe_n),
            },
        }
        model_exif_path = model_dir / "model_exif.json"
        with open(model_exif_path, "w", encoding="utf-8") as f:
            json.dump(model_exif, f, ensure_ascii=False, indent=2)

    # ---------------- APPLY ----------------
    fit_written = 0
    exif_written = 0
    t_apply = 0.0

    if do_apply:
        logger.log("INFO", "[APPLY] Writing image-fit / image-exif (optional) / comparison (optional) ...")

        if want_fit:
            ensure_dir(img_fit_dir)
        if want_exif and exif_usable:
            ensure_dir(img_exif_dir)
        if want_comp:
            ensure_dir(comp_dir)

        per_fp = open(per_image_path, "w", encoding="utf-8", errors="ignore")

        cell_w, cell_h = th_w0, th_h0

        t_apply0 = time.perf_counter()
        for idx, pair in enumerate(pairs, start=1):
            t0 = time.perf_counter()
            rgb_bgr, th_bgr = load_pair_images(pair)
            rgb_h, rgb_w = rgb_bgr.shape[:2]

            out_name = output_rgb_filename(pair)

            # Decide which crops are needed
            need_fit_crop = want_fit or want_comp
            need_exif_crop = (want_exif or want_comp) and exif_usable

            fit_box = None
            img_fit = None
            zoom_fit = None

            if need_fit_crop:
                cx_fit = rgb_w / 2.0 + fit_cx_med * rgb_w
                cy_fit = rgb_h / 2.0 + fit_cy_med * rgb_h
                fit_box = crop_box_from_fov(rgb_w, rgb_h, th_w0, th_h0, float(fit_fov_med), cx_fit, cy_fit)
                fit_crop_w = max(1, int(fit_box[2] - fit_box[0]))
                zoom_fit = float(rgb_w) / float(fit_crop_w)
                crop_fit = crop_with_pad(rgb_bgr, fit_box)
                img_fit = cv2.resize(crop_fit, (th_w0, th_h0), interpolation=cv2.INTER_AREA)

            exif_box = None
            img_exif = None
            zoom_exif = None
            if need_exif_crop:
                rgb_exif = read_exif_lens_info(pair.rgb_path)
                th_exif = read_exif_lens_info(pair.th_path)
                zr = compute_zoom_ratio(rgb_exif, th_exif)
                if zr is not None and zr > 0:
                    fov_exif_w = 1.0 / zr
                    exif_box = crop_box_from_fov(rgb_w, rgb_h, th_w0, th_h0, float(fov_exif_w),
                                                 rgb_w / 2.0, rgb_h / 2.0)
                    exif_crop_w = max(1, int(exif_box[2] - exif_box[0]))
                    zoom_exif = float(rgb_w) / float(exif_crop_w)
                    crop_exif = crop_with_pad(rgb_bgr, exif_box)
                    img_exif = cv2.resize(crop_exif, (th_w0, th_h0), interpolation=cv2.INTER_AREA)

            out_fit_path = None
            out_exif_path = None

            # Save fit output
            if want_fit and img_fit is not None:
                out_fit_path = img_fit_dir / out_name
                save_with_metadata(pair.rgb_path, out_fit_path, img_fit, th_w0, th_h0, zoom_fit, logger)
                fit_written += 1

            # Save exif output
            if want_exif and img_exif is not None and exif_usable:
                out_exif_path = img_exif_dir / out_name
                save_with_metadata(pair.rgb_path, out_exif_path, img_exif, th_w0, th_h0, zoom_exif, logger)
                exif_written += 1

            # comparison montage (optional)
            cmp_path = None
            if want_comp:
                # vis boxes needs fit_box at least
                if fit_box is None:
                    # fall back to full-frame box if fit was not computed for some reason
                    fit_box = (0, 0, rgb_w, rgb_h)
                vis_boxes = draw_boxes_on_vis(rgb_bgr, fit_box, exif_box)
                # montage expects img_fit; if missing, reuse img_exif or blank
                if img_fit is None:
                    img_fit = img_exif if img_exif is not None else cv2.resize(rgb_bgr, (th_w0, th_h0), interpolation=cv2.INTER_AREA)
                cmp_img = make_comparison_montage(vis_boxes, th_bgr, img_exif, img_fit, cell_w, cell_h)
                cmp_path = comp_dir / f"{pair.stem}.png"
                save_png(cmp_path, cmp_img)

            exp_dz_fit, exp_f35_fit = (None, None)
            exp_dz_exif, exp_f35_exif = (None, None)
            if zoom_fit is not None:
                exp_dz_fit, exp_f35_fit = _compute_zoom_updates_for_crop(pair.rgb_path, zoom_fit)
            if zoom_exif is not None:
                exp_dz_exif, exp_f35_exif = _compute_zoom_updates_for_crop(pair.rgb_path, zoom_exif)

            dt = time.perf_counter() - t0
            per_fp.write(json.dumps({
                "stem": pair.stem,
                "idx": idx,
                "total": len(pairs),
                "time_sec": round(dt, 4),
                "fit_out": str(out_fit_path) if out_fit_path else None,
                "exif_out": str(out_exif_path) if out_exif_path else None,
                "comparison": str(cmp_path) if cmp_path else None,
                "zoom_fit": float(zoom_fit) if zoom_fit is not None else None,
                "zoom_exif": float(zoom_exif) if zoom_exif is not None else None,
                "expected_dzoom_fit": float(exp_dz_fit) if exp_dz_fit is not None else None,
                "expected_f35_fit": int(exp_f35_fit) if exp_f35_fit is not None else None,
                "expected_dzoom_exif": float(exp_dz_exif) if exp_dz_exif is not None else None,
                "expected_f35_exif": int(exp_f35_exif) if exp_f35_exif is not None else None,
            }, ensure_ascii=False) + "\n")
            per_fp.flush()

            logger.log("INFO", f"[IMG] {pair.stem} ({idx}/{len(pairs)}) "
                               f"fit_out={'YES' if out_fit_path else 'NO'} "
                               f"exif_out={'YES' if out_exif_path else 'NO'} "
                               f"cmp={'YES' if cmp_path else 'NO'} ({dt:.2f}s)")

        t_apply = time.perf_counter() - t_apply0
        per_fp.close()

        # ---- v10 debug: exif audit + sample dump/diff ----
        exiftool = find_exiftool()
        if exiftool:
            try:
                exif_audit_path.write_text("", encoding="utf-8")
            except Exception:
                pass

            if want_fit and img_fit_dir.exists():
                _write_exif_audit_jsonl(exiftool, img_fit_dir, exif_audit_path, kind="fit",
                                        expected_w=th_w0, expected_h=th_h0, logger=logger)
            if want_exif and exif_usable and img_exif_dir.exists():
                _write_exif_audit_jsonl(exiftool, img_exif_dir, exif_audit_path, kind="exif",
                                        expected_w=th_w0, expected_h=th_h0, logger=logger)

            # dump + diff only for the first pair (keep debug light)
            if pairs:
                first = pairs[0]
                out_name0 = output_rgb_filename(first)
                src0 = first.rgb_path
                dst_fit0 = (img_fit_dir / out_name0) if (want_fit and (img_fit_dir / out_name0).exists()) else None
                dst_exif0 = (img_exif_dir / out_name0) if (want_exif and (img_exif_dir / out_name0).exists()) else None

                src_meta = _exiftool_dump_json(exiftool, src0, debug_dir / f"exif_src-{suffix}.json", logger)

                dst_fit_meta = None
                if dst_fit0 is not None:
                    dst_fit_meta = _exiftool_dump_json(exiftool, dst_fit0, debug_dir / f"exif_dst_fit-{suffix}.json", logger)

                dst_exif_meta = None
                if dst_exif0 is not None:
                    dst_exif_meta = _exiftool_dump_json(exiftool, dst_exif0, debug_dir / f"exif_dst_exif-{suffix}.json", logger)

                if src_meta is not None and dst_fit_meta is not None:
                    diff = _diff_meta_dict(src_meta, dst_fit_meta)
                    (debug_dir / f"exif_diff-{suffix}.json").write_text(
                        json.dumps(diff, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                        errors="ignore",
                    )
        else:
            logger.log("WARN", "exiftool not found; exif_audit/exif_dump/exif_diff skipped.")

    else:
        logger.log("INFO", "[APPLY] Skipped (stage=fit).")

    # summary
    summary = {
        "version": 10,
        "args": {
            "rgb_dir": str(rgb_dir),
            "th_dir": str(th_dir),
            "out_dir": str(out_dir),
            "samples": args.samples,
            "comparison": bool(want_comp),
            "align": args.align,
            "stage": args.stage,
        },
        "flags": {
            "want_fit": bool(want_fit),
            "want_exif": bool(want_exif),
            "do_fit": bool(do_fit),
            "do_apply": bool(do_apply),
            "exif_usable": bool(exif_usable),
        },
        "paths": {
            "out_dir": str(out_dir),
            "debug_dir": str(debug_dir),
            "model_fit": str(model_fit_path) if (need_fit_model and model_fit_path.exists()) else None,
            "model_exif": str(model_exif_path) if model_exif_path else None,
            "image_fit_dir": str(img_fit_dir) if (do_apply and want_fit) else None,
            "image_exif_dir": str(img_exif_dir) if (do_apply and want_exif and exif_usable) else None,
            "comparison_dir": str(comp_dir) if (do_apply and want_comp) else None,
            "run_log": str(debug_dir / f"run-{suffix}.log"),
            "per_image_jsonl": str(per_image_path) if do_apply else None,
            "exif_audit_jsonl": str(exif_audit_path) if do_apply else None,
        },
        "counts": {
            "pairs_found": int(pairs_found),
            "pairs_processed": int(len(pairs)),
            "fit_written": int(fit_written),
            "exif_written": int(exif_written),
        },
        "timing": {"fit_sec": round(float(t_fit), 3), "apply_sec": round(float(t_apply), 3)},
        "exif_probe": {"usable": bool(exif_usable), "probe_ok": int(exif_ok_count), "probe_n": int(probe_n)},
        "metadata_policy": {
            "read_pixels": "apply EXIF orientation to pixels via ImageOps.exif_transpose",
            "write_orientation": "force Orientation=1 (IFD0 + XMP-tiff) using 1-pass exiftool after raw XMP injection",
            "sizes": "write IFD0:ImageWidth/Height and ExifIFD:ExifImageWidth/Height only (no PixelXDimension/YDimension)",
            "digital_zoom": "DigitalZoomRatio *= zoom_factor",
            "f35mm": "FocalLengthIn35mmFormat *= zoom_factor (correct ExifTool tag name: Format)",
            "xmp": "copy -xmp:all and inject raw XMP packet from source JPEG (APP1) via exiftool -XMP<= (best effort for DJI namespaces)",
            "strip": "remove MPF/IFD1/Preview/Thumbnail blocks after crop/resize",
            "audit": "write debug/exif_audit-<suffix>.jsonl for orientation/size correctness + key GPS/XMP fields",
        }
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.log("INFO", "[SUMMARY] ------------------------------")
    logger.log("INFO", f"[SUMMARY] pairs_found={pairs_found} pairs_processed={len(pairs)} fit_written={fit_written} exif_written={exif_written}")
    if need_fit_model and model_fit_path.exists():
        logger.log("INFO", f"[SUMMARY] model_fit={model_fit_path}")
    if model_exif_path:
        logger.log("INFO", f"[SUMMARY] model_exif={model_exif_path}")
    logger.log("INFO", f"[SUMMARY] summary={summary_path}")
    logger.log("INFO", "RUN END")
    logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
