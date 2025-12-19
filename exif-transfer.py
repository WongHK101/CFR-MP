#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copy EXIF/XMP/ICC metadata from images in A (src_dir) to same-named images in B (dst_dir).

Default behavior:
- Match by basename (e.g., "PV (24).JPG") between A and B.
- Copy metadata "as-is" (no recomputation): -TagsFromFile SRC -all:all -unsafe -icc_profile:all
- Overwrite destination in-place (no *_original backups). Use --keep_backup if you want backups.

Requires: exiftool (https://exiftool.org/) available on PATH, or pass --exiftool path/to/exiftool.exe
"""

from __future__ import annotations

import argparse
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional


IMG_EXTS_DEFAULT = ["jpg", "jpeg", "tif", "tiff", "png", "heic", "dng"]


def iter_images(root: Path, recursive: bool, exts: List[str]) -> List[Path]:
    exts_lc = {e.lower().lstrip(".") for e in exts}
    if recursive:
        files = [p for p in root.rglob("*") if p.is_file()]
    else:
        files = [p for p in root.iterdir() if p.is_file()]
    out = []
    for p in files:
        suf = p.suffix.lower().lstrip(".")
        if suf in exts_lc:
            out.append(p)
    out.sort()
    return out


def build_src_index(
    src_dir: Path,
    recursive: bool,
    exts: List[str],
    match: str,
) -> Tuple[Dict[str, Path], Dict[str, List[Path]]]:
    """
    Returns:
      - src_map: key -> unique src path
      - dup_map: key -> list of src paths if duplicates exist (len>1)
    """
    src_files = iter_images(src_dir, recursive=recursive, exts=exts)
    buckets: Dict[str, List[Path]] = {}
    for p in src_files:
        if match == "basename":
            key = p.name
        elif match == "relative":
            key = str(p.relative_to(src_dir)).replace("\\", "/")
        else:
            raise ValueError(match)
        buckets.setdefault(key, []).append(p)

    src_map: Dict[str, Path] = {}
    dup_map: Dict[str, List[Path]] = {}
    for k, lst in buckets.items():
        if len(lst) == 1:
            src_map[k] = lst[0]
        else:
            dup_map[k] = lst
    return src_map, dup_map


def chunk_pairs(pairs: List[Tuple[Path, Path]], max_pairs: int = 120) -> List[List[Tuple[Path, Path]]]:
    """Chunk pairs to avoid Windows command-line length limits."""
    if max_pairs <= 0:
        return [pairs]
    return [pairs[i:i + max_pairs] for i in range(0, len(pairs), max_pairs)]


def run_exiftool_batch(
    exiftool: str,
    pairs: List[Tuple[Path, Path]],
    overwrite: bool,
    log_file: Optional[Path],
    dry_run: bool,
) -> int:
    """
    Build one exiftool command that applies multiple (src->dst) copy operations.
    Returns number of failed pairs (best-effort; if command fails, returns len(pairs)).
    """
    if not pairs:
        return 0

    cmd: List[str] = [exiftool, "-charset", "filename=utf8", "-m", "-P", "-unsafe"]
    if overwrite:
        cmd.append("-overwrite_original")
    cmd += ["-q", "-q"]  # quieter

    for src, dst in pairs:
        cmd += ["-TagsFromFile", str(src), "-all:all", "-icc_profile:all", str(dst)]

    if dry_run:
        print("[DRY RUN] Would run exiftool with", len(pairs), "pairs")
        return 0

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError:
        print(f"[ERROR] exiftool not found: {exiftool}", file=sys.stderr)
        return len(pairs)

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8", errors="replace") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"EXIFTOOL CMD (pairs={len(pairs)}):\n")
            f.write(" ".join(cmd) + "\n")
            if out:
                f.write("\n[STDOUT]\n" + out + "\n")
            if err:
                f.write("\n[STDERR]\n" + err + "\n")

    if proc.returncode != 0:
        print(f"[ERROR] exiftool failed with code {proc.returncode}.", file=sys.stderr)
        if err:
            print(err, file=sys.stderr)
        return len(pairs)

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Copy EXIF/XMP metadata from folder A to same-named images in folder B.")
    ap.add_argument("--src_dir", required=True, help="Folder A (source metadata images)")
    ap.add_argument("--dst_dir", required=True, help="Folder B (destination images to receive metadata)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders for both src/dst")
    ap.add_argument("--ext", default=",".join(IMG_EXTS_DEFAULT),
                    help="Comma-separated extensions to include (default: common photo types)")
    ap.add_argument("--match", choices=["basename", "relative"], default="basename",
                    help="How to match files between A and B (default: basename)")
    ap.add_argument("--exiftool", default="exiftool", help="Path to exiftool (default: exiftool on PATH)")
    ap.add_argument("--keep_backup", action="store_true",
                    help="Keep exiftool *_original backups (default: overwrite in-place)")
    ap.add_argument("--chunk", type=int, default=120,
                    help="Pairs per exiftool invocation (avoid Windows cmd limit). Default 120.")
    ap.add_argument("--log", default="", help="Optional log file path (append).")
    ap.add_argument("--dry_run", action="store_true", help="Don't write anything; just report matches.")
    args = ap.parse_args()

    src_dir = Path(args.src_dir).expanduser().resolve()
    dst_dir = Path(args.dst_dir).expanduser().resolve()

    if not src_dir.exists():
        print(f"[ERROR] src_dir not found: {src_dir}", file=sys.stderr)
        return 2
    if not dst_dir.exists():
        print(f"[ERROR] dst_dir not found: {dst_dir}", file=sys.stderr)
        return 2

    exts = [e.strip() for e in args.ext.split(",") if e.strip()]
    overwrite = not args.keep_backup
    log_file = Path(args.log).expanduser().resolve() if args.log else None

    src_map, dup_map = build_src_index(src_dir, recursive=args.recursive, exts=exts, match=args.match)

    if dup_map:
        print(f"[WARN] Found {len(dup_map)} duplicate keys in src_dir under match={args.match}.")
        if args.match == "basename":
            print("[WARN] Ambiguous basenames will be skipped. Consider using --match relative.", file=sys.stderr)

    dst_files = iter_images(dst_dir, recursive=args.recursive, exts=exts)
    pairs: List[Tuple[Path, Path]] = []
    missing_src: List[Path] = []
    skipped_ambiguous: List[Path] = []

    for dst in dst_files:
        key = dst.name if args.match == "basename" else str(dst.relative_to(dst_dir)).replace("\\", "/")
        if key in dup_map:
            skipped_ambiguous.append(dst)
            continue
        src = src_map.get(key)
        if src is None:
            missing_src.append(dst)
            continue
        pairs.append((src, dst))

    print(f"[INFO] dst_files={len(dst_files)} matched_pairs={len(pairs)} missing_src={len(missing_src)} "
          f"skipped_ambiguous={len(skipped_ambiguous)}")

    if missing_src:
        print("[WARN] Examples of dst without src match:")
        for p in missing_src[:10]:
            print("  -", p)

    if skipped_ambiguous:
        print("[WARN] Examples of dst skipped due to ambiguous src match:")
        for p in skipped_ambiguous[:10]:
            print("  -", p)

    total_failed = 0
    chunks = chunk_pairs(pairs, max_pairs=args.chunk)
    for idx, ch in enumerate(chunks, start=1):
        print(f"[INFO] exiftool batch {idx}/{len(chunks)}: {len(ch)} files")
        total_failed += run_exiftool_batch(
            exiftool=args.exiftool,
            pairs=ch,
            overwrite=overwrite,
            log_file=log_file,
            dry_run=args.dry_run,
        )

    if total_failed == 0:
        print("[OK] Metadata copy completed.")
        return 0

    print(f"[WARN] Completed with failures affecting up to {total_failed} images. Check log.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
