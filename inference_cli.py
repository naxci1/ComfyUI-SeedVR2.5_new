#!/usr/bin/env python3
"""SeedVR2 inference CLI rewritten for Qt runner integration."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

_STATUS_PREFIX = "__SEEDVR2_GUI_STATUS__|"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@dataclass
class QueueItem:
    path: Path
    frame_count: int


def _print(msg: str) -> None:
    print(msg, flush=True)


def _emit_gui_queue_status(
    file_path: str,
    current: int,
    total: int,
    done: int,
    remaining: int,
    remaining_frames_queue: int,
) -> None:
    payload = {
        "file_path": file_path,
        "current": int(current),
        "total": int(total),
        "done": int(done),
        "remaining": int(remaining),
        "remaining_frames_queue": int(max(0, remaining_frames_queue)),
    }
    print(f"{_STATUS_PREFIX}{json.dumps(payload, ensure_ascii=False)}", flush=True)


def _strict_batch_flush(*objs: Any) -> None:
    for obj in objs:
        try:
            del obj
        except Exception:
            pass

    gc.collect()

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def _parse_resolution(raw: str) -> Tuple[int, int]:
    value = raw.strip().lower()
    aliases = {
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "1440p": (2560, 1440),
        "4k": (3840, 2160),
    }
    if value in aliases:
        return aliases[value]

    if "x" not in value:
        raise ValueError(f"Unsupported resolution format: {raw}")

    w, h = value.split("x", 1)
    width, height = int(w), int(h)
    if width < 16 or height < 16:
        raise ValueError("Resolution too small.")
    return width, height


def _resample_mode(method: str) -> int:
    from PIL import Image

    mapping = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    return mapping[method]


def _apply_image_pipeline(
    image: "Image.Image",
    target_size: Tuple[int, int],
    resize_method: str,
    recover_detail: int,
    grain: int,
) -> "Image.Image":
    from PIL import ImageFilter
    import numpy as np

    processed = image.resize(target_size, _resample_mode(resize_method))

    if recover_detail > 0:
        processed = processed.filter(
            ImageFilter.UnsharpMask(
                radius=1.6 + recover_detail / 85.0,
                percent=min(450, 50 + recover_detail * 4),
                threshold=2,
            )
        )

    if grain > 0:
        arr = np.asarray(processed).astype(np.float32)
        sigma = max(0.2, grain / 14.0)
        noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        from PIL import Image

        processed = Image.fromarray(arr, mode=processed.mode)

    return processed


def _collect_queue(input_path: Path) -> List[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        return [input_path]

    entries = sorted(p for p in input_path.rglob("*") if p.is_file())
    return [
        p
        for p in entries
        if p.suffix.lower() in _IMAGE_SUFFIXES or p.suffix.lower() in _VIDEO_SUFFIXES
    ]


def _count_frames(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        return 1

    if suffix in _VIDEO_SUFFIXES:
        if cv2 is None:
            raise RuntimeError("OpenCV is required for video processing.")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open video: {path}")

        value = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return max(1, value)

    raise ValueError(f"Unsupported file type: {path}")


def _build_queue_items(paths: Sequence[Path]) -> List[QueueItem]:
    items: List[QueueItem] = []
    for p in paths:
        try:
            items.append(QueueItem(path=p, frame_count=_count_frames(p)))
        except Exception as exc:
            _print(f"Skipping {p}: {exc}")
    return items


def _resolve_output_path(input_file: Path, output_root: Path, multiple_inputs: bool) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)

    if multiple_inputs:
        name = input_file.stem + (".mp4" if input_file.suffix.lower() in _VIDEO_SUFFIXES else input_file.suffix)
        return output_root / name

    if output_root.suffix:
        return output_root

    return output_root / (input_file.stem + (".mp4" if input_file.suffix.lower() in _VIDEO_SUFFIXES else input_file.suffix))


def _process_image(
    item: QueueItem,
    output_path: Path,
    target_size: Tuple[int, int],
    resize_method: str,
    recover_detail: int,
    grain: int,
    done_files: int,
    remaining_files: int,
    remaining_frames_queue: int,
    flush_interval: int,
) -> None:
    from PIL import Image

    with Image.open(item.path) as src:
        image = src.convert("RGB")
        rendered = _apply_image_pipeline(image, target_size, resize_method, recover_detail, grain)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(output_path)

    _emit_gui_queue_status(str(item.path), 1, 1, done_files, remaining_files, remaining_frames_queue)
    _strict_batch_flush(image, rendered)
    if flush_interval <= 1:
        _strict_batch_flush()


def _process_video(
    item: QueueItem,
    output_path: Path,
    target_size: Tuple[int, int],
    resize_method: str,
    recover_detail: int,
    grain: int,
    done_files: int,
    remaining_files: int,
    remaining_frames_queue: int,
    flush_interval: int,
) -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for video processing.")
    import numpy as np
    from PIL import Image

    cap = cv2.VideoCapture(str(item.path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {item.path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, target_size)
    if not writer.isOpened():
        cap.release()
        writer.release()
        raise RuntimeError(f"Cannot write video: {output_path}")

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb, mode="RGB")
        rendered = _apply_image_pipeline(image, target_size, resize_method, recover_detail, grain)
        out_rgb = np.asarray(rendered, dtype=np.uint8)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        writer.write(out_bgr)

        frame_index += 1
        frames_left_after_current = max(0, item.frame_count - frame_index)
        _emit_gui_queue_status(
            str(item.path),
            frame_index,
            item.frame_count,
            done_files,
            remaining_files,
            remaining_frames_queue + frames_left_after_current,
        )

        if frame_index % flush_interval == 0:
            _strict_batch_flush(frame, rgb, image, rendered, out_rgb, out_bgr)

    cap.release()
    writer.release()

    if frame_index < item.frame_count:
        _emit_gui_queue_status(
            str(item.path),
            frame_index,
            item.frame_count,
            done_files,
            remaining_files,
            remaining_frames_queue,
        )

    _strict_batch_flush(cap, writer)


def _validate_args(args: argparse.Namespace) -> None:
    if args.recover_detail < 0 or args.recover_detail > 100:
        raise ValueError("--recover-detail must be 0..100")
    if args.grain < 0 or args.grain > 100:
        raise ValueError("--grain must be 0..100")
    if args.batch_flush_interval < 1:
        raise ValueError("--batch-flush-interval must be >= 1")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SeedVR2 runner CLI")
    parser.add_argument("--input", required=True, help="Input media file or directory")
    parser.add_argument("--output", required=True, help="Output directory or file path")
    parser.add_argument("--output-resolution", default="1920x1080", help="Target output resolution")
    parser.add_argument("--resize-method", default="lanczos", choices=["nearest", "bilinear", "bicubic", "lanczos"])
    parser.add_argument("--ai-model", default="seedvr2-pro", help="AI model preset label")
    parser.add_argument("--recover-detail", type=int, default=35, help="Recover detail strength 0..100")
    parser.add_argument("--grain", type=int, default=8, help="Film grain intensity 0..100")
    parser.add_argument("--batch-flush-interval", type=int, default=1, help="Frame interval for strict flush")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    _validate_args(args)

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    target_size = _parse_resolution(args.output_resolution)

    queue_paths = _collect_queue(input_path)
    if not queue_paths:
        raise RuntimeError("No supported media files found for processing.")

    queue_items = _build_queue_items(queue_paths)
    if not queue_items:
        raise RuntimeError("No processable files in queue.")

    _print(f"Model={args.ai_model} | Resize={args.resize_method} | Resolution={target_size[0]}x{target_size[1]} | RecoverDetail={args.recover_detail} | Grain={args.grain}")

    total_files = len(queue_items)
    for index, item in enumerate(queue_items):
        done_files = index
        remaining_files = total_files - done_files - 1
        remaining_frames_queue = sum(q.frame_count for q in queue_items[index + 1 :])

        out = _resolve_output_path(item.path, output_path, total_files > 1 or output_path.is_dir())
        _print(f"Processing {item.path} -> {out}")

        suffix = item.path.suffix.lower()
        if suffix in _IMAGE_SUFFIXES:
            _process_image(
                item,
                out,
                target_size,
                args.resize_method,
                args.recover_detail,
                args.grain,
                done_files,
                remaining_files,
                remaining_frames_queue,
                args.batch_flush_interval,
            )
        elif suffix in _VIDEO_SUFFIXES:
            _process_video(
                item,
                out,
                target_size,
                args.resize_method,
                args.recover_detail,
                args.grain,
                done_files,
                remaining_files,
                remaining_frames_queue,
                args.batch_flush_interval,
            )
        else:
            raise ValueError(f"Unsupported media type: {item.path}")

        _strict_batch_flush(item)

    _print("All files processed successfully.")
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    try:
        args = parse_arguments(argv)
        exit_code = run(args)
    except Exception as exc:
        _print(f"ERROR: {exc}")
        exit_code = 1

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
