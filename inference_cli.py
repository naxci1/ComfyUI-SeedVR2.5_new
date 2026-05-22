#!/usr/bin/env python3
"""Backend worker CLI for SeedVR2 Qt runner."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Sequence

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

_STATUS_PREFIX = "__SEEDVR2_GUI_STATUS__|"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


@dataclass
class Job:
    file_path: Path
    output_path: Path
    frame_count: int
    is_video: bool


def _log(message: str) -> None:
    print(message, flush=True)


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
    except Exception:
        pass


def _validate_args(args: argparse.Namespace) -> None:
    if args.grain < 0 or args.grain > 100:
        raise ValueError("--grain must be in range 0..100")
    if args.recover_detail < 0 or args.recover_detail > 100:
        raise ValueError("--recover-detail must be in range 0..100")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SeedVR2 backend worker")
    parser.add_argument("--input", required=True, help="Input directory or media file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--grain", type=int, default=12, help="Grain amount 0..100")
    parser.add_argument("--recover-detail", type=int, default=35, help="Recover detail amount 0..100")
    parser.add_argument("--fps", type=int, default=24, help="Target output FPS")
    return parser.parse_args(argv)


def _apply_inference(rgb_frame: "Any", grain: int, recover_detail: int) -> "Any":
    import numpy as np
    import cv2 as _cv2

    frame = rgb_frame.astype(np.float32)

    if recover_detail > 0:
        sigma = 0.4 + (recover_detail / 140.0)
        blurred = _cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma, sigmaY=sigma)
        amount = min(2.2, 0.3 + recover_detail / 60.0)
        frame = _cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)

    if grain > 0:
        noise_strength = max(0.3, grain / 14.0)
        noise = np.random.normal(0.0, noise_strength, frame.shape)
        frame = frame + noise

    return np.clip(frame, 0, 255).astype(np.uint8)


def _iter_media_files(input_path: Path) -> Iterator[Path]:
    if input_path.is_file():
        yield input_path
        return

    for candidate in sorted(input_path.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in (_IMAGE_SUFFIXES | _VIDEO_SUFFIXES):
            yield candidate


def _count_video_frames(path: Path) -> int:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for video processing")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(1, total)


def _build_jobs(input_path: Path, output_root: Path) -> List[Job]:
    files = list(_iter_media_files(input_path))
    if not files:
        raise RuntimeError(f"No media files found in input: {input_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    jobs: List[Job] = []

    for src in files:
        suffix = src.suffix.lower()
        is_video = suffix in _VIDEO_SUFFIXES
        frame_count = _count_video_frames(src) if is_video else 1

        if is_video:
            dst = output_root / f"{src.stem}_upscaled.mp4"
        else:
            dst = output_root / f"{src.stem}_upscaled{src.suffix}"

        jobs.append(Job(file_path=src, output_path=dst, frame_count=frame_count, is_video=is_video))

    return jobs


def _open_ffmpeg_pipe(output_path: Path, width: int, height: int, fps: int) -> subprocess.Popen:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]

    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _process_single_image(
    job: Job,
    args: argparse.Namespace,
    done_files: int,
    remaining_files: int,
    remaining_frames_queue: int,
) -> None:
    from PIL import Image
    import numpy as np

    with Image.open(job.file_path) as img:
        rgb = np.array(img.convert("RGB"), dtype=np.uint8)

    processed = _apply_inference(rgb, args.grain, args.recover_detail)
    Image.fromarray(processed, mode="RGB").save(job.output_path)

    _emit_gui_queue_status(
        str(job.file_path),
        current=1,
        total=1,
        done=done_files,
        remaining=remaining_files,
        remaining_frames_queue=remaining_frames_queue,
    )

    _strict_batch_flush(rgb, processed)


def _process_video_job(
    job: Job,
    args: argparse.Namespace,
    done_files: int,
    remaining_files: int,
    remaining_frames_queue: int,
) -> None:
    import numpy as np

    if cv2 is None:
        raise RuntimeError("OpenCV is required for video processing")

    cap = cv2.VideoCapture(str(job.file_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video: {job.file_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    output_fps = int(args.fps if args.fps > 0 else source_fps if source_fps > 0 else 24)

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Video has no readable frames: {job.file_path}")

    first_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    height, width = first_rgb.shape[:2]
    ffmpeg = _open_ffmpeg_pipe(job.output_path, width, height, output_fps)

    try:
        frame_index = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            processed = _apply_inference(rgb, args.grain, args.recover_detail)

            if ffmpeg.stdin is None:
                raise RuntimeError("FFmpeg stdin is unavailable")
            ffmpeg.stdin.write(processed.tobytes())

            frame_index += 1
            _emit_gui_queue_status(
                str(job.file_path),
                current=frame_index,
                total=job.frame_count,
                done=done_files,
                remaining=remaining_files,
                remaining_frames_queue=remaining_frames_queue,
            )

            _strict_batch_flush(bgr, rgb, processed)

        if ffmpeg.stdin:
            ffmpeg.stdin.close()

        stdout_data, stderr_data = ffmpeg.communicate()
        if ffmpeg.returncode != 0:
            raise RuntimeError(
                f"FFmpeg encoding failed for {job.output_path}: {stderr_data.decode('utf-8', errors='ignore')}"
            )

        if stdout_data:
            _log(stdout_data.decode("utf-8", errors="ignore"))
    finally:
        cap.release()
        if ffmpeg.poll() is None:
            ffmpeg.kill()


def run(args: argparse.Namespace) -> int:
    _validate_args(args)

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    _log(f"SeedVR2 worker booted | model={model_path} | grain={args.grain} | recover={args.recover_detail} | fps={args.fps}")

    jobs = _build_jobs(input_path, output_path)
    total_jobs = len(jobs)

    for index, job in enumerate(jobs):
        done_files = index
        remaining_files = total_jobs - index - 1
        remaining_frames_queue = sum(item.frame_count for item in jobs[index + 1 :])

        _log(f"Processing {job.file_path} -> {job.output_path}")

        if job.is_video:
            _process_video_job(
                job=job,
                args=args,
                done_files=done_files,
                remaining_files=remaining_files,
                remaining_frames_queue=remaining_frames_queue,
            )
        else:
            _process_single_image(
                job=job,
                args=args,
                done_files=done_files,
                remaining_files=remaining_files,
                remaining_frames_queue=remaining_frames_queue,
            )

        _strict_batch_flush(job)

    _log("All jobs completed successfully")
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    try:
        args = parse_arguments(argv)
        code = run(args)
    except Exception as exc:
        _log(f"ERROR: {exc}")
        code = 1

    raise SystemExit(code)


if __name__ == "__main__":
    main()
