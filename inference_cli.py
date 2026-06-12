#!/usr/bin/env python3
"""
SeedVR2 Video Upscaler - Standalone CLI Interface

Command-line interface for high-quality upscaling using SeedVR2 diffusion models.
Supports single and multi-GPU processing with advanced memory optimization.

Key Features:
    • Multi-GPU Processing: Automatic workload distribution across multiple GPUs with
      temporal overlap blending for seamless transitions
    • Streaming Mode: Memory-efficient processing of long videos in chunks, avoiding
      full video loading into RAM while maintaining temporal consistency
    • Memory Optimization: BlockSwap for limited VRAM, VAE tiling for large resolutions,
      intelligent tensor offloading between processing phases
    • Performance: Torch.compile integration, BFloat16 compute pipeline,
      efficient model caching for batch and streaming processing
    • Flexibility: Multiple output formats (MP4/PNG), advanced color correction methods,
      directory batch processing with auto-format detection
    • Quality Control: Temporal overlap blending, frame prepending for artifact reduction,
      configurable noise scales for detail preservation

Architecture:
    The CLI implements a 4-phase processing pipeline:
    1. Encode: VAE encoding with optional input noise and tiling
    2. Upscale: DiT transformer upscaling with latent space diffusion
    3. Decode: VAE decoding with optional tiling
    4. Postprocess: Color correction and temporal blending

Usage:
    python inference_cli.py video.mp4 --resolution 1080
    For complete usage examples, run: python inference_cli.py --help

Requirements:
    • Python 3.10+
    • PyTorch 2.4+ with CUDA 12.1+ (NVIDIA) or MPS (Apple Silicon)
    • 16GB+ VRAM recommended (8GB minimum with BlockSwap)
    • OpenCV, NumPy for video I/O

Model Support:
    • 3B models: seedvr2_ema_3b_fp16.safetensors (default), _fp8_e4m3fn/GGUF variants
    • 7B models: seedvr2_ema_7b_fp16.safetensors, _fp8_e4m3fn/GGUF variants
    • VAE: ema_vae_fp16.safetensors (shared across all models)
    • Auto-downloads from HuggingFace on first run with SHA256 validation
"""

# Standard library imports
import sys
import os
import gc
import json
import argparse
import time
import shlex
import platform
import threading
import multiprocessing as mp
from collections import deque
from typing import Dict, Any, List, Optional, Tuple, Literal, Generator
from datetime import datetime
from pathlib import Path

# Set up path before any other imports to fix module resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Set environment variable so all spawned processes can find modules
os.environ['PYTHONPATH'] = script_dir + ':' + os.environ.get('PYTHONPATH', '')

# Ensure safe CUDA usage with multiprocessing
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)

# ── Blackwell / PyTorch performance environment variables ────────────────────
# Must be injected BEFORE 'import torch' so the CUDA runtime picks them up.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "backend:cudaMallocAsync")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("TORCH_CUDNN_V8_API_ENABLED", "1")
os.environ.setdefault("CUDA_CACHE_MAXSIZE", "4294967296")

# Configure platform-specific memory management before heavy imports
# Must be set BEFORE import torch
if platform.system() == "Darwin":
    # MPS allocator requires: low_watermark <= high_watermark
    # Setting both to 0.0 disables PyTorch memory limits, letting macOS manage memory
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.0")
else:
    # expandable_segments lets the CUDA allocator grow/shrink segment sizes on demand,
    # preventing the VRAM fragmentation that causes OOM during Phase 3 VAE decoding
    # (10.87 GiB allocated, 1.56 GiB reserved-but-unallocated, 1.32 GiB contiguous block missing).
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Pre-parse arguments that must be handled before torch import
    _pre_parser = argparse.ArgumentParser(add_help=False)
    _pre_parser.add_argument("--cuda_device", type=str, default=None)
    _pre_args, _ = _pre_parser.parse_known_args()
    
    if _pre_args.cuda_device is not None:
        device_list_env = [x.strip() for x in _pre_args.cuda_device.split(',') if x.strip()!='']
        
        # Skip validation if CUDA_VISIBLE_DEVICES is already set (worker process)
        if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
            # Temporary torch import for CUDA device validation only
            # Must happen before setting CUDA_VISIBLE_DEVICES and before main torch import
            import torch as _torch_check
            if _torch_check.cuda.is_available():
                available_count = _torch_check.cuda.device_count()
                invalid_devices = [d for d in device_list_env if not d.isdigit() or int(d) >= available_count]
                if invalid_devices:
                    print(f"❌ [ERROR] Invalid CUDA device ID(s): {', '.join(invalid_devices)}. "
                        f"Available devices: 0-{available_count-1} (total: {available_count})")
                    sys.exit(1)
            else:
                print("❌ [ERROR] CUDA is not available on this system. Cannot use --cuda_device argument.")
                sys.exit(1)
            
            # Set CUDA_VISIBLE_DEVICES for single GPU after validation
            if len(device_list_env) == 1:
                os.environ["CUDA_VISIBLE_DEVICES"] = device_list_env[0]

# Heavy dependency imports after environment configuration
import torch
torch._inductor.config.triton.cudagraphs = False
torch._dynamo.config.suppress_errors = True
import cv2
import numpy as np
import subprocess
import shutil

# Project imports
from src.utils.downloads import download_weight
from src.utils.model_registry import get_available_dit_models, DEFAULT_DIT, DEFAULT_VAE
from src.utils.constants import SEEDVR2_FOLDER_NAME
from src.core.generation_utils import (
    setup_generation_context, 
    prepare_runner, 
    compute_generation_info, 
    log_generation_start,
    blend_overlapping_frames,
    load_text_embeddings,
    script_directory
)
from src.core.generation_phases import (
    encode_all_batches, 
    upscale_all_batches, 
    decode_all_batches, 
    postprocess_all_batches
)
from src.utils.debug import Debug
from src.optimization.memory_manager import clear_memory, get_gpu_backend, is_cuda_available
debug = Debug(enabled=False)  # Will be enabled via --debug CLI flag


# =============================================================================
# FFMPEG Class
# =============================================================================

class FFMPEGVideoWriter:
    """
    Video writer using ffmpeg subprocess for encoding with 10-bit support.
    
    Provides cv2.VideoWriter-compatible interface (write, isOpened, release) while
    using ffmpeg for encoding. Defaults to GPU-accelerated NVENC HEVC encoding.
    
    Args:
        path: Output video file path
        width: Frame width in pixels
        height: Frame height in pixels
        fps: Frames per second
        use_10bit: If True, uses yuv420p10le pixel format when default args are used.
                   If False, uses yuv420p (default: False)
        custom_video_args: Optional list of ffmpeg video encoding args to use instead of
                           the default H264/H265 codec selection. When provided, ``use_10bit``
                           is ignored for codec/pix_fmt selection (but you may still rely on it
                           elsewhere). Example: ["-c:v", "prores_ks", "-profile:v", "3",
                           "-pix_fmt", "yuv422p10le"]
    
    Raises:
        RuntimeError: If ffmpeg is not found in system PATH
    
    Note:
        Frames must be passed to write() in BGR format (same as cv2.VideoWriter).
        Internally converts to RGB for ffmpeg rawvideo input.
    """

    @staticmethod
    def _map_codec_to_nvenc(codec: str) -> str:
        lowered = (codec or "").strip().lower()
        if "libx264" in lowered or "h264" in lowered:
            return "h264_nvenc"
        if "libx265" in lowered or "h265" in lowered or "hevc" in lowered:
            return "hevc_nvenc"
        return codec

    @staticmethod
    def _extract_flag_value(args: List[str], *flags: str) -> Optional[str]:
        for idx, token in enumerate(args):
            if token in flags and idx + 1 < len(args):
                return args[idx + 1]
        return None

    @staticmethod
    def _strip_flag_with_value(args: List[str], *flags: str) -> List[str]:
        stripped: List[str] = []
        skip_next = False
        for idx, token in enumerate(args):
            if skip_next:
                skip_next = False
                continue
            if token in flags:
                if idx + 1 < len(args):
                    skip_next = True
                continue
            stripped.append(token)
        return stripped

    @classmethod
    def _normalize_video_encoder_args(cls, raw_args: List[str]) -> List[str]:
        args = list(raw_args)

        codec_idx = None
        codec_val = None
        for idx, token in enumerate(args):
            if token == "-c:v" and idx + 1 < len(args):
                codec_idx = idx + 1
                codec_val = args[idx + 1]
                break

        if codec_val is not None and codec_idx is not None:
            mapped_codec = cls._map_codec_to_nvenc(codec_val)
            args[codec_idx] = mapped_codec

            if "nvenc" in mapped_codec.lower():
                crf_val = cls._extract_flag_value(args, "-crf", "-crf:v")
                args = cls._strip_flag_with_value(args, "-crf", "-crf:v")
                args = cls._strip_flag_with_value(args, "-rc", "-cq", "-qp", "-b:v")
                if cls._extract_flag_value(args, "-spatial-aq") is None:
                    args += ["-spatial-aq", "1"]
                if crf_val is not None:
                    args += ["-cq", str(crf_val)]

                # CPU-only presets are invalid for NVENC; map them to the closest NVENC preset.
                # veryslow/slower → p7 (highest quality), veryfast/superfast/ultrafast → p1 (fastest).
                _CPU_TO_NVENC_PRESET = {
                    "veryslow": "p7",
                    "very slow": "p7",
                    "slower": "p7",
                    "veryfast": "p1",
                    "very fast": "p1",
                    "superfast": "p1",
                    "ultrafast": "p1",
                }
                preset_val = cls._extract_flag_value(args, "-preset")
                if preset_val is not None:
                    nvenc_preset = _CPU_TO_NVENC_PRESET.get(preset_val.lower())
                    if nvenc_preset is not None:
                        args = cls._strip_flag_with_value(args, "-preset")
                        args += ["-preset", nvenc_preset]

        return args
    
    def __init__(self, path: str, width: int, height: int, fps: float, use_10bit: bool = False,
                 custom_video_args: Optional[List[str]] = None,
                 lut_path: Optional[str] = None):
        if custom_video_args:
            video_enc_args = custom_video_args
        else:
            pix_fmt = 'yuv420p10le' if use_10bit else 'yuv420p'
            # Dynamic GPU default path (resolution/fps come from width/height/fps args above).
            bitrate = os.environ.get("SEEDVR2_DEFAULT_BITRATE", "8M")
            video_enc_args = ['-c:v', 'hevc_nvenc', '-preset', 'p7', '-b:v', bitrate, '-pix_fmt', pix_fmt]
        video_enc_args = self._normalize_video_encoder_args(video_enc_args)
        
        filter_args: List[str] = []
        if lut_path:
            filter_args = ['-vf', f'lut3d={lut_path}']

        ffmpeg_cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
            *filter_args, *video_enc_args, path
        ]
        if debug.enabled:
            debug.log(
                f"FFmpeg command: {' '.join(shlex.quote(str(part)) for part in ffmpeg_cmd)}",
                category="file",
                force=True,
            )

        # Persistent encoder pipe. We keep ffmpeg's stderr connected to a PIPE
        # (instead of discarding it) and drain it on a background daemon thread
        # into a bounded ring buffer. This is what makes long exports debuggable:
        # when the pipe breaks after minutes of streaming, the OS only tells us
        # "broken pipe" — the *reason* (codec rejection, disk full, bad LUT) is
        # only ever written by ffmpeg to stderr. Buffering the tail lets us report
        # the real cause instead of an opaque error.
        self._stderr_tail: "deque[str]" = deque(maxlen=50)
        self.proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name="ffmpeg-stderr-drain", daemon=True
        )
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        """Continuously read ffmpeg stderr so the pipe never fills and blocks the
        encoder, retaining only the most recent lines for diagnostics."""
        stream = self.proc.stderr if self.proc else None
        if stream is None:
            return
        try:
            for raw in iter(stream.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line:
                    self._stderr_tail.append(line)
        except (ValueError, OSError):
            pass  # Stream closed during shutdown.

    def _stderr_message(self) -> str:
        """Return the captured ffmpeg stderr tail, if any, for error reporting."""
        if not self._stderr_tail:
            return ""
        return " | ffmpeg said: " + " / ".join(list(self._stderr_tail)[-5:])

    def write(self, frame_bgr: np.ndarray):
        if not self.isOpened():
            raise RuntimeError(
                "FFMPEGVideoWriter: ffmpeg process is not running." + self._stderr_message()
            )

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            self.proc.stdin.write(frame_rgb.astype(np.uint8).tobytes())
            self.proc.stdin.flush()  # Critical: prevent buffering issues
        except BrokenPipeError:
            raise RuntimeError(
                "FFMPEGVideoWriter: ffmpeg process terminated unexpectedly. "
                "Check video path, codec support, and disk space." + self._stderr_message()
            )

    def isOpened(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def release(self):
        if self.proc:
            try:
                self.proc.stdin.close()
            except Exception:
                pass  # Ignore errors on close

            self.proc.wait()

            # Let the drain thread flush any final stderr lines before we read them.
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=2.0)

            if self.proc.returncode != 0:
                debug.log(
                    f"ffmpeg exited with code {self.proc.returncode}. "
                    "Check output file for corruption." + self._stderr_message(),
                    level="WARNING", force=True, category="file"
                )
            self.proc = None


# =============================================================================
# Device Management Helpers
# =============================================================================

def _device_id_to_name(device_id: str, platform_type: str = None) -> str:
    """
    Convert device ID to full device name.
    
    Args:
        device_id: Device ID ("0", "1") or special value ("cpu", "none")
        platform_type: Override platform type ("cuda", "mps", "cpu")
    
    Returns:
        Full device name ("cuda:0", "mps:0", "cpu", "none")
    """
    if device_id in ("cpu", "none"):
        return device_id
    
    if platform_type is None:
        platform_type = get_gpu_backend()
    
    # MPS typically doesn't use indices
    if platform_type == "mps":
        return "mps"
    
    return f"{platform_type}:{device_id}"


def _parse_offload_device(offload_arg: str, platform_type: str = None, cache_enabled: bool = False) -> Optional[str]:
    """
    Parse offload device argument to full device name.
    
    Args:
        offload_arg: Offload device argument ("none", "cpu", "0", "1", or "cuda:1")
        platform_type: Override platform type
        cache_enabled: If True and offload_arg is "none", default to "cpu"
    
    Returns:
        Full device name or None
    """
    if offload_arg == "none":
        # If caching enabled but no offload device specified, default to CPU
        return "cpu" if cache_enabled else None
    
    if offload_arg == "cpu":
        return "cpu"
    
    # If already a full device name (cuda:1, mps:0), return as-is
    if ":" in offload_arg:
        return offload_arg
    
    # Otherwise treat as device ID
    return _device_id_to_name(offload_arg, platform_type)


# =============================================================================
# Constants
# =============================================================================

# Supported file extensions
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}


# =============================================================================
# Video I/O Functions
# =============================================================================

def get_media_files(directory: str) -> List[str]:
    """
    Get all video and image files from directory, sorted alphabetically.
    
    Args:
        directory: Path to directory to scan
        
    Returns:
        Sorted list of file paths (strings) matching video or image extensions
    """
    valid_extensions = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS
    path = Path(directory)
    
    # Get all files and filter by extension (case-insensitive)
    files = [f for f in path.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
    
    return sorted([str(f) for f in files])


def extract_frames_from_image(image_path: str) -> Tuple[torch.Tensor, float]:
    """
    Extract single frame from image file and convert to tensor format.
    
    Reads image using OpenCV, converts BGR to RGB, normalizes to [0,1] range,
    and formats as single-frame video tensor for consistent processing.
    
    Args:
        image_path: Path to input image file
        
    Returns:
        Tuple containing:
            - frames_tensor: Single frame as tensor [1, H, W, C], Float16, range [0,1] (C=3 for RGB, C=4 for RGBA)
            - fps: Default FPS value (30.0) for image-to-video conversion
    
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be opened
    """
    debug.log(f"Loading image: {image_path}", category="file")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Read image with alpha channel preserved
    frame = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise ValueError(f"Cannot open image file: {image_path}")
    
    # Handle grayscale images (2D shape has no channel dimension)
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        debug.log(f"Detected grayscale image, converted to RGB", category="file")
    # Convert BGR(A) to RGB based on channel count (always output 3-channel RGB)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        debug.log(f"Detected BGRA image, converted to RGB", category="file")
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to float32 and normalize based on the source bit depth so that
    # 16-bit inputs (e.g. 16-bit TIFF preview frames) preserve their full
    # dynamic range instead of overflowing the [0,1] range.
    if np.issubdtype(frame.dtype, np.integer):
        max_val = float(np.iinfo(frame.dtype).max)
    else:
        max_val = 1.0 if frame.max() <= 1.0 else 255.0
    frame = frame.astype(np.float32) / max_val
    
    # Convert to tensor [1, H, W, C]
    frames_tensor = torch.from_numpy(frame[None, ...]).to(torch.float16)
    
    debug.log(f"Image tensor shape: {frames_tensor.shape}, dtype: {frames_tensor.dtype}", category="memory")
    
    return frames_tensor, 30.0  # Default FPS for images


def get_input_type(input_path: str) -> Literal['video', 'image', 'directory', 'unknown']:
    """
    Determine input type from file path.
    
    Args:
        input_path: Path to input file or directory
        
    Returns:
        Input type: 'video', 'image', 'directory', or 'unknown'
        
    Raises:
        FileNotFoundError: If input path doesn't exist
    """
    path = Path(input_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    if path.is_dir():
        return 'directory'
    
    ext = path.suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    elif ext in IMAGE_EXTENSIONS:
        return "image"
    else:
        return "unknown"


_IMAGE_OUTPUT_EXTS = frozenset({".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"})
_IMAGE_SEQUENCE_FORMATS = frozenset({"png", "tif", "tiff", "jpg", "jpeg", "dpx", "exr"})
_VIDEO_CONTAINER_EXTS: Dict[str, str] = {
    "mp4": ".mp4",
    "mov": ".mov",
    "mkv": ".mkv",
    "webm": ".webm",
    "avi": ".avi",
}

# Codec name substrings that mandate a specific output container format.
# Codecs absent from this map are assumed to work in any standard container.
_CODEC_REQUIRED_FORMAT: Dict[str, str] = {
    "prores": "mov",   # Apple ProRes family (prores_ks, prores, prores_lt, prores_xe …)
    "dnxhd": "mov",    # Avid DNxHD
    "dnxhr": "mov",    # Avid DNxHR (HD and UHD variants)
}

# Containers that carry an explicit codec allow-list.
# Containers NOT listed here are considered permissive (e.g. MKV accepts almost anything).
_FORMAT_ALLOWED_CODEC_PATTERNS: Dict[str, frozenset] = {
    "mp4": frozenset({
        "h264", "libx264", "hevc", "libx265", "h265", "avc",
        "h264_nvenc", "hevc_nvenc", "h264_amf", "hevc_amf",
        "h264_qsv", "hevc_qsv", "av1", "libaom", "vp9", "mpeg4",
    }),
    "webm": frozenset({"vp8", "vp9", "av1", "libaom", "libvpx"}),
}

OUTPUT_FILE_PREFIX = "seedvr_"
SECONDS_PER_MINUTE = 60.0


def _seedvr_prefixed_name(name: str) -> str:
    return name if name.startswith(OUTPUT_FILE_PREFIX) else f"{OUTPUT_FILE_PREFIX}{name}"


def _extract_codec_from_ffmpeg_args(ffmpeg_args: Optional[List[str]]) -> Optional[str]:
    """Return the value of the -c:v flag in ffmpeg_args, or None if absent."""
    if not ffmpeg_args:
        return None
    for idx, token in enumerate(ffmpeg_args):
        if token == "-c:v" and idx + 1 < len(ffmpeg_args):
            return ffmpeg_args[idx + 1]
    return None


def _resolve_output_format_for_codec(
    codec: Optional[str],
    requested_format: str,
) -> Tuple[str, Optional[str]]:
    """
    Validate that *codec* is compatible with *requested_format* container.

    Returns ``(final_format, warning_message)`` where *warning_message* is a
    human-readable string when a correction was made, or ``None`` when the
    requested format was already valid.

    Rules applied in order:
      1. Codec has a required container and *requested_format* matches → accept.
      2. Codec has a required container and *requested_format* does NOT match →
         correct to the required container and return a warning.
      3. *requested_format* has a strict allow-list and the codec is not in it →
         correct to the codec's required format (or "mov" as a safe fallback).
      4. No conflict → return *requested_format* unchanged.
    """
    if codec is None:
        return requested_format, None

    codec_lower = codec.lower()

    # Check whether this codec mandates a specific container.
    req_format: Optional[str] = None
    for pattern, fmt in _CODEC_REQUIRED_FORMAT.items():
        if pattern in codec_lower:
            req_format = fmt
            break

    if req_format is not None and req_format != requested_format:
        msg = (
            f"Codec '{codec}' requires container '{req_format}' "
            f"(requested: '{requested_format}') — output format corrected to '{req_format}'."
        )
        return req_format, msg

    # Check whether the requested container explicitly disallows this codec.
    allowed = _FORMAT_ALLOWED_CODEC_PATTERNS.get(requested_format)
    if allowed is not None:
        if not any(pat in codec_lower for pat in allowed):
            fallback = req_format or "mov"
            msg = (
                f"Codec '{codec}' is not supported in container '{requested_format}' "
                f"— output format corrected to '{fallback}'."
            )
            return fallback, msg

    return requested_format, None


def generate_output_path(input_path: str, output_format: str, output_dir: Optional[str] = None, 
                        input_type: Optional[str] = None, from_directory: bool = False) -> str:
    """
    Generate output path based on input path and format.
    
    Args:
        input_path: Source file path
        output_format: "mp4", "mov", "mkv", "webm", "png", or any video container name
        output_dir: Optional output directory (overrides default behavior)
        input_type: Optional input type ("image", "video", "directory")
        from_directory: True if processing files from a directory (batch mode)
    
    Returns:
        Absolute output path (file for single image/video, directory for sequences)
    """
    input_path_obj = Path(input_path)
    input_name = _seedvr_prefixed_name(input_path_obj.stem)
    
    # Determine base directory and whether to add suffix
    if output_dir:
        # User specified output directory - use as-is.
        base_dir = Path(output_dir)
    elif from_directory:
        # Batch mode: create sibling folder with _upscaled, using seedvr-prefixed filenames.
        original_dir = input_path_obj.parent
        base_dir = original_dir.parent / f"{original_dir.name}_upscaled"
    else:
        # Single file mode: output to the same directory with a seedvr-prefixed filename.
        base_dir = input_path_obj.parent
    
    # Image input always produces an image output regardless of the video format flag.
    # Honor a requested image format (e.g. tiff) so high-fidelity outputs are kept;
    # fall back to PNG for any non-image format.
    if input_type == "image":
        fmt_lower = (output_format or "").lower()
        img_ext = f".{fmt_lower}" if f".{fmt_lower}" in _IMAGE_OUTPUT_EXTS else ".png"
        output_path = base_dir / f"{input_name}{img_ext}"
        return str(ensure_unique_output_path(output_path).resolve())

    # Generate output path based on format
    if output_format == "png":
        # PNG sequence: output is a directory, not a file
        output_path = base_dir / input_name
    else:
        # Video output: pick the container extension
        fmt_lower = (output_format or "mp4").lower()
        ext = _VIDEO_CONTAINER_EXTS.get(fmt_lower, f".{fmt_lower}")
        output_path = base_dir / f"{input_name}{ext}"

    return str(ensure_unique_output_path(output_path).resolve())


def _normalize_output_path_for_target(
    output_path: str,
    output_format: Optional[str],
    input_type: str,
    video_backend: str,
) -> str:
    """
    Reconcile the output path's extension with the requested export target.

    The user-selected ``--output_format`` is *authoritative*. Any extension the
    user supplied on ``--output`` (for example a stray ``.png``) is stripped via
    :func:`os.path.splitext` and the correct extension for the format is
    re-appended. For video-container exports this override is **mandatory** and
    unconditional: there is no code path that lets an image extension (``.png``
    etc.) reach the ffmpeg command when a video container (mp4/mov/mkv/…) is the
    requested format, regardless of ``video_backend``.
    """
    fmt = (output_format or "mp4").lower()
    abs_path = str(Path(output_path).resolve())
    base, ext = os.path.splitext(abs_path)

    # --- Single image output (image input) ---
    if input_type == "image":
        target_ext = f".{fmt}" if f".{fmt}" in _IMAGE_OUTPUT_EXTS else ".png"
        if ext.lower() != target_ext:
            debug.log(
                f"Output extension '{ext or '<none>'}' is invalid for image output; "
                f"using '{target_ext}'.",
                level="WARNING",
                category="file",
                force=True,
            )
        return base + target_ext

    # --- Image-sequence export (video input → directory of frames) ---
    if fmt in _IMAGE_SEQUENCE_FORMATS:
        if ext:
            debug.log(
                f"Image-sequence export requested ({fmt}); removing file extension "
                f"'{ext}' to use directory output.",
                level="WARNING",
                category="file",
                force=True,
            )
            return base
        return abs_path

    # --- Video container export: extension MUST match --output_format ---
    # Look up the canonical container extension and re-append it after stripping
    # whatever the user supplied. This guarantees that, for the ffmpeg backend,
    # the filename always matches the requested format (never .png/.jpg/etc.).
    target_ext = _VIDEO_CONTAINER_EXTS.get(fmt, f".{fmt}")
    if ext.lower() != target_ext:
        debug.log(
            f"Output extension '{ext or '<none>'}' does not match "
            f"--output_format '{fmt}'; enforcing '{target_ext}'"
            + (" for ffmpeg backend." if video_backend == "ffmpeg" else "."),
            level="WARNING",
            category="file",
            force=True,
        )
    return base + target_ext


def resolve_canonical_output_path(
    input_path: str,
    output: Optional[str],
    output_format: Optional[str],
    input_type: str,
    video_backend: str,
    from_directory: bool = False,
) -> str:
    """
    Produce the single canonical output path mandated by ``output_format``.

    This is the *only* place the user-supplied ``--output`` string is reconciled
    against the master ``--output_format``. It must run in ``main()`` before any
    branching or writer initialization so that downstream consumers (including
    :func:`process_single_file`, the ``is_png`` decision, and the writers) receive
    a path whose extension/target-type already agrees with ``output_format``.

    Args:
        input_path: Source file path.
        output: User-supplied ``--output`` value (may be ``None``).
        output_format: Master format/container (already resolved to its final value).
        input_type: "image", "video", or "directory".
        video_backend: "ffmpeg" or "opencv".
        from_directory: True when processing files discovered inside a directory.

    Returns:
        Absolute, unique output path reconciled with ``output_format``.
    """
    requested_format = (output_format or "").lower()
    if from_directory:
        # Batch mode always derives the per-file path; ``output`` is the optional
        # destination directory and may be ``None``.
        output_path = generate_output_path(
            input_path, output_format, output_dir=output,
            input_type=input_type, from_directory=True,
        )
    elif output is None:
        output_path = generate_output_path(input_path, output_format, input_type=input_type)
    elif not Path(output).suffix or (requested_format in _IMAGE_SEQUENCE_FORMATS and input_type != "image"):
        # No extension or image-sequence export → treat ``output`` as a directory.
        output_path = generate_output_path(
            input_path, output_format, output_dir=output, input_type=input_type,
        )
    else:
        output_path = str(Path(output).resolve())

    # Master format dictates the extension/target type.
    output_path = _normalize_output_path_for_target(
        output_path=output_path,
        output_format=output_format,
        input_type=input_type,
        video_backend=video_backend,
    )
    return str(ensure_unique_output_path(output_path).resolve())


def ensure_unique_output_path(path: Path | str) -> Path:
    """Return a non-conflicting file or directory path by appending a numeric suffix."""
    candidate = Path(path)
    if not candidate.exists():
        return candidate

    if candidate.suffix:
        stem = candidate.stem
        suffix = candidate.suffix
        parent = candidate.parent
        counter = 1
        while True:
            probe = parent / f"{stem}_{counter}{suffix}"
            if not probe.exists():
                return probe
            counter += 1

    parent = candidate.parent
    name = candidate.name
    counter = 1
    while True:
        probe = parent / f"{name}_{counter}"
        if not probe.exists():
            return probe
        counter += 1


def _resolve_effective_resolution(args: "argparse.Namespace", input_width: int, input_height: int) -> int:
    """Compute the effective target resolution from pre_downscale + resolution_mode settings.

    Steps:
    1. Apply pre-downscale factor to determine the *processing baseline* dimensions.
    2. If resolution_mode == 'xtimes', multiply the baseline short-side by resolution_scale.
    3. If resolution_mode == 'pixel', return args.resolution unchanged (it is already the
       pixel target; the engine's own geometry logic will handle aspect ratio).

    Returns:
        Target short-side resolution in pixels to pass as args.resolution.
    """
    ds = getattr(args, "pre_downscale", 1) or 1
    baseline_h = input_height // ds
    baseline_w = input_width // ds
    short_side = min(baseline_h, baseline_w)

    mode = getattr(args, "resolution_mode", "pixel") or "pixel"
    if mode == "xtimes":
        scale = getattr(args, "resolution_scale", 2) or 2
        return short_side * scale
    # pixel mode
    return getattr(args, "resolution", 1080)


def _apply_lanczos_downscale(frames_tensor: "torch.Tensor", factor: int) -> "torch.Tensor":
    """Downscale a float32 [T, H, W, C] tensor by *factor* using Lanczos filtering.

    Uses cv2.INTER_LANCZOS4 on each frame independently.  Returns a new tensor of shape
    [T, H//factor, W//factor, C] with values still in [0, 1].
    """
    import cv2 as _cv2
    frames_np = (frames_tensor.cpu().numpy() * 255.0).astype("uint8")  # [T,H,W,C]
    T, H, W, C = frames_np.shape
    new_h, new_w = H // factor, W // factor
    out = []
    for frame in frames_np:
        frame_bgr = _cv2.cvtColor(frame, _cv2.COLOR_RGB2BGR)
        resized = _cv2.resize(frame_bgr, (new_w, new_h), interpolation=_cv2.INTER_LANCZOS4)
        out.append(_cv2.cvtColor(resized, _cv2.COLOR_BGR2RGB))
    import numpy as _np
    import torch as _torch
    return _torch.from_numpy(_np.stack(out, axis=0).astype("float32") / 255.0)


def _force_disable_cudagraphs_for_safe_mode(args: "argparse.Namespace", mode_label: str) -> None:
    """Disable CUDA Graph capture paths for dynamic-shape-safe execution modes."""
    try:
        import torch._inductor.config as _inductor_config
        _inductor_config.triton.cudagraphs = False
    except Exception as e:
        debug.log(
            f"Could not disable torch._inductor.config.triton.cudagraphs: {e}",
            level="WARNING",
            category="setup",
            force=True,
            indent_level=1,
        )

    if hasattr(args, "compile_dynamic") and not bool(getattr(args, "compile_dynamic", False)):
        args.compile_dynamic = True
        debug.log(
            f"{mode_label}: forced compile_dynamic=True for dynamic-shape safety",
            category="setup",
            force=True,
            indent_level=1,
        )

    debug.log(
        f"{mode_label}: forced CUDA Graphs OFF for safe dynamic execution",
        category="setup",
        force=True,
        indent_level=1,
    )


def _log_crash_diagnostics() -> None:
    """Log the exact GPU/RAM state at the moment of a fatal error.

    Called from the centralized error handler so that a crash report captures the
    memory picture *before* the process unwinds — by the time a traceback prints,
    allocator state is already being torn down. We never let diagnostics raise:
    a failure here must not mask the original exception.
    """
    try:
        import psutil  # Project dependency; imported lazily to keep startup light.
        vm = psutil.virtual_memory()
        proc_rss = psutil.Process(os.getpid()).memory_info().rss
        debug.log(
            f"System RAM: used {vm.used / 1e9:.2f} GB / {vm.total / 1e9:.2f} GB "
            f"({vm.percent:.0f}%), this process RSS {proc_rss / 1e9:.2f} GB",
            level="ERROR", category="generation", force=True,
        )
    except Exception:
        pass

    try:
        if is_cuda_available() and torch.cuda.is_available():
            for dev in range(torch.cuda.device_count()):
                free_b, total_b = torch.cuda.mem_get_info(dev)
                allocated = torch.cuda.memory_allocated(dev)
                reserved = torch.cuda.memory_reserved(dev)
                debug.log(
                    f"GPU {dev} ({torch.cuda.get_device_name(dev)}): "
                    f"free {free_b / 1e9:.2f} GB / total {total_b / 1e9:.2f} GB, "
                    f"allocated {allocated / 1e9:.2f} GB, reserved {reserved / 1e9:.2f} GB",
                    level="ERROR", category="generation", force=True,
                )
    except Exception:
        pass


def _is_oom_error(err: BaseException) -> bool:
    """True when an exception represents a CUDA out-of-memory condition."""
    if isinstance(err, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    return "out of memory" in str(err).lower()


# Auto Tune OOM fail-safe chain configuration.
_AUTOTUNE_BATCH_STEP = 4              # batch_size reduction per loop iteration
_AUTOTUNE_BATCH_REDUCTION_ITERS = 10  # number of batch-reduction iterations
# 4 fixed stages (1: batch=77/ds=2, 2: tiling, 3: tile=256, 4: batch=45) plus the
# 10-iteration batch-reduction loop = 14 escalation stages before termination.
_AUTOTUNE_MAX_STAGES = 4 + _AUTOTUNE_BATCH_REDUCTION_ITERS


def _apply_oom_mitigation(args: "argparse.Namespace", stage: int) -> str:
    """Apply the OOM mitigation prescribed for a given escalation *stage*.

    Implements a fixed 10-step fail-safe chain (see :func:`_run_with_auto_tune`):

      Stage 1  → batch_size=77, pre_downscale=2
      Stage 2  → enable VAE encode/decode tiling, tile_size=512, tile_overlap=32
      Stage 3  → reduce tile_size to 256
      Stage 4  → batch_size=45
      Stages 5-14 → reduce batch_size by 4 each iteration (45, 41, 37, …)

    Returns a short human-readable summary of the change applied.
    """
    changes: List[str] = []

    if stage == 1:
        args.batch_size = 77
        args.pre_downscale = 2
        changes.append("batch_size=77, pre_downscale=2")

    elif stage == 2:
        args.vae_encode_tiled = True
        args.vae_decode_tiled = True
        args.vae_encode_tile_size = 512
        args.vae_decode_tile_size = 512
        args.vae_encode_tile_overlap = 32
        args.vae_decode_tile_overlap = 32
        changes.append("enabled VAE tiling (tile_size=512, tile_overlap=32)")

    elif stage == 3:
        args.vae_encode_tile_size = 256
        args.vae_decode_tile_size = 256
        # Keep overlap strictly smaller than the (now smaller) tile.
        args.vae_encode_tile_overlap = min(getattr(args, "vae_encode_tile_overlap", 32), 32)
        args.vae_decode_tile_overlap = min(getattr(args, "vae_decode_tile_overlap", 32), 32)
        changes.append("reduced tile_size=256")

    elif stage == 4:
        args.batch_size = 45
        changes.append("batch_size=45")

    else:
        # Stages 5..14: ten iterations stepping batch_size down by 4 each time.
        iteration = stage - 4
        args.batch_size = max(1, args.batch_size - _AUTOTUNE_BATCH_STEP)
        changes.append(
            f"batch_size→{args.batch_size} "
            f"(reduction {iteration}/{_AUTOTUNE_BATCH_REDUCTION_ITERS})"
        )

    return ", ".join(changes) if changes else "no further mitigations available"


def _probe_input_short_side(input_path: str) -> Optional[int]:
    """Return the short-side pixel resolution of an image/video input, or ``None``.

    Used by the Auto Tune pre-check to decide the initial ``pre_downscale``. Never
    raises: probing failures simply return ``None`` so the pre-check is skipped.
    """
    try:
        itype = get_input_type(input_path)
        if itype == "video":
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return None
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return min(width, height) if width and height else None
        if itype == "image":
            img = cv2.imread(input_path)
            if img is None:
                return None
            height, width = img.shape[:2]
            return min(width, height)
    except Exception:
        return None
    return None


def _autotune_precheck(args: "argparse.Namespace", input_path: str) -> None:
    """Auto Tune pre-check: set ``pre_downscale`` from the input resolution.

    Runs once per input (before the retry loop) when Auto Tune is enabled:
      * input short side > 480p  → force ``pre_downscale = 2``
      * input short side <= 480p → force ``pre_downscale = 1``
    """
    if not getattr(args, "auto_tune", False):
        return
    short_side = _probe_input_short_side(input_path)
    if short_side is None:
        return
    if short_side > 480:
        args.pre_downscale = 2
        debug.log(
            f"Auto Tune pre-check: input short side {short_side}px > 480p → pre_downscale=2",
            category="setup", force=True,
        )
    else:
        args.pre_downscale = 1
        debug.log(
            f"Auto Tune pre-check: input short side {short_side}px <= 480p → pre_downscale=1",
            category="setup", force=True,
        )


def _run_with_auto_tune(process_fn, args: "argparse.Namespace",
                        on_success=None) -> int:
    """Run ``process_fn`` and, when Auto Tune is on, retry through CUDA OOM.

    On every CUDA out-of-memory error the allocator caches are flushed and the
    next stage of a fixed 10-step fail-safe chain is applied (see
    :func:`_apply_oom_mitigation`):

        1. batch_size=77, pre_downscale=2
        2. enable VAE tiling (tile_size=512, tile_overlap=32)
        3. reduce tile_size to 256
        4. batch_size=45
        5-14. reduce batch_size by 4 each iteration (10 iterations)

    If OOM persists after the 10th batch-reduction iteration the process is
    terminated with a CRITICAL MEMORY ERROR message. Non-OOM errors, or OOMs when
    Auto Tune is disabled, propagate unchanged.
    """
    stage = 0
    while True:
        try:
            frames = process_fn()
            if on_success is not None:
                on_success()
            return frames
        except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
            if not getattr(args, "auto_tune", False) or not _is_oom_error(err):
                raise
            if stage >= _AUTOTUNE_MAX_STAGES:
                # Fail-safe chain exhausted: capture state and terminate.
                _log_crash_diagnostics()
                debug.log(
                    "CRITICAL MEMORY ERROR: Unable to fit model in VRAM even with minimal settings.",
                    level="ERROR", category="generation", force=True,
                )
                sys.exit(1)
            stage += 1
            if hasattr(torch, "cuda"):
                torch.cuda.empty_cache()
            gc.collect()
            summary = _apply_oom_mitigation(args, stage)
            debug.log(
                f"Auto Tune: CUDA OOM caught — escalating fail-safe stage "
                f"{stage}/{_AUTOTUNE_MAX_STAGES} ({summary})",
                category="setup", force=True,
            )


def process_single_file(input_path: str, args: "argparse.Namespace", device_list: "List[str]",
                       output_path: "Optional[str]" = None, format_auto_detected: bool = False,
                       runner_cache: "Optional[Dict[str, Any]]" = None) -> int:
    """
    Process a single video or image file with optional model caching.
    
    For videos, supports streaming mode (chunk_size > 0) which processes in memory-bounded
    chunks with temporal overlap for seamless transitions between chunks.
    
    Args:
        input_path: Path to input file
        args: Command-line arguments with all processing settings
        device_list: List of GPU device IDs as strings
        output_path: Optional explicit output path (auto-generated if None)
        format_auto_detected: Whether output format was auto-detected
        runner_cache: Optional cache dict for model reuse across multiple files
    
    Returns:
        Number of frames written to output
    """
    cap: Optional[cv2.VideoCapture] = None
    video_writer: Optional[cv2.VideoWriter] = None
    frames_tensor: Optional[torch.Tensor] = None
    result: Optional[torch.Tensor] = None

    try:
        input_type = get_input_type(input_path)
        
        if input_type == "unknown":
            debug.log(f"Skipping unsupported file: {input_path}", level="WARNING", category="file", force=True)
            return 0

        if bool(getattr(args, "preview", False)):
            _force_disable_cudagraphs_for_safe_mode(args, "Preview mode")
        if input_type == "image":
            _force_disable_cudagraphs_for_safe_mode(args, "Single-image mode")
        
        debug.log(f"Processing {input_type}: {Path(input_path).name}", category="generation", force=True)
        
        # The output path is reconciled against the master --output_format by the
        # caller (main, via resolve_canonical_output_path) before this worker is
        # ever invoked. Trust the path as given; only fall back to reconciliation
        # here for defensive use when called without a pre-resolved path.
        if output_path is None:
            output_path = resolve_canonical_output_path(
                input_path, None, args.output_format, input_type, args.video_backend,
            )
        else:
            output_path = str(Path(output_path).resolve())
    
        # Show format with auto-detection indicator
        format_prefix = "Auto-detected" if format_auto_detected else "Requested"
        debug.log(f"{format_prefix} output format: {args.output_format}", category="info", force=True, indent_level=1)
        
        # === VIDEO PROCESSING ===
        if input_type == "video":
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Video file not found: {input_path}")
            
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {input_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            debug.log(f"Video info: {total_frames} frames, {width}x{height}, {fps:.2f} FPS", category="info")
            
           # Skip initial frames
            if args.skip_first_frames > 0:
                debug.log(f"Skipping first {args.skip_first_frames} frames", category="info")
                cap.set(cv2.CAP_PROP_POS_FRAMES, args.skip_first_frames)
            
            # Calculate frames to process (apply load_cap if set)
            frames_to_process = total_frames - args.skip_first_frames
            if args.load_cap > 0:
                frames_to_process = min(frames_to_process, args.load_cap)
            
            # Early exit for empty/exhausted video
            if frames_to_process <= 0:
                debug.log(f"No frames to process after skipping {args.skip_first_frames} of {total_frames}", 
                         level="WARNING", category="file", force=True)
                return 0

            # --- Pre-downscale & X-Times resolution math ---
            ds_factor = getattr(args, "pre_downscale", 1) or 1
            if ds_factor > 1:
                debug.log(
                    f"Pre-Downscale {ds_factor}:1 → processing baseline: "
                    f"{width // ds_factor}×{height // ds_factor} px",
                    category="info", force=True, indent_level=1,
                )
            args.resolution = _resolve_effective_resolution(args, width, height)
            debug.log(f"Effective target resolution: {args.resolution} px (short side)", category="info", force=True, indent_level=1)
            
            # Streaming mode: process in chunks
            requested_chunk_size = int(getattr(args, "chunk_size", 0) or 0)
            chunk_duration_minutes = float(getattr(args, "chunk_duration_minutes", 0.0) or 0.0)
            runtime_chunk_size = 0
            if requested_chunk_size > 0:
                runtime_chunk_size = requested_chunk_size
            elif chunk_duration_minutes > 0.0:
                runtime_chunk_size = int(round(chunk_duration_minutes * SECONDS_PER_MINUTE * fps))
            chunk_size = runtime_chunk_size if runtime_chunk_size > 0 else frames_to_process
            streaming = runtime_chunk_size > 0
            total_chunks = (frames_to_process + chunk_size - 1) // chunk_size  # ceiling division
            
            if streaming:
                if requested_chunk_size > 0:
                    debug.log(
                        f"Streaming mode: chunks of {chunk_size} frames, overlap={args.temporal_overlap}",
                        category="info", force=True, indent_level=1
                    )
                else:
                    debug.log(
                        f"Streaming mode: {chunk_duration_minutes:g} minute chunks → {chunk_size} frames at {fps:.2f} FPS, overlap={args.temporal_overlap}",
                        category="info", force=True, indent_level=1
                    )
            
            # Image sequences: detect by format extension, not just "png"
            is_png = (args.output_format or "").lower() in _IMAGE_SEQUENCE_FORMATS
            overlap = args.temporal_overlap
            frames_written = 0
            chunk_idx = 0
            base_name = Path(input_path).stem
            
            # Extract custom ffmpeg video args if supplied
            custom_video_args: Optional[List[str]] = getattr(args, "ffmpeg_video_args", None) or None
            
            # Multi-GPU: workers stream their own segments
            if len(device_list) > 1:
                cap.release()
                cap = None
                video_info = {
                    'video_path': input_path,
                    'start_frame': args.skip_first_frames,
                    'frames_to_process': frames_to_process,
                }
                result = _gpu_processing(None, device_list, args, video_info=video_info)
                # NOTE: Lanczos downscale is now applied to INPUT frames inside
                # _stream_video_chunks (called by each GPU worker), so no post-processing
                # downscale is needed here.

                # Save result
                if is_png:
                    save_frames_to_image(result, output_path, base_name,
                                         image_format=args.output_format)
                else:
                    video_writer = save_frames_to_video(result, output_path, fps,
                        video_backend=args.video_backend, use_10bit=args.use_10bit,
                        custom_video_args=custom_video_args,
                        lut_path=getattr(args, "lut", None))

                frames_written = result.shape[0]

            # Single GPU: stream in main process
            else:
                chunk_count = 0
                for result in _stream_video_chunks(
                    cap=cap,
                    frames_to_process=frames_to_process,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    args=args,
                    device_id=device_list[0],
                    debug=debug,
                    runner_cache=runner_cache,
                    log_progress=streaming,
                    total_chunks=total_chunks,
                    cleanup_timer_name="chunk_cleanup"
                ):
                    chunk_count += 1
                    # NOTE: Lanczos downscale is applied to input frames inside
                    # _stream_video_chunks, so the result here is already at the
                    # correct (upscaled) resolution.
                    # Save output
                    if is_png:
                        save_frames_to_image(result, output_path, base_name,
                                             start_index=frames_written,
                                             image_format=args.output_format)
                    else:
                        video_writer = save_frames_to_video(result, output_path, fps, writer=video_writer,
                            video_backend=args.video_backend, use_10bit=args.use_10bit,
                            custom_video_args=custom_video_args,
                            lut_path=getattr(args, "lut", None))

                    frames_written += result.shape[0]
                    del result
                    result = None

                chunk_idx = chunk_count
            
            if streaming:
                debug.log("", category="none", force=True)
                if len(device_list) > 1:
                    debug.log(f"Streaming complete: {frames_written} frames across {len(device_list)} GPUs", category="success", force=True)
                else:
                    debug.log(f"Streaming complete: {frames_written} frames in {chunk_idx} chunks", category="success", force=True)
            
            debug.log(f"Output saved to: {output_path}", category="file", force=True)
            return frames_written
        
        # === IMAGE PROCESSING ===
        frames_tensor, _ = extract_frames_from_image(input_path)

        # Apply pre-downscale Lanczos to input image and resolve effective resolution
        img_ds_factor = getattr(args, "pre_downscale", 1) or 1
        _, img_h, img_w, _ = frames_tensor.shape
        args.resolution = _resolve_effective_resolution(args, img_w, img_h)
        if img_ds_factor > 1:
            debug.log(
                f"Pre-Downscale {img_ds_factor}:1 on image: "
                f"{img_w}×{img_h} → {img_w // img_ds_factor}×{img_h // img_ds_factor} px",
                category="info", force=True, indent_level=1,
            )
            frames_tensor = _apply_lanczos_downscale(
                frames_tensor.to(torch.float32), img_ds_factor
            ).to(torch.float16)
        debug.log(
            f"Effective target resolution: {args.resolution} px (short side)",
            category="info", force=True, indent_level=1,
        )

        processing_start = time.time()
        # Process frames (multiprocessing only for multi-GPU)
        if len(device_list) > 1:
            result = _gpu_processing(frames_tensor, device_list, args)
        else:
            result = _single_gpu_direct_processing(frames_tensor, args, device_list[0], runner_cache)
        debug.log(f"Processing time: {time.time() - processing_start:.2f}s", category="timing")

        # Save single image. Pass the float result directly so 16-bit targets
        # (e.g. TIFF previews) retain full precision; the saver picks the bit
        # depth from the output extension.
        os.makedirs(Path(output_path).parent, exist_ok=True)
        frame_float = result[0].cpu().numpy()
        _save_image_bgr(frame_float, output_path)
        del frame_float

        debug.log(f"Output saved to: {output_path}", category="file", force=True)
        return 1
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            if video_writer is not None:
                video_writer.release()
        except Exception:
            pass
        if result is not None:
            del result
        if frames_tensor is not None:
            del frames_tensor
        _release_post_file_resources()


def _read_frames_from_cap(cap: cv2.VideoCapture, max_frames: int) -> Optional[torch.Tensor]:
    """
    Read up to max_frames from an already-open VideoCapture.

    This reads *only* ``max_frames`` frames (the current chunk size) and never
    the whole video: the loop strictly stops after ``max_frames`` iterations and
    ``np.stack`` is applied solely to the small chunk that was just read. This is
    what keeps RAM usage bounded to a single chunk when streaming/chunking is
    enabled, instead of materialising the entire decoded sequence at once.

    Args:
        cap: An already opened cv2.VideoCapture instance
        max_frames: Maximum number of frames to read in this call (chunk size)

    Returns:
        Tensor [T, H, W, C] float32 [0,1], or None if no frames available
    """
    # Guard against non-positive chunk sizes so we never spin or over-read.
    if max_frames <= 0:
        return None

    frames = []
    # Strictly stop at max_frames: the loop runs at most chunk_size times.
    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append(frame)

    if not frames:
        return None
    # np.stack runs only on this chunk's frames (len(frames) <= max_frames),
    # never on the full video sequence.
    return torch.from_numpy(np.stack(frames)).to(torch.float32)


def _stream_video_chunks(
    cap: cv2.VideoCapture,
    frames_to_process: int,
    chunk_size: int,
    overlap: int,
    args: argparse.Namespace,
    device_id: str,
    debug: 'Debug',
    runner_cache: Optional[Dict[str, Any]],
    log_progress: bool = False,
    total_chunks: int = 0,
    cleanup_timer_name: Optional[str] = None,
    log_prefix: str = ""
) -> Generator[torch.Tensor, None, None]:
    """
    Generator that streams and processes video chunks.
    
    Handles frame reading, temporal context prepending, processing via
    _process_frames_core, context removal from output, and memory cleanup.
    Caller is responsible for VideoCapture lifecycle and result handling.
    
    Args:
        cap: Open VideoCapture positioned at start frame
        frames_to_process: Total frames to read and process
        chunk_size: Frames per chunk (use frames_to_process for single chunk)
        overlap: Temporal overlap frames between chunks for blending
        args: Processing arguments (copied internally, prepend_frames zeroed after first chunk)
        device_id: GPU device ID for processing
        debug: Debug instance for logging
        runner_cache: Optional model cache dict for reuse across chunks
        log_progress: If True, log chunk progress with separators
        total_chunks: Total chunks for progress display (used if log_progress=True)
        cleanup_timer_name: Optional timer name for memory cleanup logging
        log_prefix: Optional prefix for log messages (e.g., "[GPU 0] " for worker identification)
    
    Yields:
        Processed frames tensor [T, H, W, C] for each chunk, context frames removed
    """
    chunk_args = argparse.Namespace(**vars(args))
    frames_read = 0
    prev_raw_tail = None
    chunk_idx = 0
    streaming = chunk_size < frames_to_process
    # Pre-downscale factor – applied to each chunk's raw frames before processing
    ds_factor = getattr(chunk_args, "pre_downscale", 1) or 1

    while frames_read < frames_to_process:
        read_count = min(chunk_size, frames_to_process - frames_read)
        new_frames = _read_frames_from_cap(cap, read_count)
        if new_frames is None:
            break
        frames_read += new_frames.shape[0]
        chunk_idx += 1

        # Apply pre-downscale Lanczos to raw input frames before processing.
        # This must happen BEFORE overlap concatenation so that prev_raw_tail
        # and new_frames are both at the downscaled resolution.
        if ds_factor > 1:
            new_frames = _apply_lanczos_downscale(new_frames, ds_factor)

        # Disable prepend_frames after first chunk
        if chunk_idx > 1:
            chunk_args.prepend_frames = 0

        # Prepend context from previous chunk
        if prev_raw_tail is not None and overlap > 0:
            context_count = min(overlap, prev_raw_tail.shape[0])
            frames = torch.cat([prev_raw_tail[-context_count:], new_frames], dim=0)
        else:
            frames = new_frames
            context_count = 0

        # Log progress if enabled
        if log_progress and streaming:
            if chunk_idx > 1:
                debug.log("", category="none", force=True)
                debug.log("━" * 60, category="none", force=True)
            debug.log("", category="none", force=True)
            debug.log(f"{log_prefix}Chunk {chunk_idx}/{total_chunks}: {new_frames.shape[0]} new + {context_count} context frames",
                     category="generation", force=True)
            debug.log("", category="none", force=True)

        # Process chunk
        result = _process_frames_core(
            frames_tensor=frames.to(torch.float16),
            args=chunk_args,
            device_id=device_id,
            debug=debug,
            runner_cache=runner_cache
        )

        # Remove context frames from output
        if context_count > 0:
            result = result[context_count:]

        # Save tail for next chunk context (downscaled frames for consistency)
        prev_raw_tail = new_frames[-overlap:].clone() if overlap > 0 else None

        # Cleanup before yield
        del frames

        yield result

        # Memory cleanup between chunks
        if streaming:
            clear_memory(debug=debug, deep=True, force=True, timer_name=cleanup_timer_name)


def _save_image_bgr(frame: np.ndarray, file_path: str) -> None:
    """
    Save a single RGB(A) frame to disk as BGR(A) for OpenCV.

    Accepts either a Float32 array in range [0,1] or an integer array. TIFF
    outputs are written as uncompressed 16-bit to preserve fidelity for previews
    and intermediate frames; all other formats are written as 8-bit.

    Args:
        frame: Frame as numpy array [H, W, C] where C is 3 (RGB) or 4 (RGBA);
            either float in [0,1] or an integer dtype.
        file_path: Output file path (extension selects the format/bit depth)
    """
    ext = os.path.splitext(file_path)[1].lower()
    sixteen_bit = ext in (".tif", ".tiff")

    # Normalize input to float [0,1] regardless of incoming dtype.
    if np.issubdtype(frame.dtype, np.floating):
        norm = np.clip(frame, 0.0, 1.0)
    elif np.issubdtype(frame.dtype, np.integer):
        norm = np.clip(frame.astype(np.float32) / float(np.iinfo(frame.dtype).max), 0.0, 1.0)
    else:
        norm = np.clip(frame.astype(np.float32), 0.0, 1.0)

    if sixteen_bit:
        scaled = (norm * 65535.0 + 0.5).astype(np.uint16)
    else:
        scaled = (norm * 255.0 + 0.5).astype(np.uint8)

    if scaled.shape[2] == 4:
        out = cv2.cvtColor(scaled, cv2.COLOR_RGBA2BGRA)
    else:
        out = cv2.cvtColor(scaled, cv2.COLOR_RGB2BGR)

    if sixteen_bit:
        # IMWRITE_TIFF_COMPRESSION = 1 → no compression (uncompressed TIFF).
        cv2.imwrite(file_path, out, [int(cv2.IMWRITE_TIFF_COMPRESSION), 1])
    else:
        cv2.imwrite(file_path, out)


def _emit_gui_queue_status(file_path: str, current: int, total: int) -> None:
    """Emit a machine-readable queue update for the GUI worker."""
    payload = {
        "file_path": file_path,
        "current": current,
        "total": total,
        "done": max(0, current - 1),
        "remaining": max(0, total - current),
    }
    print(f"__SEEDVR2_GUI_STATUS__|{json.dumps(payload, ensure_ascii=False)}", flush=True)


def _strict_batch_flush(*objs: Any) -> None:
    """Strict flush hook used by GUI/CLI runs to aggressively release memory."""
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def _release_post_file_resources() -> None:
    """Force Python and CUDA cleanup between sequential files."""
    _strict_batch_flush()


def save_frames_to_video(
    frames_tensor: torch.Tensor, 
    output_path: str, 
    fps: float = 30.0,
    writer: Optional[cv2.VideoWriter] = None,
    video_backend: str = "ffmpeg",
    use_10bit: bool = False,
    custom_video_args: Optional[List[str]] = None,
    lut_path: Optional[str] = None,
) -> Optional[cv2.VideoWriter]:
    """
    Save frames tensor to a video file.
    
    Converts tensor from Float32 [0,1] to uint8 [0,255], RGB to BGR for OpenCV,
    and writes to video file. Supports streaming mode where
    an existing writer is passed and kept open for subsequent chunks.
    
    Args:
        frames_tensor: Frames in format [T, H, W, C], Float32, range [0,1]
        output_path: Output video file path (directory created if doesn't exist)
        fps: Frames per second for output video (default: 30.0)
        writer: Existing VideoWriter for streaming (if None, creates new one)
        video_backend: reserved for backwards compatibility (FFmpeg-only export path)
        use_10bit: When video_backend=ffmpeg and custom_video_args is None, use x265/10-bit
        custom_video_args: Optional list of ffmpeg video encoding args (overrides codec/pix_fmt
                           defaults). Implies video_backend=ffmpeg.
    
    Returns:
        VideoWriter if streaming mode (caller must close), None if standalone mode
    
    Raises:
        ValueError: If video writer cannot be initialized
    """
    T, H, W, C = frames_tensor.shape

    effective_backend = video_backend if video_backend in ("ffmpeg", "opencv") else "ffmpeg"
    output_suffix = Path(output_path).suffix.lower()

    if effective_backend == "ffmpeg":
        if output_suffix in _IMAGE_OUTPUT_EXTS:
            raise ValueError(
                f"FFmpeg backend requires a video container path, got image extension '{output_suffix}' "
                f"for output '{output_path}'."
            )
        if not output_suffix:
            raise ValueError(
                f"FFmpeg backend requires an explicit video file extension for output '{output_path}'."
            )

    if writer is None:
        debug.log(f"Saving {T} frames to video: {output_path} (backend={effective_backend})", category="file")
        os.makedirs(Path(output_path).parent, exist_ok=True)
        if effective_backend == "opencv":
            # OpenCV VideoWriter: select a codec based on the output extension.
            ext = Path(output_path).suffix.lower()
            if ext in (".mp4", ".m4v"):
                fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            elif ext in (".avi",):
                fourcc = cv2.VideoWriter.fourcc(*"XVID")
            elif ext in (".mov",):
                fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            else:
                fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
            if not writer.isOpened():
                raise ValueError(f"cv2.VideoWriter cannot open: {output_path}")
        else:
            writer = FFMPEGVideoWriter(output_path, W, H, fps, use_10bit,
                                       custom_video_args=custom_video_args,
                                       lut_path=lut_path)
            if not writer.isOpened():
                raise ValueError(f"Cannot create video writer for: {output_path}")

    gc_interval = 1

    for frame_idx in range(T):
        frame_np = (
            frames_tensor[frame_idx:frame_idx + 1]
            .detach()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .to(device="cpu", dtype=torch.uint8)
            .squeeze(0)
            .contiguous()
            .numpy()
        )

        if effective_backend == "ffmpeg":
            frame_to_write = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        else:
            frame_to_write = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        writer.write(frame_to_write)
        del frame_np, frame_to_write

        completed = frame_idx + 1
        if debug.enabled and ((completed % 32 == 0) or completed == T):
            debug.log(f"Written {completed}/{T} frames", category="file")

        if (completed % gc_interval == 0) or completed == T:
            gc.collect()
            if frames_tensor.is_cuda:
                torch.cuda.empty_cache()

    return writer  # Caller always closes


def save_frames_to_image(
    frames_tensor: torch.Tensor,
    output_dir: str,
    base_name: str,
    start_index: int = 0,
    image_format: Optional[str] = None,
) -> int:
    """
    Save frames tensor as sequential image files in the chosen format.

    Each frame saved as ``{base_name}_{index:0Nd}{ext}`` with zero-padded indices.
    Converts Float32 [0,1] to BGR(A) for OpenCV; TIFF frames are written as
    uncompressed 16-bit, all other formats as 8-bit.

    Args:
        frames_tensor: Frames in format [T, H, W, C], Float32, range [0,1]
        output_dir: Directory to save image files (created if doesn't exist)
        base_name: Base name for output files (e.g., "frame" → "frame_00000.png")
        start_index: Starting index for filenames (for streaming continuation)
        image_format: Image format string from CLI ``--output_format`` (e.g. "png",
            "tiff", "tif", "jpg", "jpeg", "dpx", "exr").  Defaults to "png".

    Returns:
        Number of frames saved
    """
    # Map CLI format name → file extension
    _FMT_TO_EXT: Dict[str, str] = {
        "tiff": ".tiff", "tif": ".tiff",
        "jpg": ".jpg", "jpeg": ".jpg",
        "dpx": ".dpx",
        "exr": ".exr",
        "png": ".png",
    }
    ext = _FMT_TO_EXT.get((image_format or "png").lower(), ".png")

    os.makedirs(output_dir, exist_ok=True)

    # Keep frames as Float32 [0,1] and let _save_image_bgr pick the bit depth
    # from the extension (16-bit for TIFF, 8-bit otherwise) so high-fidelity
    # formats are not down-converted to 8-bit.
    frames_np = frames_tensor.cpu().numpy()
    total = frames_np.shape[0]

    if start_index == 0:
        debug.log(
            f"Saving {total} frames as {ext.lstrip('.')} sequence to directory: {output_dir}",
            category="file",
        )
    digits = 6  # Supports up to 999,999 frames (~11.5 hours at 24fps)

    for idx, frame in enumerate(frames_np):
        filename = f"{base_name}_{start_index + idx:0{digits}d}{ext}"
        file_path = os.path.join(output_dir, filename)
        _save_image_bgr(frame, file_path)
        gc.collect()
        if frames_tensor.is_cuda:
            torch.cuda.empty_cache()
        if debug.enabled and (idx + 1) % 100 == 0:
            debug.log(f"Saved {idx + 1}/{total} images", category="file")

    debug.log(f"Saved {total} images to '{output_dir}'", category="success")
    return total


# =============================================================================
# Core Processing Logic
# =============================================================================

def _process_frames_core(
    frames_tensor: torch.Tensor,
    args: argparse.Namespace,
    device_id: str,
    debug: Debug,
    runner_cache: Optional[Dict[str, Any]] = None
) -> torch.Tensor:
    """
    Core frame processing logic shared between worker and direct processing.
    
    Executes the complete 4-phase pipeline: encode → upscale → decode → postprocess.
    Supports both cached (direct) and non-cached (worker) execution modes.
    
    Args:
        frames_tensor: Input frames [T, H, W, C], Float16/Float32, range [0,1]
        args: Command-line arguments with all processing settings
        device_id: Device ID for inference ("0", "1", etc.)
        debug: Debug instance for logging
        runner_cache: Optional cache dict for model reuse (direct mode only)
    
    Returns:
        Upscaled frames tensor [T', H', W', C], Float32, range [0,1]
    """    
    # Determine platform and convert device IDs to full names
    platform_type = get_gpu_backend()
    inference_device = _device_id_to_name(device_id, platform_type)
    
    # Parse offload devices (with caching defaults)
    cache_dit = args.cache_dit if runner_cache is not None else False
    cache_vae = args.cache_vae if runner_cache is not None else False
    
    dit_offload = _parse_offload_device(args.dit_offload_device, platform_type, cache_dit)
    vae_offload = _parse_offload_device(args.vae_offload_device, platform_type, cache_vae)
    tensor_offload = _parse_offload_device(args.tensor_offload_device, platform_type, False)
    
    # Setup or reuse generation context
    if runner_cache is not None and 'ctx' in runner_cache:
        ctx = runner_cache['ctx']
        # Clear previous run data but keep device config
        keys_to_keep = {'dit_device', 'vae_device', 'dit_offload_device', 
                       'vae_offload_device', 'tensor_offload_device', 'compute_dtype'}
        for key in list(ctx.keys()):
            if key not in keys_to_keep:
                del ctx[key]
    else:
        ctx = setup_generation_context(
            dit_device=inference_device,
            vae_device=inference_device,
            dit_offload_device=dit_offload,
            vae_offload_device=vae_offload,
            tensor_offload_device=tensor_offload,
            debug=debug
        )
        if runner_cache is not None:
            runner_cache['ctx'] = ctx

    # Propagate Auto Tune flag into ctx so generation phases can read it.
    # Keep the legacy key for compatibility with any older code paths.
    ctx['auto_tune'] = getattr(args, 'auto_tune', False)
    ctx['auto_safeguard'] = ctx['auto_tune']

    # Build torch compile args
    torch_compile_args_dit = None
    torch_compile_args_vae = None
    if args.compile_dit:
        torch_compile_args_dit = {
            "backend": args.compile_backend,
            "mode": args.compile_mode,
            "fullgraph": args.compile_fullgraph,
            "dynamic": args.compile_dynamic,
            "dynamo_cache_size_limit": args.compile_dynamo_cache_size_limit,
            "dynamo_recompile_limit": args.compile_dynamo_recompile_limit,
        }
    if args.compile_vae:
        torch_compile_args_vae = {
            "backend": args.compile_backend,
            "mode": args.compile_mode,
            "fullgraph": args.compile_fullgraph,
            "dynamic": args.compile_dynamic,
            "dynamo_cache_size_limit": args.compile_dynamo_cache_size_limit,
            "dynamo_recompile_limit": args.compile_dynamo_recompile_limit,
        }
    
    # Prepare runner with caching support
    model_dir = args.model_dir if args.model_dir is not None else f"./models/{SEEDVR2_FOLDER_NAME}"
    
    # Use fixed IDs for CLI caching when enabled
    dit_id = "cli_dit" if cache_dit else None
    vae_id = "cli_vae" if cache_vae else None
    
    runner, cache_context = prepare_runner(
        dit_model=args.dit_model,
        vae_model=DEFAULT_VAE,
        model_dir=model_dir,
        debug=debug,
        ctx=ctx,
        dit_cache=cache_dit,
        vae_cache=cache_vae,
        dit_id=dit_id,
        vae_id=vae_id,
        block_swap_config={
            'blocks_to_swap': args.blocks_to_swap,
            'swap_io_components': args.swap_io_components,
            'offload_device': dit_offload,
        },
        encode_tiled=args.vae_encode_tiled,
        encode_tile_size=(args.vae_encode_tile_size, args.vae_encode_tile_size),
        encode_tile_overlap=(args.vae_encode_tile_overlap, args.vae_encode_tile_overlap),
        decode_tiled=args.vae_decode_tiled,
        decode_tile_size=(args.vae_decode_tile_size, args.vae_decode_tile_size),
        decode_tile_overlap=(args.vae_decode_tile_overlap, args.vae_decode_tile_overlap),
        tile_debug=args.tile_debug.lower() if args.tile_debug else "false",
        attention_mode=args.attention_mode,
        torch_compile_args_dit=torch_compile_args_dit,
        torch_compile_args_vae=torch_compile_args_vae
    )
    
    ctx['cache_context'] = cache_context
    if runner_cache is not None:
        runner_cache['runner'] = runner
    
    # Preload text embeddings before Phase 1 to avoid sync stall in Phase 2
    ctx['text_embeds'] = load_text_embeddings(script_directory, ctx['dit_device'], ctx['compute_dtype'], debug)
    debug.log("Loaded text embeddings for DiT", category="dit")
    
    # Compute generation info and log start (handles prepending internally)
    frames_tensor, gen_info = compute_generation_info(
        ctx=ctx,
        images=frames_tensor,
        resolution=args.resolution,
        max_resolution=args.max_resolution,
        batch_size=args.batch_size,
        uniform_batch_size=args.uniform_batch_size,
        seed=args.seed,
        prepend_frames=args.prepend_frames,
        temporal_overlap=args.temporal_overlap,
        debug=debug
    )
    log_generation_start(gen_info, debug)
    
    # Phase 1: Encode
    ctx = encode_all_batches(
        runner, ctx=ctx, images=frames_tensor,
        debug=debug, 
        batch_size=args.batch_size,
        uniform_batch_size=args.uniform_batch_size,
        seed=args.seed,
        progress_callback=None, 
        temporal_overlap=args.temporal_overlap,
        resolution=args.resolution,
        max_resolution=args.max_resolution,
        input_noise_scale=args.input_noise_scale,
        color_correction=args.color_correction
    )
    
    # Phase 2: Upscale
    ctx = upscale_all_batches(
        runner, ctx=ctx, debug=debug, progress_callback=None,
        seed=args.seed,
        latent_noise_scale=args.latent_noise_scale,
        cache_model=cache_dit
    )
    
    # Phase 3: Decode
    ctx = decode_all_batches(
        runner, ctx=ctx, debug=debug, progress_callback=None,
        cache_model=cache_vae,
        only_frames=args.only_frames
    )
    
    # Phase 4: Post-process
    ctx = postprocess_all_batches(
        ctx=ctx, debug=debug, progress_callback=None,
        color_correction=args.color_correction,
        prepend_frames=0,  # Worker mode handles this in main process
        temporal_overlap=args.temporal_overlap,
        batch_size=args.batch_size
    )
    
    result_tensor = ctx['final_video']
    
    # Convert to CPU and compatible dtype
    if result_tensor.is_cuda or result_tensor.is_mps:
        result_tensor = result_tensor.cpu()
    if result_tensor.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
        result_tensor = result_tensor.to(torch.float32)
    
    return result_tensor


def _worker_process(
    proc_idx: int, 
    device_id: str, 
    frames_np: Optional[np.ndarray],
    shared_args: Dict[str, Any], 
    return_queue: mp.Queue,
    done_barrier: mp.Barrier,
    video_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Worker process for multi-GPU upscaling.
    
    Supports two modes:
    1. frames_np provided: Process pre-loaded frames (for images)
    2. video_info provided: Stream video segment internally (for videos)
       - Each worker opens the video, seeks to its assigned range, and streams
         with internal chunking and model caching for memory efficiency
    
    Args:
        proc_idx: Worker index for result ordering
        device_id: GPU device ID (used for CUDA_VISIBLE_DEVICES inheritance)
        frames_np: Pre-loaded frames as numpy array, or None for video streaming
        shared_args: Serialized args namespace as dict
        return_queue: Queue for returning results to parent
        done_barrier: Barrier for synchronizing shared memory handoff
        video_info: Optional dict with 'video_path', 'start_frame', 'end_frame'
                   for video streaming mode
    """
    # Create debug instance for this worker
    worker_debug = Debug(enabled=shared_args["debug"])
    
    args = argparse.Namespace(**shared_args)
    
    # Video streaming mode: worker reads and processes its assigned segment
    if video_info is not None:
        cap = cv2.VideoCapture(video_info['video_path'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_info['start_frame'])
        
        segment_frames = video_info['end_frame'] - video_info['start_frame']
        chunk_size = args.chunk_size if args.chunk_size > 0 else segment_frames
        
        worker_debug.log(f"GPU {proc_idx}: frames {video_info['start_frame']}-{video_info['end_frame']} "
                        f"({segment_frames} frames, chunks of {chunk_size})",
                        category="generation", force=True)
        
        # Only GPU 0 uses prepend_frames (applies to video start only)
        worker_args = argparse.Namespace(**vars(args))
        if proc_idx != 0:
            worker_args.prepend_frames = 0
        
        # Enable model caching within worker only if requested
        runner_cache = {} if (args.cache_dit or args.cache_vae) else None
        
        total_chunks = (segment_frames + chunk_size - 1) // chunk_size
        results = []
        for result in _stream_video_chunks(
            cap=cap,
            frames_to_process=segment_frames,
            chunk_size=chunk_size,
            overlap=args.temporal_overlap,
            args=worker_args,
            device_id="0",
            debug=worker_debug,
            runner_cache=runner_cache,
            log_progress=total_chunks > 1,
            total_chunks=total_chunks,
            log_prefix=f"[GPU {proc_idx}] "
        ):
            results.append(result.cpu())
        
        cap.release()
        result_tensor = torch.cat(results, dim=0) if results else torch.empty(0, dtype=torch.float32)
    
    # Pre-loaded frames mode (original behavior)
    else:
        frames_tensor = torch.from_numpy(frames_np).to(torch.float16)
        result_tensor = _process_frames_core(
            frames_tensor=frames_tensor,
            args=args,
            device_id="0",
            debug=worker_debug,
            runner_cache=None
        )
    
    # Share tensor memory for efficient cross-process transfer (avoids pickling large arrays)
    return_queue.put((proc_idx, result_tensor.share_memory_()))
    
    # Wait for parent to copy shared tensors before exiting
    # (shared memory requires creating process to stay alive during access)
    done_barrier.wait()


def _single_gpu_direct_processing(
    frames_tensor: torch.Tensor,
    args: argparse.Namespace,
    device_id: str,
    runner_cache: Optional[Dict[str, Any]]
) -> torch.Tensor:
    """
    Direct single-GPU processing with model caching support.
    
    Uses main process and shared runner cache for efficient multi-file processing.
    """
    return _process_frames_core(
        frames_tensor=frames_tensor,
        args=args,
        device_id=device_id,
        debug=debug,
        runner_cache=runner_cache
    )


def _gpu_processing(
    frames_tensor: Optional[torch.Tensor],
    device_list: List[str], 
    args: argparse.Namespace,
    video_info: Optional[Dict[str, Any]] = None
) -> torch.Tensor:
    """
    Orchestrate multi-GPU parallel video upscaling with temporal overlap blending.
    
    Supports two modes:
    1. video_info provided: Workers stream their assigned video segments internally
       (each GPU reads and processes its frame range with internal chunking)
    2. frames_tensor provided: Workers process pre-loaded frame chunks
       (non streaming behavior for images or pre-loaded videos)
    
    Args:
        frames_tensor: Input frames [T, H, W, C] or None if using video_info mode
        device_list: List of device IDs as strings (e.g., ["0", "1"])
        args: Parsed command-line arguments containing all processing settings
        video_info: Optional dict with 'video_path', 'start_frame', 'frames_to_process'
                   for streaming mode where workers read video directly
    
    Returns:
        Upscaled frames tensor [T', H', W', C], Float32, range [0,1]
    """
    num_devices = len(device_list)
    overlap = args.temporal_overlap
    
    return_queue = mp.Queue(maxsize=0)
    done_barrier = mp.Barrier(num_devices + 1)
    workers = []
    shared_args = vars(args).copy()
    
    # Video streaming mode: distribute frame ranges to workers
    if video_info is not None:
        total_frames = video_info['frames_to_process']
        start_frame = video_info['start_frame']
        video_path = video_info['video_path']
        
        base_per_gpu = total_frames // num_devices
        remainder = total_frames % num_devices
        
        current_start = start_frame
        for idx, device_id in enumerate(device_list):
            gpu_frames = base_per_gpu + (1 if idx < remainder else 0)
            gpu_end = current_start + gpu_frames
            
            # Add overlap frames for blending (except last GPU)
            if idx < num_devices - 1 and overlap > 0:
                gpu_end = min(gpu_end + overlap, start_frame + total_frames)
            
            worker_video_info = {
                'video_path': video_path,
                'start_frame': current_start,
                'end_frame': gpu_end,
            }
            
            os.environ["CUDA_VISIBLE_DEVICES"] = device_id
            p = mp.Process(
                target=_worker_process,
                args=(idx, device_id, None, shared_args, return_queue, done_barrier),
                kwargs={'video_info': worker_video_info}
            )
            p.start()
            workers.append(p)
            
            current_start += gpu_frames
    
    # Pre-loaded frames mode (original behavior for images or non-streaming)
    else:
        total_frames = frames_tensor.shape[0]
        
        if overlap > 0 and num_devices > 1:
            chunk_with_overlap = total_frames // num_devices + overlap
            if args.batch_size > 1:
                chunk_with_overlap = ((chunk_with_overlap + args.batch_size - 1) // args.batch_size) * args.batch_size
            base_chunk_size = chunk_with_overlap - overlap

            chunks = []
            for i in range(num_devices):
                start_idx = i * base_chunk_size
                if i == num_devices - 1:
                    end_idx = total_frames
                else:
                    end_idx = min(start_idx + chunk_with_overlap, total_frames)
                chunks.append(frames_tensor[start_idx:end_idx])
        else:
            chunks = torch.chunk(frames_tensor, num_devices, dim=0)

        for idx, (device_id, chunk_tensor) in enumerate(zip(device_list, chunks)):
            os.environ["CUDA_VISIBLE_DEVICES"] = device_id
            p = mp.Process(
                target=_worker_process,
                args=(idx, device_id, chunk_tensor.cpu().numpy(), shared_args, return_queue, done_barrier),
            )
            p.start()
            workers.append(p)

    # Collect results before joining to prevent deadlock
    # Tensors arrive via shared memory - copy to numpy while workers still alive
    results_np = [None] * num_devices
    collected = 0
    while collected < num_devices:
        proc_idx, result_tensor = return_queue.get()
        results_np[proc_idx] = result_tensor.numpy()
        collected += 1
    
    # Release workers now that shared tensors are copied
    done_barrier.wait()
    
    # Now safe to join
    for p in workers:
        p.join()

    # Concatenate results with overlap blending using shared function
    if args.temporal_overlap > 0 and num_devices > 1:        
        overlap = args.temporal_overlap
        result_tensor = None
        
        for idx, res_np in enumerate(results_np):
            chunk_tensor = torch.from_numpy(res_np).to(torch.float32)
            
            if idx == 0:
                # First chunk: keep all frames
                result_tensor = chunk_tensor
            else:
                # Subsequent chunks: blend overlapping region with accumulated result
                if chunk_tensor.shape[0] > overlap and result_tensor.shape[0] >= overlap:
                    # Get overlapping regions
                    prev_tail = result_tensor[-overlap:]  # Last N frames from accumulated result
                    cur_head = chunk_tensor[:overlap]      # First N frames from current chunk
                    
                    # Blend using shared function
                    blended = blend_overlapping_frames(prev_tail, cur_head, overlap)
                    
                    # Replace tail of result with blended frames, then append rest of chunk
                    result_tensor = torch.cat([
                        result_tensor[:-overlap],           # Everything except the tail
                        blended,                            # Blended overlapping frames
                        chunk_tensor[overlap:]              # Non-overlapping part of current chunk
                    ], dim=0)
                else:
                    # Edge case: chunk too small, just append non-overlapping part
                    if chunk_tensor.shape[0] > overlap:
                        result_tensor = torch.cat([result_tensor, chunk_tensor[overlap:]], dim=0)
        
        if result_tensor is None:
            result_tensor = torch.from_numpy(results_np[0]).to(torch.float32)
    else:
        # Simple concatenation without overlap
        result_tensor = torch.from_numpy(np.concatenate(results_np, axis=0)).to(torch.float32)

    # Handle prepend_frames removal (multi-GPU safe - done after all workers complete)
    if args.prepend_frames > 0:
        if args.prepend_frames < result_tensor.shape[0]:
            debug.log(f"Removing {args.prepend_frames} prepended frames from output", category="generation")
            result_tensor = result_tensor[args.prepend_frames:]
        else:
            debug.log(f"prepend_frames ({args.prepend_frames}) >= total frames ({result_tensor.shape[0]}), skipping removal", 
                     level="WARNING", category="generation", force=True)
    
    return result_tensor


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse and validate command-line arguments for SeedVR2 CLI.
    
    Configures all available options including model selection, processing parameters,
    memory optimization settings, and output configuration.
    
    Returns:
        Parsed arguments namespace with all CLI parameters
    
    Note:
        - cuda_device argument only available on non-macOS systems
        - Default model directory resolves to "models/SEEDVR2" if not specified
    """
    
    # Get the actual invocation path for usage examples
    invocation = sys.argv[0]
    
    # Multi-line usage examples for --help
    usage_examples = f"""
Examples:

  Basic image upscaling:
    python {invocation} image.jpg

  Basic video upscaling with temporal consistency:
    python {invocation} video.mp4 --resolution 720 --batch_size 33
    
  Streaming mode for long videos with 10-bit video output (requires FFMPEG):
    python {invocation} long_video.mp4 --resolution 1080 --batch_size 33 --chunk_size 330 --temporal_overlap 3 --video_backend ffmpeg --10bit

  Multi-GPU processing with temporal overlap:
    python {invocation} video.mp4 --cuda_device 0,1 --resolution 1080 --batch_size 81 --uniform_batch_size --temporal_overlap 3 --prepend_frames 4 

  Memory-optimized for low VRAM (8GB):
    python {invocation} image.png --dit_model seedvr2_ema_3b-Q8_0.gguf --blocks_to_swap 32 --swap_io_components --dit_offload_device cpu --vae_offload_device cpu
    
  High resolution with VAE tiling:
    python {invocation} video.mp4 --resolution 1440 --batch_size 31 --uniform_batch_size --temporal_overlap 3 --vae_encode_tiled --vae_decode_tiled
    
  Batch directory processing:
    python {invocation} media_folder/ --output processed/ --cuda_device 0 --cache_dit --cache_vae --dit_offload_device cpu --vae_offload_device cpu --resolution 1080 --max_resolution 1920
"""
    
    parser = argparse.ArgumentParser(
        description="SeedVR2 Video Upscaler - CLI for high-quality image/video upscaling and batch processing",
        epilog=usage_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False
    )
    
    # Input/Output
    io_group = parser.add_argument_group('Input/Output options')
    io_group.add_argument("input", type=str,
                        help="Input: video file (.mp4, .avi, etc.), image file (.png, .jpg, etc.), or directory")
    io_group.add_argument("--output", type=str, default=None,
                        help="Output path (default: auto-generated in 'output/' directory)")
    io_group.add_argument("--output_format", type=str, default=None,
                        help="Output format/container: 'png' (image sequence), 'mp4', 'mov', 'mkv', 'webm', "
                             "'tiff', 'tif', 'jpg', 'jpeg', 'dpx', 'exr'. "
                             "Default: auto-detect from input type")
    io_group.add_argument("--video_backend", type=str, default="ffmpeg", choices=["ffmpeg", "opencv"],
                        help="Video encoder backend: 'ffmpeg' (recommended, requires ffmpeg in PATH) or "
                             "'opencv' (fallback, mp4/avi output only, no 10-bit support).")
    io_group.add_argument("--10bit", dest="use_10bit", action="store_true",
                        help="Save 10-bit video with x265 codec (reduces banding). Without this flag, "
                         "ffmpeg uses x264 for maximum compatibility. Requires --video_backend ffmpeg")
    io_group.add_argument("--ffmpeg_video_args", type=str, default=None,
                        help="JSON array of custom FFmpeg video encoding args, e.g. "
                             '\'["-c:v","prores_ks","-profile:v","3","-pix_fmt","yuv422p10le"]\'. '
                             "When supplied, implies --video_backend ffmpeg and overrides --10bit codec defaults.")
    io_group.add_argument("--lut", type=str, default=None,
                        help="Optional LUT file path passed to ffmpeg (uses lut3d filter when writing video).")
    io_group.add_argument("--model_dir", type=str, default=None,
                        help=f"Model directory (default: ./models/{SEEDVR2_FOLDER_NAME})")
    io_group.add_argument("--pre_downscale", type=int, default=1, choices=[1, 2, 3],
                        help="Pre-downscale input by this factor before upscaling using Lanczos filtering. "
                             "1 = no downscale (default), 2 = halve dimensions, 3 = reduce to 1/3.")
    io_group.add_argument("--resolution_mode", type=str, default="pixel", choices=["pixel", "xtimes"],
                        help="Resolution computation mode. 'pixel' (default): use --resolution as the target "
                             "short-side pixel count. 'xtimes': multiply the pre-downscaled input height "
                             "by --resolution_scale.")
    io_group.add_argument("--resolution_scale", type=int, default=2,
                        help="Multiplier used when --resolution_mode xtimes is active (default: 2).")

    # Model Selection
    model_group = parser.add_argument_group('Model selection')
    model_group.add_argument("--dit_model", type=str, default=DEFAULT_DIT,
                        choices=get_available_dit_models(),
                        help="DiT model to use. Options: 3B (fp16/fp8/GGUF) or 7B (fp16/fp8/GGUF). Default: 3B FP8")
    
    # Processing Parameters
    process_group = parser.add_argument_group('Processing parameters')
    process_group.add_argument("--resolution", type=int, default=1080,
                        help="Target short-side resolution in pixels (default: 1080)")
    process_group.add_argument("--max_resolution", type=int, default=0,
                        help="Maximum resolution for any edge. Scales down if exceeded. 0 = no limit (default: 0)")
    process_group.add_argument("--batch_size", type=int, default=81,
                        help="Frames per batch (must follow 4n+1: 1, 5, 9, 13, 17, 21,...). "
                          "Ideally matches shot length for best temporal consistency. Higher values improve "
                         "quality and speed but require more VRAM. Default: 81")
    process_group.add_argument("--uniform_batch_size", action="store_true",
                        help="Pad final batch to match batch_size. Prevents temporal artifacts caused by small "
                         "final batches. Add extra compute but recommended for optimal quality.")
    process_group.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    process_group.add_argument("--skip_first_frames", type=int, default=0,
                        help="Skip N initial frames (default: 0)")
    process_group.add_argument("--load_cap", type=int, default=0,
                        help="Load maximum N frames from video. 0 = load all (default: 0)")
    process_group.add_argument("--only_frames", type=int, default=0,
                        help="Limits the maximum number of frames processed per VAE decode chunk to prevent OOM. 0 = no limit (default: 0)")
    process_group.add_argument("--chunk_size", type=int, default=0,
                        help="Frames per chunk for streaming mode. When > 0, processes video in "
                             "memory-bounded chunks of N frames. 0 = load all frames at once (default: 0)")
    process_group.add_argument("--chunk_duration_minutes", type=float, default=0.0,
                        help="Chunk duration in minutes for streaming mode. Runtime chunk size is computed as minutes × 60 × source FPS. Ignored when --chunk_size is set.")
    process_group.add_argument("--prepend_frames", type=int, default=0,
                        help="Prepend N reversed frames to reduce start artifacts (auto-removed). Default: 0")
    process_group.add_argument("--temporal_overlap", type=int, default=0,
                        help="Frames to overlap between batches/GPUs for smooth blending (default: 0)")
    
    # Quality Control
    quality_group = parser.add_argument_group('Quality control')
    quality_group.add_argument("--color_correction", type=str, default="lab", 
                    choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"],
                    help="Color correction method: 'lab' (perceptual color matching, recommended), 'wavelet' (frequency-based), "
                    "'wavelet_adaptive' (wavelet + saturation correction), 'hsv' (hue-conditional), 'adain' (statistical transfer), "
                    "'none' (disabled) (default: lab)")
    quality_group.add_argument("--input_noise_scale", type=float, default=0.0,
                        help="Input noise injection scale (0.0-1.0). Adds variation to input images (default: 0.0)")
    quality_group.add_argument("--latent_noise_scale", type=float, default=0.0,
                        help="Latent noise injection scale (0.0-1.0). Adds variation to latent space (default: 0.0)")
    
    # Device Management
    device_group = parser.add_argument_group('Device management')
    if platform.system() != "Darwin":
        device_group.add_argument("--cuda_device", type=str, default=None,
                        help="CUDA device(s): single '0' or multi-GPU '0,1,2'. Default: device 0")
    device_group.add_argument("--dit_offload_device", type=str, default="none",
                        help="DiT offload device when idle: 'none' (keep on GPU), 'cpu' (offload to RAM), or GPU ID. "
                             "Frees VRAM between phases. Required for BlockSwap. Default: none")
    device_group.add_argument("--vae_offload_device", type=str, default="none",
                        help="VAE offload device when idle: 'none', 'cpu', or GPU ID. Frees VRAM between phases. Default: none")
    device_group.add_argument("--tensor_offload_device", type=str, default="cpu",
                        help="Intermediate tensor storage: 'cpu' (recommended), 'none' (keep on GPU), or GPU ID. Default: cpu")
    
    # Memory Optimization (BlockSwap)
    blockswap_group = parser.add_argument_group('Memory optimization (BlockSwap)')
    blockswap_group.add_argument("--blocks_to_swap", type=int, default=0,
                        help="Transformer blocks to swap for VRAM savings. 0-32 (3B) or 0-36 (7B). "
                             "Requires --dit_offload_device. Not available on macOS. Default: 0 (disabled)")
    blockswap_group.add_argument("--swap_io_components", action="store_true",
                        help="Offload DiT I/O layers for extra VRAM savings. Requires --dit_offload_device. "
                             "Not available on macOS")
    
    # VAE Tiling
    vae_group = parser.add_argument_group('VAE tiling (for high resolution upscale)')
    vae_group.add_argument("--vae_encode_tiled", action="store_true",
                        help="Enable VAE encode tiling to reduce VRAM during encoding")
    vae_group.add_argument("--vae_encode_tile_size", type=int, default=1024,
                        help="VAE encode tile size in pixels (default: 1024). Applied to both height and width. Only used if --vae_encode_tiled is set")
    vae_group.add_argument("--vae_encode_tile_overlap", type=int, default=128,
                        help="VAE encode tile overlap in pixels (default: 128). Reduces visible seams between tiles. Only used if --vae_encode_tiled is set")
    vae_group.add_argument("--vae_decode_tiled", action="store_true",
                        help="Enable VAE decode tiling to reduce VRAM during decoding")
    vae_group.add_argument("--vae_decode_tile_size", type=int, default=1024,
                        help="VAE decode tile size in pixels (default: 1024). Applied to both height and width. Only used if --vae_decode_tiled is set")
    vae_group.add_argument("--vae_decode_tile_overlap", type=int, default=128,
                        help="VAE decode tile overlap in pixels (default: 128). Reduces visible seams between tiles. Only used if --vae_decode_tiled is set")
    vae_group.add_argument("--tile_debug", type=str, default="false", choices=["false", "encode", "decode"],
                        help="Visualize tiles: 'false' (default), 'encode', or 'decode'")
    
    # Performance
    perf_group = parser.add_argument_group('Performance optimization')
    perf_group.add_argument("--attention_mode", type=str, default="sdpa",
                        choices=["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"],
                        help="Attention backend: 'sdpa' (default), 'flash_attn_2', 'flash_attn_3', 'sageattn_2', or 'sageattn_3' (Blackwell GPUs)")
    perf_group.add_argument("--compile_dit", action="store_true", 
                        help="Enable torch.compile for DiT model (20-40%% speedup, requires PyTorch 2.0+ and Triton)")
    perf_group.add_argument("--compile_vae", action="store_true",
                        help="Enable torch.compile for VAE model (15-25%% speedup, requires PyTorch 2.0+ and Triton)")
    perf_group.add_argument("--compile_backend", type=str, default="inductor", choices=["inductor"],
                        help="Compilation backend: 'inductor' (full optimization with Triton) (default: inductor)")
    perf_group.add_argument("--compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"],
                        help="Optimization level: 'default' (fast compilation), 'reduce-overhead' (lower overhead), 'max-autotune' (best runtime, slow compilation), "
                        "(default: default)")
    perf_group.add_argument("--compile_fullgraph", action="store_true",
                        help="Compile entire model as single graph (faster but less flexible). May fail with dynamic shapes (default: False)")
    perf_group.add_argument("--compile_dynamic", action="store_true",
                        help="Handle varying input shapes without recompilation. Useful for different resolutions/batch sizes (default: False)")
    perf_group.add_argument("--compile_dynamo_cache_size_limit", type=int, default=64,
                        help="Max cached compiled versions per function. Increase when using many different input shapes. Higher uses more memory (default: 64)")
    perf_group.add_argument("--compile_dynamo_recompile_limit", type=int, default=128,
                        help="Max recompilation attempts before fallback to eager mode. Safety limit to prevent compilation loops (default: 128)")
    
    # Model Caching (for batch processing)
    cache_group = parser.add_argument_group('Model caching (batch processing)')
    cache_group.add_argument("--cache_dit", action="store_true",
                        help="Keep DiT model in memory between generations. Works with single-GPU directory processing "
                             "or multi-GPU streaming (--chunk_size). Requires --dit_offload_device")
    cache_group.add_argument("--cache_vae", action="store_true",
                        help="Keep VAE model in memory between generations. Works with single-GPU directory processing "
                             "or multi-GPU streaming (--chunk_size). Requires --vae_offload_device")
    
    # Debugging
    debug_group = parser.add_argument_group('Debugging')
    debug_group.add_argument("--auto_tune", action="store_true",
                        help="Enable Auto Tune: performs pre-flight VRAM detection, forces VAE tile overlap "
                             "to 32, and on OOM automatically relaxes settings and retries (up to 5 times) by "
                             "shrinking batch size, enabling/tightening VAE tiling, and reducing temporal overlap.")
    debug_group.add_argument("--debug", action="store_true",
                        help="Enable verbose debug logging")
    
    # Auto-show help if no arguments provided
    if len(sys.argv) == 1:
        sys.argv.append('--help')

    return parser.parse_args()


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """
    Main entry point for SeedVR2 Video Upscaler CLI.
    
    Orchestrates the complete upscaling workflow:
        1. Parse and validate command-line arguments
        2. Extract frames from input video/image(s)
        3. Download required models if not cached
        4. Process frames on single or multiple GPUs
        5. Save results as video or PNG sequence
        6. Report timing and FPS (calculated from total wall-clock time)
    
    Error handling:
        - Validates tile configuration before processing
        - Provides detailed error messages with traceback
        - Ensures proper cleanup on exit (VRAM automatically freed)
    
    Raises:
        SystemExit: On argument validation failure or processing error
    """
    # Parse arguments
    args = parse_arguments()

    # --- Mandatory entry-point output path/extension override (source of truth) ---
    # When the caller supplies BOTH an explicit --output path and an explicit
    # --output_format, the output extension is forced to match the requested
    # format right here, before any other logic runs. This guarantees the user's
    # chosen container is authoritative and is never silently replaced (e.g. with
    # a stray ".png"). When --output_format is auto-detected (None) the canonical
    # reconciliation later in main() resolves the final path instead.
    if getattr(args, "output", None) and getattr(args, "output_format", None):
        base_path = os.path.splitext(args.output)[0]
        fmt = args.output_format.lower()
        if fmt in _IMAGE_SEQUENCE_FORMATS:
            # Image-sequence export writes a directory of frames, not a single
            # file, so strip any extension and use the bare base path.
            args.output = base_path
        else:
            ext = _VIDEO_CONTAINER_EXTS.get(fmt, f".{fmt}")
            args.output = f"{base_path}{ext}"

    # Update debug instance with --debug flag
    debug.enabled = args.debug

    # Auto Tune: force tile overlap to 32 for both encode and decode
    if getattr(args, "auto_tune", False):
        if hasattr(args, "vae_encode_tile_overlap"):
            args.vae_encode_tile_overlap = 32
        if hasattr(args, "vae_decode_tile_overlap"):
            args.vae_decode_tile_overlap = 32
        debug.log("Auto Tune active: VAE tile overlap forced to 32", category="setup", force=True)

    # print header
    debug.print_header(cli=True)
    
    debug.log("Arguments:", category="setup")
    for key, value in vars(args).items():
        debug.log(f"{key}: {value}", category="none", indent_level=1)

    if args.vae_encode_tiled and args.vae_encode_tile_overlap >= args.vae_encode_tile_size:
        debug.log(f"VAE encode tile overlap ({args.vae_encode_tile_overlap}) must be smaller than tile size ({args.vae_encode_tile_size})", level="ERROR", category="vae", force=True)
        sys.exit(1)
    
    if args.vae_decode_tiled and args.vae_decode_tile_overlap >= args.vae_decode_tile_size:
        debug.log(f"VAE decode tile overlap ({args.vae_decode_tile_overlap}) must be smaller than tile size ({args.vae_decode_tile_size})", level="ERROR", category="vae", force=True)
        sys.exit(1)

    # Validate ffmpeg availability if selected or needed for custom video args
    if args.ffmpeg_video_args:
        try:
            import json as _json
            parsed_ffmpeg_args = _json.loads(args.ffmpeg_video_args)
            if not isinstance(parsed_ffmpeg_args, list):
                raise ValueError("must be a JSON array")
            args.ffmpeg_video_args = parsed_ffmpeg_args  # replace string with list
        except Exception as exc:
            debug.log(f"--ffmpeg_video_args JSON parse error: {exc}", level="ERROR", category="setup", force=True)
            sys.exit(1)
        # Force ffmpeg backend when custom args are given
        args.video_backend = "ffmpeg"

        # Codec-container pre-check is intentionally disabled.
        # args.output_format is used as provided by the caller without modification.
    
    if args.video_backend == "ffmpeg" and shutil.which("ffmpeg") is None:
        debug.log("--video_backend ffmpeg requires ffmpeg in PATH. Install ffmpeg and retry.", 
                 level="ERROR", category="setup", force=True)
        sys.exit(1)
    
    # Inform about caching defaults
    if args.cache_dit and args.dit_offload_device == "none":
        offload_target = "system memory (CPU)" if get_gpu_backend() != "mps" else "unified memory"
        debug.log(
            f"DiT caching enabled: Using default {offload_target} for offload. "
            "Set --dit_offload_device explicitly to use a different device.",
            category="cache", force=True
        )
    
    if args.cache_vae and args.vae_offload_device == "none":
        offload_target = "system memory (CPU)" if get_gpu_backend() != "mps" else "unified memory"
        debug.log(
            f"VAE caching enabled: Using default {offload_target} for offload. "
            "Set --vae_offload_device explicitly to use a different device.",
            category="cache", force=True
        )

    chunking_requested = bool((args.chunk_size or 0) > 0 or (args.chunk_duration_minutes or 0.0) > 0.0)

    if args.debug:
        if platform.system() == "Darwin":
            debug.log("You are running on macOS and will use the MPS backend!", category="info", force=True)
        else:
            # Show actual CUDA device visibility
            debug.log(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set (all)')}", category="device")
            if is_cuda_available():
                debug.log(f"torch.cuda.device_count(): {torch.cuda.device_count()}", category="device")
                debug.log(f"Using device index 0 inside script (mapped to selected GPU)", category="device")
    
    try:
        start_time = time.time()
        
        # Parse GPU list
        if platform.system() == "Darwin":
            device_list = ["0"]
        else:
            if args.cuda_device:
                device_list = [d.strip() for d in str(args.cuda_device).split(',') if d.strip()]
            else:
                device_list = ["0"]
        if args.debug:
            debug.log(f"Using devices: {device_list}", category="device")
        
        # Download models once before processing
        if not download_weight(dit_model=args.dit_model, vae_model=DEFAULT_VAE, model_dir=args.model_dir, debug=debug):
            debug.log("Failed to download required models. Check console output above.", level="ERROR", category="download", force=True)
            sys.exit(1)
        
        # Determine input type and process accordingly
        input_type = get_input_type(args.input)

        # Track total frames for FPS calculation (time tracked via start_time)
        total_frames_processed = 0
        
        # Track if output format was user-specified or auto-detected
        format_auto_detected = args.output_format is None
        
        if input_type == 'directory':
            media_files = get_media_files(args.input)
            if not media_files:
                debug.log(f"No video or image files found in directory: {args.input}", 
                        level="ERROR", category="file", force=True)
                sys.exit(1)
            
            debug.log(f"Found {len(media_files)} media files to process", category="file", force=True)
            
            # Multi-GPU caching requires streaming (workers cache within their chunk loops)
            if (args.cache_dit or args.cache_vae) and len(device_list) > 1 and not chunking_requested:
                debug.log(
                    "Model caching requires streaming mode (--chunk_size > 0 or --chunk_duration_minutes > 0) for multi-GPU. "
                    "Disabling caching for this run.",
                    level="WARNING", category="cache", force=True
                )
                args.cache_dit = False
                args.cache_vae = False
            
            # Single-GPU: runner_cache persists across files; multi-GPU: workers cache internally
            runner_cache = {} if (args.cache_dit or args.cache_vae) and len(device_list) == 1 else None
            
            for idx, file_path in enumerate(media_files, 1):
                # Visual separation between files (except before first file)
                if idx > 1:
                    debug.log("", category="none", force=True)
                    debug.log("━" * 60, category="none", force=True)
                    debug.log("", category="none", force=True)
                
                total_files = len(media_files)
                _emit_gui_queue_status(file_path, idx, total_files)
                debug.log(
                    f"Processing: {file_path} | File {idx} of {total_files} "
                    f"[Done: {idx - 1}, Remaining: {total_files - idx}]",
                    category="generation", force=True
                )
                
                # Auto-detect format per file if not user-specified.
                # When --ffmpeg_video_args is provided, derive the default container from
                # the codec rather than blindly using "mp4" — this lets ProRes, DNxHD, etc.
                # select the correct container automatically.
                if format_auto_detected:
                    file_type = get_input_type(file_path)
                    if file_type == "video":
                        _codec = _extract_codec_from_ffmpeg_args(
                            getattr(args, "ffmpeg_video_args", None)
                        )
                        file_output_format, _warn = _resolve_output_format_for_codec(_codec, "mp4")
                        if _warn:
                            debug.log(_warn, level="WARNING", category="setup", force=True)
                    else:
                        file_output_format = "png"
                else:
                    file_output_format = args.output_format
                
                # Temporarily override args.output_format for this file
                original_format = args.output_format
                args.output_format = file_output_format
                
                # Reconcile the path against the now-final per-file format before
                # the worker is invoked. file_output_format is the master here.
                output_path = resolve_canonical_output_path(
                    file_path, args.output, file_output_format,
                    input_type=get_input_type(file_path),
                    video_backend=args.video_backend, from_directory=True,
                )
                
                # Process with explicit output path and runner cache. Auto Tune
                # (when enabled) recovers from OOM by relaxing settings and retrying.
                _autotune_precheck(args, file_path)
                total_frames_processed += _run_with_auto_tune(
                    lambda: process_single_file(
                        file_path, args, device_list, output_path,
                        format_auto_detected=format_auto_detected,
                        runner_cache=runner_cache),
                    args,
                    on_success=_release_post_file_resources,
                )
                
                # Restore original format
                args.output_format = original_format

        elif input_type in ("video", "image"):
            # Auto-detect output format for single file if not specified.
            # When --ffmpeg_video_args carries a codec, derive the container from it
            # rather than defaulting to "mp4" unconditionally.
            if format_auto_detected:
                if input_type == "video":
                    _codec = _extract_codec_from_ffmpeg_args(
                        getattr(args, "ffmpeg_video_args", None)
                    )
                    args.output_format, _warn = _resolve_output_format_for_codec(_codec, "mp4")
                    if _warn:
                        debug.log(_warn, level="WARNING", category="setup", force=True)
                else:
                    args.output_format = "png"
            
            # args.output_format is now final/authoritative. Reconcile args.output
            # against it exactly once, before any worker call or writer init, so the
            # master format mandates the path extension/target type.
            args.output = resolve_canonical_output_path(
                args.input, args.output, args.output_format,
                input_type=input_type, video_backend=args.video_backend,
            )
            
            # Caching: single-GPU streaming uses runner_cache, multi-GPU streaming workers cache internally
            runner_cache = None
            streaming = chunking_requested
            
            if args.cache_dit or args.cache_vae:
                if len(device_list) > 1:
                    if not streaming:
                        debug.log(
                            "Model caching requires streaming mode (--chunk_size > 0 or --chunk_duration_minutes > 0) for multi-GPU. "
                            "Disabling caching for this run.",
                            level="WARNING", category="cache", force=True
                        )
                        args.cache_dit = False
                        args.cache_vae = False
                elif streaming:
                    runner_cache = {}
                else:
                    debug.log(
                        "Model caching has no benefit for single file processing (only useful for directories or streaming mode). "
                        "Consider removing --cache_dit/--cache_vae for single files.",
                        category="tip", force=True
                    )
            
            _autotune_precheck(args, args.input)
            total_frames_processed += _run_with_auto_tune(
                lambda: process_single_file(
                    args.input, args, device_list, args.output,
                    format_auto_detected=format_auto_detected,
                    runner_cache=runner_cache),
                args,
            )

        else:
            debug.log(f"Unsupported input type: {args.input}", level="ERROR", category="file", force=True)
            sys.exit(1)
        
        # Calculate total execution time
        total_time = time.time() - start_time
        
        debug.log("", category="none", force=True)
        debug.log(f"All upscaling processes completed successfully in {total_time:.2f}s", category="success", force=True)
        
        # Calculate and display FPS based on overall wall-clock time
        if total_time > 0 and total_frames_processed > 0:
            fps = total_frames_processed / total_time
            debug.log(f"Average FPS: {fps:.2f} frames/sec", category="timing", force=True)
        
    except Exception as e:
        debug.log(f"Error during processing: {e}", level="ERROR", category="generation", force=True)
        # Capture the exact GPU/RAM state before unwinding so crash reports are actionable.
        _log_crash_diagnostics()
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        debug.log(f"Process {os.getpid()} terminating - VRAM will be automatically freed", category="cleanup", force=True)

        # print footer
        debug.print_footer()

if __name__ == "__main__":
    main()
