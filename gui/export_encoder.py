"""
Dedicated export encoder — builds correct ffmpeg commands for each codec.
Import this in workers.py and export_dialog.py.
"""

import os
import subprocess

# Codec → ffmpeg encoder mapping
CODEC_MAP = {
    "ProRes": {"encoder": "prores_ks", "container": "mov"},
    "DNxHR": {"encoder": "dnxhd", "container": "mov"},
    "H264": {"encoder": "auto", "container": "mp4"},
    "H265": {"encoder": "auto", "container": "mp4"},
    "AV1": {"encoder": "libsvtav1", "container": "mp4"},
    "FFV1": {"encoder": "ffv1", "container": "mkv"},
    "VP9": {"encoder": "libvpx-vp9", "container": "webm"},
    "QuickTime V210": {"encoder": "v210", "container": "mov"},
    "QuickTime R210": {"encoder": "r210", "container": "mov"},
    "QuickTime Animation": {"encoder": "qtrle", "container": "mov"},
}

PRORES_PROFILES = {
    "Proxy": 0, "LT": 1, "Standard": 2,
    "HQ": 3, "4444": 4, "4444 XQ": 5,
}

QUALITY_MAP = {
    "ProRes": {"low": "Proxy", "medium": "HQ", "high": "4444"},
    "DNxHR": {"low": "LB", "medium": "SQ", "high": "HQX"},
    "H264": {"low": 28, "medium": 23, "high": 18},
    "H265": {"low": 28, "medium": 23, "high": 18},
    "AV1": {"low": 35, "medium": 25, "high": 18},
    "VP9": {"low": 35, "medium": 25, "high": 18},
    "FFV1": {"low": 1, "medium": 3, "high": 4},
}


def detect_hw_encoder(codec: str) -> str:
    """Detect best hardware encoder for H264/H265."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True, text=True, timeout=5, check=False
        )
        encoders = result.stdout
    except Exception:
        encoders = ""

    prefix = "h264" if codec == "H264" else "hevc"

    # Priority: NVENC > QSV > AMF > Software
    if f"{prefix}_nvenc" in encoders:
        return f"{prefix}_nvenc"
    if f"{prefix}_qsv" in encoders:
        return f"{prefix}_qsv"
    if f"{prefix}_amf" in encoders:
        return f"{prefix}_amf"

    return "libx264" if codec == "H264" else "libx265"


def build_ffmpeg_command(
    input_path: str,
    output_path: str,
    codec: str,
    profile: str = "",
    quality_level: str = "medium",
    bitrate_mode: str = "dynamic",
    bitrate_mbps: float = 0,
    audio_mode: str = "copy",
    source_fps: float = 30.0,
    use_10bit: bool = True,
    trim_in: int = -1,
    trim_out: int = -1,
    input_frames_dir: str = "",
) -> list:
    """Build complete ffmpeg command for export."""

    cmd = ["ffmpeg", "-y"]

    # Input
    if input_frames_dir:
        cmd += ["-framerate", str(source_fps),
                "-i", os.path.join(input_frames_dir, "%08d.tiff")]
    else:
        cmd += ["-i", input_path]

    # Trim
    if trim_in >= 0:
        cmd += ["-ss", str(trim_in / source_fps)]
    if trim_out >= 0 and trim_in >= 0:
        duration = (trim_out - trim_in) / source_fps
        cmd += ["-t", str(duration)]

    # Video codec
    encoder = CODEC_MAP.get(codec, {}).get("encoder", "libx265")

    if codec in ("H264", "H265") and encoder == "auto":
        encoder = detect_hw_encoder(codec)

    cmd += ["-c:v", encoder]

    # Codec-specific quality parameters
    ql = quality_level.lower()

    if codec == "ProRes":
        pname = QUALITY_MAP["ProRes"].get(ql, "HQ")
        profile_num = PRORES_PROFILES.get(pname, 3)
        cmd += ["-profile:v", str(profile_num), "-vendor", "apl0"]
        if use_10bit:
            cmd += ["-pix_fmt", "yuva444p10le" if profile_num >= 4
                    else "yuv422p10le"]

    elif codec == "DNxHR":
        pname = QUALITY_MAP["DNxHR"].get(ql, "SQ")
        cmd += ["-profile:v", pname.lower()]
        if pname in ("HQX", "444"):
            cmd += ["-pix_fmt", "yuv422p10le"]

    elif codec in ("H264", "H265"):
        cq = QUALITY_MAP[codec].get(ql, 23)
        if "nvenc" in encoder:
            cmd += ["-preset", "p5"]
            if bitrate_mode == "constant" and bitrate_mbps > 0:
                cmd += ["-rc", "cbr", "-b:v", f"{bitrate_mbps}M"]
            else:
                cmd += ["-rc", "vbr", "-cq", str(cq)]
        elif "qsv" in encoder:
            if bitrate_mode == "constant" and bitrate_mbps > 0:
                cmd += ["-b:v", f"{bitrate_mbps}M"]
            else:
                cmd += ["-global_quality", str(cq), "-look_ahead", "1"]
        elif "amf" in encoder:
            quality_map = {"low": "speed", "medium": "balanced", "high": "quality"}
            cmd += ["-quality", quality_map.get(ql, "balanced")]
            if bitrate_mode == "constant" and bitrate_mbps > 0:
                cmd += ["-rc", "cbr", "-b:v", f"{bitrate_mbps}M"]
            else:
                cmd += ["-qp_i", str(cq)]
        else:  # software libx264/libx265
            cmd += ["-preset", "medium"]
            if bitrate_mode == "constant" and bitrate_mbps > 0:
                cmd += ["-b:v", f"{bitrate_mbps}M"]
            else:
                cmd += ["-crf", str(cq)]
        # 10-bit for H265
        if codec == "H265" and use_10bit:
            cmd += ["-pix_fmt", "yuv420p10le", "-profile:v", "main10"]
        elif codec == "H264":
            cmd += ["-pix_fmt", "yuv420p"]

    elif codec == "AV1":
        cq = QUALITY_MAP["AV1"].get(ql, 25)
        cmd += ["-crf", str(cq), "-preset", "6"]
        if bitrate_mode == "constant" and bitrate_mbps > 0:
            cmd += ["-b:v", f"{bitrate_mbps}M"]

    elif codec == "VP9":
        cq = QUALITY_MAP["VP9"].get(ql, 25)
        cmd += ["-crf", str(cq), "-b:v", "0"]
        if bitrate_mode == "constant" and bitrate_mbps > 0:
            cmd += ["-b:v", f"{bitrate_mbps}M"]

    elif codec == "FFV1":
        level = QUALITY_MAP["FFV1"].get(ql, 3)
        cmd += ["-level", str(level), "-slicecrc", "1"]

    elif codec == "QuickTime V210":
        cmd += ["-pix_fmt", "yuv422p10le"]

    elif codec == "QuickTime R210":
        cmd += ["-pix_fmt", "rgb48be"]

    elif codec == "QuickTime Animation":
        pass  # qtrle needs no extra params

    # FPS — preserve source
    cmd += ["-r", str(source_fps)]

    # Audio (raw frame sequence has no audio stream; never copy audio in that case)
    if input_frames_dir:
        cmd += ["-an"]
    elif audio_mode == "none" or audio_mode == "None":
        cmd += ["-an"]
    elif audio_mode == "copy":
        cmd += ["-c:a", "copy"]
    elif "aac" in audio_mode:
        bitrate = audio_mode.replace("aac_", "").replace("kbps", "")
        cmd += ["-c:a", "aac", "-b:a", f"{bitrate}k"]
    elif audio_mode == "flac":
        cmd += ["-c:a", "flac"]
    else:
        cmd += ["-c:a", "copy"]

    # Output
    cmd.append(output_path)

    return cmd


def get_export_extension(codec: str, container: str = "") -> str:
    """Get correct file extension for codec."""
    if container:
        return f".{container}"
    return f".{CODEC_MAP.get(codec, {}).get('container', 'mp4')}"
