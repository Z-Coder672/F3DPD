#!/usr/bin/env python3
"""
F3DPD Orchestrator: headless stream + AI + Telegram DM

Behavior:
- Start the camera streamer headlessly with infinite retries (80s interval after initial backoff).
- Maintain a rolling 30-second buffer of frames at 30 fps (900 frames).
- Every 30s (time.time scheduling), run AI classifier on current frame.
- On failure detection: stitch a 30s video (buffer leading up to detection), send to Telegram.
- After alert, enter 5-minute cooldown.
- Send success notification when connection is established after failures.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import tempfile
from collections import deque
from typing import Deque, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from eufy_rtsp_streamer import EufyRTSPStreamer
from notifier import TelegramNotifier
from predictor_clip import predict_failed_from_bgr


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


FRAME_BUFFER_SIZE = 900  # 30 seconds at 30 fps
TARGET_FPS = 30


def decode_jpeg_to_bgr(jpeg_bytes: bytes) -> Optional[np.ndarray]:
    try:
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


def exponential_backoff_attempts() -> List[int]:
    return [5, 10, 20, 40, 80]


def _estimate_jpeg_buffer_bytes(jpeg_frames: Sequence[bytes]) -> int:
    return sum(len(b) for b in jpeg_frames)


def _ffmpeg_encode_rawvideo(
    first_frame_bgr: np.ndarray,
    rest_frames_bgr: Iterable[np.ndarray],
    output_path: str,
    fps: int,
) -> bool:
    """Encode BGR frames to MP4 using ffmpeg (more compatible than OpenCV VideoWriter)."""
    height, width = first_frame_bgr.shape[:2]

    # Prefer widely-compatible settings for Telegram:
    # - yuv420p: most decoders support it
    # - +faststart: moov atom at beginning for streaming playback
    # Encoders to try (in order): hardware (if present) then software, then a very compatible MPEG-4 fallback.
    encoder_variants: List[List[str]] = [
        ["-c:v", "h264_v4l2m2m", "-pix_fmt", "yuv420p", "-b:v", "2500k"],
        ["-c:v", "h264_omx", "-pix_fmt", "yuv420p", "-b:v", "2500k"],
        ["-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p"],
        ["-c:v", "mpeg4", "-q:v", "6"],
    ]

    base_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-movflags",
        "+faststart",
        output_path,
    ]

    # Try each encoder config by injecting it before output_path (end of cmd)
    for enc_args in encoder_variants:
        cmd = base_cmd[:-1]  # drop output_path
        # Insert encoder args before movflags/output
        # base_cmd layout: ... -i pipe:0 -movflags +faststart output
        insert_at = cmd.index("-movflags")
        cmd = cmd[:insert_at] + enc_args + cmd[insert_at:]
        cmd.append(output_path)

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            logger.error("ffmpeg not found; cannot encode video")
            return False
        except Exception as e:
            logger.error("Failed to start ffmpeg encoder: %s", e)
            continue

        assert proc.stdin is not None
        try:
            # Write first frame then stream the rest; keeps memory bounded.
            proc.stdin.write(first_frame_bgr.tobytes())
            for f in rest_frames_bgr:
                if f.shape[0] != height or f.shape[1] != width:
                    f = cv2.resize(f, (width, height), interpolation=cv2.INTER_AREA)
                proc.stdin.write(f.tobytes())
            proc.stdin.close()
            stderr = proc.stderr.read() if proc.stderr is not None else b""
            rc = proc.wait(timeout=120)
            if rc == 0:
                return True
            logger.warning("ffmpeg encode failed (rc=%s): %s", rc, stderr.decode(errors="ignore").strip())
        except Exception as e:
            logger.warning("ffmpeg encode exception: %s", e)
            try:
                proc.kill()
            except Exception:
                pass
        finally:
            try:
                if proc.stderr is not None:
                    proc.stderr.close()
            except Exception:
                pass

    return False


def create_video_from_jpegs(jpeg_frames: Sequence[bytes], output_path: str, fps: int = 30) -> bool:
    """Create MP4 video from list of JPEG-encoded frames (ffmpeg preferred; OpenCV fallback)."""
    if not jpeg_frames:
        return False
    try:
        # Decode first frame to establish size
        first_bgr = decode_jpeg_to_bgr(jpeg_frames[0])
        if first_bgr is None:
            logger.error("Failed to decode first frame; cannot create video")
            return False
        height, width = first_bgr.shape[:2]

        def _iter_rest_frames() -> Iterable[np.ndarray]:
            for jb in jpeg_frames[1:]:
                f = decode_jpeg_to_bgr(jb)
                if f is None:
                    continue
                if f.shape[0] != height or f.shape[1] != width:
                    f = cv2.resize(f, (width, height), interpolation=cv2.INTER_AREA)
                yield f

        # Stream into ffmpeg to avoid ever holding 900 decoded frames in RAM (prevents OOM-kills on Pi).
        if _ffmpeg_encode_rawvideo(first_bgr, _iter_rest_frames(), output_path, fps=fps):
            logger.info("Encoded video via ffmpeg")
            return True

        codecs_to_try: List[Tuple[str, str]] = [
            ("mp4v", "MPEG-4"),
            ("avc1", "H.264/AVC1"),
            ("H264", "H264"),
            ("x264", "X264"),
        ]

        out: Optional[cv2.VideoWriter] = None
        for codec_str, codec_name in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_str)
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if out.isOpened():
                    logger.info("Using %s codec for video encoding", codec_name)
                    break
                out = None
            except Exception:
                out = None
                continue

        if out is None or not out.isOpened():
            logger.error("Failed to open video writer with any codec")
            return False

        # OpenCV fallback: decode/write on the fly (still bounded memory)
        out.write(first_bgr)
        for jb in jpeg_frames[1:]:
            f = decode_jpeg_to_bgr(jb)
            if f is None:
                continue
            if f.shape[0] != height or f.shape[1] != width:
                f = cv2.resize(f, (width, height), interpolation=cv2.INTER_AREA)
            out.write(f)

        out.release()
        return True
    except Exception as e:
        logger.error("Failed to create video: %s", e)
        return False


def _sleep_until(next_time: float, max_sleep: float = 0.05) -> float:
    """Sleep until next_time (monotonic seconds). Returns current monotonic time."""
    now = time.monotonic()
    if now < next_time:
        time.sleep(min(max_sleep, next_time - now))
        return time.monotonic()
    return now


def main() -> None:
    notifier = TelegramNotifier()

    # Load configuration for rotation angle
    rotation_angle = 0
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if key.strip() == 'ROTATION_ANGLE':
                        rotation_angle = int(value.strip())
                        break

    # Infinite retry loop for startup
    streamer = None
    connection_succeeded = False

    while True:
        # Start headless streamer
        streamer = EufyRTSPStreamer(rtsp_url="", headless=True, rotation_angle=rotation_angle)
        started = streamer.start_headless()
        
        if not started:
            logger.error("Failed to start camera stream; retrying in 80s")
            time.sleep(80)
            continue
        
        # Attempt to capture first frame with exponential backoff
        first_frame = None
        for wait_s in exponential_backoff_attempts():
            frame_bytes = streamer.get_current_frame()
            if frame_bytes:
                first_frame = decode_jpeg_to_bgr(frame_bytes)
                if first_frame is not None:
                    break
            logger.info("No frame yet; retrying in %ds", wait_s)
            time.sleep(wait_s)
        
        if first_frame is None:
            logger.error("Could not capture frame; retrying connection in 80s")
            streamer.stop_stream()
            time.sleep(80)
            continue
        
        # Success! Notify if this is a recovery
        if connection_succeeded:
            try:
                notifier.send_text("F3DPD: Camera connection restored successfully.")
            except Exception as e:
                logger.error("Failed to send reconnection notification: %s", e)
        
        connection_succeeded = True
        logger.info("Camera connected; building initial 30-second buffer before monitoring")
        break
    
    # Rolling frame buffer: store JPEG bytes (much smaller than decoded BGR)
    frame_buffer: Deque[bytes] = deque(maxlen=FRAME_BUFFER_SIZE)
    
    # Build initial buffer
    BUFFER_UPDATE_INTERVAL = 1.0 / TARGET_FPS  # ~33ms for 30 fps
    buffer_fill_start = time.monotonic()
    next_frame_at = time.monotonic()
    
    while len(frame_buffer) < FRAME_BUFFER_SIZE:
        now = _sleep_until(next_frame_at)
        if now < next_frame_at:
            continue
        next_frame_at = now + BUFFER_UPDATE_INTERVAL

        frame_bytes = streamer.get_current_frame()
        if frame_bytes:
            frame_buffer.append(frame_bytes)
        
        # Safety timeout: 45 seconds max for initial buffer
        if time.monotonic() - buffer_fill_start > 45:
            logger.warning("Initial buffer fill timeout; starting with %d frames", len(frame_buffer))
            break
    
    logger.info("Initial buffer ready with %d frames; starting AI monitoring", len(frame_buffer))
    
    # Monitoring loop with cooldown
    cooldown_until = 0.0
    next_check_at = 0.0
    CHECK_INTERVAL = 30.0
    COOLDOWN_SECONDS = 300.0
    next_frame_at = time.monotonic()
    
    while True:
        now = time.monotonic()
        
        # Update frame buffer at ~30 fps
        if now >= next_frame_at:
            next_frame_at = now + BUFFER_UPDATE_INTERVAL
            frame_bytes = streamer.get_current_frame()
            if frame_bytes:
                frame_buffer.append(frame_bytes)
        
        # Respect cooldown: skip checks during cooldown window
        if now < cooldown_until:
            _sleep_until(next_frame_at, max_sleep=0.02)
            continue
        
        # monotonic cadence for AI checks
        if now < next_check_at:
            _sleep_until(min(next_check_at, next_frame_at), max_sleep=0.02)
            continue
        next_check_at = now + CHECK_INTERVAL
        
        # Freeze the buffer - take snapshot of last 30 seconds (JPEG bytes)
        frozen_buffer: List[bytes] = list(frame_buffer)
        if len(frozen_buffer) < 10:
            logger.warning("Buffer too small for detection; skipping check")
            continue
        
        # Get the most recent frame for classification
        last_jpeg = frozen_buffer[-1]
        frame_bgr = decode_jpeg_to_bgr(last_jpeg)
        if frame_bgr is None:
            logger.warning("Failed to decode latest frame; skipping check")
            continue
        
        try:
            is_failed, score = predict_failed_from_bgr(frame_bgr)
        except Exception as e:
            logger.error("Prediction error: %s", e)
            continue
        
        if is_failed:
            logger.info("Failure detected (score: %.3f); sending 30-second video", score)
            
            # Use the frozen buffer (30 seconds up to detection moment)
            all_frames = frozen_buffer
            approx_mb = _estimate_jpeg_buffer_bytes(all_frames) / (1024 * 1024)
            logger.info("Creating 30s video from %d JPEG frames (~%.1f MiB)", len(all_frames), approx_mb)
            
            # Create temporary video file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                video_path = tmp.name
            
            if create_video_from_jpegs(all_frames, video_path, fps=TARGET_FPS):
                try:
                    with open(video_path, 'rb') as vf:
                        video_bytes = vf.read()
                    os.unlink(video_path)
                    
                    caption = "F3DPD has detected a failed print."
                    notifier.send_video(video_bytes, caption)
                    logger.info("Failure alert video sent")
                except Exception as e:
                    logger.error("Failed to send Telegram video: %s", e)
                    # Try cleaning up temp file
                    try:
                        os.unlink(video_path)
                    except:
                        pass
            else:
                logger.error("Failed to create video; skipping alert")
            
            cooldown_until = time.monotonic() + COOLDOWN_SECONDS
            logger.info("Entering cooldown for %.0fs", COOLDOWN_SECONDS)
        else:
            logger.info("Print status: Successful (score: %.3f)", score)


if __name__ == "__main__":
    main()
