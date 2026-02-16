#!/usr/bin/env python3
"""
F3DPD Orchestrator: headless stream + AI + Telegram DM

Behavior:
- Start the camera streamer headlessly with infinite retries (80s interval after initial backoff).
- Maintain a rolling 10-second buffer of frames at 30 fps (300 frames).
- Every 10s (time.time scheduling), run AI classifier on current frame.
- On failure detection: stitch a 10s video (buffer leading up to detection), send to Telegram.
- After alert, enter 5-minute cooldown.
- Send success notification when connection is established after failures.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
import tempfile
from collections import deque
from typing import Deque, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from rpi_ws281x import PixelStrip, Color

from eufy_rtsp_streamer import EufyRTSPStreamer
from notifier import TelegramNotifier
from predictor_clip import predict_failed_from_bgr
from printer_gcode import send_pause, send_resume, send_m119, send_m27, send_m661, send_m23, parse_m119, parse_m27, parse_m661, parse_m23
import telegram_bot_sender


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


FRAME_BUFFER_SIZE = 300  # 10 seconds at 30 fps
TARGET_FPS = 30

# --- WS2812B LED strip configuration ---
LED_COUNT = 8
LED_PIN = 18
LED_FREQ_HZ = 800000
LED_DMA = 10
LED_BRIGHTNESS = 255
LED_INVERT = False
LED_CHANNEL = 0
PURE_WHITE = Color(255, 255, 255)
GREYSCALE_INFERENCES_BEFORE_RECHECK = 12
GREYSCALE_SATURATION_THRESHOLD = 3


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


def is_frame_greyscale(frame_bgr: np.ndarray) -> bool:
    b, g, r = cv2.split(frame_bgr)
    b = b.astype(np.int16)
    g = g.astype(np.int16)
    r = r.astype(np.int16)
    max_ch = np.maximum(np.maximum(r, g), b)
    min_ch = np.minimum(np.minimum(r, g), b)
    mean_saturation = float(np.mean(max_ch - min_ch))
    return mean_saturation < GREYSCALE_SATURATION_THRESHOLD


def _init_led_strip() -> PixelStrip:
    strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA,
                       LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
    strip.begin()
    return strip


def _leds_on(strip: PixelStrip) -> None:
    for i in range(LED_COUNT):
        strip.setPixelColor(i, PURE_WHITE)
    strip.show()


def _leds_off(strip: PixelStrip) -> None:
    for i in range(LED_COUNT):
        strip.setPixelColor(i, Color(0, 0, 0))
    strip.show()


def _grab_bgr(streamer, frame_buffer: Deque[bytes]) -> Optional[np.ndarray]:
    jpeg = streamer.get_current_frame()
    if jpeg is None and frame_buffer:
        jpeg = frame_buffer[-1]
    if jpeg is None:
        return None
    return decode_jpeg_to_bgr(jpeg)


def _wait_updating_buffer(
    seconds: float,
    streamer,
    frame_buffer: Deque[bytes],
    buffer_interval: float,
) -> None:
    end = time.monotonic() + seconds
    next_frame = time.monotonic()
    while True:
        now = time.monotonic()
        if now >= end:
            break
        if now >= next_frame:
            next_frame = now + buffer_interval
            fb = streamer.get_current_frame()
            if fb:
                frame_buffer.append(fb)
        time.sleep(min(0.02, max(0, end - time.monotonic())))


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
        logger.info("Camera connected; building initial 10-second buffer before monitoring")
        break
    
    # Initialize LED strip and ensure lights are off at start
    strip = _init_led_strip()
    _leds_off(strip)
    logger.info("LED strip initialized and turned off")

    # Rolling frame buffer: store JPEG bytes (much smaller than decoded BGR)
    frame_buffer: Deque[bytes] = deque(maxlen=FRAME_BUFFER_SIZE)

    def _on_frame_request(chat_id: int) -> None:
        """Handle /frame command: send the latest frame to the requesting user."""
        jpeg = streamer.get_current_frame()
        if jpeg is None and frame_buffer:
            jpeg = frame_buffer[-1]
        if jpeg is None:
            logger.warning("/frame: no frame available for chat_id %s", chat_id)
            telegram_bot_sender.send_message(chat_id, "⚠️ No frame available yet.")
            return
        success = telegram_bot_sender.send_photo(chat_id, jpeg, caption="F3DPD: Live frame")
        if success:
            logger.info("/frame: sent frame to chat_id %s", chat_id)
        else:
            logger.error("/frame: failed to send frame to chat_id %s (timed out or API error)", chat_id)

    telegram_bot_sender.set_frame_callback(_on_frame_request)

    def _verify_pause_resume(chat_id: int, action: str, delays: List[int]) -> None:
        """Background verification loop for pause/resume commands."""
        if action == "pause":
            expect_move_mode = "PAUSED"
        else:
            expect_machine = "BUILDING_FROM_SD"

        for delay in delays:
            time.sleep(delay)
            raw = send_m119()
            if raw is None:
                telegram_bot_sender.send_message(chat_id, "Failed to query printer status.")
                return
            parsed = parse_m119(raw)
            if parsed is None:
                telegram_bot_sender.send_message(chat_id, "Failed to parse printer status.")
                return

            if action == "pause" and parsed["move_mode"] == expect_move_mode:
                telegram_bot_sender.send_message(chat_id, "Printer paused successfully.")
                return
            if action == "resume" and parsed["machine_status"] == expect_machine and parsed["move_mode"] != "PAUSED":
                telegram_bot_sender.send_message(chat_id, "Printer resumed successfully.")
                return

            remaining = delays[delays.index(delay) + 1:]
            if remaining:
                next_delay = remaining[0]
                telegram_bot_sender.send_message(
                    chat_id,
                    f"Command sent, but printer not {action}d. Checking again in {next_delay} seconds."
                )
            else:
                telegram_bot_sender.send_message(
                    chat_id,
                    f"Command sent, but printer not {action}d. Please check the printer manually."
                )

    def _on_pause_request(chat_id: int) -> None:
        send_pause()
        threading.Thread(
            target=_verify_pause_resume,
            args=(chat_id, "pause", [5, 10, 20]),
            daemon=True,
        ).start()

    def _on_resume_request(chat_id: int) -> None:
        raw = send_m119()
        if raw is not None:
            parsed = parse_m119(raw)
            if parsed is not None:
                ms = parsed["machine_status"]
                mm = parsed["move_mode"]
                if ms == "BUILDING_FROM_SD" and mm != "PAUSED":
                    telegram_bot_sender.send_message(chat_id, "Printer already printing.")
                    return
                if ms == "READY":
                    telegram_bot_sender.send_message(chat_id, "Printer not paused. Status: Ready.")
                    return
        send_resume()
        threading.Thread(
            target=_verify_pause_resume,
            args=(chat_id, "resume", [5, 10, 20]),
            daemon=True,
        ).start()

    def _on_status_request(chat_id: int) -> None:
        raw_m119 = send_m119()
        raw_m27 = send_m27()

        if raw_m119 is None:
            telegram_bot_sender.send_message(chat_id, "Failed to query printer (M119).")
            return
        parsed_119 = parse_m119(raw_m119)
        if parsed_119 is None:
            telegram_bot_sender.send_message(chat_id, "Failed to parse printer status.")
            return

        move_mode = parsed_119["move_mode"]
        machine_status = parsed_119["machine_status"]
        current_file = parsed_119["current_file"]

        is_printing = machine_status == "BUILDING_FROM_SD"

        if move_mode == "PAUSED":
            display_mode = "Paused"
        elif is_printing:
            display_mode = "Printing"
        else:
            display_mode = machine_status[0].upper() + machine_status[1:].lower().replace("_", " ") if machine_status else "Unknown"

        lines = [display_mode]

        if is_printing or move_mode == "PAUSED":
            if current_file:
                lines.append(f"File: {current_file}")
            if raw_m27 is not None:
                parsed_27 = parse_m27(raw_m27)
                if parsed_27 is not None:
                    lines.append(f"Layer: {parsed_27['layer_current']}/{parsed_27['layer_total']}")

        telegram_bot_sender.send_message(chat_id, "\n".join(lines))

    def _on_listfiles_request(chat_id: int) -> None:
        raw = send_m661()
        if raw is None:
            telegram_bot_sender.send_message(chat_id, "Failed to query printer file list.")
            return
        files = parse_m661(raw)
        if not files:
            telegram_bot_sender.send_message(chat_id, "No files found on printer.")
            return
        telegram_bot_sender.send_message(chat_id, "\n".join(files))

    def _verify_print_started(chat_id: int, filename: str, delays: List[int]) -> None:
        for delay in delays:
            time.sleep(delay)
            raw = send_m119()
            if raw is None:
                telegram_bot_sender.send_message(chat_id, "Failed to query printer status.")
                return
            parsed = parse_m119(raw)
            if parsed is None:
                telegram_bot_sender.send_message(chat_id, "Failed to parse printer status.")
                return

            if parsed["machine_status"] == "BUILDING_FROM_SD":
                telegram_bot_sender.send_message(chat_id, f"Printing started: {filename}")
                return

            remaining = delays[delays.index(delay) + 1:]
            if remaining:
                next_delay = remaining[0]
                telegram_bot_sender.send_message(
                    chat_id,
                    f"Print command sent, but printer not printing yet. Checking again in {next_delay}s."
                )
            else:
                telegram_bot_sender.send_message(
                    chat_id,
                    "Print command sent, but printer did not start. Please check the printer manually."
                )

    def _on_print_request(chat_id: int, filename: str) -> None:
        filepath = f"0:/user/{filename}"
        raw = send_m23(filepath)
        if raw is None:
            telegram_bot_sender.send_message(chat_id, "Failed to send print command to printer.")
            return
        parsed = parse_m23(raw)
        if parsed is None:
            telegram_bot_sender.send_message(chat_id, "Failed to parse printer response.")
            return
        if parsed["size"] == 0:
            telegram_bot_sender.send_message(chat_id, f"File not found: {filename}")
            return
        telegram_bot_sender.send_message(
            chat_id,
            f"File selected: {parsed['filename']} ({parsed['size']} bytes). Verifying print starts..."
        )
        threading.Thread(
            target=_verify_print_started,
            args=(chat_id, filename, [5, 10, 20]),
            daemon=True,
        ).start()

    telegram_bot_sender.set_pause_callback(_on_pause_request)
    telegram_bot_sender.set_resume_callback(_on_resume_request)
    telegram_bot_sender.set_status_callback(_on_status_request)
    telegram_bot_sender.set_listfiles_callback(_on_listfiles_request)
    telegram_bot_sender.set_print_callback(_on_print_request)

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
    confirmation_at = 0.0
    next_check_at = 0.0
    CHECK_INTERVAL = 10.0
    COOLDOWN_SECONDS = 300.0
    next_frame_at = time.monotonic()

    # LED / greyscale state
    led_on = False
    led_on_time = 0.0
    inferences_with_led = 0
    
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
            # Confirmation check 10s into cooldown
            if confirmation_at > 0 and now >= confirmation_at:
                confirmation_at = 0.0
                confirm_jpeg = streamer.get_current_frame()
                if confirm_jpeg is None and frame_buffer:
                    confirm_jpeg = frame_buffer[-1]
                if confirm_jpeg is not None:
                    confirm_bgr = decode_jpeg_to_bgr(confirm_jpeg)
                    if confirm_bgr is not None:
                        try:
                            confirm_failed, confirm_score = predict_failed_from_bgr(confirm_bgr)
                        except Exception as e:
                            logger.error("Confirmation prediction error: %s", e)
                            confirm_failed = False
                            confirm_score = 0.0
                        if confirm_failed:
                            logger.info("Confirmation check ALSO failed (score: %.3f); auto-pausing print", confirm_score)
                            send_pause()
                            notifier.send_text(
                                "F3DPD repeatedly detected failure. Print has been auto-paused."
                            )
                        else:
                            logger.info("Confirmation check passed (score: %.3f); no auto-pause", confirm_score)
            _sleep_until(next_frame_at, max_sleep=0.02)
            continue
        elif cooldown_until > 0:
            logger.info("Cooldown ended; resuming AI monitoring")
            cooldown_until = 0.0
        
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
        
        # --- Greyscale / LED logic (before inference) ---
        if is_frame_greyscale(frame_bgr):
            if not led_on:
                logger.info("Frame is greyscale (IR mode); turning on LEDs")
                _leds_on(strip)
                led_on = True
                led_on_time = time.monotonic()
                inferences_with_led = 0

                _wait_updating_buffer(5, streamer, frame_buffer, BUFFER_UPDATE_INTERVAL)
                check = _grab_bgr(streamer, frame_buffer)
                if check is not None and is_frame_greyscale(check):
                    logger.info("Still greyscale after 5s; waiting 5 more")
                    _wait_updating_buffer(5, streamer, frame_buffer, BUFFER_UPDATE_INTERVAL)
                    check = _grab_bgr(streamer, frame_buffer)
                    if check is not None and is_frame_greyscale(check):
                        logger.info("Still greyscale after 10s; waiting 5 more")
                        _wait_updating_buffer(5, streamer, frame_buffer, BUFFER_UPDATE_INTERVAL)

                frame_bgr = _grab_bgr(streamer, frame_buffer)
                if frame_bgr is None:
                    continue
                next_check_at = time.monotonic() + CHECK_INTERVAL
            elif time.monotonic() - led_on_time < 15:
                logger.debug("LED on < 15s and frame still greyscale; skipping inference")
                continue

        try:
            is_failed, score = predict_failed_from_bgr(frame_bgr)
        except Exception as e:
            logger.error("Prediction error: %s", e)
            continue
        
        # --- Post-inference LED recheck (every 12 inferences) ---
        if led_on:
            inferences_with_led += 1
            if inferences_with_led >= GREYSCALE_INFERENCES_BEFORE_RECHECK:
                logger.info("12 inferences with LED on; turning off to recheck ambient light")
                _leds_off(strip)
                led_on = False
                inferences_with_led = 0
                _wait_updating_buffer(15, streamer, frame_buffer, BUFFER_UPDATE_INTERVAL)
                recheck = _grab_bgr(streamer, frame_buffer)
                if recheck is not None and is_frame_greyscale(recheck):
                    logger.info("Still greyscale after recheck; turning LEDs back on")
                    _leds_on(strip)
                    led_on = True
                    led_on_time = time.monotonic()
                    _wait_updating_buffer(5, streamer, frame_buffer, BUFFER_UPDATE_INTERVAL)
                else:
                    logger.info("Ambient light sufficient; LEDs staying off")
                next_check_at = time.monotonic() + CHECK_INTERVAL

        if is_failed:
            logger.info("Failure detected (score: %.3f); sending 10-second video", score)
            
            # Use the frozen buffer (10 seconds up to detection moment)
            all_frames = frozen_buffer
            approx_mb = _estimate_jpeg_buffer_bytes(all_frames) / (1024 * 1024)
            logger.info("Creating 10s video from %d JPEG frames (~%.1f MiB)", len(all_frames), approx_mb)
            
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
            confirmation_at = time.monotonic() + 10.0
            logger.info("Entering cooldown for %.0fs; confirmation check in 10s", COOLDOWN_SECONDS)
        else:
            logger.info("Print status: Successful (score: %.3f)", score)


if __name__ == "__main__":
    main()
