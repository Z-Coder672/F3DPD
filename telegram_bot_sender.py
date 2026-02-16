#!/usr/bin/env python3
"""
Telegram Bot Message Sender
Uses a bot token for persistent, no-2FA authentication
Supports auto-registration via /register command
"""

import io
import json
import logging
import os
import threading
import time
from typing import Callable, Dict, List, Optional, Set

import requests

logger = logging.getLogger(__name__)

# Bot token from @BotFather - this never expires!
# Loaded from config file (more reliable than env vars at boot)
_BOT_TOKEN: Optional[str] = None
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'telegram_config.json')
WHITELIST_FILE = os.path.join(os.path.dirname(__file__), 'telegram_whitelist.json')
ENV_FILE = os.path.join(os.path.dirname(__file__), '.env')

_pending_registration: Dict[int, str] = {}
_frame_timestamps: Dict[int, List[float]] = {}
FRAME_RATE_LIMIT = 10
FRAME_RATE_WINDOW = 60.0

_status_timestamps: Dict[int, List[float]] = {}
STATUS_RATE_LIMIT = 15
STATUS_RATE_WINDOW = 60.0

_pause_resume_timestamps: Dict[int, List[float]] = {}
PAUSE_RESUME_RATE_LIMIT = 5
PAUSE_RESUME_RATE_WINDOW = 60.0

_listfiles_timestamps: Dict[int, List[float]] = {}
LISTFILES_RATE_LIMIT = 10
LISTFILES_RATE_WINDOW = 60.0

_print_timestamps: Dict[int, List[float]] = {}
PRINT_RATE_LIMIT = 10
PRINT_RATE_WINDOW = 60.0

_pending_print: Dict[int, bool] = {}


def _load_env_password() -> str:
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if key.strip() == 'REGISTER_PASSWORD':
                        return value.strip()
    return os.getenv('REGISTER_PASSWORD', '')


def load_whitelist() -> Set[int]:
    if os.path.exists(WHITELIST_FILE):
        try:
            with open(WHITELIST_FILE, 'r') as f:
                data = json.load(f)
            return set(data)
        except Exception:
            return set()
    return set()


def save_whitelist(wl: Set[int]) -> None:
    with open(WHITELIST_FILE, 'w') as f:
        json.dump(sorted(wl), f, indent=2)


def is_whitelisted(chat_id: int) -> bool:
    return chat_id in load_whitelist()


def get_bot_token() -> str:
    """Get bot token from config file, fallback to environment"""
    global _BOT_TOKEN
    if _BOT_TOKEN is None:
        # Try config file first (most reliable at boot)
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                _BOT_TOKEN = config.get('bot_token', '')
            except Exception as e:
                logger.warning("Failed to read config file: %s", e)
        
        # Fallback to environment variable
        if not _BOT_TOKEN:
            _BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
        
        if not _BOT_TOKEN:
            logger.error("Bot token not found in config file or TELEGRAM_BOT_TOKEN env var")
    return _BOT_TOKEN

# Chat IDs file - stores users who have messaged the bot
CHAT_IDS_FILE = os.path.join(os.path.dirname(__file__), 'telegram_chat_ids.json')

# Track last processed update to avoid duplicates
_last_update_id = 0
_polling_thread: Optional[threading.Thread] = None
_polling_stop = threading.Event()

# Callback for /frame command: signature (chat_id: int) -> None
_frame_callback: Optional[Callable[[int], None]] = None
_pause_callback: Optional[Callable[[int], None]] = None
_resume_callback: Optional[Callable[[int], None]] = None
_status_callback: Optional[Callable[[int], None]] = None
_listfiles_callback: Optional[Callable[[int], None]] = None
_print_callback: Optional[Callable[[int, str], None]] = None


def set_frame_callback(callback: Callable[[int], None]) -> None:
    """Register a callback invoked when a user sends /frame."""
    global _frame_callback
    _frame_callback = callback


def set_pause_callback(callback: Callable[[int], None]) -> None:
    global _pause_callback
    _pause_callback = callback


def set_resume_callback(callback: Callable[[int], None]) -> None:
    global _resume_callback
    _resume_callback = callback


def set_status_callback(callback: Callable[[int], None]) -> None:
    global _status_callback
    _status_callback = callback


def set_listfiles_callback(callback: Callable[[int], None]) -> None:
    global _listfiles_callback
    _listfiles_callback = callback


def set_print_callback(callback: Callable[[int, str], None]) -> None:
    global _print_callback
    _print_callback = callback


def load_chat_ids() -> Dict[str, int]:
    """Load saved chat IDs"""
    if os.path.exists(CHAT_IDS_FILE):
        try:
            with open(CHAT_IDS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_chat_ids(chat_ids: Dict[str, int]) -> None:
    """Save chat IDs"""
    with open(CHAT_IDS_FILE, 'w') as f:
        json.dump(chat_ids, f, indent=2)


def send_message(chat_id: int, message: str) -> bool:
    """
    Send a text message to a specific chat ID
    
    Args:
        chat_id: Telegram chat ID (user or group)
        message: Message text to send
    
    Returns:
        True if successful, False otherwise
    """
    token = get_bot_token()
    if not token:
        return False
        
    url = f'https://api.telegram.org/bot{token}/sendMessage'
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'HTML'
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        result = response.json()
        
        if result.get('ok'):
            return True
        else:
            logger.error("Telegram API error: %s", result.get('description', 'Unknown error'))
            return False
    except Exception as e:
        logger.error("Error sending message: %s", e)
        return False


def send_photo(chat_id: int, photo_bytes: bytes, caption: str = "") -> bool:
    """Send a photo to a specific chat ID"""
    token = get_bot_token()
    if not token:
        return False
        
    url = f'https://api.telegram.org/bot{token}/sendPhoto'
    
    try:
        files = {'photo': ('photo.jpg', io.BytesIO(photo_bytes), 'image/jpeg')}
        data = {'chat_id': chat_id, 'caption': caption}
        
        response = requests.post(url, data=data, files=files, timeout=60)
        result = response.json()
        
        if result.get('ok'):
            return True
        else:
            logger.error("Telegram API error: %s", result.get('description', 'Unknown error'))
            return False
    except Exception as e:
        logger.error("Error sending photo: %s", e)
        return False


def send_video(chat_id: int, video_bytes: bytes, caption: str = "") -> bool:
    """Send a video to a specific chat ID"""
    token = get_bot_token()
    if not token:
        return False
        
    url = f'https://api.telegram.org/bot{token}/sendVideo'
    
    try:
        files = {'video': ('video.mp4', io.BytesIO(video_bytes), 'video/mp4')}
        data = {'chat_id': chat_id, 'caption': caption, 'supports_streaming': True}
        
        response = requests.post(url, data=data, files=files, timeout=300)
        result = response.json()
        
        if result.get('ok'):
            return True
        else:
            logger.error("Telegram API error: %s", result.get('description', 'Unknown error'))
            return False
    except Exception as e:
        logger.error("Error sending video: %s", e)
        return False


def send_to_all(message: str) -> Dict[str, bool]:
    """Send a message to all registered AND whitelisted chat IDs"""
    chat_ids = load_chat_ids()
    wl = load_whitelist()
    results = {}
    
    for name, chat_id in chat_ids.items():
        if chat_id not in wl:
            continue
        success = send_message(chat_id, message)
        results[name] = success
        logger.info("%s Sent to %s (%s)", '✓' if success else '✗', name, chat_id)
    
    return results


def send_photo_to_all(photo_bytes: bytes, caption: str = "") -> Dict[str, bool]:
    """Send a photo to all registered AND whitelisted chat IDs"""
    chat_ids = load_chat_ids()
    wl = load_whitelist()
    results = {}
    
    for name, chat_id in chat_ids.items():
        if chat_id not in wl:
            continue
        success = send_photo(chat_id, photo_bytes, caption)
        results[name] = success
        logger.info("%s Sent photo to %s (%s)", '✓' if success else '✗', name, chat_id)
    
    return results


def send_video_to_all(video_bytes: bytes, caption: str = "") -> Dict[str, bool]:
    """Send a video to all registered AND whitelisted chat IDs"""
    chat_ids = load_chat_ids()
    wl = load_whitelist()
    results = {}
    
    for name, chat_id in chat_ids.items():
        if chat_id not in wl:
            continue
        success = send_video(chat_id, video_bytes, caption)
        results[name] = success
        logger.info("%s Sent video to %s (%s)", '✓' if success else '✗', name, chat_id)
    
    return results


def get_updates(offset: int = 0) -> List[dict]:
    """Get recent messages to the bot"""
    token = get_bot_token()
    if not token:
        return []
        
    url = f'https://api.telegram.org/bot{token}/getUpdates'
    params = {'timeout': 30}
    if offset:
        params['offset'] = offset
    
    try:
        response = requests.get(url, params=params, timeout=35)
        result = response.json()
        
        if result.get('ok'):
            return result.get('result', [])
        else:
            logger.error("Error getting updates: %s", result.get('description'))
            return []
    except Exception as e:
        logger.error("Error getting updates: %s", e)
        return []


def _check_rate_limit(chat_id: int, timestamps_dict: Dict[int, List[float]], limit: int, window: float) -> bool:
    """Return True if the request is allowed, False if rate-limited."""
    now = time.time()
    timestamps = timestamps_dict.get(chat_id, [])
    timestamps = [t for t in timestamps if now - t < window]
    if len(timestamps) >= limit:
        timestamps_dict[chat_id] = timestamps
        return False
    timestamps.append(now)
    timestamps_dict[chat_id] = timestamps
    return True


def process_update(update: dict) -> None:
    """Process a single update - handle /register and /unregister commands"""
    global _last_update_id
    
    update_id = update.get('update_id', 0)
    if update_id > _last_update_id:
        _last_update_id = update_id
    
    message = update.get('message', {})
    if not message:
        return
    
    chat = message.get('chat', {})
    chat_id = chat.get('id')
    if not chat_id:
        return
    
    text = message.get('text', '').strip()
    first_name = chat.get('first_name', '') or chat.get('title', '')
    username = message.get('from', {}).get('username', '')
    name = first_name or username or f'User_{chat_id}'
    
    chat_ids = load_chat_ids()
    wl = load_whitelist()

    if chat_id in _pending_print:
        _pending_print.pop(chat_id)
        if not text:
            send_message(chat_id, "No filename provided. /print cancelled.")
            return
        if text.startswith('/'):
            send_message(chat_id, "Print cancelled.")
        else:
            if _print_callback is not None:
                try:
                    _print_callback(chat_id, text)
                except Exception as e:
                    logger.error("/print failed for %s (ID: %s): %s", name, chat_id, e)
                    send_message(chat_id, "⚠️ Failed to start print.")
            else:
                send_message(chat_id, "⚠️ Printer control not available.")
            return

    if chat_id in _pending_registration:
        expected_password = _load_env_password()
        if text == expected_password:
            pending_name = _pending_registration.pop(chat_id)
            chat_ids[pending_name] = chat_id
            save_chat_ids(chat_ids)
            wl.add(chat_id)
            save_whitelist(wl)
            send_message(chat_id, f"✅ Registered! You'll now receive F3DPD print failure alerts.\n\nYour chat ID: <code>{chat_id}</code>")
            logger.info("Registered & whitelisted user: %s (ID: %s)", pending_name, chat_id)
        else:
            _pending_registration.pop(chat_id)
            send_message(chat_id, "❌ Incorrect password. Registration cancelled. Send /register to try again.")
            logger.info("Failed registration attempt for %s (ID: %s)", name, chat_id)
        return
    
    if text.lower() == '/register':
        if chat_id in wl and str(chat_id) in [str(v) for v in chat_ids.values()]:
            send_message(chat_id, "✅ You're already registered for F3DPD alerts!")
        else:
            _pending_registration[chat_id] = name
            send_message(chat_id, "🔐 Please reply with the registration password:")
            logger.info("Registration started for %s (ID: %s); awaiting password", name, chat_id)
    
    elif text.lower() == '/unregister':
        to_remove = [k for k, v in chat_ids.items() if v == chat_id]
        if to_remove:
            for k in to_remove:
                del chat_ids[k]
            save_chat_ids(chat_ids)
            wl.discard(chat_id)
            save_whitelist(wl)
            send_message(chat_id, "🔕 Unregistered. You'll no longer receive F3DPD alerts.")
            logger.info("Unregistered user: %s (ID: %s)", name, chat_id)
        else:
            send_message(chat_id, "You weren't registered.")
    
    elif text.lower() == '/frame':
        logger.info("/frame received from %s (ID: %s)", name, chat_id)
        if not is_whitelisted(chat_id):
            logger.warning("/frame rejected: %s (ID: %s) not whitelisted", name, chat_id)
            send_message(chat_id, "❌ You must /register first.")
            return
        if not _check_rate_limit(chat_id, _frame_timestamps, FRAME_RATE_LIMIT, FRAME_RATE_WINDOW):
            logger.warning("/frame rate-limited: %s (ID: %s)", name, chat_id)
            return
        if _frame_callback is not None:
            try:
                _frame_callback(chat_id)
            except Exception as e:
                logger.error("/frame failed for %s (ID: %s): %s", name, chat_id, e)
                send_message(chat_id, "⚠️ Failed to capture frame.")
        else:
            logger.warning("/frame: camera stream not available for %s (ID: %s)", name, chat_id)
            send_message(chat_id, "⚠️ Camera stream not available.")

    elif text.lower() == '/pause':
        logger.info("/pause received from %s (ID: %s)", name, chat_id)
        if not is_whitelisted(chat_id):
            send_message(chat_id, "❌ You must /register first.")
            return
        if not _check_rate_limit(chat_id, _pause_resume_timestamps, PAUSE_RESUME_RATE_LIMIT, PAUSE_RESUME_RATE_WINDOW):
            logger.warning("/pause rate-limited: %s (ID: %s)", name, chat_id)
            send_message(chat_id, "⚠️ Rate limited. Try again shortly.")
            return
        if _pause_callback is not None:
            try:
                _pause_callback(chat_id)
            except Exception as e:
                logger.error("/pause failed for %s (ID: %s): %s", name, chat_id, e)
                send_message(chat_id, "⚠️ Failed to send pause command.")
        else:
            send_message(chat_id, "⚠️ Printer control not available.")

    elif text.lower() == '/resume':
        logger.info("/resume received from %s (ID: %s)", name, chat_id)
        if not is_whitelisted(chat_id):
            send_message(chat_id, "❌ You must /register first.")
            return
        if not _check_rate_limit(chat_id, _pause_resume_timestamps, PAUSE_RESUME_RATE_LIMIT, PAUSE_RESUME_RATE_WINDOW):
            logger.warning("/resume rate-limited: %s (ID: %s)", name, chat_id)
            send_message(chat_id, "⚠️ Rate limited. Try again shortly.")
            return
        if _resume_callback is not None:
            try:
                _resume_callback(chat_id)
            except Exception as e:
                logger.error("/resume failed for %s (ID: %s): %s", name, chat_id, e)
                send_message(chat_id, "⚠️ Failed to send resume command.")
        else:
            send_message(chat_id, "⚠️ Printer control not available.")

    elif text.lower() == '/status':
        logger.info("/status received from %s (ID: %s)", name, chat_id)
        if not is_whitelisted(chat_id):
            send_message(chat_id, "❌ You must /register first.")
            return
        if not _check_rate_limit(chat_id, _status_timestamps, STATUS_RATE_LIMIT, STATUS_RATE_WINDOW):
            logger.warning("/status rate-limited: %s (ID: %s)", name, chat_id)
            send_message(chat_id, "⚠️ Rate limited. Try again shortly.")
            return
        if _status_callback is not None:
            try:
                _status_callback(chat_id)
            except Exception as e:
                logger.error("/status failed for %s (ID: %s): %s", name, chat_id, e)
                send_message(chat_id, "⚠️ Failed to get printer status.")
        else:
            send_message(chat_id, "⚠️ Printer control not available.")

    elif text.lower() == '/listfiles':
        logger.info("/listfiles received from %s (ID: %s)", name, chat_id)
        if not is_whitelisted(chat_id):
            send_message(chat_id, "❌ You must /register first.")
            return
        if not _check_rate_limit(chat_id, _listfiles_timestamps, LISTFILES_RATE_LIMIT, LISTFILES_RATE_WINDOW):
            logger.warning("/listfiles rate-limited: %s (ID: %s)", name, chat_id)
            send_message(chat_id, "⚠️ Rate limited. Try again shortly.")
            return
        if _listfiles_callback is not None:
            try:
                _listfiles_callback(chat_id)
            except Exception as e:
                logger.error("/listfiles failed for %s (ID: %s): %s", name, chat_id, e)
                send_message(chat_id, "⚠️ Failed to list files.")
        else:
            send_message(chat_id, "⚠️ Printer control not available.")

    elif text.lower() == '/print':
        logger.info("/print received from %s (ID: %s)", name, chat_id)
        if not is_whitelisted(chat_id):
            send_message(chat_id, "❌ You must /register first.")
            return
        if not _check_rate_limit(chat_id, _print_timestamps, PRINT_RATE_LIMIT, PRINT_RATE_WINDOW):
            logger.warning("/print rate-limited: %s (ID: %s)", name, chat_id)
            send_message(chat_id, "⚠️ Rate limited. Try again shortly.")
            return
        if _print_callback is not None:
            _pending_print[chat_id] = True
            send_message(chat_id, "📄 Reply with the filename to print (e.g. file.gcode):")
        else:
            send_message(chat_id, "⚠️ Printer control not available.")

    elif text.lower() == '/start':
        send_message(chat_id, 
            "Welcome to F3DPD Bot!\n\n"
            "Commands:\n"
            "/register - Subscribe to print failure alerts\n"
            "/unregister - Stop receiving alerts\n"
            "/frame - Get a live frame from the camera\n"
            "/status - Get printer status\n"
            "/pause - Pause the print\n"
            "/resume - Resume the print\n"
            "/listfiles - List files on the printer\n"
            "/print - Start printing a file"
        )


def _polling_loop() -> None:
    """Background polling loop for auto-registration"""
    global _last_update_id
    
    logger.info("Starting Telegram bot polling for /register commands...")
    
    # Wait for token to become available (may take time at boot)
    token = get_bot_token()
    retry_count = 0
    while not token and retry_count < 12:
        retry_count += 1
        logger.warning("Bot token not available, retrying in 5s... (%d/12)", retry_count)
        time.sleep(5)
        # Force re-read from environment
        global _BOT_TOKEN
        _BOT_TOKEN = None
        token = get_bot_token()
    
    if not token:
        logger.error("Bot token not available after retries, polling disabled")
        return
    
    logger.info("Bot token loaded (starts with: %s...)", token[:10] if len(token) > 10 else "???")
    
    # Get initial offset to skip old messages
    updates = get_updates()
    if updates:
        _last_update_id = updates[-1].get('update_id', 0)
    
    while not _polling_stop.is_set():
        try:
            updates = get_updates(offset=_last_update_id + 1)
            for update in updates:
                process_update(update)
        except Exception as e:
            logger.error("Polling error: %s", e)
            time.sleep(5)


def start_polling() -> None:
    """Start background thread for auto-registration polling"""
    global _polling_thread
    
    if _polling_thread is not None and _polling_thread.is_alive():
        return
    
    _polling_stop.clear()
    _polling_thread = threading.Thread(target=_polling_loop, daemon=True)
    _polling_thread.start()


def stop_polling() -> None:
    """Stop background polling thread"""
    _polling_stop.set()
    if _polling_thread is not None:
        _polling_thread.join(timeout=5)


if __name__ == '__main__':
    import sys
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if not get_bot_token():
        print("ERROR: Set TELEGRAM_BOT_TOKEN environment variable!")
        print("1. Message @BotFather on Telegram")
        print("2. Send /newbot and follow instructions")
        print("3. export TELEGRAM_BOT_TOKEN='your_token_here'")
        sys.exit(1)
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        
        if cmd == 'poll':
            # Run polling in foreground (for testing)
            print("Polling for /register commands... (Ctrl+C to stop)")
            print("Tell users to message @F3DPD_bot and send /register")
            try:
                _polling_loop()
            except KeyboardInterrupt:
                print("\nStopped.")
                
        elif cmd == 'list':
            chat_ids = load_chat_ids()
            print("Registered chat IDs:")
            for name, cid in chat_ids.items():
                print(f"  {name}: {cid}")
                
        elif cmd == 'add' and len(sys.argv) == 4:
            chat_ids = load_chat_ids()
            chat_ids[sys.argv[2]] = int(sys.argv[3])
            save_chat_ids(chat_ids)
            print(f"Added {sys.argv[2]}: {sys.argv[3]}")
            
        elif cmd == 'test':
            print("Sending test message to all registered users...")
            send_to_all("🔧 Test from F3DPD - Print detector is online!")
            
        else:
            # Send a message
            message = ' '.join(sys.argv[1:])
            send_to_all(message)
    else:
        print("Usage:")
        print("  python telegram_bot_sender.py poll     - Listen for /register commands")
        print("  python telegram_bot_sender.py list     - List registered chat IDs")
        print("  python telegram_bot_sender.py add NAME CHAT_ID  - Add a chat ID manually")
        print("  python telegram_bot_sender.py test     - Send test message to all")
        print("  python telegram_bot_sender.py MESSAGE  - Send message to all users")
