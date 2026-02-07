#!/usr/bin/env python3
"""
Telegram Bot notifier utilities for F3DPD

Uses Telegram Bot API for persistent, no-2FA authentication.
Sends messages to all registered chat IDs.
"""

import logging
from typing import List

import telegram_bot_sender

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Notifier that sends alerts to all registered Telegram users via Bot API"""
    
    def __init__(self, start_polling: bool = True) -> None:
        """
        Initialize the notifier.
        
        Args:
            start_polling: If True, start background polling for /register commands
        """
        if start_polling:
            telegram_bot_sender.start_polling()
    
    def send_text(self, text: str) -> None:
        """Send text message to all registered users"""
        telegram_bot_sender.send_to_all(text)
    
    def send_photo(self, photo_bytes: bytes, caption: str = "") -> None:
        """Send photo to all registered users"""
        telegram_bot_sender.send_photo_to_all(photo_bytes, caption)
    
    def send_video(self, video_bytes: bytes, caption: str = "") -> None:
        """Send video to all registered users"""
        telegram_bot_sender.send_video_to_all(video_bytes, caption)
    
    # Legacy methods for backward compatibility with orchestrator.py
    def send_text_to_phones(self, phones: List[str], text: str) -> None:
        """Legacy: phones parameter is ignored; sends to all registered chat IDs"""
        self.send_text(text)
    
    def send_photo_to_phones(self, phones: List[str], photo_bytes: bytes, caption: str) -> None:
        """Legacy: phones parameter is ignored; sends to all registered chat IDs"""
        self.send_photo(photo_bytes, caption)
    
    def send_video_to_phones(self, phones: List[str], video_bytes: bytes, caption: str) -> None:
        """Legacy: phones parameter is ignored; sends to all registered chat IDs"""
        self.send_video(video_bytes, caption)
