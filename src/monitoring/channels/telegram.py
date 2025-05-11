"""
Telegram Alert Channel
"""

import aiohttp
import logging
from typing import Dict
from ..alerts import AlertChannel


class TelegramChannel(AlertChannel):
    """Telegram notification channel"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.logger = logging.getLogger(__name__)
        self.api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    async def send_alert(self, alert: Dict):
        """Send alert via Telegram"""
        message = self.format_message(alert)
        
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"Telegram API error: {error_text}")
                        raise Exception(f"Telegram API returned {response.status}")
        except Exception as e:
            self.logger.error(f"Failed to send Telegram alert: {e}")
            raise
