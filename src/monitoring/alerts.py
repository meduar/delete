"""
Alert Management Module
Handles notifications via various channels (Telegram, Email, etc.)
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum


class AlertPriority(Enum):
    """Alert priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts"""
    TRADE_EXECUTED = "trade_executed"
    POSITION_CLOSED = "position_closed"
    RISK_LIMIT_REACHED = "risk_limit_reached"
    SYSTEM_ERROR = "system_error"
    STRATEGY_SIGNAL = "strategy_signal"
    ACCOUNT_BALANCE = "account_balance"
    MARKET_EVENT = "market_event"


class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize notification channels
        self.channels = {}
        
        # Telegram channel
        if self.config.get('telegram', {}).get('enabled', False):
            from .channels.telegram import TelegramChannel
            self.channels['telegram'] = TelegramChannel(
                bot_token=self.config['telegram']['bot_token'],
                chat_id=self.config['telegram']['chat_id']
            )
        
        # Email channel (if configured)
        if self.config.get('email', {}).get('enabled', False):
            from .channels.email import EmailChannel
            self.channels['email'] = EmailChannel(self.config['email'])
        
        # Alert history
        self.alert_history: List[Dict] = []
        self.max_history = self.config.get('max_history', 1000)
        
        # Rate limiting
        self.rate_limits = self.config.get('rate_limits', {})
        self.last_sent = {}
    
    async def send_alert(
        self,
        alert_type: AlertType,
        priority: AlertPriority,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """Send an alert through configured channels"""
        
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type.value,
            'priority': priority.value,
            'message': message,
            'data': data or {}
        }
        
        # Add to history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]
        
        # Check rate limits
        if not self._check_rate_limit(alert_type, priority):
            self.logger.debug(f"Alert rate limited: {alert_type.value}")
            return
        
        # Send through channels
        for channel_name, channel in self.channels.items():
            try:
                if self._should_send_to_channel(channel_name, priority):
                    await channel.send_alert(alert)
                    self.logger.debug(f"Alert sent via {channel_name}: {message}")
            except Exception as e:
                self.logger.error(f"Failed to send alert via {channel_name}: {e}")
        
        # Update rate limit tracker
        self._update_rate_limit(alert_type, priority)
    
    def _check_rate_limit(self, alert_type: AlertType, priority: AlertPriority) -> bool:
        """Check if alert exceeds rate limit"""
        key = f"{alert_type.value}_{priority.value}"
        
        if key not in self.rate_limits:
            return True
        
        limit_config = self.rate_limits[key]
        max_per_minute = limit_config.get('max_per_minute', 60)
        
        now = datetime.now()
        last_sent_times = self.last_sent.get(key, [])
        
        # Remove old entries (older than 1 minute)
        recent_times = [t for t in last_sent_times if (now - t).total_seconds() < 60]
        
        return len(recent_times) < max_per_minute
    
    def _update_rate_limit(self, alert_type: AlertType, priority: AlertPriority):
        """Update rate limit tracker"""
        key = f"{alert_type.value}_{priority.value}"
        now = datetime.now()
        
        if key not in self.last_sent:
            self.last_sent[key] = []
        
        self.last_sent[key].append(now)
        
        # Clean up old entries
        self.last_sent[key] = [
            t for t in self.last_sent[key] 
            if (now - t).total_seconds() < 60
        ]
    
    def _should_send_to_channel(self, channel_name: str, priority: AlertPriority) -> bool:
        """Check if alert should be sent to specific channel based on priority"""
        channel_config = self.config.get('channels', {}).get(channel_name, {})
        min_priority = AlertPriority(channel_config.get('min_priority', AlertPriority.LOW.value))
        
        priority_order = {
            AlertPriority.LOW: 0,
            AlertPriority.MEDIUM: 1,
            AlertPriority.HIGH: 2,
            AlertPriority.CRITICAL: 3
        }
        
        return priority_order[priority] >= priority_order[min_priority]
    
    async def send_trade_alert(self, trade_info: Dict):
        """Send trade execution alert"""
        message = f"Trade executed: {trade_info['action']} {trade_info['symbol']} @ {trade_info['price']}"
        await self.send_alert(
            AlertType.TRADE_EXECUTED,
            AlertPriority.MEDIUM,
            message,
            trade_info
        )
    
    async def send_risk_alert(self, risk_event: Dict):
        """Send risk-related alert"""
        message = f"Risk alert: {risk_event['reason']}"
        priority = AlertPriority.HIGH if risk_event.get('severity', 'medium') == 'high' else AlertPriority.MEDIUM
        
        await self.send_alert(
            AlertType.RISK_LIMIT_REACHED,
            priority,
            message,
            risk_event
        )
    
    async def send_error_alert(self, error_info: Dict):
        """Send system error alert"""
        message = f"System error: {error_info['message']}"
        await self.send_alert(
            AlertType.SYSTEM_ERROR,
            AlertPriority.HIGH,
            message,
            error_info
        )
    
    async def send_balance_alert(self, balance_info: Dict):
        """Send account balance alert"""
        message = f"Account balance: ${balance_info['balance']:.2f}"
        priority = AlertPriority.CRITICAL if balance_info.get('below_threshold', False) else AlertPriority.LOW
        
        await self.send_alert(
            AlertType.ACCOUNT_BALANCE,
            priority,
            message,
            balance_info
        )
    
    def get_alert_history(self, limit: int = 100) -> List[Dict]:
        """Get recent alert history"""
        return self.alert_history[-limit:] if limit > 0 else self.alert_history
    
    def get_alert_stats(self) -> Dict:
        """Get statistics about sent alerts"""
        stats = {
            'total_alerts': len(self.alert_history),
            'by_type': {},
            'by_priority': {},
            'by_channel': {}
        }
        
        for alert in self.alert_history:
            alert_type = alert['type']
            priority = alert['priority']
            
            stats['by_type'][alert_type] = stats['by_type'].get(alert_type, 0) + 1
            stats['by_priority'][priority] = stats['by_priority'].get(priority, 0) + 1
        
        return stats


# Base alert channel class
class AlertChannel:
    """Base class for alert notification channels"""
    
    async def send_alert(self, alert: Dict):
        """Send alert through this channel"""
        raise NotImplementedError
    
    def format_message(self, alert: Dict) -> str:
        """Format alert message for this channel"""
        timestamp = alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        priority_emoji = {
            'low': '🔵',
            'medium': '🟡',
            'high': '🟠',
            'critical': '🔴'
        }
        
        emoji = priority_emoji.get(alert['priority'], '⚪')
        
        return f"{emoji} {timestamp} - {alert['message']}"


# Create empty files for channels
import os
os.makedirs('src/monitoring/channels', exist_ok=True)

# Create __init__.py for channels
channels_init = '''"""
Alert Channels Package
"""
'''

with open('src/monitoring/channels/__init__.py', 'w') as f:
    f.write(channels_init)

# Create telegram channel
telegram_channel = '''"""
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
'''

with open('src/monitoring/channels/telegram.py', 'w') as f:
    f.write(telegram_channel)

# Create email channel
email_channel = '''"""
Email Alert Channel
"""

import smtplib
import ssl
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import logging
from typing import Dict
from ..alerts import AlertChannel


class EmailChannel(AlertChannel):
    """Email notification channel"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.smtp_server = config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = config.get('smtp_port', 587)
        self.sender_email = config['sender_email']
        self.sender_password = config['sender_password']
        self.recipient_email = config['recipient_email']
        self.use_tls = config.get('use_tls', True)
    
    async def send_alert(self, alert: Dict):
        """Send alert via email"""
        message = self.format_email_message(alert)
        
        try:
            context = ssl.create_default_context()
            
            if self.use_tls:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                server.starttls(context=context)
            else:
                server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context)
            
            server.login(self.sender_email, self.sender_password)
            server.send_message(message)
            server.quit()
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
            raise
    
    def format_email_message(self, alert: Dict) -> MimeMultipart:
        """Format alert as email message"""
        message = MimeMultipart()
        message["From"] = self.sender_email
        message["To"] = self.recipient_email
        message["Subject"] = f"Trading Alert: {alert['type']} - {alert['priority'].upper()}"
        
        body = self.format_message(alert)
        
        # Add data if available
        if alert.get('data'):
            body += "\\n\\nAdditional Data:\\n"
            for key, value in alert['data'].items():
                body += f"{key}: {value}\\n"
        
        message.attach(MimeText(body, "plain"))
        
        return message
'''

with open('src/monitoring/channels/email.py', 'w') as f:
    f.write(email_channel)