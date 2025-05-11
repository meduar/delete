"""
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
            body += "\n\nAdditional Data:\n"
            for key, value in alert['data'].items():
                body += f"{key}: {value}\n"
        
        message.attach(MimeText(body, "plain"))
        
        return message
