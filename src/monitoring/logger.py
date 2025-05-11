import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

class TradingLogger:
    """Enhanced logging system for the trading framework"""
    
    def __init__(self, log_level: str = "INFO", log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Set up main logger
        self.logger = logging.getLogger("trading_framework")
        self.logger.setLevel(getattr(logging, log_level))
        
        # Prevent duplicate logs
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handlers
        self._setup_file_handlers()
        
        # Structured logs
        self.structured_logs = []
        
    def _setup_file_handlers(self):
        """Set up file handlers for different log levels"""
        
        # Main log file
        main_handler = logging.FileHandler(self.log_dir / "trading.log")
        main_handler.setLevel(logging.DEBUG)
        main_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'
        )
        main_handler.setFormatter(main_format)
        self.logger.addHandler(main_handler)
        
        # Error log file
        error_handler = logging.FileHandler(self.log_dir / "errors.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(main_format)
        self.logger.addHandler(error_handler)
        
        # Trade log file
        self.trade_logger = logging.getLogger("trading_framework.trades")
        trade_handler = logging.FileHandler(self.log_dir / "trades.log")
        trade_format = logging.Formatter('%(asctime)s - %(message)s')
        trade_handler.setFormatter(trade_format)
        self.trade_logger.addHandler(trade_handler)
        self.trade_logger.setLevel(logging.INFO)
        
        # Performance log file
        self.perf_logger = logging.getLogger("trading_framework.performance")
        perf_handler = logging.FileHandler(self.log_dir / "performance.log")
        perf_handler.setFormatter(trade_format)
        self.perf_logger.addHandler(perf_handler)
        self.perf_logger.setLevel(logging.INFO)
    
    def log_trade(self, trade_data: Dict[str, Any]):
        """Log trade information"""
        trade_log = {
            "timestamp": datetime.now().isoformat(),
            "type": "TRADE",
            **trade_data
        }
        
        self.trade_logger.info(json.dumps(trade_log))
        self.structured_logs.append(trade_log)
        
        # Also log to main logger
        self.logger.info(f"Trade executed: {trade_data['symbol']} {trade_data['action']} @ {trade_data['price']}")
    
    def log_signal(self, signal_data: Dict[str, Any]):
        """Log trading signal"""
        signal_log = {
            "timestamp": datetime.now().isoformat(),
            "type": "SIGNAL",
            **signal_data
        }
        
        self.structured_logs.append(signal_log)
        self.logger.debug(f"Signal generated: {signal_data}")
    
    def log_performance(self, metrics: Dict[str, float]):
        """Log performance metrics"""
        perf_log = {
            "timestamp": datetime.now().isoformat(),
            "type": "PERFORMANCE",
            **metrics
        }
        
        self.perf_logger.info(json.dumps(perf_log))
        self.structured_logs.append(perf_log)
    
    def log_risk_event(self, event: Dict[str, Any]):
        """Log risk management events"""
        risk_log = {
            "timestamp": datetime.now().isoformat(),
            "type": "RISK_EVENT",
            **event
        }
        
        self.logger.warning(f"Risk event: {event}")
        self.structured_logs.append(risk_log)
    
    def export_structured_logs(self, output_file: str = None):
        """Export structured logs to JSON file"""
        if output_file is None:
            output_file = self.log_dir / f"structured_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.structured_logs, f, indent=2)
        
        return output_file
