import asyncio
import time
from typing import Dict, Any, Optional
from collections import defaultdict, deque
from datetime import datetime
import psutil

class MetricsCollector:
    """Collect and track system and trading metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.metrics_history = defaultdict(lambda: deque(maxlen=max_history))
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.start_time = time.time()
        
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        timestamp = datetime.now()
        metric_data = {
            'timestamp': timestamp,
            'value': value,
            'tags': tags or {}
        }
        
        self.metrics_history[name].append(metric_data)
        self.gauges[name] = value
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter metric"""
        self.counters[name] += value
    
    def timing_decorator(self, metric_name: str):
        """Decorator to measure function execution time"""
        def decorator(func):
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                self.record_metric(metric_name, duration * 1000)  # Convert to milliseconds
                return result
            
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                self.record_metric(metric_name, duration * 1000)
                return result
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        metrics = {
            'system.cpu.usage': cpu_usage,
            'system.memory.usage': memory.percent,
            'system.memory.available': memory.available,
            'system.uptime': time.time() - self.start_time
        }
        
        # Record system metrics
        for name, value in metrics.items():
            self.record_metric(name, value)
        
        return metrics
    
    def get_trading_metrics(self) -> Dict[str, Any]:
        """Get current trading metrics"""
        return {
            'trades.total': self.counters['trades.total'],
            'trades.winning': self.counters['trades.winning'],
            'trades.losing': self.counters['trades.losing'],
            'signals.generated': self.counters['signals.generated'],
            'orders.placed': self.counters['orders.placed'],
            'orders.filled': self.counters['orders.filled'],
            'current.equity': self.gauges.get('portfolio.equity', 0),
            'current.drawdown': self.gauges.get('portfolio.drawdown', 0)
        }
    
    def get_metric_summary(self, metric_name: str) -> Dict[str, float]:
        """Get statistical summary of a metric"""
        if metric_name not in self.metrics_history:
            return {}
        
        values = [m['value'] for m in self.metrics_history[metric_name]]
        if not values:
            return {}
        
        return {
            'count': len(values),
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'last': values[-1],
            'std': self._calculate_std(values)
        }
    
    def _calculate_std(self, values: list) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def export_metrics(self, output_file: str):
        """Export all metrics to JSON file"""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'history_summary': {
                name: self.get_metric_summary(name)
                for name in self.metrics_history.keys()
            }
        }
        
        with open(output_file, 'w') as f:
            import json
            json.dump(export_data, f, indent=2)