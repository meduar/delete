"""
Configuration Management Module
Handles loading and managing configuration from YAML files and environment variables
"""

import yaml
import os
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging

class Config:
    """Configuration management with environment variable support"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load_config()
        
    def load_config(self) -> None:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file) or {}
            self._resolve_env_vars()
            logging.info(f"Configuration loaded from {self.config_path}")
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {self.config_path}")
            self.config = {}
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML config: {e}")
            self.config = {}
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            self.config = {}
            
    def _resolve_env_vars(self) -> None:
        """Resolve environment variables in config values"""
        self.config = self._resolve_dict(self.config)
        
    def _resolve_dict(self, d: Dict) -> Dict:
        """Recursively resolve environment variables in dictionary"""
        resolved = {}
        for key, value in d.items():
            if isinstance(value, dict):
                resolved[key] = self._resolve_dict(value)
            elif isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                env_value = os.getenv(env_var)
                if env_value is None:
                    logging.warning(f"Environment variable {env_var} not found, using original value")
                    resolved[key] = value
                else:
                    # Try to convert to appropriate type
                    resolved[key] = self._convert_env_value(env_value)
            elif isinstance(value, list):
                resolved[key] = self._resolve_list(value)
            else:
                resolved[key] = value
        return resolved
    
    def _resolve_list(self, lst: list) -> list:
        """Recursively resolve environment variables in list"""
        resolved = []
        for item in lst:
            if isinstance(item, dict):
                resolved.append(self._resolve_dict(item))
            elif isinstance(item, str) and item.startswith('${') and item.endswith('}'):
                env_var = item[2:-1]
                env_value = os.getenv(env_var)
                if env_value is None:
                    logging.warning(f"Environment variable {env_var} not found, using original value")
                    resolved.append(item)
                else:
                    resolved.append(self._convert_env_value(env_value))
            else:
                resolved.append(item)
        return resolved
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable value to appropriate type"""
        # Try to convert to number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Try to convert to boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        elif value.lower() in ('false', 'no', '0'):
            return False
        
        # Return as string
        return value
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default"""
        return self.config.get(key, default)
        

    def get_nested(self, *keys, default: Any = None) -> Any:
        """
        Get nested configuration value
        
        Args:
            *keys: Sequence of keys to traverse
            default: Default value if key not found
        
        Returns:
            Value at the nested location or default
        """
        current = self.config
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
                if current is None:
                    return default
            else:
                return default
        # Don't check if current is None again - just return it
        return current
        
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        self.config.update(updates)
        
    def save(self, path: Optional[str] = None) -> None:
        """
        Save configuration to YAML file
        
        Args:
            path: Path to save file (uses original path if not specified)
        """
        save_path = Path(path) if path else self.config_path
        
        try:
            with open(save_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)
            logging.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logging.error(f"Error saving config: {e}")
            raise
        
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary"""
        return self.config.copy()
    
    def validate_required_keys(self, required_keys: Dict[str, type]) -> bool:
        """
        Validate that required keys exist and have correct types
        
        Args:
            required_keys: Dictionary mapping key paths to expected types
        
        Returns:
            True if all required keys exist with correct types
        """
        for key_path, expected_type in required_keys.items():
            keys = key_path.split('.')
            value = self.get_nested(*keys)
            
            if value is None:
                logging.error(f"Required key not found: {key_path}")
                return False
                
            if not isinstance(value, expected_type):
                logging.error(f"Key {key_path} has wrong type. Expected {expected_type}, got {type(value)}")
                return False
                
        return True
    
    def merge(self, other_config: 'Config') -> 'Config':
        """
        Merge with another configuration
        
        Args:
            other_config: Another Config instance to merge
        
        Returns:
            New Config instance with merged values
        """
        merged = Config.__new__(Config)
        merged.config_path = self.config_path
        merged.config = self._deep_merge(self.config.copy(), other_config.config)
        return merged
    
    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self.config[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style setting"""
        self.config[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Allow 'in' operator"""
        return key in self.config
    
    def reload(self) -> None:
        """Reload configuration from file"""
        self.load_config()

# Global configuration instance (optional)
_global_config: Optional[Config] = None

def get_global_config() -> Optional[Config]:
    """Get the global configuration instance"""
    return _global_config

def set_global_config(config: Config) -> None:
    """Set the global configuration instance"""
    global _global_config
    _global_config = config

def load_global_config(config_path: str) -> Config:
    """Load and set the global configuration"""
    config = Config(config_path)
    set_global_config(config)
    return config