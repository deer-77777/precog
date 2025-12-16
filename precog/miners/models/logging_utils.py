"""
Simple Logging Utility for LSTM Training

Provides consistent logging across all LSTM modules.
"""

from datetime import datetime


class SimpleLogger:
    """Simple logger that prints to console with colors."""
    
    COLORS = {
        "INFO": "\033[94m",     # Blue
        "DEBUG": "\033[90m",    # Gray
        "WARN": "\033[93m",     # Yellow
        "ERROR": "\033[91m",    # Red
        "SUCCESS": "\033[92m",  # Green
        "TRACE": "\033[90m",    # Gray
        "RESET": "\033[0m",     # Reset
    }
    
    @staticmethod
    def _log(level: str, msg: str):
        color = SimpleLogger.COLORS.get(level, "")
        reset = SimpleLogger.COLORS["RESET"]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{color}[{timestamp}] [{level}] {msg}{reset}", flush=True)
    
    @staticmethod
    def info(msg): SimpleLogger._log("INFO", msg)
    
    @staticmethod
    def debug(msg): SimpleLogger._log("DEBUG", msg)
    
    @staticmethod
    def warning(msg): SimpleLogger._log("WARN", msg)
    
    @staticmethod
    def error(msg): SimpleLogger._log("ERROR", msg)
    
    @staticmethod
    def success(msg): SimpleLogger._log("SUCCESS", msg)
    
    @staticmethod
    def trace(msg): SimpleLogger._log("TRACE", msg)


# Global logger instance
logger = SimpleLogger()

