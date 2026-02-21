"""
Non-locking logger for OneDrive-safe file operations.
Opens file, writes, closes immediately - no persistent locks.
Updates: Added buffering to reduce open/close frequency (anti-locking).
"""
import os
import sys
import time
from datetime import datetime
from threading import Lock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(PROJECT_ROOT, "scan.log")
_LOG_LOCK = Lock()

class NonLockingLogger:
    """Logger that buffers output and writes in batches to be OneDrive safe."""
    
    def __init__(self, log_path: str = None, max_lines: int = 5000, buffer_size: int = 50, flush_interval: float = 3.0):
        self.log_path = log_path or LOG_FILE
        self.max_lines = max_lines
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        
        # Buffer processing
        self.msg_buffer = []
        self.last_flush_time = time.time()
    
    def log(self, message: str):
        """Buffer a log line and write if buffer full or timeout."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{timestamp}] {message}\n"
        
        with _LOG_LOCK:
            self.msg_buffer.append(line)
            
            # Check conditions to flush
            time_since_flush = time.time() - self.last_flush_time
            if len(self.msg_buffer) >= self.buffer_size or time_since_flush >= self.flush_interval:
                self._flush_unsafe()

    def _flush_unsafe(self):
        """Write buffer to disk. Assumes lock is held."""
        if not self.msg_buffer:
            return
            
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.writelines(self.msg_buffer)
                f.flush()
            
            # Clear buffer and reset timer
            self.msg_buffer = []
            self.last_flush_time = time.time()
            
        except Exception as e:
            # Fall back to stdout if file write fails (and we aren't redirecting stdout)
             if self._original_stdout and self._original_stdout != sys.stdout:
                self._original_stdout.write(f"[LOG ERROR] {e}: {len(self.msg_buffer)} lines dropped.\n")

    def flush(self):
        """Force flush buffer to disk."""
        with _LOG_LOCK:
            self._flush_unsafe()
    
    def clear(self):
        """Clear the log file."""
        with _LOG_LOCK:
             self.msg_buffer = [] # Clear buffer too
             try:
                 with open(self.log_path, 'w', encoding='utf-8') as f:
                     f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === SCAN STARTED ===\n")
             except Exception:
                 pass
    
    def install_stdout_redirect(self):
        """Redirect stdout/stderr to also log to file."""
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = _TeeWriter(self._original_stdout, self)
        sys.stderr = _TeeWriter(self._original_stderr, self)
    
    def restore_stdout(self):
        """Restore original stdout/stderr and flush remaining logs."""
        self.flush() # Ensure everything is written before detaching
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


class _TeeWriter:
    """Writes to both original stream AND log file."""
    
    def __init__(self, original, logger: NonLockingLogger):
        self.original = original
        self.logger = logger
        self.buffer = ""
    
    def write(self, text):
        # Write to original immediately
        try:
            self.original.write(text)
        except:
            pass
        
        # Buffer for log (write complete lines only)
        self.buffer += text
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            if line.strip():  # Only log non-empty lines
                self.logger.log(line)
    
    def flush(self):
        try:
            self.original.flush()
        except:
            pass
        # We generally don't force logger flush here to avoid re-introducing locking frequency,
        # unless it's critical. Let the logger manage its own interval.


# Global logger instance
_LOGGER = None

def get_logger() -> NonLockingLogger:
    """Get or create the global non-locking logger."""
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = NonLockingLogger()
    return _LOGGER

def log(message: str):
    """Convenience function to log a message."""
    get_logger().log(message)

def start_logging():
    """Start logging with stdout/stderr redirect."""
    logger = get_logger()
    logger.clear()
    logger.install_stdout_redirect()
    return logger

def stop_logging():
    """Stop logging redirect."""
    logger = get_logger()
    logger.flush() # Explicit flush
    logger.restore_stdout()
