import time
import random
import threading

class TokenBucket:
    """Token Bucket algorithm for rate limiting (thread-safe)."""
    def __init__(self, tokens, fill_rate):
        """
        tokens: Capacity of the bucket (max burst).
        fill_rate: Tokens added per second.
        """
        self.capacity = float(tokens)
        self._tokens = float(tokens)
        self.fill_rate = float(fill_rate)
        self.timestamp = time.time()
        self._lock = threading.Lock()

    def consume(self, tokens=1):
        """Returns True if tokens can be consumed, False otherwise. Thread-safe."""
        with self._lock:
            now = time.time()
            # Refill tokens based on time passed
            delta = now - self.timestamp
            self._tokens = min(self.capacity, self._tokens + self.fill_rate * delta)
            self.timestamp = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

class APICircuitBreaker:
    """Circuit Breaker for failing APIs."""
    def __init__(self, name, failure_threshold=5, recovery_timeout=300):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED (working), OPEN (broken)

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
            print(f"  [CircuitBreaker] {self.name} OPENED! Too many failures ({self.failures}). Cooldown for {self.recovery_timeout}s.")

    def record_success(self):
        if self.state == "HALF-OPEN":
            self.state = "CLOSED"
            self.failures = 0
            print(f"  [CircuitBreaker] {self.name} CLOSED. Recovered.")
        else:
            self.failures = 0

    def allow_request(self):
        if self.state == "CLOSED":
            return True
        
        # Check if recovery timeout has passed
        if time.time() - self.last_failure_time > self.recovery_timeout:
            self.state = "HALF-OPEN"  # Allow one test request
            return True
            
        return False
