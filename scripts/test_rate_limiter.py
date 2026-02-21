import sys
import os
import time
import unittest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from rate_limiter import TokenBucket, APICircuitBreaker

class TestRateLimiter(unittest.TestCase):
    
    def test_token_bucket(self):
        print("\nTesting TokenBucket...")
        # 2 tokens per second, capacity 2
        bucket = TokenBucket(tokens=2, fill_rate=2)
        
        # Consume all tokens
        self.assertTrue(bucket.consume(1))
        self.assertTrue(bucket.consume(1))
        self.assertFalse(bucket.consume(1)) # Should assume empty
        
        print("Waiting 0.6s...")
        time.sleep(0.6)
        # Should have refilled ~1.2 tokens -> 1 token available
        self.assertTrue(bucket.consume(1))
        self.assertFalse(bucket.consume(1)) 
        
        print("TokenBucket tests passed.")

    def test_circuit_breaker(self):
        print("\nTesting APICircuitBreaker...")
        # Threshold 2, Recovery 1s
        cb = APICircuitBreaker("TestAPI", failure_threshold=2, recovery_timeout=1)
        
        # Initial state
        self.assertTrue(cb.allow_request())
        self.assertEqual(cb.state, "CLOSED")
        
        # Fail once
        cb.record_failure()
        self.assertEqual(cb.state, "CLOSED")
        
        # Fail twice -> Open
        cb.record_failure()
        self.assertEqual(cb.state, "OPEN")
        self.assertFalse(cb.allow_request())
        
        print("Waiting 1.1s for recovery...")
        time.sleep(1.1)
        
        # Should allow trial request (HALF-OPEN)
        self.assertTrue(cb.allow_request())
        self.assertEqual(cb.state, "HALF-OPEN")
        
        # Success closes it
        cb.record_success()
        self.assertEqual(cb.state, "CLOSED")
        
        print("CircuitBreaker tests passed.")

if __name__ == '__main__':
    unittest.main()
