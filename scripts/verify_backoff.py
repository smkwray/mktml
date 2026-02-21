
import sys
import time
import unittest
from unittest.mock import patch, MagicMock

# Add scripts dir to path to import update_qual_features
sys.path.append('scripts')
import update_qual_features

class TestBackoff(unittest.TestCase):
    
    @patch('update_qual_features.run_gemini_cli')
    @patch('update_qual_features.time.sleep')
    def test_backoff_success(self, mock_sleep, mock_gemini):
        print("\n--- Testing Backoff Logic ---")
        
        # Scenario: 
        # 1. First call returns "rate_limit"
        # 2. Second call returns valid JSON
        
        mock_gemini.side_effect = [
            ("Error: 429 Too Many Requests", "rate_limit"),
            ('{"ticker": "TEST", "sector": "Technology", "industry": "Software", "classifications": {}, "sources": [{"url": "http"}]}', None)
        ]
        
        # Run process_ticker
        result, error = update_qual_features.process_ticker("TEST")
        
        # Verify
        self.assertIsNotNone(result)
        self.assertIsNone(error)
        self.assertEqual(result['ticker'], "TEST")
        
        # Verify sleep was called
        mock_sleep.assert_called()
        print(f"Sleep called {mock_sleep.call_count} times.")
        print("Success: Backoff logic handled rate limit and succeeded on retry.")

    @patch('update_qual_features.run_gemini_cli')
    @patch('update_qual_features.time.sleep')
    def test_hard_quota_stop(self, mock_sleep, mock_gemini):
        print("\n--- Testing Hard Quota Stop ---")
        
        # Scenario:
        # 1. First call returns "daily_quota"
        
        mock_gemini.side_effect = [
            ("Error: Quota Exceeded", "daily_quota")
        ]
        
        result, error = update_qual_features.process_ticker("TEST")
        
        self.assertIsNone(result)
        self.assertEqual(error, "daily_quota")
        print("Success: Hard quota stopped immediately.")

if __name__ == '__main__':
    unittest.main()
