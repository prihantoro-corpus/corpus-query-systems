import unittest
import pandas as pd
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.modules.word_profiler import load_wordlist, run_word_profiler_analysis

class TestWordProfiler(unittest.TestCase):
    def setUp(self):
        # Create a mock wordlist
        self.plain_wl_content = "the\nand\nto\nof\nin"
        self.cat_wl_content = "the\tA1\nand\tA1\nto\tA1\nof\tA2\nin\tA2"
        
        # Paths to hypothetical wordlists created earlier
        self.test_plain_path = r"c:\Users\priha\Documents\cortex\wordlist\test_plain.txt"
        self.test_cat_path = r"c:\Users\priha\Documents\cortex\wordlist\test_categorized.txt"

    def test_load_wordlist_plain(self):
        wl = load_wordlist(self.plain_wl_content, is_file=False)
        self.assertEqual(wl['the'], 'Coverage')
        self.assertEqual(wl['and'], 'Coverage')
        self.assertEqual(len(wl), 5)

    def test_load_wordlist_categorized(self):
        wl = load_wordlist(self.cat_wl_content, is_file=False)
        self.assertEqual(wl['the'], 'A1')
        self.assertEqual(wl['of'], 'A2')
        self.assertEqual(len(wl), 5)

    def test_load_wordlist_from_file(self):
        if os.path.exists(self.test_plain_path):
            wl = load_wordlist(self.test_plain_path)
            self.assertIn('the', wl)
            self.assertEqual(wl['the'], 'Coverage')

    def test_run_analysis_mock(self):
        # We need a sample DB to test run_word_profiler_analysis
        # For now, we'll just check if it returns an empty DF if path is None
        df = run_word_profiler_analysis(None, {})
        self.assertTrue(df.empty)

if __name__ == '__main__':
    unittest.main()
