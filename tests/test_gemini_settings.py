import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.ai_service import _resolve_ai_settings, test_gemini_connection

class TestGeminiSettings(unittest.TestCase):
    def test_settings_resolver_default(self):
        settings = _resolve_ai_settings()
        self.assertEqual(settings["ai_provider"], "Ollama")
        self.assertEqual(settings["gemini_model"], "gemini-2.5-flash")
        self.assertEqual(settings["gemini_api_key"], "")

    def test_settings_resolver_explicit(self):
        settings = _resolve_ai_settings(
            ai_provider="Gemini",
            gemini_api_key="test_key",
            gemini_model="gemini-2.0-flash",
            ollama_url="http://localhost:11434",
            ollama_model="llama3"
        )
        self.assertEqual(settings["ai_provider"], "Gemini")
        self.assertEqual(settings["gemini_api_key"], "test_key")
        self.assertEqual(settings["gemini_model"], "gemini-2.0-flash")
        self.assertEqual(settings["ollama_url"], "http://localhost:11434")
        self.assertEqual(settings["ollama_model"], "llama3")

    @patch('requests.post')
    def test_gemini_connection_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {"content": {"parts": [{"text": "Hello response"}]}}
            ]
        }
        mock_post.return_value = mock_response

        success, msg = test_gemini_connection("valid_key", "gemini-2.5-flash")
        self.assertTrue(success)
        self.assertIn("Successfully connected", msg)

    @patch('requests.post')
    def test_gemini_connection_failure(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.value = {
            "error": {"message": "API key not valid"}
        }
        mock_response.json.return_value = {
            "error": {"message": "API key not valid"}
        }
        mock_post.return_value = mock_response

        success, msg = test_gemini_connection("invalid_key", "gemini-2.5-flash")
        self.assertFalse(success)
        self.assertIn("API Error", msg)

if __name__ == '__main__':
    unittest.main()
