import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock genai before importing app to avoid API calls during test
sys.modules["google.generativeai"] = MagicMock()

# Mock other dependencies that might require system resources
sys.modules["pytesseract"] = MagicMock()
sys.modules["pdf2image"] = MagicMock()

# Now import app
from app import app, TASKS

class TestExport(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        
        # Add a dummy task
        self.task_id = "test-task-123"
        TASKS[self.task_id] = {
            "status": "done",
            "progress": 100,
            "summary": "This is a medical summary.",
            "extracted": "Raw text",
            "language": "en"
        }

    @patch("app.translate_with_gemini")
    def test_download_item_translation(self, mock_translate):
        # Mock translation return
        mock_translate.return_value = "Este es un resumen médico."
        
        # Request download in Spanish
        response = self.app.get(f"/download_item/{self.task_id}?language=es")
        
        # Check status
        self.assertEqual(response.status_code, 200)
        
        # Check content type
        self.assertEqual(response.content_type, "application/pdf")
        
        # Check if translation was called
        mock_translate.assert_called_with("This is a medical summary.", "es")
        
        print("Test passed: Download with translation works.")

    def test_download_item_same_language(self):
        # Request download in English (same as task)
        response = self.app.get(f"/download_item/{self.task_id}?language=en")
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "application/pdf")
        
        print("Test passed: Download same language works.")

if __name__ == "__main__":
    unittest.main()
