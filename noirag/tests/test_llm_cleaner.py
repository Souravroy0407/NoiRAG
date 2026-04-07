"""
Tests for Local Ollama LLM Cleaner.
"""
import pytest
from unittest.mock import patch, MagicMock
from noirag.preprocessing.hybrid.llm_cleaner import LLMCleaner

@patch('requests.post')
def test_successful_ollama_cleaning(mock_post):
    cleaner = LLMCleaner()
    
    # Mock a successful local Ollama response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'message': {'content': 'The revenue was great.'}
    }
    mock_post.return_value = mock_response
    
    noisy_text = "The rvnue was grt."
    result = cleaner.clean(noisy_text)
    
    assert result == "The revenue was great."
    mock_post.assert_called_once()

@patch('requests.post')
def test_ollama_connection_failure_returns_original(mock_post):
    cleaner = LLMCleaner()
    
    import requests
    mock_post.side_effect = requests.exceptions.ConnectionError("Connection Refused")
    
    noisy_text = "Bad text"
    result = cleaner.clean(noisy_text)
    
    # Needs to gracefully bypass and survive if Ollama isn't started
    assert result == "Bad text"
    assert mock_post.call_count == 1
