"""
Tests for Hybrid Cleaner.
"""
import pytest
from noirag.preprocessing.hybrid.hybrid_cleaner import HybridCleaner

@pytest.fixture(scope="module")
def cleaner():
    # Lower thresholds slightly to ensure our small test sentences trigger them
    return HybridCleaner(formatting_threshold=0.01, semantic_threshold=0.10)

def test_clean_text_bypass(cleaner):
    text = "This is a perfectly clean sentence with no typos and valid formatting."
    cleaned_text, metadata = cleaner.clean(text)
    
    assert cleaned_text == text
    assert metadata["applied_cleaners"] == []

def test_routes_formatting_only(cleaner):
    text = "Here is some text • with garbage ■ in it [...] and ==="
    cleaned_text, metadata = cleaner.clean(text)
    
    assert "rule_based" in metadata["applied_cleaners"]
    assert "statistical" not in metadata["applied_cleaners"]
    assert "■" not in cleaned_text
    
def test_routes_semantic_only(cleaner):
    text = "The computars ran teh t3st bdy."
    cleaned_text, metadata = cleaner.clean(text)
    
    assert "statistical" in metadata["applied_cleaners"]
    assert "rule_based" not in metadata["applied_cleaners"]
    # Check that spellchecking actually occurred
    assert "computer" in cleaned_text.lower() or "computers" in cleaned_text.lower()
    
def test_routes_both(cleaner):
    text = "The computars ran teh t3st bdy. • with garbage ■ in it ==="
    cleaned_text, metadata = cleaner.clean(text)
    
    # Needs to trigger both thresholds
    assert "rule_based" in metadata["applied_cleaners"]
    assert "statistical" in metadata["applied_cleaners"]
    
    # Check both rules applied
    assert "■" not in cleaned_text
    assert "computer" in cleaned_text.lower() or "computers" in cleaned_text.lower()
