"""
Tests for Rule-Based Cleaner.
"""
import pytest
from noirag.preprocessing.rule_based.cleaner import RuleBasedCleaner

@pytest.fixture
def cleaner():
    return RuleBasedCleaner()

def test_remove_garbage_strings(cleaner):
    text = "Here is some text • with garbage ■ in it [...] and ==="
    cleaned = cleaner.remove_garbage_strings(text)
    assert "•" not in cleaned
    assert "■" not in cleaned
    assert "===" not in cleaned
    assert "[...]" not in cleaned
    assert "Here is some text   with garbage   in it   and  " == cleaned

def test_fix_punctuation_spacing(cleaner):
    text = "This is a test.This is another test,with no spaces."
    cleaned = cleaner.fix_punctuation_spacing(text)
    assert cleaned == "This is a test. This is another test, with no spaces."

def test_repair_broken_lines(cleaner):
    text = "This is a normal line.\nThis line was brok\nen in the middle.\nAnd another one."
    cleaned = cleaner.repair_broken_lines(text)
    assert "brok\nen" not in cleaned
    assert "broken in the middle" in cleaned

def test_repair_broken_lines_with_space(cleaner):
    text = "This is a normal line.\nThis line was broken \nin the middle.\nAnd another one."
    cleaned = cleaner.repair_broken_lines(text)
    assert "broken \nin" not in cleaned
    assert "broken in the middle" in cleaned

def test_normalize_whitespace(cleaner):
    text = "This    has  way   too   many    spaces.\n\n\n\nAnd too many newlines."
    cleaned = cleaner.normalize_whitespace(text)
    assert "This has way too many spaces." in cleaned
    assert "\n\n\n" not in cleaned
    assert "spaces.\n\nAnd" in cleaned

def test_remove_duplicate_lines(cleaner):
    text = "Line 1\nLine 2\nLine 2\nLine 3"
    cleaned = cleaner.remove_duplicate_lines(text)
    assert cleaned == "Line 1\nLine 2\nLine 3"

def test_clean_pipeline(cleaner):
    text = "This is a noisy\ntext.With bad formatting • and\nduplicates.\ntext.With bad formatting • and\nduplicates."
    cleaned = cleaner.clean(text)
    # checking no crash and output exists
    assert len(cleaned) > 0
    assert "•" not in cleaned
