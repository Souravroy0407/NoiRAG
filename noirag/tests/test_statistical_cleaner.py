"""
Tests for Statistical Cleaner.
"""
import pytest
from noirag.preprocessing.statistical.spell_cleaner import StatisticalCleaner

@pytest.fixture(scope="module")
def cleaner():
    return StatisticalCleaner(max_dictionary_edit_distance=3)

def test_correct_simple_typo(cleaner):
    text = "computar"
    cleaned = cleaner.clean(text)
    assert cleaned == "computer"

def test_correct_in_sentence(cleaner):
    text = "Ths is a tst of teh spell checker"
    cleaned = cleaner.clean(text)
    assert cleaned == "The is a test of the spell checker"

def test_ignore_acronyms(cleaner):
    text = "The EBITDA of the company is high"
    cleaned = cleaner.clean(text)
    assert "EBITDA" in cleaned

def test_ignore_numbers(cleaner):
    text = "Revenue in 2023 Q1 was 1.5M"
    cleaned = cleaner.clean(text)
    assert "2023" in cleaned
    assert "Q1" in cleaned
    assert "1.5M" in cleaned

def test_preserve_punctuation(cleaner):
    text = "\"Hello, world!\", he said."
    cleaned = cleaner.clean(text)
    assert "\"Hello, world!\", he said." == cleaned

def test_preserve_capitalization(cleaner):
    text = "Wlcome to New Yrk"
    cleaned = cleaner.clean(text)
    assert cleaned == "Welcome to New York"

def test_complex_sentence(cleaner):
    text = "The quick brwn fox jumpd over teh lazy dog."
    cleaned = cleaner.clean(text)
    # Note: 'jumpd' -> 'jump' (edit dist 1) is higher frequency than 'jumped' (edit dist 1)
    assert cleaned == "The quick brown fox jump over the lazy dog."

def test_ignore_short_words(cleaner):
    # Short unknown words < 3 chars should generally be ignored by our filter
    text = "xy zq ab"
    cleaned = cleaner.clean(text)
    assert cleaned == "xy zq ab"
