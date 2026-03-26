"""
Tests for Quality Scorer.
"""
import pytest
from noirag.preprocessing.hybrid.quality_scorer import QualityScorer

@pytest.fixture(scope="module")
def scorer():
    return QualityScorer()

def test_clean_text_score(scorer):
    text = "This is a perfectly clean sentence with no typos and valid formatting."
    result = scorer.score(text)
    
    # Needs to be very low score
    assert result["overall_score"] < 0.1
    assert result["oov_ratio"] == 0.0
    assert result["garbage_density"] == 0.0
    assert result["formatting_anomaly_rate"] == 0.0

def test_semantic_noisy_text(scorer):
    # 'computars' is out of vocabulary, 'teh' is out of vocabulary (depending on dict, often corrected), 't3st' is OOV
    text = "The computars ran teh t3st bdy."
    result = scorer.score(text)
    
    # Semantic errors heavily influence overall score
    assert result["oov_ratio"] > 0.3
    assert result["garbage_density"] == 0.0
    assert result["overall_score"] > 0.15

def test_formatting_noisy_text(scorer):
    text = "Here is some text • with garbage ■ in it [...] and ===\n\n\n  and   lots of spaces."
    result = scorer.score(text)
    
    assert result["garbage_density"] > 0.0
    assert result["formatting_anomaly_rate"] > 0.0
    assert result["overall_score"] > 0.0

def test_acronym_ignorance(scorer):
    text = "The EBITDA is solid and the NASA launch was okay."
    result = scorer.score(text)
    
    # Acronyms should be ignored by the OOV checker, so the ratio should be 0.0
    assert result["oov_ratio"] == 0.0
    
def test_number_ignorance(scorer):
    text = "Revenues hit 1.5M in 2023 Q1."
    result = scorer.score(text)
    
    # Numbers should be ignored by the OOV checker, so the ratio should be 0.0
    assert result["oov_ratio"] == 0.0

def test_broken_lines(scorer):
    text = "This sentence is broken\nin the middle of nowhere."
    result_broken = scorer.score(text)
    
    text_clean = "This sentence is clean.\nIn the middle of nowhere."
    result_clean = scorer.score(text_clean)
    
    assert result_broken["formatting_anomaly_rate"] > result_clean["formatting_anomaly_rate"]
