import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.modules.readability import (
    count_syllables_english,
    calculate_formulas,
    map_score_to_level
)

def test_syllable_counting():
    print("Testing syllable counting...")
    test_cases = {
        "the": 1,
        "cat": 1,
        "sat": 1,
        "on": 1,
        "mat": 1,
        "stable": 2,
        "syllable": 3,
        "readability": 5,
        "very": 2,
        "easy": 2,
        "difficult": 3
    }
    
    success = True
    for word, expected in test_cases.items():
        actual = count_syllables_english(word)
        if actual != expected:
            print(f"FAIL: word '{word}' expected {expected} syllables, got {actual}")
            success = False
        else:
            print(f"PASS: word '{word}' -> {actual} syllables")
            
    return success

def test_formulas():
    print("\nTesting formulas...")
    # 100 words, 5 sentences, 150 syllables, 400 characters, 15 complex words
    metrics = calculate_formulas(
        words=100,
        sentences=5,
        syllables=150,
        characters=400,
        complex_words=15
    )
    
    print("Calculated metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v} (Level: {map_score_to_level(v)})")
        
    # Check that they return positive numbers
    assert all(v >= 0 for v in metrics.values()), "All metrics must be positive"
    print("PASS: Formula calculation succeeded!")

if __name__ == "__main__":
    s_ok = test_syllable_counting()
    f_ok = True
    try:
        test_formulas()
    except Exception as e:
        print(f"FAIL: Formula test failed: {e}")
        f_ok = False
        
    if s_ok and f_ok:
        print("\nALL READABILITY TESTS PASSED!")
        sys.exit(0)
    else:
        print("\nREADABILITY TESTS FAILED!")
        sys.exit(1)
