import sys
import os
import duckdb
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.modules.concordance import generate_kwic

def create_test_db():
    db_path = "test_symbol_filtering.duckdb"
    if os.path.exists(db_path):
        os.remove(db_path)
    con = duckdb.connect(db_path)
    con.execute("""
        CREATE TABLE corpus (
            id INTEGER, 
            token VARCHAR, 
            lemma VARCHAR, 
            pos VARCHAR, 
            _token_low VARCHAR,
            sent_id INTEGER
        )
    """)
    
    # data:
    # 0-2: "lightest weight ."
    # 3-5: "lightest . ."
    # 6-8: "lightest color ."
    data = [
        (0, "lightest", "light", "JJS", "lightest", 1),
        (1, "weight", "weight", "NN", "weight", 1),
        (2, ".", ".", ".", ".", 1),
        
        (3, "lightest", "light", "JJS", "lightest", 2),
        (4, ".", ".", ".", ".", 2),
        (5, ".", ".", ".", ".", 2),
        
        (6, "lightest", "light", "JJS", "lightest", 3),
        (7, "color", "color", "NN", "color", 3),
        (8, ".", ".", ".", ".", 3),
    ]
    con.executemany("INSERT INTO corpus VALUES (?, ?, ?, ?, ?, ?)", data)
    con.close()
    return db_path

def run_tests():
    db_path = create_test_db()
    try:
        # Test 1: Unfiltered wildcard search for "lightest *"
        # Should return all 3 matches: "lightest weight", "lightest .", and "lightest color"
        rows, total, _, _, _, _ = generate_kwic(
            corpus_db_path=db_path,
            raw_target_input="lightest *",
            kwic_left=1,
            kwic_right=1,
            corpus_name="Test",
            limit=10,
            hide_symbols=False
        )
        print(f"Unfiltered search 'lightest *' count: {total}")
        for r in rows:
            print(f"  Match: {r['Node']}")
        assert total == 3, f"Expected 3 matches, got {total}"
        print("Test 1 (hide_symbols=False) PASS")

        # Test 2: Filtered wildcard search for "lightest *"
        # Should return only 2 matches ("lightest weight" and "lightest color") and hide "lightest ."
        rows, total, _, _, _, _ = generate_kwic(
            corpus_db_path=db_path,
            raw_target_input="lightest *",
            kwic_left=1,
            kwic_right=1,
            corpus_name="Test",
            limit=10,
            hide_symbols=True
        )
        print(f"Filtered search 'lightest *' count: {total}")
        for r in rows:
            print(f"  Match: {r['Node']}")
        assert total == 2, f"Expected 2 matches, got {total}"
        print("Test 2 (hide_symbols=True) PASS")

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

if __name__ == "__main__":
    run_tests()
