import sys
import os
import duckdb
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.modules.concordance import generate_kwic

def create_test_db():
    db_path = "test_concordance_restriction.duckdb"
    if os.path.exists(db_path):
        os.remove(db_path)
    con = duckdb.connect(db_path)
    con.execute("""
        CREATE TABLE corpus (
            id INTEGER, 
            token VARCHAR, 
            lemma VARCHAR, 
            pos VARCHAR, 
            speaker VARCHAR, 
            year INTEGER, 
            _token_low VARCHAR,
            sent_id INTEGER
        )
    """)
    
    # Speaker A: "I like apple" (id 0-2)
    # Speaker B: "I like orange" (id 3-5)
    data = [
        (0, "I", "I", "PRP", "A", 2000, "i", 1),
        (1, "like", "like", "VBP", "A", 2000, "like", 1),
        (2, "apple", "apple", "NN", "A", 2000, "apple", 1),
        (3, "I", "I", "PRP", "B", 2005, "i", 2),
        (4, "like", "like", "VBP", "B", 2005, "like", 2),
        (5, "orange", "orange", "NN", "B", 2005, "orange", 2),
    ]
    con.executemany("INSERT INTO corpus VALUES (?, ?, ?, ?, ?, ?, ?, ?)", data)
    con.close()
    return db_path

def run_tests():
    db_path = create_test_db()
    try:
        # Test 1: Unrestricted search for "like"
        # Should return 2 matches
        rows, total, _, _, _, _ = generate_kwic(
            corpus_db_path=db_path,
            raw_target_input="like",
            kwic_left=1,
            kwic_right=1,
            corpus_name="Test",
            limit=10
        )
        print(f"Unrestricted search 'like' count: {total}")
        assert total == 2, f"Expected 2 matches, got {total}"

        # Test 2: Restricted search by Speaker = 'A'
        # Should return 1 match
        rows, total, _, _, _, _ = generate_kwic(
            corpus_db_path=db_path,
            raw_target_input="like",
            kwic_left=1,
            kwic_right=1,
            corpus_name="Test",
            limit=10,
            xml_where_clause=" AND speaker = ?",
            xml_params=['A']
        )
        print(f"Restricted search 'like' (Speaker='A') count: {total}")
        assert total == 1, f"Expected 1 match, got {total}"
        print("Test 2 (Speaker='A') PASS")

        # Test 3: Restricted search by Year = 2005
        # Should return 1 match
        rows, total, _, _, _, _ = generate_kwic(
            corpus_db_path=db_path,
            raw_target_input="like",
            kwic_left=1,
            kwic_right=1,
            corpus_name="Test",
            limit=10,
            xml_where_clause=" AND TRY_CAST(year AS BIGINT) BETWEEN ? AND ?",
            xml_params=[2005, 2005]
        )
        print(f"Restricted search 'like' (Year=2005) count: {total}")
        assert total == 1, f"Expected 1 match, got {total}"
        print("Test 3 (Year=2005) PASS")

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

if __name__ == "__main__":
    run_tests()
