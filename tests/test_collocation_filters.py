import sys
import os
import duckdb
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.modules.collocation import generate_collocation_results

def test_collocation_filters():
    db_path = "test_filters_temp.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    con = duckdb.connect(db_path)
    con.execute("""
    CREATE TABLE corpus (
        id INTEGER,
        token TEXT,
        _token_low TEXT,
        pos TEXT,
        lemma TEXT
    )
    """)
    
    # "the happy boy bought a very smart dog"
    data = [
        (0, "the", "the", "DT", "the"),
        (1, "happy", "happy", "JJ", "happy"),
        (2, "boy", "boy", "NN", "boy"),
        (3, "bought", "bought", "VBD", "buy"),
        (4, "a", "a", "DT", "a"),
        (5, "very", "very", "RB", "very"),
        (6, "smart", "smart", "JJ", "smart"),
        (7, "dog", "dog", "NN", "dog"),
    ]
    for row in data:
        con.execute("INSERT INTO corpus VALUES (?, ?, ?, ?, ?)", row)
    con.close()

    try:
        # Test Case 1: (JJ|NN) union positive filter
        df, _, _ = generate_collocation_results(
            db_path, "boy", coll_window=2, mi_min_freq=1, max_collocates=10, is_raw_mode=False,
            pos_filter="(JJ|NN)"
        )
        collocates = df['Collocate'].tolist()
        pos_tags = df['POS'].tolist()
        assert "happy" in collocates, f"happy should be included: {collocates}"
        for tag in pos_tags:
            assert tag in ("JJ", "NN"), f"Only JJ or NN should be shown but got tag {tag} in {pos_tags}"
        print("[OK] Union (JJ|NN) correctly matched JJ and NN only")

        # Test Case 2: (-JJ|-NN|-RB) union negative filter (inside brackets)
        df, _, _ = generate_collocation_results(
            db_path, "boy", coll_window=2, mi_min_freq=1, max_collocates=10, is_raw_mode=False,
            pos_filter="(-JJ|-NN|-RB)"
        )
        collocates = df['Collocate'].tolist()
        pos_tags = df['POS'].tolist()
        assert "happy" not in collocates, f"happy (JJ) should be excluded: {collocates}"
        for tag in pos_tags:
            assert tag not in ("JJ", "NN", "RB"), f"JJ, NN, RB should be excluded but got tag {tag} in {pos_tags}"
        print("[OK] Union (-JJ|-NN|-RB) correctly excluded JJ, NN, RB")

        # Test Case 3: -(JJ|NN|RB) negation outside parenthesis
        df, _, _ = generate_collocation_results(
            db_path, "boy", coll_window=2, mi_min_freq=1, max_collocates=10, is_raw_mode=False,
            pos_filter="-(JJ|NN|RB)"
        )
        collocates = df['Collocate'].tolist()
        pos_tags = df['POS'].tolist()
        assert "happy" not in collocates, f"happy (JJ) should be excluded: {collocates}"
        for tag in pos_tags:
            assert tag not in ("JJ", "NN", "RB"), f"JJ, NN, RB should be excluded but got tag {tag} in {pos_tags}"
        print("[OK] Distributive union negation -(JJ|NN|RB) correctly worked")

        # Test Case 4: Case-insensitive union
        df, _, _ = generate_collocation_results(
            db_path, "boy", coll_window=2, mi_min_freq=1, max_collocates=10, is_raw_mode=False,
            pos_filter="-(jj|nn)"
        )
        collocates = df['Collocate'].tolist()
        pos_tags = df['POS'].tolist()
        assert "happy" not in collocates, f"happy should be excluded: {collocates}"
        for tag in pos_tags:
            assert tag not in ("JJ", "NN"), f"JJ, NN should be excluded but got tag {tag} in {pos_tags}"
        print("[OK] Case-insensitive union -(jj|nn) worked correctly")

        # Test Case 5: Token filter with union and negation
        df, _, _ = generate_collocation_results(
            db_path, "boy", coll_window=2, mi_min_freq=1, max_collocates=10, is_raw_mode=False,
            token_filter="(-happy|-the)"
        )
        collocates = df['Collocate'].tolist()
        assert "happy" not in collocates, f"happy should be excluded: {collocates}"
        assert "the" not in collocates, f"the should be excluded: {collocates}"
        assert "bought" in collocates, f"bought should be included: {collocates}"
        print("[OK] Token filter union negation (-happy|-the) worked correctly")

        # Test Case 6: Lemma filter with union and negation outside parenthesis
        df, _, _ = generate_collocation_results(
            db_path, "boy", coll_window=2, mi_min_freq=1, max_collocates=10, is_raw_mode=False,
            lemma_filter="-(buy|the)"
        )
        collocates = df['Collocate'].tolist()
        lemmas = df['Lemma'].tolist()
        assert "buy" not in lemmas, f"buy lemma should be excluded: {lemmas}"
        assert "the" not in lemmas, f"the lemma should be excluded: {lemmas}"
        assert "happy" in collocates, f"happy collocate should be included: {collocates}"
        print("[OK] Lemma filter union negation -(buy|the) worked correctly")

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

if __name__ == "__main__":
    test_collocation_filters()
