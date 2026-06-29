import sys
import os
import duckdb
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.modules.concordance import generate_kwic
from core.modules.collocation import generate_collocation_results
from core.modules.distribution import calculate_distribution
from core.modules.statistical_testing import preview_query_matches, compare_groups_by_word
from core.modules.ngram import generate_n_grams_v2

def create_test_db(filename="test_lemma_pos.duckdb"):
    if os.path.exists(filename):
        try: os.remove(filename)
        except: pass
    con = duckdb.connect(filename)
    con.execute("""
        CREATE TABLE corpus (
            id INTEGER, 
            token VARCHAR, 
            lemma VARCHAR, 
            pos VARCHAR, 
            group_id VARCHAR, 
            _token_low VARCHAR,
            sent_id INTEGER
        )
    """)
    
    # 1. studies (noun) - speaker/group: A
    # 2. studying (verb) - speaker/group: B
    # 3. playing (verb) - speaker/group: A
    data = [
        (0, "studies", "study", "NNS", "A", "studies", 1),
        (1, "studying", "study", "VBG", "B", "studying", 2),
        (2, "playing", "play", "VBG", "A", "playing", 3),
        (3, "common", "common", "JJ", "A", "common", 4),
        (4, "common", "common", "JJ", "B", "common", 5),
    ]
    con.executemany("INSERT INTO corpus VALUES (?, ?, ?, ?, ?, ?, ?)", data)
    con.close()
    return filename

def test_lemma_pos():
    db_path = create_test_db()
    try:
        # 1. Concordance Tests
        print("--- Testing Concordance ---")
        # [study] (lemma search) -> should match studies and studying (2)
        _, total_study, _, _, _, _ = generate_kwic(db_path, "[study]", 1, 1, "Test")
        print(f"Concordance [study] matches: {total_study}")
        assert total_study == 2, f"Expected 2 matches, got {total_study}"

        # [study_NNS] (lemma + POS search) -> should match studies only (1)
        _, total_study_nns, _, _, _, _ = generate_kwic(db_path, "[study_NNS]", 1, 1, "Test")
        print(f"Concordance [study_NNS] matches: {total_study_nns}")
        assert total_study_nns == 1, f"Expected 1 match, got {total_study_nns}"

        # [study_VB*] (lemma + POS wildcard search) -> should match studying only (1)
        _, total_study_vb, _, _, _, _ = generate_kwic(db_path, "[study_VB*]", 1, 1, "Test")
        print(f"Concordance [study_VB*] matches: {total_study_vb}")
        assert total_study_vb == 1, f"Expected 1 match, got {total_study_vb}"

        # 2. Collocation Tests
        print("\n--- Testing Collocation ---")
        # Collocation node query: [study_VB*]
        df, freq, node_mwu = generate_collocation_results(db_path, "[study_VB*]", 1, 1, 10, False)
        print(f"Collocation [study_VB*] matches: {freq}, node: {node_mwu}")
        assert freq == 1, f"Expected 1 match, got {freq}"
        assert node_mwu == "[study_VB*]", f"Expected node to be [study_VB*], got {node_mwu}"

        # 3. Distribution Tests
        print("\n--- Testing Distribution ---")
        dist_df, _ = calculate_distribution(db_path, "[study_NNS]")
        print(f"Distribution [study_NNS] matches sum: {dist_df['Absolute Frequency'].sum()}")
        assert dist_df['Absolute Frequency'].sum() == 1, f"Expected 1 match, got {dist_df['Absolute Frequency'].sum()}"

        # 4. Statistical Testing Tests
        print("\n--- Testing Statistical Testing ---")
        preview = preview_query_matches(db_path, "[study_NNS]", min_freq=1)
        print(f"Stat Preview [study_NNS] total freq: {preview.get('total_freq')}")
        assert preview.get('total_freq') == 1, f"Expected 1, got {preview.get('total_freq')}"

        stats_compare = compare_groups_by_word(db_path, "[study]", "group_id", ["A", "B"], min_freq=1)
        print(f"Stat Compare [study] output size: {len(stats_compare)}")
        assert len(stats_compare) == 2, f"Expected 2 groups/words, got {len(stats_compare)}"

        # 5. N-Gram Tests
        print("\n--- Testing N-Gram ---")
        # Generate unigrams with filter [study_VB*]
        df_ngrams = generate_n_grams_v2(
            corpus_db_path=db_path,
            n_size=1,
            n_gram_filters={'1': '[study_VB*]'},
            is_raw_mode=False,
            corpus_name="Test",
            basis='Token'
        )
        print(f"N-Gram [study_VB*] output: {df_ngrams.to_dict('records')}")
        assert len(df_ngrams) == 1, f"Expected 1 n-gram, got {len(df_ngrams)}"
        assert df_ngrams.iloc[0]['N-Gram'] == 'studying', f"Expected studying, got {df_ngrams.iloc[0]['N-Gram']}"

        print("\n[ALL PASS] [lemma_POS] Query Syntax Verified Successfully!")

    finally:
        if os.path.exists(db_path):
            try: os.remove(db_path)
            except: pass

if __name__ == "__main__":
    test_lemma_pos()
