import os
import sys
import duckdb
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.modules.collocation import generate_collocation_results

def test_measures():
    db_path = "test_colloc_measures.db"
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
    
    # Corpus: target word "cat" co-occurring with:
    # "fish" 5 times (joint freq = 5, cat freq = 10, fish freq = 15, N = 100)
    # "milk" 3 times (joint freq = 3, cat freq = 10, milk freq = 8, N = 100)
    # "dog" 1 time
    data = []
    id_cnt = 0
    
    # Helper to insert a sentence: cat <colloc> .
    # Window is 2, so putting "." separates cats by 2 tokens, preventing overlap.
    def add_cooccurrence(colloc):
        nonlocal id_cnt
        data.extend([
            (id_cnt, "cat", "cat", "NN", "cat"),
            (id_cnt+1, colloc, colloc.lower(), "NN", colloc.lower()),
            (id_cnt+2, ".", ".", ".", ".")
        ])
        id_cnt += 3

    # Add 5 "cat fish"
    for _ in range(5):
        add_cooccurrence("fish")
    # Add 3 "cat milk"
    for _ in range(3):
        add_cooccurrence("milk")
    # Add 1 "cat dog"
    for _ in range(1):
        add_cooccurrence("dog")
        
    # Fill up the rest with filler words to make N = 200, cat freq = 10 (9 + 1 single cat), fish = 15 (5 + 10 single fish), etc.
    # cat frequency: we already have 9 cats. Let's add 1 single "cat"
    data.extend([
        (id_cnt, "cat", "cat", "NN", "cat"),
        (id_cnt+1, ".", ".", ".", ".")
    ])
    id_cnt += 2
    
    # fish frequency: we already have 5 fish. Let's add 10 single "fish"
    for _ in range(10):
        data.extend([
            (id_cnt, "fish", "fish", "NN", "fish"),
            (id_cnt+1, ".", ".", ".", ".")
        ])
        id_cnt += 2
        
    # milk frequency: we already have 3 milk. Let's add 5 single "milk"
    for _ in range(5):
        data.extend([
            (id_cnt, "milk", "milk", "NN", "milk"),
            (id_cnt+1, ".", ".", ".", ".")
        ])
        id_cnt += 2

    # Fill remaining up to 200 total tokens with "the"
    while id_cnt < 200:
        data.append((id_cnt, "the", "the", "DT", "the"))
        id_cnt += 1

    for row in data:
        con.execute("INSERT INTO corpus VALUES (?, ?, ?, ?, ?)", row)
    
    con.close()

    try:
        # 1. Test sorting and calculation by Log-Likelihood (default)
        print("--- Testing Log-Likelihood ---")
        df_ll, freq, _ = generate_collocation_results(
            db_path, "cat", coll_window=2, mi_min_freq=1, max_collocates=10, is_raw_mode=False,
            stat_measure="log-likelihood"
        )
        print(df_ll[['Collocate', 'Observed', 'Total_Freq', 'LL', 'MI', 'Dice', 'Log-Dice']])
        assert df_ll.iloc[0]['Collocate'] == 'fish' or df_ll.iloc[0]['Collocate'] == 'milk'
        assert 'Dice' in df_ll.columns
        assert 'Log-Dice' in df_ll.columns

        # 2. Test sorting and calculation by Log-Dice
        print("\n--- Testing Log-Dice ---")
        df_ld, _, _ = generate_collocation_results(
            db_path, "cat", coll_window=2, mi_min_freq=1, max_collocates=10, is_raw_mode=False,
            stat_measure="log-dice"
        )
        print(df_ld[['Collocate', 'Observed', 'Total_Freq', 'Log-Dice']])
        # Verify first is sorted by Log-Dice descending
        for i in range(len(df_ld) - 1):
            assert df_ld.iloc[i]['Log-Dice'] >= df_ld.iloc[i+1]['Log-Dice']

        # 3. Test sorting and calculation by Dice Coefficient
        print("\n--- Testing Dice Coefficient ---")
        df_d, _, _ = generate_collocation_results(
            db_path, "cat", coll_window=2, mi_min_freq=1, max_collocates=10, is_raw_mode=False,
            stat_measure="dice coefficient"
        )
        print(df_d[['Collocate', 'Observed', 'Total_Freq', 'Dice']])
        for i in range(len(df_d) - 1):
            assert df_d.iloc[i]['Dice'] >= df_d.iloc[i+1]['Dice']

        # 4. Test sorting and calculation by Mutual Information
        print("\n--- Testing Mutual Information ---")
        df_mi, _, _ = generate_collocation_results(
            db_path, "cat", coll_window=2, mi_min_freq=1, max_collocates=10, is_raw_mode=False,
            stat_measure="mutual information"
        )
        print(df_mi[['Collocate', 'Observed', 'Total_Freq', 'MI']])
        for i in range(len(df_mi) - 1):
            assert df_mi.iloc[i]['MI'] >= df_mi.iloc[i+1]['MI']

        print("\nAll collocation association measure tests passed successfully!")
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

if __name__ == "__main__":
    test_measures()
