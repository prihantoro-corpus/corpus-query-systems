import duckdb
import os
import pandas as pd
import numpy as np
from core.modules.keyword import generate_keyword_list

def setup_test_databases():
    target_db = "test_stable_target.duckdb"
    ref_db = "test_stable_ref.duckdb"
    
    # Clean up any existing files
    for db in [target_db, ref_db]:
        if os.path.exists(db):
            try: os.remove(db)
            except: pass

    # Create target database
    # N1 = 100 total tokens
    con_t = duckdb.connect(target_db)
    con_t.execute("CREATE TABLE corpus (_token_low VARCHAR)")
    data_t = []
    for _ in range(50): data_t.append(("the",))      # Stable
    for _ in range(30): data_t.append(("apple",))    # Positive
    for _ in range(2): data_t.append(("banana",))    # Negative
    for _ in range(18): data_t.append(("other",))    # Filler
    con_t.executemany("INSERT INTO corpus VALUES (?)", data_t)
    con_t.close()

    # Create reference database
    # N2 = 100 total tokens
    con_r = duckdb.connect(ref_db)
    con_r.execute("CREATE TABLE corpus (_token_low VARCHAR)")
    data_r = []
    for _ in range(50): data_r.append(("the",))      # Stable
    for _ in range(2): data_r.append(("apple",))     # Positive in target
    for _ in range(30): data_r.append(("banana",))   # Negative in target
    for _ in range(18): data_r.append(("other",))    # Filler
    con_r.executemany("INSERT INTO corpus VALUES (?)", data_r)
    con_r.close()

    return target_db, ref_db

def test_stable_keywords():
    target_db, ref_db = setup_test_databases()
    
    try:
        df = generate_keyword_list(target_db, ref_db_path=ref_db, min_freq=1)
        
        print("\n--- Keyword Generation Output ---")
        print(df[['token', 'freq_t', 'freq_r', 'LL', 'LogRatio', 'Type', 'Significance']])
        
        # Verify 'the' is Stable
        the_row = df[df['token'] == 'the']
        assert not the_row.empty, "'the' should be in the results"
        assert the_row.iloc[0]['Type'] == 'Stable', f"'the' should be Stable, got {the_row.iloc[0]['Type']}"
        assert the_row.iloc[0]['LL'] < 3.84, f"'the' LL should be < 3.84, got {the_row.iloc[0]['LL']}"
        
        # Verify 'apple' is Positive
        apple_row = df[df['token'] == 'apple']
        assert not apple_row.empty, "'apple' should be in the results"
        assert apple_row.iloc[0]['Type'] == 'Positive', f"'apple' should be Positive, got {apple_row.iloc[0]['Type']}"
        assert apple_row.iloc[0]['LL'] >= 3.84, f"'apple' LL should be >= 3.84, got {apple_row.iloc[0]['LL']}"
        
        # Verify 'banana' is Negative
        banana_row = df[df['token'] == 'banana']
        assert not banana_row.empty, "'banana' should be in the results"
        assert banana_row.iloc[0]['Type'] == 'Negative', f"'banana' should be Negative, got {banana_row.iloc[0]['Type']}"
        assert banana_row.iloc[0]['LL'] >= 3.84, f"'banana' LL should be >= 3.84, got {banana_row.iloc[0]['LL']}"

        print("[PASS] Stable Keyword Logic Verified Successfully!")

    finally:
        # Clean up database files
        for db in [target_db, ref_db]:
            if os.path.exists(db):
                try: os.remove(db)
                except: pass

if __name__ == "__main__":
    test_stable_keywords()
