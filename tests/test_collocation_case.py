import os
import sys
import duckdb
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.modules.collocation import generate_collocation_results

def test_case_insensitive_collocation_grouping():
    db_path = "test_colloc_case.db"
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
    
    # 3 "money bank ." + 2 "money Bank ." (separated by filler to prevent window overlap)
    data = []
    id_cnt = 1
    def add_pair(colloc):
        nonlocal id_cnt
        data.extend([
            (id_cnt, "money", "money", "NN", "money"),
            (id_cnt+1, colloc, colloc.lower(), "NN", colloc.lower()),
            (id_cnt+2, ".", ".", ".", "."),
            (id_cnt+3, ".", ".", ".", ".")
        ])
        id_cnt += 4

    for _ in range(3):
        add_pair("bank")
    for _ in range(2):
        add_pair("Bank")

    for row in data:
        con.execute("INSERT INTO corpus VALUES (?, ?, ?, ?, ?)", row)
    
    con.close()
    
    try:
        df_coll, freq, _ = generate_collocation_results(
            db_path, "money", coll_window=2, mi_min_freq=1, max_collocates=10, is_raw_mode=False
        )
        
        # Verify that 'bank' and 'Bank' are merged into a single collocate row ('bank')
        assert len(df_coll) == 1, f"Expected 1 merged collocate row, got {len(df_coll)}"
        row = df_coll.iloc[0]
        assert row['Collocate'] == 'bank'
        assert row['Observed'] == 5, f"Expected combined Observed count of 5, got {row['Observed']}"
        assert row['Obs_R'] == 5
        print("[OK] Case-insensitive collocation grouping and score adaptation verified successfully!")
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

if __name__ == "__main__":
    test_case_insensitive_collocation_grouping()
