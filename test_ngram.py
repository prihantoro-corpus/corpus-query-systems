import sys
sys.path.append(r'C:\Users\priha\Documents\cortex')
from core.modules.ngram import generate_n_grams_v2
import pandas as pd

corpus_db_path = r'C:\Users\priha\Documents\cortex\dummy.duckdb'

try:
    df = generate_n_grams_v2(
        corpus_db_path=corpus_db_path,
        n_size=4,
        n_gram_filters={},
        is_raw_mode=False,
        corpus_name="Test",
        basis="Token",
        positional_bases={'1': 'Token', '2': 'Token', '3': 'Part-of-Speech', '4': 'Token'},
    )
    print("DataFrame generated successfully")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
    if not df.empty:
        print("First NGram value:", df['N-Gram'].iloc[0])
except Exception as e:
    print("Error:", e)
