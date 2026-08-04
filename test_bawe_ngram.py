import sys
import duckdb
sys.path.append(r'C:\Users\priha\Documents\cortex')
from core.modules.ngram import generate_n_grams_v2

# Find the BAWE db path by checking cortex config or just search for it
import glob
import os

db_files = glob.glob(r'C:\Users\priha\Documents\cortex\data\*\*.duckdb')
if not db_files:
    db_files = glob.glob(r'C:\Users\priha\Documents\cortex\*\*.duckdb')

print("Found DBs:", db_files)

# Let's run on the first one that seems like english bawe
for db in db_files:
    if 'bawe' in db.lower() or 'english' in db.lower():
        print(f"\n--- Testing on {db} ---")
        try:
            # Check POS column content
            con = duckdb.connect(db)
            print("Sample POS tags:", con.execute("SELECT pos FROM corpus WHERE pos IS NOT NULL LIMIT 5").fetchall())
            con.close()
            
            # Run ngram
            df = generate_n_grams_v2(
                corpus_db_path=db,
                n_size=4,
                n_gram_filters={},
                is_raw_mode=False,
                corpus_name="Test Corpus",
                basis="Token",
                positional_bases={'1': 'Token', '2': 'Token', '3': 'Part-of-Speech', '4': 'Token'}
            )
            print("First few N-Grams with POS on pos 3:")
            print(df.head())
            break
        except Exception as e:
            print(f"Error on {db}: {e}")
