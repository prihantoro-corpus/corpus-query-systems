import duckdb
import os
import tempfile
import glob

# Try to find recent duckdb files in temp
temp_dir = tempfile.gettempdir()
db_files = glob.glob(os.path.join(temp_dir, "corpus_*.duckdb"))
db_files.sort(key=os.path.getmtime, reverse=True)

if not db_files:
    print("No recent corpus databases found in temp.")
else:
    db_path = db_files[0]
    print(f"Inspecting database: {db_path}")
    con = duckdb.connect(db_path, read_only=True)
    try:
        cols = con.execute("PRAGMA table_info(corpus)").fetch_df()
        print("Columns in 'corpus' table:")
        print(cols[['name', 'type']])
        
        # Check for dep_rel values
        if 'dep_rel' in cols['name'].values:
            stats = con.execute("SELECT dep_rel, COUNT(*) as count FROM corpus WHERE dep_rel IS NOT NULL GROUP BY dep_rel ORDER BY count DESC LIMIT 10").fetch_df()
            print("\nTop 10 Dependency Relations:")
            print(stats)
        else:
            print("\n'dep_rel' column NOT found.")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        con.close()
