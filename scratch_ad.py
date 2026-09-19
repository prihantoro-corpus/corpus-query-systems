import duckdb
import time

print("Starting Auto-Discover test...")
start = time.time()

con = duckdb.connect('C:/Users/priha/Documents/cortex/corpora/english/EN-BPPT-tagged.db', read_only=True)
query = """
SELECT lower(lemma) as l
FROM corpus 
WHERE 1=1  AND regexp_matches(pos, 'NN|NP|V|JJ|RB')
GROUP BY l 
HAVING COUNT(*) >= 20
ORDER BY COUNT(*) DESC
LIMIT 1000
"""
nodes_list = [row[0] for row in con.execute(query).fetchall() if row[0]]
con.close()

print(f"Auto-Discover took: {time.time() - start:.2f}s")
