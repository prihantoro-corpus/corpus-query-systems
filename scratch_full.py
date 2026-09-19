import time
import sys
sys.path.append('C:/Users/priha/Documents/cortex')

from core.modules.divergence import calculate_divergence_for_nodes

print("Testing full RAM calculate_divergence_for_nodes...")

corpora_details = [
    {'path': 'C:/Users/priha/Documents/cortex/corpora/english/EN-BPPT-tagged.db', 'name': 'Corpus 1', 'xml_where': '', 'xml_params': [], 'stats': None},
    {'path': 'C:/Users/priha/Documents/cortex/corpora/english/EN-BPPT-tagged.db', 'name': 'Corpus 2', 'xml_where': '', 'xml_params': [], 'stats': None}
]

import duckdb
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

start = time.time()
df = calculate_divergence_for_nodes(
    corpora_details=corpora_details,
    nodes_list=nodes_list,
    window_size=5,
    assoc_measure='Log-Likelihood',
    min_pair_freq=5,
    min_collocate_freq=10,
    top_k=20,
    metric='Jensen-Shannon Divergence',
    is_raw_mode=False,
    token_filter="-(the,and,of,to,in,a,is,that,it,for)",
    pos_filter="NN*|NP*|V*|JJ*|RB*",
    lemma_filter="",
    min_node_collocates=10
)
print(f"calculate_divergence_for_nodes took {time.time() - start:.2f}s")
print(f"Results rows: {len(df)}")
