import duckdb
import time
import re
from collections import defaultdict

print("Starting test...")
start = time.time()

con = duckdb.connect('C:/Users/priha/Documents/cortex/corpora/english/EN-BPPT-tagged.db', read_only=True)
query = "SELECT filename, id, lower(lemma) as w, pos as p FROM corpus ORDER BY filename, id"
tokens = con.execute(query).fetchall()
con.close()

print(f"DuckDB fetch took: {time.time() - start:.2f}s")
fetch_end = time.time()

pos_re = re.compile('^(?:NN.*|NP.*|V.*|JJ.*|RB.*)$', re.IGNORECASE)

valid_cache = {}
def check_valid(w, p):
    if not w: return False
    key = (w, p)
    if key in valid_cache: return valid_cache[key]
    res = bool(pos_re.match(p))
    valid_cache[key] = res
    return res

valid_array = [check_valid(w, p) for _, _, w, p in tokens]
print(f"Valid array build took: {time.time() - fetch_end:.2f}s")
val_end = time.time()

target_nodes = set([w for _, _, w, p in tokens[:2000] if w])

pairs = defaultdict(lambda: defaultdict(int))
node_freq = defaultdict(int)
coll_freq = defaultdict(int)

N_tokens = len(tokens)
window_size = 5

for i in range(N_tokens):
    if not valid_array[i]: continue
    
    fn_i, id_i, w_i, p_i = tokens[i]
    coll_freq[w_i] += 1
        
    if w_i not in target_nodes:
        continue
        
    node_freq[w_i] += 1
    
    # Scan Left
    for j in range(max(0, i - window_size), i):
        if valid_array[j]:
            fn_j, id_j, w_j, p_j = tokens[j]
            if fn_j == fn_i:
                pairs[w_i][w_j] += 1
            
    # Scan Right
    for j in range(i + 1, min(N_tokens, i + window_size + 1)):
        if valid_array[j]:
            fn_j, id_j, w_j, p_j = tokens[j]
            if fn_j == fn_i:
                pairs[w_i][w_j] += 1

print(f"Python loop took: {time.time() - val_end:.2f}s")
print(f"Total time: {time.time() - start:.2f}s")
