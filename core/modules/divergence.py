import pandas as pd
import numpy as np
import duckdb
from collections import defaultdict
import re
import math
import time

def calculate_divergence_for_nodes(
    corpora_details, 
    nodes_list, # Not used in RAM mode if auto_discover is fully memory driven, but we'll adapt.
    window_size, 
    assoc_measure, 
    min_pair_freq, 
    min_collocate_freq, 
    top_k, 
    metric, 
    is_raw_mode,
    token_filter="", 
    pos_filter="", 
    lemma_filter="",
    min_node_collocates=10
):
    """
    In-memory calculation mirroring Paul Baker's SWAT RAM approach.
    Fetches corpus strings once, counts collocates in Python dictionaries for massive speedup.
    """
    
    # 1. Prepare Regexes for filters
    import re
    
    def compile_cortex_regex(f_str):
        if not f_str: return None, None
        f_str = f_str.replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace('|', ',')
        items = [i.strip() for i in f_str.split(',') if i.strip()]
        pos_items = [i for i in items if not i.startswith('-')]
        neg_items = [i[1:] for i in items if i.startswith('-')]
        
        pos_re = None
        neg_re = None
        
        if pos_items:
            escaped = [re.escape(i).replace(r'\*', '.*').replace(r'\?', '.') for i in pos_items]
            pos_re = re.compile('^(?:' + '|'.join(escaped) + ')$', re.IGNORECASE)
        if neg_items:
            escaped = [re.escape(i).replace(r'\*', '.*').replace(r'\?', '.') for i in neg_items]
            neg_re = re.compile('^(?:' + '|'.join(escaped) + ')$', re.IGNORECASE)
            
        return pos_re, neg_re

    coll_pos_re, coll_pos_neg = compile_cortex_regex(pos_filter)
    coll_tok_re, coll_tok_neg = compile_cortex_regex(token_filter)
    
    def is_valid_collocate(word, pos):
        if coll_pos_re and not coll_pos_re.match(pos): return False
        if coll_pos_neg and coll_pos_neg.match(pos): return False
        if coll_tok_re and not coll_tok_re.match(word): return False
        if coll_tok_neg and coll_tok_neg.match(word): return False
        return True

    # Master dictionaries
    corpus_matrices = {}
    
    # 2. Load and count in RAM for each corpus
    for idx, corpus in enumerate(corpora_details):
        c_name = corpus['name']
        c_path = corpus['path']
        xml_where = corpus.get('xml_where', "")
        xml_params = corpus.get('xml_params', [])
        
        con = duckdb.connect(c_path, read_only=True)
        cols_info = con.execute("PRAGMA table_info(corpus)").fetchall()
        cols = [c[1] for c in cols_info]
        has_lemma = 'lemma' in cols
        has_pos = 'pos' in cols
        
        node_col = "lower(lemma)" if has_lemma else "_token_low"
        pos_col = "pos" if has_pos else "''"
        
        query = f"SELECT filename, id, {node_col} as w, {pos_col} as p FROM corpus"
        if xml_where:
            query += f" WHERE 1=1 {xml_where}" if xml_where.strip().upper().startswith('AND') else f" WHERE {xml_where}"
        query += " ORDER BY filename, id"
        
        tokens = con.execute(query, xml_params).fetchall()
        con.close()
        
        if not tokens:
            corpus_matrices[c_name] = {'pairs': {}, 'node_freq': {}, 'coll_freq': {}, 'N': 0}
            continue
            
        pairs = defaultdict(lambda: defaultdict(int))
        node_freq = defaultdict(int)
        coll_freq = defaultdict(int)
        
        # Build node set for O(1) lookup
        target_nodes = set(nodes_list) if nodes_list else set()
        
        # Precompute valid array to avoid regex overhead in inner loop
        valid_cache = {}
        def check_valid(w, p):
            if not w: return False
            key = (w, p)
            if key in valid_cache: return valid_cache[key]
            res = is_valid_collocate(w, p)
            valid_cache[key] = res
            return res
            
        valid_array = [check_valid(w, p) for _, _, w, p in tokens]
        
        # In-memory sliding window
        N_tokens = len(tokens)
        
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
                    
        corpus_matrices[c_name] = {
            'pairs': pairs,
            'node_freq': node_freq,
            'coll_freq': coll_freq,
            'N': N_tokens
        }

    # 3. Calculate Association Scores & Divergence Math
    result_rows = []
    
    measure_lower = assoc_measure.lower()
    if "log-likelihood" in measure_lower: sort_col = "LL"
    elif "log-dice" in measure_lower: sort_col = "Log-Dice"
    elif "dice" in measure_lower: sort_col = "Dice"
    elif "mi" in measure_lower: sort_col = "MI"
    else: sort_col = "LL"
    
    def safe_ll(o, e):
        if o == 0 or e == 0: return 0.0
        return o * math.log(o / e)
        
    for node_input in nodes_list:
        corpus_top_k = {}
        valid_corpora_count = 0
        
        for corpus in corpora_details:
            c_name = corpus['name']
            mat = corpus_matrices[c_name]
            
            pairs = mat['pairs']
            node_freq = mat['node_freq'].get(node_input, 0)
            coll_freq = mat['coll_freq']
            N = mat['N']
            
            if node_freq == 0:
                corpus_top_k[c_name] = []
                continue
                
            # Score all collocates
            scored_collocates = []
            if node_input in pairs:
                for c, obs in pairs[node_input].items():
                    if obs < min_pair_freq: continue
                    
                    c_freq = coll_freq.get(c, 0)
                    c_freq = max(c_freq, obs) # Safety
                    if c_freq < min_collocate_freq: continue
                    
                    # Math
                    k11 = obs
                    k12 = max(0, node_freq - k11)
                    k21 = max(0, c_freq - k11)
                    k22 = max(0, N - (k11 + k12 + k21))
                    
                    R1 = k11 + k12
                R2 = k21 + k22
                C1 = k11 + k21
                C2 = k12 + k22
                
                E11 = (R1 * C1) / N if N > 0 else 0
                E12 = (R1 * C2) / N if N > 0 else 0
                E21 = (R2 * C1) / N if N > 0 else 0
                E22 = (R2 * C2) / N if N > 0 else 0
                
                ll = 2 * (safe_ll(k11, E11) + safe_ll(k12, E12) + safe_ll(k21, E21) + safe_ll(k22, E22))
                mi = math.log2(k11 / E11) if E11 > 0 and k11 > 0 else 0
                dice = (2 * k11) / (node_freq + c_freq) if (node_freq + c_freq) > 0 else 0
                log_dice = 14 + math.log2(dice) if dice > 0 else 0
                
                scores = {'LL': ll, 'MI': mi, 'Dice': dice, 'Log-Dice': log_dice}
                scored_collocates.append((c, scores[sort_col]))
                
            if len(scored_collocates) < min_node_collocates:
                corpus_top_k[c_name] = []
                continue
                
            # Sort and Top-K
            scored_collocates.sort(key=lambda x: x[1], reverse=True)
            top = scored_collocates[:top_k]
            corpus_top_k[c_name] = top
            valid_corpora_count += 1
            
        if valid_corpora_count < 2:
            continue
            
        # 4. Math for Divergence
        all_collocates = set()
        for top in corpus_top_k.values():
            all_collocates.update([x[0] for x in top])
            
        master_list = list(all_collocates)
        
        if metric == 'Generalized Jaccard':
            sets = [set([x[0] for x in top]) for top in corpus_top_k.values()]
            intersection = set.intersection(*sets) if sets else set()
            union = set.union(*sets) if sets else set()
            jaccard_score = len(intersection) / len(union) if union else 0
            div_score = 1.0 - jaccard_score
            
            row = {'Node': node_input, 'Divergence Score': round(div_score, 4)}
            for c_name, top in corpus_top_k.items():
                row[f'{c_name} Collocates'] = ", ".join([x[0] for x in top])
            result_rows.append(row)
            
        elif metric == 'Jensen-Shannon Divergence':
            distributions = []
            for c_name, top in corpus_top_k.items():
                dist = []
                if not top:
                    dist = [0.0] * len(master_list)
                else:
                    score_map = dict(top)
                    for word in master_list:
                        val = max(0.0, score_map.get(word, 0.0))
                        dist.append(val)
                dist = np.array(dist)
                s = dist.sum()
                if s > 0: dist = dist / s
                else: dist = np.zeros(len(master_list))
                distributions.append(dist)
                
            distributions = np.array(distributions)
            M = np.mean(distributions, axis=0)
            
            jsd_sum = 0.0
            n_corpora = len(corpus_top_k)
            for P in distributions:
                mask = P > 0
                if np.any(mask):
                    jsd_sum += np.sum(P[mask] * np.log2(P[mask] / M[mask]))
            jsd_score = jsd_sum / n_corpora
            
            row = {'Node': node_input, 'Divergence Score': round(jsd_score, 4)}
            for c_name, top in corpus_top_k.items():
                row[f'{c_name} Collocates'] = ", ".join([x[0] for x in top])
            result_rows.append(row)

    if not result_rows:
        return pd.DataFrame()
        
    final_df = pd.DataFrame(result_rows)
    return final_df.sort_values('Divergence Score', ascending=False)
