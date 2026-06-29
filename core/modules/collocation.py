import duckdb
import pandas as pd
import numpy as np
import re
import streamlit as st # Only for @st.cache_data? No, avoiding st dependency in core.

def apply_smart_filter(df, col_name, filter_str):
    """
    Applies inclusive and/or exclusive filtering based on user input.
    Format: 'word1, word2' (include only these) AND/OR '-word1, -word2' (exclude these).
    Supports * and ? wildcards (e.g., '*al' for words ending in al).
    Comparisons are case-insensitive.
    """
    if not filter_str or df.empty:
        return df
    
    # Distribute minus sign outside parenthesis, e.g. -(JJ|NN) -> -JJ,-NN
    import re
    filter_str = re.sub(r'-\(([^)]+)\)', lambda m: ",".join(f"-{p.strip()}" if not p.strip().startswith('-') else p.strip() for p in re.split(r'[|,]', m.group(1)) if p.strip()), filter_str)
    filter_str = re.sub(r'-\[([^\]]+)\]', lambda m: ",".join(f"-{p.strip()}" if not p.strip().startswith('-') else p.strip() for p in re.split(r'[|,]', m.group(1)) if p.strip()), filter_str)

    # Support union by replacing | with , and stripping ()
    filter_str = filter_str.replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace('|', ',')
    
    items = [i.strip() for i in filter_str.split(',') if i.strip()]
    if not items:
        return df
    
    pos_items = []
    neg_items = []
    for item in items:
        if item.startswith('-'):
            clean = item[1:].strip()
            if clean:
                neg_items.append(clean)
        else:
            clean = item.strip()
            if clean:
                pos_items.append(clean)
                
    # Helper to check matches for a list of items (mixed wildcard and plain)
    def get_matches_mask(items_list):
        has_wildcard = any('*' in i or '?' in i for i in items_list)
        if has_wildcard:
            escaped_items = [re.escape(i).replace(r'\*', '.*').replace(r'\?', '.') for i in items_list]
            regex_pat = '^(?:' + '|'.join(escaped_items) + ')$'
            return df[col_name].astype(str).str.contains(regex_pat, flags=re.IGNORECASE, regex=True)
        else:
            items_lower = [i.lower() for i in items_list]
            return df[col_name].astype(str).str.lower().isin(items_lower)

    if pos_items:
        df = df[get_matches_mask(pos_items)]
        
    if neg_items:
        df = df[~get_matches_mask(neg_items)]
        
    return df

def generate_collocation_results(corpus_db_path, raw_target_input, coll_window, mi_min_freq, max_collocates, is_raw_mode, 
                                 token_filter="", pos_filter="", lemma_filter="",
                                 corpus_stats=None, xml_where_clause="", xml_params=[],
                                 stat_measure="log-likelihood"):
    """
    Generalized function to run collocation analysis using DuckDB.
    Returns: (stats_df_sorted, freq, primary_target_mwu)
    """
    try:
        con = duckdb.connect(corpus_db_path, read_only=True)
        import math
        total_tokens = con.execute("SELECT count(*) FROM corpus").fetchone()[0]
        
        # 1. Robust Query Tokenization
        query_pattern = r'<[^>]+>|[^\s]+'
        search_terms = re.findall(query_pattern, raw_target_input)
        primary_target_len = len(search_terms)
        
        # Check raw mode (introspect table if metadata not provided)
        is_raw_mode_active = is_raw_mode
        if not is_raw_mode_active:
            try:
                cols_info = con.execute("PRAGMA table_info(corpus)").fetchall()
                cols = [c[1] for c in cols_info]
                if 'pos' in cols:
                    raw_count = con.execute("SELECT count(*) FROM corpus WHERE pos LIKE '##%'").fetchone()[0]
                    total_rows = con.execute("SELECT count(*) FROM corpus").fetchone()[0]
                    if total_rows > 0:
                        is_raw_mode_active = (raw_count / total_rows) > 0.99
            except: pass

        def parse_term(term):
            # XML Tag Check: Handles <TAG>, <TAG ATTR="VAL">, and <TAG="VAL">
            xml_tag_match = re.match(r'<(\w+)(?:[=\s](.+?))?>', term, re.IGNORECASE)
            if xml_tag_match:
                tag_name = xml_tag_match.group(1).lower()
                val_or_attrs = xml_tag_match.group(2)
                attrs = {}
                if val_or_attrs:
                    # Case 1: <TAG="VALUE">
                    if not re.search(r'\w+=', val_or_attrs):
                        val = val_or_attrs.strip()
                        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                            val = val[1:-1]
                        attrs['value'] = val
                    else:
                        # Case 2: <TAG ATTR="VALUE">
                        attr_pattern = r'(\w+)=(["\'])([^"\']*)\2'
                        for match in re.finditer(attr_pattern, val_or_attrs):
                            attr_key = match.group(1).lower()
                            attr_val = match.group(3)
                            attrs[attr_key] = attr_val
                return {'type': 'xml_tag', 'tag': tag_name, 'attrs': attrs}
            
            # Same logic as concordance.py
            lemma_match = re.search(r"\[(.*?)\]", term)
            # Find starting index of bracket if any
            bracket_idx = term.find('[')
            
            if lemma_match:
                inner_val = lemma_match.group(1).strip()
                if '_' in inner_val and not inner_val.startswith('_'):
                    parts = inner_val.rsplit('_', 1)
                    if len(parts) == 2 and parts[1]:
                        prefix = term[:bracket_idx] if bracket_idx > 0 else ""
                        suffix = term[term.find(']')+1:]
                        return {
                            'type': 'lemma_pos',
                            'lemma': prefix + parts[0].lower() + suffix,
                            'pos': parts[1]
                        }
                val = inner_val.lower()
                # Check for suffix/prefix wildcards outside brackets
                prefix = term[:bracket_idx] if bracket_idx > 0 else ""
                suffix = term[term.find(']')+1:]
                return {'type': 'lemma', 'val': prefix + val + suffix}
            
            # Combined Token_POS Check (e.g. light_V*)
            if '_' in term and not term.startswith('_'):
                 bracket_idx = term.find('[')
                 if bracket_idx == -1: # Only if not lemma
                     parts = term.rsplit('_', 1)
                     if len(parts) == 2 and parts[1]:
                          return {'type': 'token_pos', 'token': parts[0].lower(), 'pos': parts[1]}

            pos_match = re.search(r"\_([A-Za-z0-9\*|\\%_-]+)", term)
            underscore_idx = term.find('_')
            if pos_match:
                val = pos_match.group(1).strip()
                prefix = term[:underscore_idx] if underscore_idx > 0 else ""
                suffix = term[term.find(val)+len(val):]
                return {'type': 'pos', 'val': prefix + val + suffix}
                
            return {'type': 'word', 'val': term.lower()}

        search_components = [parse_term(term) for term in search_terms]

        # 2. Build Query with Dynamic Lengths
        current_offset_exprs = []
        query_joins = ""
        
        for k, comp in enumerate(search_components):
            alias = f"c{k}"
            
            if comp['type'] == 'xml_tag':
                tag_name = comp['tag']
                tag_len_col = f"{tag_name}_len"
                if tag_len_col in cols:
                    current_offset_exprs.append(f"COALESCE({alias}.{tag_len_col}, 1)")
                else:
                    current_offset_exprs.append("1")
            else:
                current_offset_exprs.append("1")
            
            if k > 0:
                prev_alias = f"c{k-1}"
                prev_offset = current_offset_exprs[k-1]
                query_joins += f" JOIN corpus {alias} ON {alias}.id = {prev_alias}.id + {prev_offset} "
        
        # Calculate TOTAL match length expression
        total_len_expr = " + ".join(current_offset_exprs)
        
        query_select = f"SELECT DISTINCT c0.id, {total_len_expr} as total_len FROM corpus c0"
        query_where = []
        query_params = []
        
        for k, comp in enumerate(search_components):
            alias = f"c{k}"
            
            if comp['type'] == 'token_pos':
                 t_val = comp['token']
                 p_val = comp['pos']
                 
                 # Token
                 pat = re.escape(t_val).replace(r'\*', '.*').replace(r'\%', '.*').replace(r'\_', '.')
                 query_where.append(f"regexp_matches({alias}._token_low, ?)")
                 query_params.append('^' + pat + '$')
                 
                 # POS
                 if not is_raw_mode_active:
                     pat_pos = re.escape(p_val).replace(r'\*', '.*').replace(r'\%', '.*').replace(r'\_', '.')
                     if '|' in p_val:
                         pat_pos = pat_pos.replace(r'\|', '|')
                     query_where.append(f"regexp_matches({alias}.pos, ?)")
                     query_params.append('(?i)^' + pat_pos + '$')
            
            elif comp['type'] == 'lemma_pos':
                 l_val = comp['lemma']
                 p_val = comp['pos']
                 
                 # Lemma
                 if not is_raw_mode_active:
                     pat = re.escape(l_val).replace(r'\*', '.*').replace(r'\%', '.*').replace(r'\_', '.')
                     if '|' in l_val:
                         pat = pat.replace(r'\|', '|')
                     query_where.append(f"regexp_matches(lower({alias}.lemma), ?)")
                     query_params.append('^' + pat + '$')
                 else:
                     pat = re.escape(l_val).replace(r'\*', '.*').replace(r'\%', '.*').replace(r'\_', '.')
                     if '|' in l_val:
                         pat = pat.replace(r'\|', '|')
                     query_where.append(f"regexp_matches({alias}._token_low, ?)")
                     query_params.append('^' + pat + '$')
                 
                 # POS
                 if not is_raw_mode_active:
                     pat_pos = re.escape(p_val).replace(r'\*', '.*').replace(r'\%', '.*').replace(r'\_', '.')
                     if '|' in p_val:
                         pat_pos = pat_pos.replace(r'\|', '|')
                     query_where.append(f"regexp_matches({alias}.pos, ?)")
                     query_params.append('(?i)^' + pat_pos + '$')
            
            elif comp['type'] == 'xml_tag':
                tag_name = comp['tag']
                attrs = comp['attrs']
                
                # Dynamic check: Is this a span-based tag or a token-level property?
                tag_start_col = f"in_{tag_name}_start"
                if tag_start_col in cols:
                    # Span-based tag (standard XML)
                    query_where.append(f"{alias}.{tag_start_col} = TRUE")
                    for attr_key, attr_val in attrs.items():
                        attr_col = f"{tag_name}_{attr_key}"
                        if '*' in attr_val:
                            regex_pat = '^' + re.escape(attr_val).replace(r'\*', '.*') + '$'
                            query_where.append(f"regexp_matches({alias}.{attr_col}, ?)")
                            query_params.append(regex_pat)
                        else:
                            query_where.append(f"{alias}.{attr_col} = ?")
                            query_params.append(attr_val)
                elif tag_name in cols:
                    # Token-level property (e.g. <dep_rel="nsubj">)
                    target_val = attrs.get('value') or (list(attrs.values())[0] if attrs else None)
                    if target_val:
                        if '*' in target_val:
                            regex_pat = '^' + re.escape(target_val).replace(r'\*', '.*') + '$'
                            query_where.append(f"regexp_matches({alias}.{tag_name}, ?)")
                            query_params.append(regex_pat)
                        else:
                            query_where.append(f"{alias}.{tag_name} = ?")
                            query_params.append(target_val)
                    else:
                        query_where.append(f"{alias}.{tag_name} IS NOT NULL")
                else:
                    query_where.append("1=0")
            
            else:
                val = comp['val']
                # Treat *, %, _ as wildcards, and | as alternation
                is_wildcard = any(c in val for c in ('*', '%', '_', '|'))
                
                col_target = f"{alias}._token_low"
                if comp['type'] == 'lemma' and not is_raw_mode_active: col_target = f"lower({alias}.lemma)"
                if comp['type'] == 'pos' and not is_raw_mode_active: col_target = f"{alias}.pos"
                
                if is_wildcard:
                    # Convert glob/SQL wildcards to Regex
                    pat = re.escape(val).replace(r'\*', '.*').replace(r'\%', '.*').replace(r'\_', '.')
                    if '|' in val:
                        pat = pat.replace(r'\|', '|')
                    
                    regex_pat = ('(?i)^' if comp['type'] == 'pos' else '^') + pat + '$'
                    query_where.append(f"regexp_matches({col_target}, ?)")
                    query_params.append(regex_pat)
                else:
                    if comp['type'] == 'pos' and not is_raw_mode_active:
                        query_where.append(f"regexp_matches({col_target}, ?)")
                        query_params.append('(?i)^' + re.escape(val) + '$')
                    else:
                        query_where.append(f"{col_target} = ?")
                        query_params.append(val)

        try: con.execute("DROP TABLE IF EXISTS search_matches")
        except: pass
        
        # Safely inject alias into xml_where_clause to avoid ambiguous column errors
        c0_xml_where = ""
        if xml_where_clause:
            try:
                cols_info = con.execute("PRAGMA table_info(corpus)").fetchall()
                meta_cols = [c[1] for c in cols_info if c[1] not in ('id', 'token', 'pos', 'lemma', 'sent_id', '_token_low', 'filename')]
                c0_xml_where = xml_where_clause
                for col in meta_cols:
                    c0_xml_where = re.sub(rf'"{col}"|\b{col}\b', lambda m: f'c0.{m.group(0)}', c0_xml_where, flags=re.IGNORECASE)
            except:
                c0_xml_where = xml_where_clause

        match_query = f"CREATE TEMP TABLE search_matches AS {query_select} {query_joins}"
        if query_where:
            match_query += " WHERE " + " AND ".join(query_where)
            if c0_xml_where: match_query += c0_xml_where
        elif c0_xml_where:
            match_query += " WHERE " + c0_xml_where.strip()[4:]
        
        full_params = query_params + xml_params
        
        try:
            with open(r"c:\Users\priha\Documents\cortex\debug_query.log", "a", encoding="utf-8") as f_debug:
                f_debug.write(f"--- COLLOCATION START ---\nTarget: {raw_target_input}\nQuery: {match_query}\nParams: {full_params}\n")
        except Exception as log_err:
            pass
            
        con.execute(match_query, full_params)
        
        if xml_where_clause:
            freq = con.execute("SELECT count(*) FROM search_matches").fetchone()[0]
            N = con.execute(f"SELECT count(*) FROM corpus WHERE 1=1 {xml_where_clause}", xml_params).fetchone()[0]
        else:
            freq = con.execute("SELECT count(*) FROM search_matches").fetchone()[0]
            N = total_tokens
        
        if freq == 0:
            con.close()
            return (pd.DataFrame(), 0, raw_target_input)

        collocation_query = f"""
        SELECT 
            c2.token as w, 
            c2.pos as p, 
            lower(c2.lemma) as l, 
            CASE WHEN c2.id < m.id THEN 'L' ELSE 'R' END as direction,
            count(*) as obs
        FROM search_matches m
        JOIN corpus c2 ON c2.id BETWEEN m.id - {coll_window} AND m.id + m.total_len + {coll_window} - 1
        WHERE 
            (c2.id < m.id OR c2.id >= m.id + m.total_len) -- Exclude node
            AND NOT regexp_matches(c2._token_low, '^[[:punct:]]+$') 
            AND NOT regexp_matches(c2._token_low, '^[0-9]+$') 
        GROUP BY 1, 2, 3, 4
        HAVING count(*) >= {mi_min_freq}
        """
        
        df_coll_counts = con.execute(collocation_query).fetch_df()
        
        # If restricted, we MUST fetch frequencies of these collocates WITHIN THE REGION
        region_freqs = {}
        if not df_coll_counts.empty and xml_where_clause:
            unique_colls = df_coll_counts['w'].str.lower().unique().tolist()
            # To avoid massive IN clauses, we can use a temp table or chunked query. 
            # For typical max_collocates (100-500), chunking or a simple temp table is safe.
            con.execute("CREATE TEMP TABLE coll_names(name VARCHAR)")
            # DuckDB allows bulk insert from list
            # Build values list separately to avoid escaping issues
            values_parts = []
            for n in unique_colls:
                escaped_n = n.replace("'", "''")
                values_parts.append(f"('{escaped_n}')")
            con.execute("INSERT INTO coll_names SELECT * FROM (VALUES " + ", ".join(values_parts) + ")")

            
            freq_sql = f"""
            SELECT _token_low, count(*) as f 
            FROM corpus 
            WHERE _token_low IN (SELECT name FROM coll_names) {xml_where_clause} 
            GROUP BY 1
            """
            region_freqs = dict(con.execute(freq_sql, xml_params).fetchall())
            con.execute("DROP TABLE coll_names")

        try: con.execute("DROP TABLE search_matches")
        except: pass
        con.close()
        
        if df_coll_counts.empty:
            return (pd.DataFrame(), freq, raw_target_input)

        stats_df = df_coll_counts.pivot_table(
            index=['w', 'p', 'l'], 
            columns='direction', 
            values='obs', 
            aggfunc='sum', 
            fill_value=0
        ).reset_index()
        
        if 'L' not in stats_df.columns: stats_df['L'] = 0
        if 'R' not in stats_df.columns: stats_df['R'] = 0
        
        stats_df['Observed'] = stats_df['L'] + stats_df['R']
        stats_df.rename(columns={'w': 'Collocate', 'p': 'POS', 'l': 'Lemma', 'L': 'Obs_L', 'R': 'Obs_R'}, inplace=True)
        stats_df['Collocate_low'] = stats_df['Collocate'].str.lower()
        
        if mi_min_freq > 1:
            stats_df = stats_df[stats_df['Observed'] >= mi_min_freq]

        if stats_df.empty: 
            return (pd.DataFrame(), freq, raw_target_input)

        if xml_where_clause:
            stats_df['Total_Freq'] = stats_df['Collocate_low'].map(region_freqs).fillna(0).astype(int)
        else:
            token_counts_unfiltered = corpus_stats.get('token_counts', {}) if corpus_stats else {}
            stats_df['Total_Freq'] = stats_df['Collocate_low'].map(token_counts_unfiltered).fillna(0).astype(int)
        
        stats_df['Total_Freq'] = np.maximum(stats_df['Total_Freq'], stats_df['Observed'])
        
        from core.statistics.association import safe_ll_term, vec_sig
        
        # N is already set above (either global or restricted)
        k11 = stats_df['Observed']
        k12 = freq - k11
        k21 = stats_df['Total_Freq'] - k11
        k22 = N - (k11 + k12 + k21).clip(lower=0)
        
        R1 = k11 + k12; R2 = k21 + k22
        C1 = k11 + k21; C2 = k12 + k22
        
        with np.errstate(divide='ignore', invalid='ignore'):
            E11 = (R1 * C1) / N
            # E12, E21, E22 calculation if needed, but safe_ll_term only needs observed and expected for the cell
            # Formula for LL is sum of 4 cells.
            E12 = (R1 * C2) / N
            E21 = (R2 * C1) / N
            E22 = (R2 * C2) / N

            ll_vec = 2 * (safe_ll_term(k11, E11) + safe_ll_term(k12, E12) + safe_ll_term(k21, E21) + safe_ll_term(k22, E22))
            mi_vec = np.log2(k11 / E11)
            dice_vec = (2 * k11) / (freq + stats_df['Total_Freq'])
            log_dice_vec = 14 + np.log2(dice_vec)

        stats_df['LL'] = np.nan_to_num(ll_vec).round(6)
        stats_df['MI'] = np.where((k11 > 0) & (E11 > 0), mi_vec, 0.0).round(6)
        stats_df['Dice'] = np.nan_to_num(dice_vec).round(6)
        stats_df['Log-Dice'] = np.where(dice_vec > 0, np.nan_to_num(log_dice_vec), 0.0).round(6)

        stats_df['Direction'] = np.where(stats_df['Obs_R'] > stats_df['Obs_L'], 'R', 
                                np.where(stats_df['Obs_L'] > stats_df['Obs_R'], 'L', 'B'))
        
        stats_df['Significance'] = stats_df['LL'].apply(vec_sig)
        
        # Apply Smart Filters
        stats_df = apply_smart_filter(stats_df, 'Collocate', token_filter)
        stats_df = apply_smart_filter(stats_df, 'POS', pos_filter)
        stats_df = apply_smart_filter(stats_df, 'Lemma', lemma_filter)
        
        if stats_df.empty:
            return (pd.DataFrame(), freq, raw_target_input)
            
        # Determine sorting column based on selected measure
        measure_lower = stat_measure.lower()
        if "log-likelihood" in measure_lower:
            sort_col = "LL"
        elif "log-dice" in measure_lower:
            sort_col = "Log-Dice"
        elif "dice coefficient" in measure_lower or measure_lower == "dice":
            sort_col = "Dice"
        elif "mutual information" in measure_lower or measure_lower == "mi":
            sort_col = "MI"
        else:
            sort_col = "LL"

        stats_df_sorted = stats_df.sort_values(sort_col, ascending=False)
        if max_collocates > 0:
            stats_df_sorted = stats_df_sorted.head(int(max_collocates))
        
        return (stats_df_sorted, freq, raw_target_input)

    except Exception as e:
        import streamlit as st
        st.error(f"Collocation Analysis Error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return (pd.DataFrame(), 0, raw_target_input)
