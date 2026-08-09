import duckdb
import pandas as pd
import numpy as np
from core.statistics.association import safe_ll_term, vec_sig

def generate_keyword_list(target_db_path, ref_db_path=None, target_xml_where="", target_xml_params=[], ref_xml_where="", ref_xml_params=[], min_freq=3, ref_freq_df=None, ref_total_tokens=0):
    """
    Generates a keyword list by comparing the target corpus against a reference corpus.
    Calculates Log Likelihood (LL) and Log Ratio.
    
    Args:
        target_db_path (str): Path to target corpus DB.
        ref_db_path (str, optional): Path to reference corpus DB.
        target_xml_where (str): XML filter clause for target.
        ref_xml_where (str): XML filter clause for reference.
        min_freq (int): Minimum frequency in target corpus to be considered.
        ref_freq_df (pd.DataFrame, optional): Pre-calculated frequency list for reference.
        ref_total_tokens (int, optional): Total token count for reference if ref_freq_df is used.
    """
    if not target_db_path or (not ref_db_path and ref_freq_df is None):
        return pd.DataFrame()

    try:
        # 1. Get Target Counts
        con_t = duckdb.connect(target_db_path, read_only=True)
        
        sql_t = f"""
        SELECT _token_low as token, count(*) as freq 
        FROM corpus 
        WHERE NOT regexp_matches(_token_low, '^[[:punct:]]+$') 
          AND NOT regexp_matches(_token_low, '^[0-9]+$')
          {target_xml_where}
        GROUP BY 1
        HAVING count(*) >= ?
        """
        df_target = con_t.execute(sql_t, target_xml_params + [min_freq]).fetch_df()
        
        if target_xml_where:
            total_target = con_t.execute(f"SELECT count(*) FROM corpus WHERE 1=1 {target_xml_where}", target_xml_params).fetchone()[0]
        else:
            total_target = con_t.execute("SELECT count(*) FROM corpus").fetchone()[0]
        
        con_t.close()
        
        if df_target.empty:
            return pd.DataFrame()
        df_target = df_target.rename(columns={'freq': 'freq_t'})
            
        # 2. Get Reference Counts
        if ref_freq_df is not None:
            df_ref = ref_freq_df.copy()
            total_ref = ref_total_tokens if ref_total_tokens > 0 else df_ref['freq'].sum()
        else:
            con_r = duckdb.connect(ref_db_path, read_only=True)
            sql_r = f"""
            SELECT _token_low as token, count(*) as freq 
            FROM corpus 
            WHERE NOT regexp_matches(_token_low, '^[[:punct:]]+$') 
              AND NOT regexp_matches(_token_low, '^[0-9]+$')
              {ref_xml_where}
            GROUP BY 1
            """
            df_ref = con_r.execute(sql_r, ref_xml_params).fetch_df()
            
            if ref_xml_where:
                total_ref = con_r.execute(f"SELECT count(*) FROM corpus WHERE 1=1 {ref_xml_where}", ref_xml_params).fetchone()[0]
            else:
                total_ref = con_r.execute("SELECT count(*) FROM corpus").fetchone()[0]
                
            con_r.close()
        df_ref = df_ref.rename(columns={'freq': 'freq_r'})

        # 3. Optimize: Only merge target tokens and top 1000 most frequent reference tokens
        # This keeps the merge size reasonable and mathematically captures all possible significant negative keywords.
        top_ref_tokens = set()
        if not df_ref.empty:
            top_ref_tokens = set(df_ref.sort_values('freq_r', ascending=False).head(1000)['token'])
            
        target_tokens = set(df_target['token'])
        keep_tokens = target_tokens.union(top_ref_tokens)
        df_ref_filtered = df_ref[df_ref['token'].isin(keep_tokens)].copy()
        
        merged = pd.merge(df_target, df_ref_filtered, on='token', how='outer')
        merged['freq_t'] = merged['freq_t'].fillna(0)
        merged['freq_r'] = merged['freq_r'].fillna(0)
        
        O1 = merged['freq_t']
        O2 = merged['freq_r']
        N1 = total_target
        N2 = total_ref
        
        E1 = N1 * (O1 + O2) / (N1 + N2)
        E2 = N2 * (O1 + O2) / (N1 + N2)
        
        term1 = np.where(O1 > 0, O1 * np.log(O1/E1), 0.0)
        term2 = np.where(O2 > 0, O2 * np.log(O2/E2), 0.0)
        merged['LL'] = 2 * (term1 + term2)
        
        rel_t_smooth = (O1 + 0.5) / (N1 + 0.5)
        rel_r_smooth = (O2 + 0.5) / (N2 + 0.5)
        merged['LogRatio'] = np.log2(rel_t_smooth / rel_r_smooth)
        
        # Vectorized significance check
        conditions = [
            merged['LL'] >= 15.13,
            merged['LL'] >= 10.83,
            merged['LL'] >= 3.84
        ]
        choices = ['*** (p<0.001)', '** (p<0.01)', ' * (p<0.05)']
        merged['Significance'] = np.select(conditions, choices, default='ns')
        
        # Categorize
        merged['Type'] = np.where(
            merged['LL'] < 3.84,
            'Stable',
            np.where(merged['LogRatio'] > 0, 'Positive', 'Negative')
        )
        
        # Sort by LL (absolute strength of keyness)
        merged = merged.sort_values("LL", ascending=False)
        
        return merged
        
    except Exception as e:
        print(f"Keyword calc error: {e}")
        return pd.DataFrame()

def generate_grouped_keyword_list(target_db_path, group_by_col, ref_db_path=None, target_xml_where="", target_xml_params=[], ref_xml_where="", ref_xml_params=[], min_freq=3, ref_freq_df=None, ref_total_tokens=0):
    """
    Generates a dictionary of keyword lists, grouped by a specific column (e.g., filename, author).
    Returns: { 'group_value': pd.DataFrame }
    """
    if not target_db_path or not group_by_col:
        return {}

    results = {}
    
    try:
        con_t = duckdb.connect(target_db_path, read_only=True)
        
        # 1. Verify column exists first
        try:
             con_t.execute(f"SELECT \"{group_by_col}\" FROM corpus LIMIT 1")
        except:
             print(f"Column {group_by_col} not found in corpus.")
             con_t.close()
             return {}

        # 2. Get Global Reference Counts (Once)
        if ref_freq_df is not None:
            df_ref = ref_freq_df.copy()
            total_ref = ref_total_tokens if ref_total_tokens > 0 else df_ref['freq'].sum()
        else:
            con_r = duckdb.connect(ref_db_path, read_only=True)
            sql_r = f"""
            SELECT _token_low as token, count(*) as freq 
            FROM corpus 
            WHERE NOT regexp_matches(_token_low, '^[[:punct:]]+$') 
              AND NOT regexp_matches(_token_low, '^[0-9]+$')
              {ref_xml_where}
            GROUP BY 1
            """
            df_ref = con_r.execute(sql_r, ref_xml_params).fetch_df()
            
            if ref_xml_where:
                total_ref = con_r.execute(f"SELECT count(*) FROM corpus WHERE 1=1 {ref_xml_where}", ref_xml_params).fetchone()[0]
            else:
                total_ref = con_r.execute("SELECT count(*) FROM corpus").fetchone()[0]
            con_r.close()
        df_ref = df_ref.rename(columns={'freq': 'freq_r'})

        # Get top 500 reference tokens by frequency to capture significant negative keywords when absent in target
        top_ref_tokens = set()
        if not df_ref.empty:
            top_ref_tokens = set(df_ref.sort_values('freq_r', ascending=False).head(500)['token'])

        # 3. Pull all target counts and totals grouped by group_by_col in single queries (much faster than looping DuckDB queries)
        sql_target_all = f"""
        SELECT "{group_by_col}" as group_val, _token_low as token, count(*) as freq 
        FROM corpus 
        WHERE NOT regexp_matches(_token_low, '^[[:punct:]]+$') 
          AND NOT regexp_matches(_token_low, '^[0-9]+$')
          AND "{group_by_col}" IS NOT NULL
          {target_xml_where}
        GROUP BY 1, 2
        HAVING count(*) >= ?
        """
        df_target_all = con_t.execute(sql_target_all, target_xml_params + [min_freq]).fetch_df()

        sql_totals = f"""
        SELECT "{group_by_col}" as group_val, count(*) as total
        FROM corpus
        WHERE "{group_by_col}" IS NOT NULL
          {target_xml_where}
        GROUP BY 1
        """
        df_totals = con_t.execute(sql_totals, target_xml_params).fetch_df()

        con_t.close()

        if df_target_all.empty or df_totals.empty:
            return {}

        totals_dict = dict(zip(df_totals['group_val'], df_totals['total']))
        groups = df_totals['group_val'].tolist()

        # 4. Iterate Groups and compile results in memory
        for group_val in groups:
            df_target = df_target_all[df_target_all['group_val'] == group_val][['token', 'freq']].rename(columns={'freq': 'freq_t'})
            if df_target.empty:
                continue

            total_target = totals_dict.get(group_val, 0)
            if total_target == 0:
                continue

            # Optimize merge: Only join target tokens + top 500 reference tokens
            target_tokens = set(df_target['token'])
            keep_tokens = target_tokens.union(top_ref_tokens)
            df_ref_filtered = df_ref[df_ref['token'].isin(keep_tokens)].copy()

            # Merge and Calculate
            merged = pd.merge(df_target, df_ref_filtered, on='token', how='outer')
            merged['freq_t'] = merged['freq_t'].fillna(0)
            merged['freq_r'] = merged['freq_r'].fillna(0)
            
            O1 = merged['freq_t']
            O2 = merged['freq_r']
            N1 = total_target
            N2 = total_ref
            
            E1 = N1 * (O1 + O2) / (N1 + N2)
            E2 = N2 * (O1 + O2) / (N1 + N2)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                term1 = np.where(O1 > 0, O1 * np.log(O1/E1), 0.0)
                term2 = np.where(O2 > 0, O2 * np.log(O2/E2), 0.0)
                merged['LL'] = 2 * (term1 + term2)
                
                rel_t_smooth = (O1 + 0.5) / (N1 + 0.5)
                rel_r_smooth = (O2 + 0.5) / (N2 + 0.5)
                merged['LogRatio'] = np.log2(rel_t_smooth / rel_r_smooth)
            
            # Vectorized significance check
            conditions = [
                merged['LL'] >= 15.13,
                merged['LL'] >= 10.83,
                merged['LL'] >= 3.84
            ]
            choices = ['*** (p<0.001)', '** (p<0.01)', ' * (p<0.05)']
            merged['Significance'] = np.select(conditions, choices, default='ns')

            merged['Type'] = np.where(
                merged['LL'] < 3.84,
                'Stable',
                np.where(merged['LogRatio'] > 0, 'Positive', 'Negative')
            )
            merged = merged.sort_values("LL", ascending=False)
            
            results[group_val] = merged
            
        return results
        
    except Exception as e:
        print(f"Grouped Keyword Gen Error: {e}")
        return {}
