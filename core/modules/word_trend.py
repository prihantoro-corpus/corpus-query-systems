import duckdb
import pandas as pd
import json

def get_available_metadata_attributes(db_path):
    """
    Returns a list of metadata columns from the corpus table that can be used as time attributes.
    Excludes standard structural columns.
    """
    if not db_path: return []
    con = duckdb.connect(db_path, read_only=True)
    try:
        cols_info = con.execute("PRAGMA table_info(corpus)").fetch_df()
        standard = {'id', 'token', 'pos', 'lemma', 'sent_id', '_token_low', 'filename', 'topic', 'sentiment'}
        meta_cols = [c for c in cols_info['name'].tolist() if c.lower() not in standard]
        return sorted(meta_cols)
    except Exception as e:
        print(f"Error getting metadata attributes: {e}")
        return []
    finally:
        con.close()

def get_metadata_values(db_path, attribute):
    """
    Returns a sorted list of unique non-null values for a specific metadata attribute.
    """
    if not db_path or not attribute: return []
    con = duckdb.connect(db_path, read_only=True)
    try:
        query = f"SELECT DISTINCT {attribute} FROM corpus WHERE {attribute} IS NOT NULL"
        res = con.execute(query).fetchall()
        values = [str(r[0]) for r in res if r[0] is not None]
        # Try to sort numerically if possible, otherwise alphabetically
        try:
            values = sorted(values, key=lambda x: float(x))
        except ValueError:
            values = sorted(values)
        return values
    except Exception as e:
        print(f"Error getting metadata values: {e}")
        return []
    finally:
        con.close()

def get_emerging_words(db_path, time_attr, ordered_time_values, pos_mode, pos_tags, top_n=10, comparison_mode='chronological'):
    """
    Computes emerging words over the ordered time periods.
    pos_mode: 'include' or 'exclude'
    pos_tags: list of POS tags
    comparison_mode: 'chronological' (finds first appearance) or 'exclusive' (finds words unique to a period)
    """
    if not db_path or not time_attr or not ordered_time_values:
        return pd.DataFrame()

    con = duckdb.connect(db_path, read_only=True)
    try:
        # Build period ranks table
        ranks = []
        for rank, val in enumerate(ordered_time_values, start=1):
            # Escape single quotes in value
            safe_val = val.replace("'", "''")
            ranks.append(f"SELECT '{safe_val}' as time_val, {rank} as rank")
        
        period_ranks_sql = " UNION ALL ".join(ranks)
        
        # Build POS filter clause
        pos_filter = ""
        if pos_tags:
            safe_tags = ", ".join([f"'{t.replace(chr(39), chr(39)+chr(39))}'" for t in pos_tags])
            if pos_mode == 'exclude':
                pos_filter = f" AND pos NOT IN ({safe_tags})"
            elif pos_mode == 'include':
                pos_filter = f" AND pos IN ({safe_tags})"
                
        if comparison_mode == 'exclusive':
            query = f"""
            WITH period_ranks AS (
                {period_ranks_sql}
            ),
            filtered_corpus AS (
                SELECT c._token_low, ANY_VALUE(c.token) as display_token, p.time_val
                FROM corpus c
                JOIN period_ranks p ON CAST(c.{time_attr} AS VARCHAR) = p.time_val
                WHERE NOT regexp_matches(c._token_low, '^[[:punct:]]+$') 
                  AND NOT regexp_matches(c._token_low, '^[0-9]+$')
                  {pos_filter}
                GROUP BY c._token_low, p.time_val
            ),
            token_distribution AS (
                SELECT _token_low, COUNT(DISTINCT time_val) as num_periods, ANY_VALUE(time_val) as exclusive_time_val
                FROM filtered_corpus
                GROUP BY _token_low
            ),
            exclusive_tokens AS (
                SELECT _token_low, exclusive_time_val
                FROM token_distribution
                WHERE num_periods = 1
            ),
            period_totals AS (
                SELECT p.time_val, CAST(COUNT(*) AS DOUBLE) as total_tokens
                FROM corpus c
                JOIN period_ranks p ON CAST(c.{time_attr} AS VARCHAR) = p.time_val
                GROUP BY p.time_val
            ),
            exclusive_frequencies AS (
                SELECT fc.time_val, fc.display_token as token, fc._token_low, CAST(COUNT(*) AS DOUBLE) as freq, pt.total_tokens
                FROM corpus c
                JOIN period_ranks p ON CAST(c.{time_attr} AS VARCHAR) = p.time_val
                JOIN exclusive_tokens et ON c._token_low = et._token_low AND p.time_val = et.exclusive_time_val
                JOIN filtered_corpus fc ON c._token_low = fc._token_low AND p.time_val = fc.time_val
                JOIN period_totals pt ON p.time_val = pt.time_val
                WHERE NOT regexp_matches(c._token_low, '^[[:punct:]]+$') 
                  AND NOT regexp_matches(c._token_low, '^[0-9]+$')
                  {pos_filter}
                GROUP BY fc.time_val, fc.display_token, fc._token_low, pt.total_tokens
            ),
            ranked_exclusive AS (
                SELECT time_val, token, (freq / total_tokens * 1000000) as rel_freq,
                       ROW_NUMBER() OVER(PARTITION BY time_val ORDER BY (freq / total_tokens) DESC, token ASC) as rn
                FROM exclusive_frequencies
            )
            SELECT time_val, token, rel_freq
            FROM ranked_exclusive
            WHERE rn <= {top_n}
            ORDER BY time_val ASC, rel_freq DESC
            """
            
            full_query = f"""
            WITH period_ranks AS (
                {period_ranks_sql}
            ),
            filtered_corpus AS (
                SELECT c._token_low, ANY_VALUE(c.token) as display_token, p.time_val
                FROM corpus c
                JOIN period_ranks p ON CAST(c.{time_attr} AS VARCHAR) = p.time_val
                WHERE NOT regexp_matches(c._token_low, '^[[:punct:]]+$') 
                  AND NOT regexp_matches(c._token_low, '^[0-9]+$')
                  {pos_filter}
                GROUP BY c._token_low, p.time_val
            ),
            token_distribution AS (
                SELECT _token_low, COUNT(DISTINCT time_val) as num_periods, ANY_VALUE(time_val) as exclusive_time_val
                FROM filtered_corpus
                GROUP BY _token_low
            ),
            exclusive_tokens AS (
                SELECT _token_low, exclusive_time_val
                FROM token_distribution
                WHERE num_periods = 1
            ),
            period_totals AS (
                SELECT p.time_val, CAST(COUNT(*) AS DOUBLE) as total_tokens
                FROM corpus c
                JOIN period_ranks p ON CAST(c.{time_attr} AS VARCHAR) = p.time_val
                GROUP BY p.time_val
            ),
            exclusive_frequencies AS (
                SELECT fc.time_val, fc.display_token as token, fc._token_low, CAST(COUNT(*) AS DOUBLE) as freq, pt.total_tokens
                FROM corpus c
                JOIN period_ranks p ON CAST(c.{time_attr} AS VARCHAR) = p.time_val
                JOIN exclusive_tokens et ON c._token_low = et._token_low AND p.time_val = et.exclusive_time_val
                JOIN filtered_corpus fc ON c._token_low = fc._token_low AND p.time_val = fc.time_val
                JOIN period_totals pt ON p.time_val = pt.time_val
                WHERE NOT regexp_matches(c._token_low, '^[[:punct:]]+$') 
                  AND NOT regexp_matches(c._token_low, '^[0-9]+$')
                  {pos_filter}
                GROUP BY fc.time_val, fc.display_token, fc._token_low, pt.total_tokens
            )
            SELECT time_val as Time, token as "Unique Word", (freq / total_tokens * 1000000) as "Relative Frequency (pmw)"
            FROM exclusive_frequencies
            ORDER BY time_val ASC, "Relative Frequency (pmw)" DESC
            """
        else:
            query = f"""
            WITH period_ranks AS (
                {period_ranks_sql}
            ),
            filtered_corpus AS (
                SELECT c._token_low, ANY_VALUE(c.token) as display_token, p.rank, p.time_val
                FROM corpus c
                JOIN period_ranks p ON CAST(c.{time_attr} AS VARCHAR) = p.time_val
                WHERE NOT regexp_matches(c._token_low, '^[[:punct:]]+$') 
                  AND NOT regexp_matches(c._token_low, '^[0-9]+$')
                  {pos_filter}
                GROUP BY c._token_low, p.rank, p.time_val
            ),
            token_emergence AS (
                SELECT _token_low, MIN(rank) as first_rank
                FROM filtered_corpus
                GROUP BY _token_low
            ),
            period_totals AS (
                SELECT p.time_val, CAST(COUNT(*) AS DOUBLE) as total_tokens
                FROM corpus c
                JOIN period_ranks p ON CAST(c.{time_attr} AS VARCHAR) = p.time_val
                GROUP BY p.time_val
            ),
            emerging_frequencies AS (
                SELECT fc.time_val, fc.display_token as token, fc._token_low, CAST(COUNT(*) AS DOUBLE) as freq, te.first_rank, pt.total_tokens
                FROM corpus c
                JOIN period_ranks p ON CAST(c.{time_attr} AS VARCHAR) = p.time_val
                JOIN token_emergence te ON c._token_low = te._token_low AND p.rank = te.first_rank
                JOIN filtered_corpus fc ON c._token_low = fc._token_low AND p.rank = fc.rank
                JOIN period_totals pt ON p.time_val = pt.time_val
                WHERE NOT regexp_matches(c._token_low, '^[[:punct:]]+$') 
                  AND NOT regexp_matches(c._token_low, '^[0-9]+$')
                  {pos_filter}
                GROUP BY fc.time_val, fc.display_token, fc._token_low, te.first_rank, pt.total_tokens
            ),
            ranked_emerging AS (
                SELECT time_val, token, (freq / total_tokens * 1000000) as rel_freq, first_rank,
                       ROW_NUMBER() OVER(PARTITION BY time_val ORDER BY (freq / total_tokens) DESC, token ASC) as rn
                FROM emerging_frequencies
                WHERE first_rank > 1
            )
            SELECT time_val, token, rel_freq
            FROM ranked_emerging
            WHERE rn <= {top_n}
            ORDER BY first_rank ASC, rel_freq DESC
            """
            
            full_query = f"""
            WITH period_ranks AS (
                {period_ranks_sql}
            ),
            filtered_corpus AS (
                SELECT c._token_low, ANY_VALUE(c.token) as display_token, p.rank, p.time_val
                FROM corpus c
                JOIN period_ranks p ON CAST(c.{time_attr} AS VARCHAR) = p.time_val
                WHERE NOT regexp_matches(c._token_low, '^[[:punct:]]+$') 
                  AND NOT regexp_matches(c._token_low, '^[0-9]+$')
                  {pos_filter}
                GROUP BY c._token_low, p.rank, p.time_val
            ),
            token_emergence AS (
                SELECT _token_low, MIN(rank) as first_rank
                FROM filtered_corpus
                GROUP BY _token_low
            ),
            period_totals AS (
                SELECT p.time_val, CAST(COUNT(*) AS DOUBLE) as total_tokens
                FROM corpus c
                JOIN period_ranks p ON CAST(c.{time_attr} AS VARCHAR) = p.time_val
                GROUP BY p.time_val
            ),
            emerging_frequencies AS (
                SELECT fc.time_val, fc.display_token as token, fc._token_low, CAST(COUNT(*) AS DOUBLE) as freq, te.first_rank, pt.total_tokens
                FROM corpus c
                JOIN period_ranks p ON CAST(c.{time_attr} AS VARCHAR) = p.time_val
                JOIN token_emergence te ON c._token_low = te._token_low AND p.rank = te.first_rank
                JOIN filtered_corpus fc ON c._token_low = fc._token_low AND p.rank = fc.rank
                JOIN period_totals pt ON p.time_val = pt.time_val
                WHERE NOT regexp_matches(c._token_low, '^[[:punct:]]+$') 
                  AND NOT regexp_matches(c._token_low, '^[0-9]+$')
                  {pos_filter}
                GROUP BY fc.time_val, fc.display_token, fc._token_low, te.first_rank, pt.total_tokens
            )
            SELECT time_val as Time, token as "Emerging Word", (freq / total_tokens * 1000000) as "Relative Frequency (pmw)"
            FROM emerging_frequencies
            WHERE first_rank > 1
            ORDER BY first_rank ASC, "Relative Frequency (pmw)" DESC
            """
        
        df_display = con.execute(query).fetch_df()
        df_full = con.execute(full_query).fetch_df()
        
        return df_display, df_full

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error computing emerging words: {e}")
        return pd.DataFrame(), pd.DataFrame()
    finally:
        con.close()

def get_word_tracker_data(db_path, time_attr, words):
    """
    Computes relative frequency (per million words) over time for a specific list of words.
    Returns a pivot table DataFrame suitable for line charting: 
    Index = time_val, Columns = words, Values = relative frequency.
    """
    if not db_path or not time_attr or not words:
        return pd.DataFrame()
        
    # Clean and lowercase the words
    clean_words = [w.strip().lower() for w in words if w.strip()]
    if not clean_words:
        return pd.DataFrame()

    con = duckdb.connect(db_path, read_only=True)
    try:
        # Escape strings for SQL IN clause
        words_sql = ", ".join([f"'{w.replace(chr(39), chr(39)+chr(39))}'" for w in clean_words])
        
        query = f"""
        WITH period_totals AS (
            SELECT CAST({time_attr} AS VARCHAR) as time_val, CAST(COUNT(*) AS DOUBLE) as total_tokens
            FROM corpus
            WHERE {time_attr} IS NOT NULL
            GROUP BY CAST({time_attr} AS VARCHAR)
        ),
        word_counts AS (
            SELECT CAST({time_attr} AS VARCHAR) as time_val, _token_low, CAST(COUNT(*) AS DOUBLE) as freq
            FROM corpus
            WHERE _token_low IN ({words_sql}) AND {time_attr} IS NOT NULL
            GROUP BY CAST({time_attr} AS VARCHAR), _token_low
        )
        SELECT p.time_val, w._token_low, 
               (COALESCE(w.freq, 0) / p.total_tokens * 1000000) as rel_freq
        FROM period_totals p
        JOIN word_counts w ON p.time_val = w.time_val
        """
        
        df = con.execute(query).fetch_df()
        if df.empty:
            return pd.DataFrame()
            
        # Pivot the data for the line chart (Time on index, Words as columns)
        # First, ensure all time periods have entries for all tracked words even if 0
        df_pivot = df.pivot(index='time_val', columns='_token_low', values='rel_freq').fillna(0)
        
        # Sort index if it can be cast to numeric (e.g. Years)
        try:
            df_pivot.index = pd.to_numeric(df_pivot.index)
            df_pivot = df_pivot.sort_index()
            df_pivot.index = df_pivot.index.astype(str)
        except ValueError:
            df_pivot = df_pivot.sort_index()
            
        return df_pivot
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error computing word tracker data: {e}")
        return pd.DataFrame()
    finally:
        con.close()

def compute_tracker_statistics(df, stat_type):
    """
    Computes inferential statistics across the words in the tracker DataFrame.
    Returns a dictionary containing DataFrames for results and rubrics.
    """
    import numpy as np
    from scipy import stats
    import pandas as pd
    import itertools
    
    if df.empty or len(df.columns) < 2 or len(df) < 3:
        return {"error": "Insufficient data to perform statistical analysis. Need at least 2 words and 3 time periods."}
        
    words = list(df.columns)
    output = {}
    
    try:
        if stat_type in ["Correlation", "Prediction"]:
            rubric_data = [
                {"Score Range (Abs)", "Category", "Description"}, # Header is implicit in pandas, this is bad format
            ]
            rubric_df = pd.DataFrame([
                {"Range (Absolute)": "0.00 - 0.19", "Strength": "Very Weak", "Description": "Virtually no relationship"},
                {"Range (Absolute)": "0.20 - 0.39", "Strength": "Weak", "Description": "Weak relationship"},
                {"Range (Absolute)": "0.40 - 0.59", "Strength": "Moderate", "Description": "Noticeable relationship"},
                {"Range (Absolute)": "0.60 - 0.79", "Strength": "Strong", "Description": "Strong relationship"},
                {"Range (Absolute)": "0.80 - 1.00", "Strength": "Very Strong", "Description": "Highly consistent relationship"}
            ])
            
            results = []
            
            if stat_type == "Correlation":
                for w1, w2 in itertools.combinations(words, 2):
                    corr, p = stats.spearmanr(df[w1], df[w2])
                    if pd.isna(corr): corr = 0
                    
                    abs_c = abs(corr)
                    if abs_c < 0.2: strength = "Very Weak"
                    elif abs_c < 0.4: strength = "Weak"
                    elif abs_c < 0.6: strength = "Moderate"
                    elif abs_c < 0.8: strength = "Strong"
                    else: strength = "Very Strong"
                    
                    direction = "Positive" if corr > 0 else ("Negative" if corr < 0 else "None")
                    sig = "Yes" if p < 0.05 else "No"
                    
                    results.append({
                        "Pair": f"{w1} & {w2}",
                        "Score (r)": round(corr, 3),
                        "p-value": round(p, 4),
                        "Significant (p<0.05)": sig,
                        "Strength": strength,
                        "Direction": direction
                    })
            else: # Prediction
                for w1, w2 in itertools.permutations(words, 2):
                    w1_data = df[w1].iloc[:-1]
                    w2_data = df[w2].iloc[1:]
                    
                    if w1_data.std() > 0 and w2_data.std() > 0:
                        corr, p = stats.spearmanr(w1_data, w2_data)
                        if pd.isna(corr): corr = 0
                    else:
                        corr, p = 0, 1.0
                        
                    abs_c = abs(corr)
                    if abs_c < 0.2: strength = "Very Weak"
                    elif abs_c < 0.4: strength = "Weak"
                    elif abs_c < 0.6: strength = "Moderate"
                    elif abs_c < 0.8: strength = "Strong"
                    else: strength = "Very Strong"
                    
                    direction = "Positive" if corr > 0 else ("Negative" if corr < 0 else "None")
                    sig = "Yes" if p < 0.05 else "No"
                    
                    results.append({
                        "Hypothesis": f"{w1} predicts {w2}",
                        "Score (r)": round(corr, 3),
                        "p-value": round(p, 4),
                        "Significant (p<0.05)": sig,
                        "Strength": strength,
                        "Direction": direction
                    })
                    
            output = {
                "title": f"{stat_type} Analysis (Spearman Rank)",
                "results_df": pd.DataFrame(results),
                "rubric_df": rubric_df,
                "highlight_col": "Strength"
            }
                
        elif stat_type == "Trend Comparison":
            rubric_df = pd.DataFrame([
                {"Slope Range", "Category", "Description"}, # Bad
            ])
            rubric_df = pd.DataFrame([
                {"Slope Range": "> 1.0", "Category": "Rapid Growth", "Description": "Increasing very quickly"},
                {"Slope Range": "0.1 to 1.0", "Category": "Moderate Growth", "Description": "Increasing steadily"},
                {"Slope Range": "-0.1 to 0.1", "Category": "Stable", "Description": "Virtually flat / No significant change"},
                {"Slope Range": "-1.0 to -0.1", "Category": "Moderate Decline", "Description": "Decreasing steadily"},
                {"Slope Range": "< -1.0", "Category": "Rapid Decline", "Description": "Decreasing very quickly"}
            ])
            
            x = np.arange(len(df))
            results = []
            for w in words:
                m, b = np.polyfit(x, df[w].values, 1)
                
                if m > 1.0: cat = "Rapid Growth"
                elif m > 0.1: cat = "Moderate Growth"
                elif m >= -0.1: cat = "Stable"
                elif m >= -1.0: cat = "Moderate Decline"
                else: cat = "Rapid Decline"
                
                results.append({
                    "Word": w,
                    "Slope": round(m, 3),
                    "Category": cat
                })
                
            output = {
                "title": "Trend Comparison (Linear Regression Slope)",
                "results_df": pd.DataFrame(results),
                "rubric_df": rubric_df,
                "highlight_col": "Category"
            }
            
        elif stat_type == "Variance/Volatility":
            rubric_df = pd.DataFrame([
                {"CV Range": "< 0.25", "Category": "Low Volatility", "Description": "Very stable over time"},
                {"CV Range": "0.25 - 0.75", "Category": "Moderate Volatility", "Description": "Noticeable fluctuations"},
                {"CV Range": "> 0.75", "Category": "High Volatility", "Description": "Extremely erratic / unstable peaks"}
            ])
            
            results = []
            arrays = []
            for w in words:
                mean_val = df[w].mean()
                std_val = df[w].std()
                cv = (std_val / mean_val) if mean_val > 0 else 0
                arrays.append(df[w].values)
                
                if cv < 0.25: cat = "Low Volatility"
                elif cv <= 0.75: cat = "Moderate Volatility"
                else: cat = "High Volatility"
                
                results.append({
                    "Word": w,
                    "Mean Freq": round(mean_val, 2),
                    "Std Dev": round(std_val, 2),
                    "CV (Score)": round(cv, 3),
                    "Category": cat
                })
                
            # Levene
            p_val_text = "N/A"
            if len(arrays) >= 2:
                stat, p = stats.levene(*arrays)
                p_val_text = f"{p:.4f} ({'Significant' if p < 0.05 else 'Not Significant'})"
                
            output = {
                "title": "Variance & Volatility (Coefficient of Variation)",
                "subtitle": f"Levene's Test for Equal Variances p-value: {p_val_text}",
                "results_df": pd.DataFrame(results),
                "rubric_df": rubric_df,
                "highlight_col": "Category"
            }
            
        return output
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Error computing statistics: {e}"}
