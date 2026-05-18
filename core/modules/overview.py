import duckdb
import pandas as pd

def get_corpus_files(db_path):
    """Fetches unique filenames from the corpus."""
    if not db_path: return []
    con = duckdb.connect(db_path, read_only=True)
    try:
        res = con.execute("SELECT DISTINCT filename FROM corpus ORDER BY filename").fetchall()
        return [r[0] for r in res]
    except:
        return []
    finally:
        con.close()

def get_restricted_stats(db_path, xml_where_clause="", xml_params=[]):
    """
    Calculates tokens, types and TTR for a specific XML restricted region.
    """
    if not db_path:
        return {}
    
    con = duckdb.connect(db_path, read_only=True)
    try:
        # total_tokens
        sql_total = f"SELECT count(*) FROM corpus WHERE 1=1 {xml_where_clause}"
        total_tokens = con.execute(sql_total, xml_params).fetchone()[0]
        
        # unique_types
        sql_types = f"SELECT count(DISTINCT _token_low) FROM corpus WHERE 1=1 {xml_where_clause}"
        unique_types = con.execute(sql_types, xml_params).fetchone()[0]
        
        ttr = (unique_types / total_tokens) if total_tokens > 0 else 0
        
        return {
            'total_tokens': total_tokens,
            'unique_types': unique_types,
            'ttr': round(ttr, 4)
        }
    finally:
        con.close()

def calculate_corpus_statistics(corpus_stats, db_path=None):
    """
    Calculates display metrics like Type/Token Ratio.
    Includes a self-healing check to query the DB if stats are missing or 0.
    """
    if not corpus_stats and not db_path:
        return {}
        
    c_stats = corpus_stats if corpus_stats else {}
    
    total_tokens = c_stats.get('total_tokens', 0)
    type_count = c_stats.get('unique_tokens', 0)
    
    if type_count == 0 and 'token_counts' in c_stats:
        type_count = len(c_stats['token_counts'])
    
    # Self-healing: if we have a DB but types/tokens are suspiciously low, re-calculate
    if db_path and (total_tokens == 0 or type_count == 0):
        try:
            con = duckdb.connect(db_path, read_only=True)
            res = con.execute("SELECT count(*), count(DISTINCT _token_low) FROM corpus").fetchone()
            total_tokens = res[0]
            type_count = res[1]
            con.close()
        except:
            pass

    ttr = (type_count / total_tokens) if total_tokens > 0 else 0
    
    return {
        'total_tokens': total_tokens,
        'unique_types': type_count,
        'ttr': round(ttr, 4)
    }

def get_top_frequencies_v2(db_path, limit=100, xml_where_clause="", xml_params=[]):
    """
    Fetches the top frequency tokens with their POS (if available).
    """
    con = duckdb.connect(db_path, read_only=True)
    try:
        # Check if POS exists
        cols = [c[1] for c in con.execute("PRAGMA table_info(corpus)").fetchall()]
        has_pos = 'pos' in cols
        
        # Base query (exclude punctuation and purely numeric strings)
        filter_clause = "WHERE NOT regexp_matches(_token_low, '^[[:punct:]]+$') AND NOT regexp_matches(_token_low, '^[0-9]+$')"
        
        if xml_where_clause:
            filter_clause += xml_where_clause
            
        if has_pos:
            query = f"SELECT token, pos, count(*) as frequency FROM corpus {filter_clause} GROUP BY token, pos ORDER BY frequency DESC LIMIT {limit}"
        else:
            query = f"SELECT token, count(*) as frequency FROM corpus {filter_clause} GROUP BY token ORDER BY frequency DESC LIMIT {limit}"
            
        df = con.execute(query, xml_params).fetch_df()
        return df
    finally:
        con.close()

def get_unique_pos_tags(db_path, xml_where_clause="", xml_params=[]):
    """
    Fetches unique POS tags from the corpus, excluding dummy or empty tags.
    """
    con = duckdb.connect(db_path, read_only=True)
    try:
        cols = [c[1] for c in con.execute("PRAGMA table_info(corpus)").fetchall()]
        if 'pos' not in cols:
            return []
            
        query = "SELECT DISTINCT pos FROM corpus WHERE pos NOT IN ('##', '###', 'O', '', 'TAG') AND pos NOT LIKE '##%'"
        
        if xml_where_clause:
            query += xml_where_clause
            
        tags = [r[0] for r in con.execute(query, xml_params).fetchall()]
        return sorted(tags)
    finally:
        con.close()

def get_pos_definitions(db_path):
    """
    Fetches the POS definitions dictionary from the database.
    """
    if not db_path: return {}
    con = duckdb.connect(db_path, read_only=True)
    try:
        tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
        if 'pos_definitions' not in tables:
            return {}
        rows = con.execute("SELECT tag, definition FROM pos_definitions").fetchall()
        return {r[0]: r[1] for r in rows}
    except:
        return {}
    finally:
        con.close()

def save_pos_definitions(db_path, definitions):
    """
    Saves the POS definitions dictionary to the database.
    """
    if not db_path or not definitions: return False
    con = duckdb.connect(db_path)
    try:
        con.execute("CREATE TABLE IF NOT EXISTS pos_definitions (tag VARCHAR PRIMARY KEY, definition VARCHAR)")
        con.execute("DELETE FROM pos_definitions")
        data = [(k, v) for k, v in definitions.items() if v and v.strip()]
        if data:
            con.executemany("INSERT INTO pos_definitions VALUES (?, ?)", data)
        return True
    except Exception as e:
        print(f"Error saving definitions: {e}")
        return False
    finally:
        con.close()

def get_corpus_language(db_path):
    """Retrieves the corpus language from metadata table."""
    if not db_path: return "English"
    con = duckdb.connect(db_path, read_only=True)
    try:
        tables = con.execute("SHOW TABLES").fetchall()
        if ('corpus_metadata',) not in tables:
            return "English"
        res = con.execute("SELECT value FROM corpus_metadata WHERE key='language'").fetchone()
        return res[0] if res else "English"
    except:
        return "English"
    finally:
        con.close()

def set_corpus_language(db_path, language):
    """Saves the corpus language to metadata table."""
    if not db_path: return False
    con = duckdb.connect(db_path)
    try:
        con.execute("CREATE TABLE IF NOT EXISTS corpus_metadata (key VARCHAR PRIMARY KEY, value VARCHAR)")
        con.execute("INSERT INTO corpus_metadata VALUES ('language', ?) ON CONFLICT (key) DO UPDATE SET value=excluded.value", [language])
        return True
    except Exception as e:
        print(f"Error setting language: {e}")
        return False
    finally:
        con.close()

def apply_metadata_to_files(db_path, metadata_df):
    """
    Applies metadata from a DataFrame to the corpus.
    metadata_df should have 'filename' and other columns as attributes.
    """
    if not db_path or metadata_df.empty: return False
    import re
    con = duckdb.connect(db_path)
    try:
        # 1. Identify attribute columns (excluding filename)
        attr_cols = [c for c in metadata_df.columns if c != 'filename']
        
        # 2. Ensure columns exist
        cols_info = con.execute("PRAGMA table_info(corpus)").fetch_df()
        existing_cols = set(cols_info['name'].tolist())
        
        for col in attr_cols:
            safe_col = re.sub(r'\W+', '_', col)
            if safe_col not in existing_cols:
                con.execute(f"ALTER TABLE corpus ADD COLUMN {safe_col} VARCHAR")
        
        # 3. Update database
        for _, row in metadata_df.iterrows():
            fname = row['filename']
            for col in attr_cols:
                safe_col = re.sub(r'\W+', '_', col)
                val = str(row[col]) if pd.notna(row[col]) else None
                con.execute(f"UPDATE corpus SET {safe_col} = ? WHERE filename = ?", [val, fname])
        
        return True
    except Exception as e:
        print(f"Error applying metadata: {e}")
        return False
    finally:
        con.close()

def get_file_sentences(db_path, filename):
    """Retrieves all sentences for a given file, grouped by sent_id."""
    if not db_path or not filename: return []
    con = duckdb.connect(db_path, read_only=True)
    try:
        # Get all tokens for the file, ordered by id
        res = con.execute("""
            SELECT sent_id, string_agg(token, ' ' ORDER BY id) as text
            FROM corpus
            WHERE filename = ?
            GROUP BY sent_id
            ORDER BY MIN(id)
        """, [filename]).fetchall()
        return [{"sent_id": r[0], "text": r[1]} for r in res]
    except Exception as e:
        print(f"Error getting file sentences: {e}")
        return []
    finally:
        con.close()

def get_file_word_count(db_path, filename):
    """Returns the total token count for a file."""
    if not db_path or not filename: return 0
    con = duckdb.connect(db_path, read_only=True)
    try:
        res = con.execute("SELECT count(*) FROM corpus WHERE filename = ?", [filename]).fetchone()
        return res[0] if res else 0
    except:
        return 0
    finally:
        con.close()

def apply_segmental_metadata(db_path, filename, sent_ids, meta_dict):
    """
    Applies attribute-value pairs to specific sentences in a file.
    meta_dict: {attribute: value, ...}
    """
    if not db_path or not filename or not sent_ids or not meta_dict: return False
    import re
    con = duckdb.connect(db_path)
    try:
        # Ensure columns exist
        cols_info = con.execute("PRAGMA table_info(corpus)").fetch_df()
        existing_cols = set(cols_info['name'].tolist())
        
        for attr in meta_dict.keys():
            safe_col = re.sub(r'\W+', '_', attr)
            if safe_col not in existing_cols:
                con.execute(f"ALTER TABLE corpus ADD COLUMN {safe_col} VARCHAR")
                existing_cols.add(safe_col)

        # Update sentences
        for attr, val in meta_dict.items():
            safe_col = re.sub(r'\W+', '_', attr)
            # Use IN clause for multiple sent_ids
            placeholders = ", ".join(["?"] * len(sent_ids))
            con.execute(f"""
                UPDATE corpus 
                SET {safe_col} = ? 
                WHERE filename = ? AND sent_id IN ({placeholders})
            """, [val, filename] + list(sent_ids))
            
        return True
    except Exception as e:
        print(f"Error applying segmental metadata: {e}")
        return False
    finally:
        con.close()

def slice_corpus_file(db_path, filename, max_words=5000):
    """
    Slices a file into multiple parts if it exceeds max_words.
    Renames the filename in the corpus table to filename_part1, filename_part2, etc.
    """
    if not db_path or not filename: return False
    con = duckdb.connect(db_path)
    try:
        # Get all token IDs for this file in order
        ids = [r[0] for r in con.execute("SELECT id FROM corpus WHERE filename = ? ORDER BY id", [filename]).fetchall()]
        
        if len(ids) <= max_words:
            return True # No slicing needed
            
        # Divide IDs into chunks
        chunks = [ids[i:i + max_words] for i in range(0, len(ids), max_words)]
        
        for i, chunk_ids in enumerate(chunks):
            new_filename = f"{filename}_part{i+1}"
            placeholders = ", ".join(["?"] * len(chunk_ids))
            con.execute(f"UPDATE corpus SET filename = ? WHERE id IN ({placeholders})", [new_filename] + chunk_ids)
            
        return True
    except Exception as e:
        print(f"Error slicing file: {e}")
        return False
    finally:
        con.close()

def get_sentence_metadata(db_path, filename):
    """Retrieves all metadata for sentences in a file."""
    if not db_path or not filename: return pd.DataFrame()
    con = duckdb.connect(db_path, read_only=True)
    try:
        cols_info = con.execute("PRAGMA table_info(corpus)").fetch_df()
        standard = {'id', 'token', 'pos', 'lemma', 'sent_id', '_token_low', 'filename', 'topic', 'sentiment'}
        meta_cols = [c for c in cols_info['name'].tolist() if c.lower() not in standard]
        
        if not meta_cols:
            return pd.DataFrame()
            
        select_cols = ", ".join([f"MAX({c}) as {c}" for c in meta_cols])
        query = f"""
            SELECT sent_id, {select_cols} 
            FROM corpus 
            WHERE filename = ? 
            GROUP BY sent_id
        """
        df = con.execute(query, [filename]).fetch_df()
        return df
    except Exception as e:
        print(f"Error getting sentence metadata: {e}")
        return pd.DataFrame()
    finally:
        con.close()

def get_file_tokens(db_path, filename):
    """Retrieves all tokens for a given file in order."""
    if not db_path or not filename: return []
    con = duckdb.connect(db_path, read_only=True)
    try:
        # Get all tokens with their ID and any existing metadata
        cols_info = con.execute("PRAGMA table_info(corpus)").fetch_df()
        standard = {'id', 'token', 'pos', 'lemma', 'sent_id', '_token_low', 'filename', 'topic', 'sentiment'}
        meta_cols = [c for c in cols_info['name'].tolist() if c.lower() not in standard]
        
        select_clause = "id, token, sent_id"
        if meta_cols:
            select_clause += ", " + ", ".join(meta_cols)
            
        res = con.execute(f"SELECT {select_clause} FROM corpus WHERE filename = ? ORDER BY id", [filename]).fetch_df()
        return res
    except Exception as e:
        print(f"Error getting file tokens: {e}")
        return pd.DataFrame()
    finally:
        con.close()

def apply_token_metadata(db_path, token_ids, meta_dict):
    """
    Applies attribute-value pairs to specific tokens.
    token_ids: List of integer IDs.
    """
    if not db_path or not token_ids or not meta_dict: return False
    import re
    con = duckdb.connect(db_path)
    try:
        # Ensure columns exist
        cols_info = con.execute("PRAGMA table_info(corpus)").fetch_df()
        existing_cols = set(cols_info['name'].tolist())
        
        for attr in meta_dict.keys():
            safe_col = re.sub(r'\W+', '_', attr)
            if safe_col not in existing_cols:
                con.execute(f"ALTER TABLE corpus ADD COLUMN {safe_col} VARCHAR")
                existing_cols.add(safe_col)

        # Update tokens
        for attr, val in meta_dict.items():
            safe_col = re.sub(r'\W+', '_', attr)
            placeholders = ", ".join(["?"] * len(token_ids))
            con.execute(f"""
                UPDATE corpus 
                SET {safe_col} = ? 
                WHERE id IN ({placeholders})
            """, [val] + list(token_ids))
            
        return True
    except Exception as e:
        print(f"Error applying token metadata: {e}")
        return False
    finally:
        con.close()
