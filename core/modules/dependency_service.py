import duckdb
import pandas as pd
import spacy
import os
from core.modules.ner_service import ensure_spacy_model

def run_dependency_parsing(db_path, model_name="en_core_web_sm"):
    """
    Runs spaCy dependency parsing on the corpus database.
    Updates the 'corpus' table with 'dep_rel' and 'dep_head_id' columns.
    """
    nlp = ensure_spacy_model(model_name)
    
    # 1. Ensure columns exist
    con = duckdb.connect(db_path, read_only=True)
    try:
        cols_info = con.execute("PRAGMA table_info(corpus)").fetchall()
        existing_cols = {c[1].lower() for c in cols_info}
        
        if 'dep_rel' not in existing_cols:
            con.execute("ALTER TABLE corpus ADD COLUMN dep_rel VARCHAR")
        if 'dep_head_id' not in existing_cols:
            con.execute("ALTER TABLE corpus ADD COLUMN dep_head_id INTEGER")
        if 'dep_head_token' not in existing_cols:
            con.execute("ALTER TABLE corpus ADD COLUMN dep_head_token VARCHAR")
            
        # 2. Fetch all sentences with their token IDs
        # We need the tokens in order to match them correctly to the parse
        df_tokens = con.execute("""
            SELECT id, token, sent_id, filename 
            FROM corpus 
            ORDER BY filename, sent_id, id
        """).fetch_df()
        
        if df_tokens.empty:
            return False
            
        # Group by sentence
        grouped = df_tokens.groupby(['filename', 'sent_id'])
        
        texts = []
        ids_list = []
        for (fname, sid), group in grouped:
            tokens_in_sent = group['token'].tolist()
            ids_in_sent = group['id'].tolist()
            text = " ".join([str(t) for t in tokens_in_sent])
            if text.strip():
                texts.append(text)
                ids_list.append(ids_in_sent)
                
        updates = []
        # Process in batches using nlp.pipe for massive speedup
        for i, doc in enumerate(nlp.pipe(texts, batch_size=200)):
            ids_in_sent = ids_list[i]
            if len(doc) == len(ids_in_sent):
                for j, token in enumerate(doc):
                    rel = token.dep_
                    head_idx = token.head.i
                    head_global_id = ids_in_sent[head_idx]
                    head_text = token.head.text
                    updates.append({
                        'dep_rel': rel, 
                        'dep_head_id': int(head_global_id), 
                        'dep_head_token': head_text, 
                        'id': int(ids_in_sent[j])
                    })

        # 3. Batch Update using DataFrame join (100x faster than executemany in DuckDB)
        if updates:
            df_updates = pd.DataFrame(updates)
            con.register('df_updates', df_updates)
            con.execute("""
                UPDATE corpus 
                SET dep_rel = df_updates.dep_rel, 
                    dep_head_id = df_updates.dep_head_id, 
                    dep_head_token = df_updates.dep_head_token 
                FROM df_updates
                WHERE corpus.id = df_updates.id
            """)
            con.commit()
            return True
            
        return False
        
    except Exception as e:
        print(f"Dependency Parsing Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        con.close()

def get_dependency_stats(db_path):
    """
    Returns high-level statistics about dependency relations in the corpus.
    """
    con = duckdb.connect(db_path, read_only=True)
    try:
        # Check if column exists
        cols = [c[1] for c in con.execute("PRAGMA table_info(corpus)").fetchall()]
        if 'dep_rel' not in cols:
            return None
            
        df_stats = con.execute("""
            SELECT dep_rel as Relation, COUNT(*) as Frequency
            FROM corpus
            WHERE dep_rel IS NOT NULL
            GROUP BY dep_rel
            ORDER BY Frequency DESC
        """).fetch_df()
        
        return df_stats
    except:
        return None
    finally:
        con.close()
