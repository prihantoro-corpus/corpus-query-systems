import duckdb
import pandas as pd
import re
import spacy
import os

def ensure_spacy_model(model_name="en_core_web_sm"):
    """
    Checks if a spaCy model is installed, and downloads it if missing.
    """
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"Downloading spaCy model '{model_name}'...")
        spacy.cli.download(model_name)
        return spacy.load(model_name)

def run_spacy_ner(db_path, model_name="en_core_web_sm"):
    """
    Runs spaCy dependency-based NER on the corpus database.
    Returns (df_flat, df_matrix_files, df_matrix_top, all_entities)
    """
    nlp = ensure_spacy_model(model_name)
    
    con = duckdb.connect(db_path, read_only=True)
    try:
        df_sents = con.execute("""
            SELECT filename, sent_id, string_agg(token, ' ' ORDER BY id) as text
            FROM corpus
            GROUP BY filename, sent_id
            ORDER BY filename, sent_id
        """).fetch_df()
    finally:
        con.close()
        
    if df_sents.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []
        
    all_entities = []
    
    # Process text in batches for spaCy performance
    for idx, row in df_sents.iterrows():
        text = row['text']
        fname = row['filename']
        sent_id = row['sent_id']
        
        if not text or not text.strip():
            continue
            
        try:
            doc = nlp(text)
        except ValueError as e:
            if "[E088]" in str(e):
                raise ValueError(f"Text length {len(text):,} exceeds the maximum limit for Named Entity Recognition. This usually happens if the corpus lacks proper sentence segmentation, resulting in massive text chunks. Please segment your corpus into sentences before processing.") from e
            raise
        for ent in doc.ents:
            all_entities.append({
                'Entity': ent.text.strip(),
                'Category': ent.label_,
                'Filename': fname,
                'sent_id': sent_id
            })
            
    df_flat, df_matrix_files, df_matrix_top = process_entities_to_outputs(all_entities)
    return df_flat, df_matrix_files, df_matrix_top, all_entities

def run_regex_ner(db_path, patterns_dict):
    """
    Runs Regex-based NER matching on the corpus database using user patterns.
    patterns_dict: {Category_Label: regex_pattern_string}
    Returns (df_flat, df_matrix_files, df_matrix_top, all_entities)
    """
    con = duckdb.connect(db_path, read_only=True)
    try:
        df_sents = con.execute("""
            SELECT filename, sent_id, string_agg(token, ' ' ORDER BY id) as text
            FROM corpus
            GROUP BY filename, sent_id
            ORDER BY filename, sent_id
        """).fetch_df()
    finally:
        con.close()
        
    if df_sents.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []
        
    all_entities = []
    
    # Compile all regexes
    compiled_patterns = {}
    for cat, pat in patterns_dict.items():
        try:
            compiled_patterns[cat] = re.compile(pat)
        except Exception as e:
            # Skip invalid regex patterns silently or handle them gracefully
            print(f"Skipping invalid regex pattern '{pat}' for category '{cat}': {e}")
            continue
            
    for idx, row in df_sents.iterrows():
        text = row['text']
        fname = row['filename']
        sent_id = row['sent_id']
        
        if not text or not text.strip():
            continue
            
        for cat, rx in compiled_patterns.items():
            for match in rx.finditer(text):
                matched_text = match.group(0).strip()
                if matched_text:
                    all_entities.append({
                        'Entity': matched_text,
                        'Category': cat,
                        'Filename': fname,
                        'sent_id': sent_id
                    })
                    
    df_flat, df_matrix_files, df_matrix_top = process_entities_to_outputs(all_entities)
    return df_flat, df_matrix_files, df_matrix_top, all_entities

def process_entities_to_outputs(all_entities):
    """
    Aggregates list of entity matches into flat frequencies, File Matrix, and Top-Entities Matrix.
    """
    if not all_entities:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    df_raw = pd.DataFrame(all_entities)
    
    # 1. Flat list of entity frequencies
    df_flat = df_raw.groupby(['Entity', 'Category']).size().reset_index(name='Frequency')
    df_flat = df_flat.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
    
    # 2. Matrix Type A: Filename vs Category Counts
    df_matrix_files = df_raw.pivot_table(
        index='Filename', 
        columns='Category', 
        aggfunc='size', 
        fill_value=0
    ).reset_index()
    
    # Ensure Categories are sorted for consistency
    categories = sorted([c for c in df_matrix_files.columns if c != 'Filename'])
    df_matrix_files = df_matrix_files[['Filename'] + categories]
    
    # 3. Matrix Type B: Wide Category Top Entity Grid
    # Get top entities per category
    category_dfs = []
    for cat in categories:
        df_cat = df_flat[df_flat['Category'] == cat].head(20).copy()
        df_cat = df_cat.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
        
        # Rename columns to show category clearly
        df_cat = df_cat.rename(columns={
            'Entity': f'{cat} Entity',
            'Frequency': f'{cat} Freq'
        })[['{cat} Entity'.format(cat=cat), '{cat} Freq'.format(cat=cat)]]
        category_dfs.append(df_cat)
        
    if category_dfs:
        # Concatenate horizontally
        df_matrix_top = pd.concat(category_dfs, axis=1)
        # Fill NaN values with empty strings
        df_matrix_top = df_matrix_top.fillna('')
    else:
        df_matrix_top = pd.DataFrame()
        
    return df_flat, df_matrix_files, df_matrix_top


def find_token_span(tokens, entity_text):
    """
    Finds the starting database token ID and length of a substring in a list of tokens.
    """
    def simplify(text):
        return re.sub(r'[^\w]', '', text).lower()
        
    simplified_entity = simplify(entity_text)
    if not simplified_entity:
        return None
        
    n = len(tokens)
    
    # Try exact word count match first (split by space)
    entity_words = entity_text.split()
    m = len(entity_words)
    
    for i in range(n - m + 1):
        window = tokens[i:i+m]
        window_text = "".join([t[1] for t in window])
        if simplify(window_text) == simplified_entity:
            return window[0][0], m
            
    # Try soft match (wider/narrower windows)
    for i in range(n):
        for j in range(i + 1, n + 1):
            span = tokens[i:j]
            span_text = "".join([t[1] for t in span])
            if simplify(span_text) == simplified_entity:
                return span[0][0], len(span)
                
    return None


def annotate_ner_tags_in_db(db_path, all_entities):
    """
    Writes NER annotations back to the corpus database under the `<NER>` tag schema.
    For category PERSON, adds column ner_person with the entity value.
    """
    if not all_entities:
        return False
        
    con = duckdb.connect(db_path, read_only=True)
    try:
        # 1. Group entities by (Filename, sent_id) for efficient lookup
        from collections import defaultdict
        sent_entities = defaultdict(list)
        for ent in all_entities:
            key = (ent['Filename'], ent['sent_id'])
            sent_entities[key].append(ent)
            
        # 2. Dynamic column addition check
        cols_info = con.execute("PRAGMA table_info(corpus)").fetchall()
        existing_cols = {c[1].lower() for c in cols_info}
        
        # Ensure base columns exist
        if 'in_ner_start' not in existing_cols:
            con.execute("ALTER TABLE corpus ADD COLUMN in_ner_start VARCHAR")
        if 'ner_len' not in existing_cols:
            con.execute("ALTER TABLE corpus ADD COLUMN ner_len INTEGER")
            
        unique_categories = {ent['Category'].lower() for ent in all_entities}
        for cat in unique_categories:
            col_name = f"ner_{cat}"
            if col_name not in existing_cols:
                con.execute(f"ALTER TABLE corpus ADD COLUMN {col_name} VARCHAR")
                
        # 3. Match entities to token spans and gather updates
        updates = []  # List of tuples: (ner_len, ner_val, token_id, category_col)
        keys = list(sent_entities.keys())
        
        for fname, sid in keys:
            # Fetch tokens for this sentence ordered by id
            tokens = con.execute("""
                SELECT id, token 
                FROM corpus 
                WHERE filename = ? AND sent_id = ? 
                ORDER BY id
            """, [fname, sid]).fetchall() # list of (id, token_text)
            
            if not tokens:
                continue
                
            for ent in sent_entities[(fname, sid)]:
                entity_text = ent['Entity']
                cat = ent['Category'].lower()
                col_name = f"ner_{cat}"
                
                # Align entity to tokens
                span = find_token_span(tokens, entity_text)
                if span:
                    start_id, length = span
                    updates.append((length, entity_text, start_id, col_name))
                    
        # 4. Apply updates
        if updates:
            col_updates = defaultdict(list)
            for length, val, tid, col in updates:
                col_updates[col].append((length, val, tid))
                
            for col, values in col_updates.items():
                con.executemany(f"""
                    UPDATE corpus 
                    SET in_ner_start = 'TRUE', 
                        ner_len = ?, 
                        {col} = ? 
                    WHERE id = ?
                """, values)
                
            con.commit()
            return True
            
        return False
    except Exception as e:
        print(f"Annotation Error: {e}")
        return False
    finally:
        con.close()
