import math
import re
import duckdb
import pandas as pd

def count_syllables_english(word):
    """
    Estimates the number of syllables in an English word.
    """
    word = word.lower().strip()
    if not word:
        return 0
        
    # Remove non-alphabetic characters
    word = "".join([c for c in word if c.isalpha()])
    if not word:
        return 0
        
    vowels = "aeiouy"
    count = 0
    
    # Handle silent 'e' at the end of word (unless ending in '-le')
    if word.endswith('e'):
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            pass
        else:
            word = word[:-1]
            
    if not word:
        return 1
        
    in_vowel = False
    for char in word:
        if char in vowels:
            if not in_vowel:
                count += 1
                in_vowel = True
        else:
            in_vowel = False
            
    # Adjust for common silent endings like '-es' and '-ed'
    if word.endswith('es') or word.endswith('ed'):
        if len(word) > 2:
            ending = word[-3:]
            if ending not in ['ted', 'ded', 'ses', 'zes', 'ches', 'shes']:
                count = max(1, count - 1)
                
    return max(1, count)

def calculate_formulas(words, sentences, syllables, characters, complex_words):
    """
    Applies the five readability grade level formulas.
    """
    if words == 0 or sentences == 0:
        return {
            "Flesch-Kincaid Grade Level": 0.0,
            "Gunning Fog": 0.0,
            "Coleman-Liau": 0.0,
            "ARI": 0.0,
            "SMOG": 0.0
        }
    
    # 1. Flesch-Kincaid Grade Level
    fkgl = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
    fkgl = max(0.0, fkgl)
    
    # 2. Gunning Fog Index
    pct_complex = (complex_words / words) * 100
    gunning_fog = 0.4 * ((words / sentences) + pct_complex)
    gunning_fog = max(0.4, gunning_fog)
    
    # 3. Coleman-Liau Index
    L = (characters / words) * 100
    S_val = (sentences / words) * 100
    coleman_liau = 0.0588 * L - 0.296 * S_val - 15.8
    coleman_liau = max(0.0, coleman_liau)
    
    # 4. Automated Readability Index (ARI)
    ari = 4.71 * (characters / words) + 0.5 * (words / sentences) - 21.43
    ari = max(0.0, ari)
    
    # 5. SMOG Grade
    smog = 1.0430 * math.sqrt(max(0, complex_words * 30 / sentences)) + 3.1291
    
    return {
        "Flesch-Kincaid Grade Level": round(fkgl, 2),
        "Gunning Fog": round(gunning_fog, 2),
        "Coleman-Liau": round(coleman_liau, 2),
        "ARI": round(ari, 2),
        "SMOG": round(smog, 2)
    }

def map_score_to_level(score):
    """
    Maps a grade level score to the Unified Readability Scale brackets.
    """
    if score <= 6.0:
        return "1. Very Easy (Grades 4-6)"
    elif score <= 8.0:
        return "2. Easy (Grades 6-8)"
    elif score <= 12.0:
        return "3. Standard (Grades 8-12)"
    elif score <= 16.0:
        return "4. Difficult (Grades 12-16)"
    else:
        return "5. Very Difficult (Grades 16+)"

def get_sentence_stats(db_path):
    """
    Retrieves all tokens from the corpus database and aggregates word/syllable/character/complex word
    counts per sentence, preserving metadata columns.
    """
    if not db_path:
        return pd.DataFrame(columns=['filename', 'sent_id', 'words', 'syllables', 'characters', 'complex_words'])
        
    con = duckdb.connect(db_path, read_only=True)
    try:
        # Determine columns in corpus table
        cols_info = con.execute("PRAGMA table_info(corpus)").fetchall()
        db_cols = [c[1] for c in cols_info]
        
        # Keep filename, sent_id, token, and any custom metadata columns
        ignored_cols = {'id', 'pos', 'lemma', '_token_low'}
        keep_cols = [c for c in db_cols if c not in ignored_cols]
        
        query = f"SELECT {', '.join(keep_cols)} FROM corpus ORDER BY id"
        df = con.execute(query).fetch_df()
    except Exception as e:
        print(f"Error querying corpus: {e}")
        return pd.DataFrame(columns=['filename', 'sent_id', 'words', 'syllables', 'characters', 'complex_words'])
    finally:
        con.close()
        
    if df.empty:
        return pd.DataFrame(columns=['filename', 'sent_id', 'words', 'syllables', 'characters', 'complex_words'])
        
    # unique token optimization (speeds up syllable counting by ~20x)
    unique_tokens = df['token'].unique()
    token_stats = {}
    alphanumeric_pattern = re.compile(r'\w')
    
    for tok in unique_tokens:
        if not isinstance(tok, str):
            token_stats[tok] = (0, 0, 0, 0)
            continue
            
        clean_word = "".join([c for c in tok if c.isalnum()])
        if not clean_word or not alphanumeric_pattern.search(clean_word):
            token_stats[tok] = (0, 0, 0, 0)
            continue
            
        chars = len(clean_word)
        syllables = count_syllables_english(clean_word)
        is_complex = 1 if syllables >= 3 else 0
        token_stats[tok] = (1, syllables, chars, is_complex)
        
    # Map back to lists
    words_col = []
    syllables_col = []
    chars_col = []
    complex_col = []
    
    for tok in df['token']:
        w, sy, c, comp = token_stats.get(tok, (0, 0, 0, 0))
        words_col.append(w)
        syllables_col.append(sy)
        chars_col.append(c)
        complex_col.append(comp)
        
    df['words'] = words_col
    df['syllables'] = syllables_col
    df['characters'] = chars_col
    df['complex_words'] = complex_col
    
    # Identify custom metadata columns
    non_meta = {'filename', 'sent_id', 'token', 'words', 'syllables', 'characters', 'complex_words'}
    meta_cols = [c for c in keep_cols if c not in non_meta]
    
    # Aggregation rules
    agg_dict = {
        'words': 'sum',
        'syllables': 'sum',
        'characters': 'sum',
        'complex_words': 'sum'
    }
    for mcol in meta_cols:
        agg_dict[mcol] = 'first'
        
    # Group by sentence key
    grouped = df.groupby(['filename', 'sent_id']).agg(agg_dict).reset_index()
    grouped['sentences'] = 1
    
    return grouped

def compute_readability_metrics(sentence_df):
    """
    Computes aggregated readability metrics for:
    1. Overall corpus
    2. Each file
    3. Each sub-corpus category (grouped by metadata columns)
    """
    if sentence_df.empty:
        return {
            'overall': {'counts': {}, 'metrics': calculate_formulas(0, 0, 0, 0, 0)},
            'files': {},
            'subcorpora': {}
        }
        
    # 1. Overall Corpus
    overall_sum = sentence_df[['words', 'sentences', 'syllables', 'characters', 'complex_words']].sum()
    overall_metrics = calculate_formulas(
        int(overall_sum['words']),
        int(overall_sum['sentences']),
        int(overall_sum['syllables']),
        int(overall_sum['characters']),
        int(overall_sum['complex_words'])
    )
    
    # 2. Each File
    file_groups = sentence_df.groupby('filename')[['words', 'sentences', 'syllables', 'characters', 'complex_words']].sum().reset_index()
    file_metrics = {}
    for _, row in file_groups.iterrows():
        fname = row['filename']
        file_metrics[fname] = calculate_formulas(
            int(row['words']),
            int(row['sentences']),
            int(row['syllables']),
            int(row['characters']),
            int(row['complex_words'])
        )
        
    # 3. Sub-corpora (metadata columns)
    non_meta = {'filename', 'sent_id', 'words', 'syllables', 'characters', 'complex_words', 'sentences'}
    meta_cols = [c for c in sentence_df.columns if c not in non_meta]
    
    subcorpora_metrics = {}
    for mcol in meta_cols:
        sub_groups = sentence_df.groupby(mcol)[['words', 'sentences', 'syllables', 'characters', 'complex_words']].sum().reset_index()
        subcorpora_metrics[mcol] = {}
        for _, row in sub_groups.iterrows():
            gval = str(row[mcol]).strip()
            if not gval or gval.lower() == 'nan':
                gval = "Unclassified"
            subcorpora_metrics[mcol][gval] = calculate_formulas(
                int(row['words']),
                int(row['sentences']),
                int(row['syllables']),
                int(row['characters']),
                int(row['complex_words'])
            )
            
    return {
        'overall': {
            'counts': overall_sum.to_dict(),
            'metrics': overall_metrics
        },
        'files': file_metrics,
        'subcorpora': subcorpora_metrics
    }

def apply_reading_ease_annotation(db_path, filenames, sent_ids, levels):
    """
    Applies the annotated levels back into the DuckDB database under the reading_ease_level column.
    """
    con = duckdb.connect(db_path)
    try:
        # Create column if missing
        try:
            con.execute("ALTER TABLE corpus ADD COLUMN reading_ease_level VARCHAR")
        except:
            pass # Column already exists
            
        update_df = pd.DataFrame({
            'filename': filenames,
            'sent_id': sent_ids,
            'new_level': levels
        })
        
        con.register('update_df', update_df)
        con.execute("""
            UPDATE corpus
            SET reading_ease_level = update_df.new_level
            FROM update_df
            WHERE corpus.filename = update_df.filename
              AND corpus.sent_id = update_df.sent_id
        """)
        con.unregister('update_df')
        
        try:
            con.execute("CREATE INDEX IF NOT EXISTS idx_reading_ease_level ON corpus(reading_ease_level)")
        except:
            pass
            
        return True
    except Exception as e:
        print(f"Error applying reading ease level: {e}")
        return False
    finally:
        con.close()

def annotate_reading_ease_by_chunks(db_path, chunk_size=1000):
    """
    Groups the corpus into sequential chunks of chunk_size words,
    calculates readability metrics for each chunk, and writes the level
    to the `reading_ease_level` column in the corpus table.
    """
    import duckdb
    import pandas as pd
    import re
    
    con = duckdb.connect(db_path)
    try:
        # Fetch all tokens in order of ID
        df = con.execute("SELECT id, token, filename, sent_id FROM corpus ORDER BY id").fetch_df()
        if df.empty:
            return False
            
        unique_tokens = df['token'].unique()
        token_stats = {}
        alphanumeric_pattern = re.compile(r'\w')
        
        for tok in unique_tokens:
            if not isinstance(tok, str):
                token_stats[tok] = (0, 0, 0, 0)
                continue
            clean_word = "".join([c for c in tok if c.isalnum()])
            if not clean_word or not alphanumeric_pattern.search(clean_word):
                token_stats[tok] = (0, 0, 0, 0)
                continue
            chars = len(clean_word)
            syllables = count_syllables_english(clean_word)
            is_complex = 1 if syllables >= 3 else 0
            token_stats[tok] = (1, syllables, chars, is_complex)
            
        words_list = []
        syllables_list = []
        chars_list = []
        complex_list = []
        
        for tok in df['token']:
            w, sy, c, comp = token_stats.get(tok, (0, 0, 0, 0))
            words_list.append(w)
            syllables_list.append(sy)
            chars_list.append(c)
            complex_list.append(comp)
            
        df['is_word'] = words_list
        df['syllables'] = syllables_list
        df['characters'] = chars_list
        df['complex_words'] = complex_list
        
        # Divide into chunks of chunk_size actual words
        word_indices = df['is_word'].cumsum()
        df['chunk_idx'] = ((word_indices - 1) // chunk_size).clip(lower=0)
        
        chunk_groups = df.groupby('chunk_idx')
        chunk_levels = {}
        
        for cidx, group in chunk_groups:
            words = int(group['is_word'].sum())
            syllables = int(group['syllables'].sum())
            characters = int(group['characters'].sum())
            complex_words = int(group['complex_words'].sum())
            
            # Count unique sentences
            sentences = group.groupby(['filename', 'sent_id']).ngroups
            sentences = max(1, sentences)
            
            metrics = calculate_formulas(words, sentences, syllables, characters, complex_words)
            avg_score = sum(metrics.values()) / len(metrics)
            level = map_score_to_level(avg_score)
            chunk_levels[cidx] = level
            
        df['level'] = df['chunk_idx'].map(chunk_levels)
        
        try:
            con.execute("ALTER TABLE corpus ADD COLUMN reading_ease_level VARCHAR")
        except:
            pass
            
        update_df = df[['id', 'level']].rename(columns={'level': 'new_level'})
        con.register('update_df', update_df)
        con.execute("""
            UPDATE corpus
            SET reading_ease_level = update_df.new_level
            FROM update_df
            WHERE corpus.id = update_df.id
        """)
        con.unregister('update_df')
        
        try:
            con.execute("CREATE INDEX IF NOT EXISTS idx_reading_ease_level ON corpus(reading_ease_level)")
        except:
            pass
            
        return True
    except Exception as e:
        print(f"Error annotating reading ease by chunk: {e}")
        return False
    finally:
        con.close()

def get_chunk_readability_stats(db_path, chunk_size=1000):
    """
    Splits the corpus into sequential chunks of chunk_size words,
    calculates readability metrics for each chunk, and returns a list of dicts.
    """
    import duckdb
    import pandas as pd
    import re
    
    con = duckdb.connect(db_path, read_only=True)
    try:
        # Fetch tokens in order of ID
        df = con.execute("SELECT id, token, filename, sent_id FROM corpus ORDER BY id").fetch_df()
    finally:
        con.close()
        
    if df.empty:
        return []
        
    unique_tokens = df['token'].unique()
    token_stats = {}
    alphanumeric_pattern = re.compile(r'\w')
    
    for tok in unique_tokens:
        if not isinstance(tok, str):
            token_stats[tok] = (0, 0, 0, 0)
            continue
        clean_word = "".join([c for c in tok if c.isalnum()])
        if not clean_word or not alphanumeric_pattern.search(clean_word):
            token_stats[tok] = (0, 0, 0, 0)
            continue
        chars = len(clean_word)
        syllables = count_syllables_english(clean_word)
        is_complex = 1 if syllables >= 3 else 0
        token_stats[tok] = (1, syllables, chars, is_complex)
        
    words_list = []
    syllables_list = []
    chars_list = []
    complex_list = []
    
    for tok in df['token']:
        w, sy, c, comp = token_stats.get(tok, (0, 0, 0, 0))
        words_list.append(w)
        syllables_list.append(sy)
        chars_list.append(c)
        complex_list.append(comp)
        
    df['is_word'] = words_list
    df['syllables'] = syllables_list
    df['characters'] = chars_list
    df['complex_words'] = complex_list
    
    # Cumulative word count
    word_indices = df['is_word'].cumsum()
    df['chunk_idx'] = ((word_indices - 1) // chunk_size).clip(lower=0)
    
    chunk_groups = df.groupby('chunk_idx')
    results = []
    
    for cidx, group in chunk_groups:
        words = int(group['is_word'].sum())
        if words == 0:
            continue
            
        syllables = int(group['syllables'].sum())
        characters = int(group['characters'].sum())
        complex_words = int(group['complex_words'].sum())
        
        # Count unique sentences
        sentences = group.groupby(['filename', 'sent_id']).ngroups
        sentences = max(1, sentences)
        
        metrics = calculate_formulas(words, sentences, syllables, characters, complex_words)
        avg_score = sum(metrics.values()) / len(metrics)
        level = map_score_to_level(avg_score)
        
        # Determine word index range
        start_word_idx = cidx * chunk_size + 1
        end_word_idx = min(word_indices.max(), (cidx + 1) * chunk_size)
        
        results.append({
            'Chunk': f"Chunk {cidx + 1}",
            'Word Range': f"{start_word_idx:,} - {end_word_idx:,}",
            'Flesch-Kincaid': metrics['Flesch-Kincaid Grade Level'],
            'Gunning Fog': metrics['Gunning Fog'],
            'Coleman-Liau': metrics['Coleman-Liau'],
            'ARI': metrics['ARI'],
            'SMOG': metrics['SMOG'],
            'Average GL': round(avg_score, 2),
            'Difficulty Level': level
        })
        
    return results
