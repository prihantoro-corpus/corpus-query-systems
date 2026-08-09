import duckdb
import pandas as pd
import os
import re

def natural_sort_key(s):
    """Sort strings containing numbers naturally (e.g. 'Sublist 2' < 'Sublist 10')"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def load_wordlist_from_file_object(file_obj, filename):
    """
    Loads a wordlist from an uploaded file object.
    Supports .txt, .csv, and .xlsx/.xls.
    Returns a dictionary: {word: category}
    """
    ext = os.path.splitext(filename)[1].lower()
    wordlist = {}
    
    if ext == '.csv':
        try:
            # Try to read CSV
            df = pd.read_csv(file_obj, header=None)
            return parse_dataframe_to_wordlist(df)
        except Exception:
            return {}
    elif ext in ['.xlsx', '.xls']:
        try:
            # Try to read Excel
            df = pd.read_excel(file_obj, header=None)
            return parse_dataframe_to_wordlist(df)
        except Exception:
            return {}
    else: # Default text file (.txt)
        try:
            content = file_obj.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            lines = [line.rstrip('\r\n') for line in content.split('\n') if line.strip()]
            start_idx = 0
            if lines:
                first_line_parts = [p.strip().lower() for p in lines[0].split('\t')]
                if any(h in first_line_parts[0] for h in ['word', 'token', 'lemma', 'collocate']) or \
                   (len(first_line_parts) >= 2 and 'category' in first_line_parts[1]):
                    start_idx = 1
            for line in lines[start_idx:]:
                parts = line.split('\t')
                if len(parts) >= 2:
                    word = parts[0].strip().lower()
                    category = parts[1].strip()
                    if word and word != 'nan':
                        wordlist[word] = category
                    if len(parts) >= 3:
                        lemma = parts[2].strip().lower()
                        if lemma and lemma != 'nan':
                            wordlist[lemma] = category
                else:
                    word = line.strip().lower()
                    if word and word != 'nan':
                        wordlist[word] = 'Coverage'
            return wordlist
        except Exception:
            return {}

def parse_dataframe_to_wordlist(df):
    wordlist = {}
    if df.empty:
        return wordlist
    
    # If first row contains header-like strings, drop it
    first_row_str = df.iloc[0].astype(str).str.lower().tolist()
    start_idx = 0
    if any(header in first_row_str[0] for header in ['word', 'token', 'lemma', 'collocate']) or \
       (len(first_row_str) >= 2 and 'category' in first_row_str[1]):
        start_idx = 1
        
    num_cols = len(df.columns)
    for idx in range(start_idx, len(df)):
        row = df.iloc[idx]
        if num_cols >= 2:
            word = str(row[0]).strip().lower()
            category = str(row[1]).strip()
            if word and word != 'nan':
                wordlist[word] = category
            if num_cols >= 3:
                lemma = str(row[2]).strip().lower()
                if lemma and lemma != 'nan':
                    wordlist[lemma] = category
        else:
            word = str(row[0]).strip().lower()
            if word and word != 'nan':
                wordlist[word] = 'Coverage'
    return wordlist

def load_wordlist(file_path_or_content, is_file=True):
    """
    Loads a wordlist from a file path or string content.
    Supports plain (1 col), categorized (2 cols), and three column (word, category, lemma).
    Returns a dictionary: {word: category}
    """
    wordlist = {}
    content = ""
    
    if is_file:
        if not os.path.exists(file_path_or_content):
            return None
        with open(file_path_or_content, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        content = file_path_or_content

    lines = [line.rstrip('\r\n') for line in content.split('\n') if line.strip()]
    start_idx = 0
    if lines:
        first_line_parts = [p.strip().lower() for p in lines[0].split('\t')]
        if any(h in first_line_parts[0] for h in ['word', 'token', 'lemma', 'collocate']) or \
           (len(first_line_parts) >= 2 and 'category' in first_line_parts[1]):
            start_idx = 1
    for line in lines[start_idx:]:
        parts = line.split('\t')
        if len(parts) >= 2:
            word = parts[0].strip().lower()
            category = parts[1].strip()
            if word and word != 'nan':
                wordlist[word] = category
            if len(parts) >= 3:
                lemma = parts[2].strip().lower()
                if lemma and lemma != 'nan':
                    wordlist[lemma] = category
        else:
            word = line.strip().lower()
            if word and word != 'nan':
                wordlist[word] = 'Coverage'
            
    return wordlist

def run_word_profiler_analysis(db_path, wordlist, basis='Whole Corpus', metadata_col=None, xml_where_clause="", xml_params=[]):
    """
    Runs the word profiler analysis.
    Returns a DataFrame with the results.
    """
    if not db_path or not wordlist:
        return pd.DataFrame()

    con = duckdb.connect(db_path, read_only=True)
    try:
        # Prepare the categories
        categories = sorted(list(set(wordlist.values())), key=natural_sort_key)
        if 'Coverage' in categories and len(categories) == 1:
            is_plain = True
        else:
            is_plain = False

        # Build the query
        group_col = None
        if basis == 'By Filename':
            group_col = 'filename'
        elif basis == 'By Metadata' and metadata_col:
            group_col = metadata_col

        where_clause = "WHERE 1=1"
        if xml_where_clause:
            where_clause += xml_where_clause

        if group_col:
            query = f"SELECT {group_col}, _token_low, lemma, count(*) as freq FROM corpus {where_clause} GROUP BY {group_col}, _token_low, lemma"
        else:
            query = f"SELECT _token_low, lemma, count(*) as freq FROM corpus {where_clause} GROUP BY _token_low, lemma"

        df_tokens = con.execute(query, xml_params).fetch_df()
        
        if df_tokens.empty:
            return pd.DataFrame()

        # Map tokens to categories by checking both token and lemma
        def get_category(row):
            token_val = str(row['_token_low']).lower()
            lemma_val = str(row['lemma']).lower() if 'lemma' in row and not pd.isna(row['lemma']) else ''
            
            # Check token first
            if token_val in wordlist:
                return wordlist[token_val]
            # Check lemma second
            if lemma_val in wordlist:
                return wordlist[lemma_val]
            return 'OOV'

        df_tokens['Category'] = df_tokens.apply(get_category, axis=1)

        # Aggregate by Segment and Category
        if group_col:
            results = df_tokens.groupby([group_col, 'Category'])['freq'].sum().reset_index()
            # Pivot to have Categories as columns
            pivot_df = results.pivot(index=group_col, columns='Category', values='freq').fillna(0)
        else:
            results = df_tokens.groupby('Category')['freq'].sum().reset_index()
            # For whole corpus, we want a single row
            pivot_df = results.set_index('Category').T
            pivot_df.index = ['Whole Corpus']

        # Ensure all wordlist categories + OOV are present
        all_cats = categories + ['OOV']
        for cat in all_cats:
            if cat not in pivot_df.columns:
                pivot_df[cat] = 0

        # Reorder columns: wordlist categories first, then OOV
        final_cols = [cat for cat in categories if cat in pivot_df.columns] + ['OOV']
        pivot_df = pivot_df[final_cols]

        # Calculate Percentages
        pivot_df['Total'] = pivot_df.sum(axis=1)
        
        # Create a display DataFrame with Freq and %
        display_rows = []
        for idx, row in pivot_df.iterrows():
            total = row['Total']
            res_row = {'Segment': idx}
            for cat in final_cols:
                freq = int(row[cat])
                perc = (freq / total * 100) if total > 0 else 0
                res_row[f"{cat} Freq"] = freq
                res_row[f"{cat} %"] = round(perc, 2)
            res_row['Total Tokens'] = int(total)
            display_rows.append(res_row)

        return pd.DataFrame(display_rows)

    finally:
        con.close()
