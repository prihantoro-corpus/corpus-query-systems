import duckdb
import pandas as pd
import os

def load_wordlist(file_path_or_content, is_file=True):
    """
    Loads a wordlist from a file path or string content.
    Supports plain (1 col) and categorized (2 cols).
    Returns a dictionary: {word: category}
    For plain wordlists, category is 'Coverage'.
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

    lines = [line.strip() for line in content.split('\n') if line.strip()]
    for line in lines:
        parts = line.split('\t')
        if len(parts) >= 2:
            word = parts[0].strip().lower()
            category = parts[1].strip()
            wordlist[word] = category
        else:
            word = line.strip().lower()
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
        categories = sorted(list(set(wordlist.values())))
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
            query = f"SELECT {group_col}, _token_low, count(*) as freq FROM corpus {where_clause} GROUP BY {group_col}, _token_low"
        else:
            query = f"SELECT _token_low, count(*) as freq FROM corpus {where_clause} GROUP BY _token_low"

        df_tokens = con.execute(query, xml_params).fetch_df()
        
        if df_tokens.empty:
            return pd.DataFrame()

        # Map tokens to categories
        def get_category(token):
            return wordlist.get(str(token).lower(), 'OOV')

        df_tokens['Category'] = df_tokens['_token_low'].apply(get_category)

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
