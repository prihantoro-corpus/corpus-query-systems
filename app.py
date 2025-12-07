# app.py
# CORTEX Corpus Explorer v17.16 - N-Gram Relative Frequency
import streamlit as st
import pandas as pd
import numpy as np
import math
from collections import Counter
from io import BytesIO, StringIO 
import tempfile 
import os       
import re       
import requests 
import matplotlib.pyplot as plt 
from wordcloud import WordCloud 
from pyvis.network import Network
import streamlit.components.v1 as components 

# We explicitly exclude external LLM libraries for the free, stable version.
# The interpret_results_llm function is replaced with a placeholder.

st.set_page_config(page_title="CORTEX - Corpus Explorer v17.16", layout="wide") 

# --- CONSTANTS ---
KWIC_MAX_DISPLAY_LINES = 100
KWIC_INITIAL_DISPLAY_HEIGHT = 10 
KWIC_COLLOC_DISPLAY_LIMIT = 20 # Limit for KWIC examples below collocation tables

# ---------------------------
# Initializing Session State
# ---------------------------
# CHANGE: Default view is now 'select_module' to prevent showing settings/empty modules on load
if 'view' not in st.session_state:
    st.session_state['view'] = 'select_module' 
if 'trigger_analyze' not in st.session_state:
    st.session_state['trigger_analyze'] = False
if 'initial_load_complete' not in st.session_state:
    st.session_state['initial_load_complete'] = False
if 'collocate_pos_regex' not in st.session_state: 
    st.session_state['collocate_pos_regex'] = ''
if 'pattern_collocate_pos' not in st.session_state: 
    st.session_state['pattern_collocate_pos'] = ''
if 'collocate_lemma' not in st.session_state: 
    st.session_state['collocate_lemma'] = ''
if 'llm_interpretation_result' not in st.session_state:
    st.session_state['llm_interpretation_result'] = None
# --- Input State (must be initialized for keyed widgets) ---
if 'dict_word_input_main' not in st.session_state: 
    st.session_state['dict_word_input_main'] = ''
if 'collocate_regex_input' not in st.session_state: 
    st.session_state['collocate_regex_input'] = ''
if 'pattern_collocate_input' not in st.session_state:
    st.session_state['pattern_collocate_input'] = ''
if 'pattern_collocate_pos_input' not in st.session_state:
     st.session_state['pattern_collocate_pos_input'] = ''
if 'typed_target_input' not in st.session_state:
     st.session_state['typed_target_input'] = ''
if 'max_collocates' not in st.session_state:
    st.session_state['max_collocates'] = 20
if 'coll_window' not in st.session_state:
    st.session_state['coll_window'] = 5
if 'mi_min_freq' not in st.session_state:
    st.session_state['mi_min_freq'] = 1
# --- N-Gram State ---
if 'n_gram_size' not in st.session_state:
    st.session_state['n_gram_size'] = 2
if 'n_gram_filters' not in st.session_state:
    st.session_state['n_gram_filters'] = {} # Dictionary to hold positional filters: {'1': 'pattern', '2': 'pattern', ...}
if 'n_gram_trigger_analyze' not in st.session_state:
    st.session_state['n_gram_trigger_analyze'] = False
if 'n_gram_results_df' not in st.session_state:
    st.session_state['n_gram_results_df'] = pd.DataFrame()
# --- NEW: Parallel Corpus State ---
if 'parallel_df' not in st.session_state:
    st.session_state['parallel_df'] = None
if 'parallel_col_name' not in st.session_state:
    st.session_state['parallel_col_name'] = None


# ---------------------------
# Built-in Corpus Configuration
# ---------------------------
BUILT_IN_CORPORA = {
    "Select built-in corpus...": None,
    "Europarl 1M Only": "https://raw.githubusercontent.com/prihantoro-corpus/corpus-query-systems/main/europarl_en-1M-only%20v2.txt",
    "sample speech 13kb only": "https://raw.githubusercontent.com/prihantoro-corpus/corpus-query-systems/main/Speech%20address.txt",
}

# Define color map constants globally (used for both graph and word cloud)
POS_COLOR_MAP = {
    'N': '#33CC33',  # Noun (Green)
    'V': '#3366FF',  # Verb (Blue)
    'J': '#FF33B5',  # Adjective (Pink)
    'R': '#FFCC00',  # Adverb (Yellow)
    '#': '#AAAAAA',  # Nonsense Tag / Raw (Gray)
    'O': '#AAAAAA'   # Other (Gray)
}

PUNCTUATION = {'.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}', '"', "'", '---', '--', '-', '...', '«', '»', '—'}

# --- NAVIGATION FUNCTIONS ---
def set_view(view_name):
    st.session_state['view'] = view_name
    st.session_state['llm_interpretation_result'] = None
    
def reset_analysis():
    st.cache_data.clear()
    # CHANGE: Reset view to 'select_module' on corpus change
    st.session_state['view'] = 'select_module'
    st.session_state['trigger_analyze'] = False
    st.session_state['n_gram_trigger_analyze'] = False
    st.session_state['initial_load_complete'] = False
    st.session_state['llm_interpretation_result'] = None
    st.session_state['parallel_df'] = None # NEW: Reset parallel data
    st.session_state['parallel_col_name'] = None # NEW: Reset parallel column

# --- Analysis Trigger Callback (for implicit Enter/change) ---
def trigger_analysis_callback():
    st.session_state['trigger_analyze'] = True
    st.session_state['llm_interpretation_result'] = None

# --- N-Gram Analysis Trigger Callback ---
def trigger_n_gram_analysis_callback():
    st.session_state['n_gram_trigger_analyze'] = True

# --- Dictionary Helper: Get all forms by lemma ---
@st.cache_data
def get_all_lemma_forms_details(df_corpus, target_word):
    """Finds all unique tokens/POS pairs sharing the target word's lemma(s)."""
    target_lower = target_word.lower()
    matching_rows = df_corpus[df_corpus['token'].str.lower() == target_lower]
    
    if matching_rows.empty or 'lemma' not in df_corpus.columns:
        return pd.DataFrame(), [], []
        
    unique_lemmas = matching_rows['lemma'].unique()
    
    # Filter out nonsense tags
    valid_lemmas = [l for l in unique_lemmas if l not in ('##', '###')]
    if not valid_lemmas:
        return pd.DataFrame(), [], []

    # Get all forms sharing these valid lemmas
    all_forms_df = df_corpus[df_corpus['lemma'].isin(valid_lemmas)][['token', 'pos', 'lemma']].copy()
    
    # Keep only unique token-pos-lemma combinations, sorted by token name
    forms_list = all_forms_df.drop_duplicates().sort_values('token').reset_index(drop=True)
    
    # Also return the unique POS and Lemma lists for the summary header (re-using old logic)
    return forms_list, all_forms_df['pos'].unique(), valid_lemmas

# --- Regex Forms Helper (Caching Removed for Bug Fix) ---
def get_related_forms_by_regex(df_corpus, target_word):
    # Construct a broad regex for related forms: .*<target_word>.* (case insensitive)
    pattern_str = f".*{re.escape(target_word)}.*"
    pattern = re.compile(pattern_str, re.IGNORECASE)
    
    all_unique_tokens = df_corpus['token'].unique()
    
    related_forms = []
    for token in all_unique_tokens:
        if pattern.fullmatch(token):
            related_forms.append(token)
            
    target_lower = target_word.lower()
    final_forms = [w for w in related_forms if w.lower() != target_lower]
    
    return sorted(list(set(final_forms)))

# --- LLM PLACEHOLDER ---
def interpret_results_llm(target_word, analysis_type, data_description, data):
    """Placeholder for LLM function."""
    mock_response = f"""
    ### 🧠 Feature Currently Unavailable

    The external LLM interpretation feature is temporarily disabled due to stability and congestion issues with free public APIs (Gemini/Hugging Face).

    **CORTEX recommends focusing on the raw linguistic data** provided in the Concordance, Collocation, and Dictionary modules to draw your own expert conclusion.

    **Analysis Context:**
    * Target: **"{target_word}"**
    * Analysis Type: **{analysis_type}**
    """
    st.session_state['llm_interpretation_result'] = mock_response
    st.warning("LLM Feature Disabled. See 'LLM Interpretation' expander for details.")
    return mock_response
    
# --- KWIC/Concordance Helper Function (Reusable by Dictionary) ---
@st.cache_data(show_spinner=False)
def generate_kwic(df_corpus, raw_target_input, kwic_left, kwic_right, pattern_collocate_input="", pattern_collocate_pos_input="", pattern_window=0, limit=KWIC_MAX_DISPLAY_LINES, random_sample=False, parallel_df=None, translation_col=None): # UPDATED SIGNATURE
    """
    Generalized function to generate KWIC lines based on target and optional collocate filter.
    Returns: (list_of_kwic_rows, total_matches, primary_target_mwu, literal_freq)
    """
    total_tokens = len(df_corpus)
    tokens_lower = df_corpus["_token_low"].tolist()
    
    # --- MWU/WILDCARD/STRUCTURAL RESOLUTION (Unified Search) ---
    search_terms = raw_target_input.split()
    primary_target_len = len(search_terms)
    is_raw_mode = 'pos' not in df_corpus.columns or df_corpus['pos'].str.contains('##', na=False).sum() > 0.99 * len(df_corpus)
    is_structural_search = not is_raw_mode and any('[' in t or '_' in t for t in search_terms)
    
    def create_structural_matcher(term):
        lemma_pattern = None
        pos_pattern = None
        lemma_match = re.search(r"\[(.*?)\]", term)
        if lemma_match:
            lemma_input = lemma_match.group(1).strip().lower()
            if lemma_input:
                lemma_pattern_str = re.escape(lemma_input).replace(r'\*', '.*')
                lemma_pattern = re.compile(f"^{lemma_pattern_str}$")
        pos_match = re.search(r"\_([\w\*|]+)", term)
        if pos_match:
            pos_input = pos_match.group(1).strip()
            if pos_input:
                pos_patterns = [p.strip() for p in pos_input.split('|') if p.strip()]
                full_pos_regex_list = [re.escape(p).replace(r'\*', '.*') for p in pos_patterns]
                pos_pattern = re.compile("^(" + "|".join(full_pos_regex_list) + ")$")
        if lemma_pattern or pos_pattern:
             return {'type': 'structural', 'lemma_pattern': lemma_pattern, 'pos_pattern': pos_pattern}
        pattern = re.escape(term.lower()).replace(r'\*', '.*')
        return {'type': 'word', 'pattern': re.compile(f"^{pattern}$")}

    search_components = [create_structural_matcher(term) for term in search_terms]
    all_target_positions = []
    
    # Execute Search Loop (find all positions)
    if primary_target_len == 1 and not is_structural_search:
        target_pattern = search_components[0]['pattern']
        for i, token in enumerate(tokens_lower):
            if target_pattern.fullmatch(token):
                all_target_positions.append(i)
    else:
        for i in range(len(tokens_lower) - primary_target_len + 1):
            match = True
            for k, component in enumerate(search_components):
                corpus_index = i + k
                if corpus_index >= len(df_corpus): break
                
                if component['type'] == 'word':
                    if not component['pattern'].fullmatch(tokens_lower[corpus_index]):
                        match = False; break
                        
                elif component['type'] == 'structural':
                    current_lemma = df_corpus["lemma"].iloc[corpus_index].lower()
                    current_pos = df_corpus["pos"].iloc[corpus_index]
                    
                    lemma_match = component['lemma_pattern'] is None or component['lemma_pattern'].fullmatch(current_lemma)
                    pos_match = component['pos_pattern'] is None or component['pos_pattern'].fullmatch(current_pos)
                    
                    if not (lemma_match and pos_match):
                        match = False; break
                        
            if match:
                all_target_positions.append(i)
                
    literal_freq = len(all_target_positions)
    if literal_freq == 0:
        return ([], 0, raw_target_input, 0)

    # --- Apply Pattern Filtering ---
    final_positions = all_target_positions
    
    # Check if a pattern filter is provided
    is_pattern_search_active = pattern_collocate_input or pattern_collocate_pos_input

    if is_pattern_search_active and pattern_window > 0:
        final_positions = []
        collocate_word_regex = re.compile(re.escape(pattern_collocate_input).replace(r'\*', '.*')) if pattern_collocate_input else None
        collocate_pos_regex = None 
        
        if pattern_collocate_pos_input and not is_raw_mode:
            pos_patterns = [p.strip() for p in pattern_collocate_pos_input.split('|') if p.strip()]
            if pos_patterns:
                full_pos_regex = re.compile("^(" + "|".join([re.escape(p).replace(r'\*', '.*') for p in pos_patterns]) + ")$")
                collocate_pos_regex = full_pos_regex

        for i in all_target_positions:
            start_index = max(0, i - pattern_window)
            end_index = min(len(tokens_lower), i + primary_target_len + pattern_window)
            
            found_collocate = False
            for j in range(start_index, end_index):
                if i <= j < i + primary_target_len: continue # Skip node word(s)
                
                token_lower = tokens_lower[j]
                token_pos = df_corpus["pos"].iloc[j]
                
                word_matches = collocate_word_regex is None or collocate_word_regex.fullmatch(token_lower)
                pos_matches = collocate_pos_regex is None or (collocate_pos_regex.fullmatch(token_pos) if not is_raw_mode else False)
                
                if word_matches and pos_matches:
                    found_collocate = True
                    break
            
            if found_collocate:
                final_positions.append(i)
                
    total_matches = len(final_positions)
    if total_matches == 0:
        return ([], 0, raw_target_input, 0)

    # --- Sample Positions ---
    if random_sample:
        import random
        random.seed(42) # Consistent random sample
        sample_size = min(total_matches, limit)
        display_positions = random.sample(final_positions, sample_size)
    else:
        display_positions = final_positions[:limit]
    
    # --- Format KWIC lines ---
    kwic_rows = []
    
    # Use pattern_window for context display if pattern search is active
    current_kwic_left = pattern_window if is_pattern_search_active and pattern_window > 0 else kwic_left
    current_kwic_right = pattern_window if is_pattern_search_active and pattern_window > 0 else kwic_right
    
    # Re-initialize regex for highlighting purposes (needs to be local)
    collocate_word_regex_highlight = re.compile(re.escape(pattern_collocate_input).replace(r'\*', '.*')) if pattern_collocate_input else None
    
    # Re-generate POS regex if necessary
    collocate_pos_regex_highlight = None
    if pattern_collocate_pos_input and not is_raw_mode:
        pos_patterns = [p.strip() for p in pattern_collocate_pos_input.split('|') if p.strip()]
        if pos_patterns:
            full_pos_regex = re.compile("^(" + "|".join([re.escape(p).replace(r'\*', '.*') for p in pos_patterns]) + ")$")
            collocate_pos_regex_highlight = full_pos_regex
    
    for i in display_positions:
        kwic_start = max(0, i - current_kwic_left)
        kwic_end = min(total_tokens, i + primary_target_len + current_kwic_right)
        full_line_tokens = df_corpus["token"].iloc[kwic_start:kwic_end].tolist()
        
        formatted_line = []
        node_orig_tokens = []
        collocate_to_display = ""
        
        for k, token in enumerate(full_line_tokens):
            token_index_in_corpus = kwic_start + k
            token_lower = token.lower()
            token_pos = df_corpus["pos"].iloc[token_index_in_corpus]
            
            is_node_word = (i <= token_index_in_corpus < i + primary_target_len)
            
            is_collocate_match = False
            if is_pattern_search_active and not is_node_word:
                 word_matches_highlight = collocate_word_regex_highlight is None or collocate_word_regex_highlight.fullmatch(token_lower)
                 pos_matches_highlight = collocate_pos_regex_highlight is None or (collocate_pos_regex_highlight.fullmatch(token_pos) if not is_raw_mode else False)
                 
                 if word_matches_highlight and pos_matches_highlight:
                    is_collocate_match = True
                    if collocate_to_display == "": # Capture the first matching collocate
                        collocate_to_display = token # Use the original token case
            
            if is_node_word:
                node_orig_tokens.append(token)
                formatted_line.append("") 
                
            elif is_collocate_match:
                # Collocate BOLDED and BRIGHT YELLOW HIGHLIGHTED
                html_token = f"<b><span style='color: black; background-color: #FFEA00;'>{token}</span></b>"
                formatted_line.append(html_token)
            else:
                formatted_line.append(token)

        node_start_rel = i - kwic_start
        node_end_rel = node_start_rel + primary_target_len

        left_context = formatted_line[:node_start_rel]
        right_context = formatted_line[node_end_rel:]
        node_orig = " ".join(node_orig_tokens)

        # NEW: Get Translation
        translation_text = ""
        if parallel_df is not None and translation_col is not None and 'sent_id' in df_corpus.columns:
            node_sent_id = df_corpus["sent_id"].iloc[i]
            
            # Use the sentence ID as the row index for the parallel DF (0-indexed alignment)
            if node_sent_id < len(parallel_df):
                # Ensure the column exists before trying to access
                if translation_col in parallel_df.columns:
                    translation_text = str(parallel_df.iloc[node_sent_id][translation_col])
                else:
                    translation_text = "[Error: Translation Column Not Found]"
            else:
                 translation_text = "[Translation ID Out of Bounds]"

        
        kwic_rows.append({
            "Left": " ".join(left_context), 
            "Node": node_orig, 
            "Right": " ".join(right_context),
            "Collocate": collocate_to_display, # Only filled if pattern search is active
            "Translation": translation_text # NEW FIELD
        })
        
    return (kwic_rows, total_matches, raw_target_input, literal_freq)

# --- Word Cloud Function ---
@st.cache_data
# ... (create_word_cloud remains the same) ...

# --- Statistical Helpers ---
# ... (compute_ll, compute_mi, significance_from_ll remain the same) ...

# --- IO / Data Helpers ---
# ... (df_to_excel_bytes, create_pyvis_graph remain the same) ...

@st.cache_data
def download_file_to_bytesio(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() 
        return BytesIO(response.content)
    except Exception as e:
        st.error(f"Failed to download built-in corpus from {url}. Ensure the file is public and the URL is a RAW content link.")
        return None

# NEW: Function to add sentence ID to the main corpus DF
def add_sentence_id(df):
    """Adds a sent_id column based on common sentence-final punctuation (.,!,?)."""
    if df is None or 'sent_id' in df.columns:
        return df
    
    # Simple rule: New sentence after ., !, ?
    sentence_enders = {'.', '!', '?'}
    # Identify tokens that are sentence enders
    is_sent_end = df['token'].apply(lambda t: t.strip() in sentence_enders)
    
    # Calculate sent_id: cumsum on the shifted 'is_sent_end' to count breaks
    # Start at 0, increment after a sentence ender.
    df['sent_id'] = is_sent_end.shift(1).fillna(False).cumsum().astype(int)
    
    return df

@st.cache_data
def load_corpus_file(file_source, sep=r"\s+"):
    if file_source is None: return None
    # ... (existing file loading logic) ...
    # ... (existing dataframe creation logic, including the raw text fallback) ...
    
    # NEW: Call add_sentence_id before returning
    if df is not None:
        df = add_sentence_id(df)
        
    return df
    # ... (rest of load_corpus_file remains the same) ...


# -----------------------------------------------------
# N-GRAM LOGIC (UPDATED FOR STRUCTURAL FILTERING & RELATIVE FREQUENCY)
# -----------------------------------------------------
# ... (generate_n_grams remains the same) ...

# -----------------------------------------------------
# Function to display KWIC examples for collocates
# -----------------------------------------------------
def display_collocation_kwic_examples(df_corpus, node_word, top_collocates_df, window, limit_per_collocate):
    # Get parallel info from session state
    parallel_df = st.session_state.get('parallel_df')
    translation_col = st.session_state.get('parallel_col_name')
    
    colloc_list = top_collocates_df.head(limit_per_collocate * KWIC_COLLOC_DISPLAY_LIMIT)['Collocate'].tolist()
    
    # ... (existing logic) ...
    
    for rank, collocate_word in enumerate(colloc_list):
        # ... (existing logic) ...
        
        # UPDATED CALL TO generate_kwic
        kwic_rows, total_matches, _, _ = generate_kwic(
            df_corpus, 
            node_word, 
            window, 
            window, 
            pattern_collocate_input=collocate_word, 
            pattern_collocate_pos_input="", 
            pattern_window=window, 
            limit=limit_per_collocate,
            parallel_df=parallel_df, # NEW ARGUMENT
            translation_col=translation_col # NEW ARGUMENT
        )
        
        if kwic_rows:
            kwic_row = kwic_rows[0]
            collex_rows_total.append({
                "Collocate": f"{rank+1}. {collocate_word}", 
                "Left Context": kwic_row['Left'], 
                "Node": kwic_row['Node'], 
                "Right Context": kwic_row['Right'],
                "Translation": kwic_row['Translation'] # NEW FIELD
            })

    if collex_rows_total:
        collex_df = pd.DataFrame(collex_rows_total)
        # UPDATED HEADER: added Translation column
        header = "<tr><th>Collocate (Rank)</th><th>Left Context</th><th>Node</th><th>Right Context</th><th>Translation</th></tr>" 
        html_table = collex_df.to_html(header=False, escape=False, classes=['collex-table-inner'], index=False)
        html_table = html_table.replace("<thead></thead>", f"<thead>{header}</thead>", 1)
        scrollable_html = f"<div class='collex-table-container-fixed'>{html_table}</div>"
        st.markdown(scrollable_html, unsafe_allow_html=True)
        st.caption(f"Context window is set to **±{window} tokens** (Collocation window). Matching collocate is **bolded and highlighted bright yellow**.")
    else:
        st.info(f"No specific KWIC examples found for the top {len(colloc_list)} collocates within the ±{window} window.")

# -----------------------------------------------------
# COLLOCATION LOGIC (generate_collocation_results remains the same)
# -----------------------------------------------------


# ---------------------------
# UI: header
# ---------------------------
st.title("CORTEX - Corpus Texts Explorer v17.16")
st.caption("Upload vertical corpus (**token POS lemma**) or **raw horizontal text**. Add an optional **sentence-aligned parallel corpus** for translation display.") # UPDATED CAPTION

# ---------------------------
# Panel: upload and corpus info
# ---------------------------
corpus_source = None
corpus_name = "Uploaded File" 

# --- SIDEBAR: CORPUS SELECTION, NAVIGATION, & MODULE SETTINGS ---
with st.sidebar:
    # 1. CORPUS SELECTION (TOP)
    st.header("1a. Primary Corpus Source") # UPDATED HEADER
    selected_corpus_name = st.selectbox(
        "Select a pre-loaded corpus:",
        options=list(BUILT_IN_CORPORA.keys()),
        key="corpus_select",
        on_change=reset_analysis
    )
    uploaded_file = st.file_uploader(
        "OR Upload your own corpus file", 
        type=["txt","csv"], 
        key="file_upload",
        on_change=reset_analysis
    )
    
    # Determine the corpus source
    if uploaded_file is not None:
        corpus_source = uploaded_file
        corpus_name = uploaded_file.name
    elif selected_corpus_name != "Select built-in corpus...":
        corpus_url = BUILT_IN_CORPORA[selected_corpus_name]
        if 'initial_load_complete' not in st.session_state or st.session_state['initial_load_complete'] == False:
            st.info(f"Downloading {selected_corpus_name}...")
        corpus_source = download_file_to_bytesio(corpus_url)
        corpus_name = selected_corpus_name
        
    # Load the main corpus
    df = load_corpus_file(corpus_source)
    
    # NEW: PARALLEL CORPUS UPLOAD SECTION
    st.markdown("---")
    st.header("1b. Parallel Corpus (Optional)")
    parallel_file = st.file_uploader(
        "Upload sentence-aligned parallel file (CSV/TSV/XLSX)", 
        type=["txt", "csv", "tsv", "xlsx"], 
        key="parallel_file_upload",
        on_change=reset_analysis
    )
    
    # Load and process parallel file
    parallel_df = None
    if parallel_file is not None:
        try:
            parallel_file.seek(0)
            
            if parallel_file.name.endswith('.xlsx'):
                parallel_df = pd.read_excel(parallel_file)
            else:
                # Try TSV, then CSV
                try:
                    parallel_df = pd.read_csv(parallel_file, sep='\t', header=0, dtype=str, keep_default_na=False)
                except Exception:
                    parallel_file.seek(0)
                    parallel_df = pd.read_csv(parallel_file, sep=',', header=0, dtype=str, keep_default_na=False)

            if df is not None and 'sent_id' in df.columns:
                max_sent_id = df['sent_id'].max()
                # Truncate to match sentence count of main corpus
                if len(parallel_df) > max_sent_id + 1:
                     parallel_df = parallel_df.head(max_sent_id + 1)
            
            st.session_state['parallel_df'] = parallel_df
            
            col_options = [col for col in parallel_df.columns if col.strip()]
            if col_options:
                default_col_index = min(1, len(col_options) - 1) 
                
                st.session_state['parallel_col_name'] = st.selectbox(
                    "Select Translation Column", 
                    options=col_options, 
                    index=default_col_index,
                    key="parallel_col_select",
                    help="This column's content will be displayed as the translation."
                )
                st.success(f"Parallel corpus loaded with {len(parallel_df)} sentences. Translation: **{st.session_state['parallel_col_name']}**")
            else:
                st.warning("Parallel file loaded but no usable columns found.")
                st.session_state['parallel_df'] = None
                
        except Exception as e:
            st.error(f"Error loading parallel file: {e}")
            st.session_state['parallel_df'] = None
    else:
        st.session_state['parallel_df'] = None
    
    # End Corpus Selection block
    
    # Check if df is ready for analysis to decide on next steps
    if df is not None and not df.empty:
        total_tokens = len(df)
        is_raw_mode_sidebar = 'pos' not in df.columns or df['pos'].str.contains('##', na=False).sum() > 0.99 * len(df)
        unique_types = df[~df['token'].str.lower().isin(PUNCTUATION)]['token'].nunique()
        unique_lemmas = df['lemma'].nunique() if 'lemma' in df.columns else 'N/A'
        
        # 2. NAVIGATION (MODULE SELECTION)
        st.header("2. Analysis Module")
        
        module_options = ['Select a module...', 'Corpus Overview', 'Concordance (KWIC)', 'Collocation', 'N-Gram', 'Dictionary']
        
        selected_view_index = module_options.index('Select a module...')
        if st.session_state['view'] != 'select_module' and st.session_state['view'] != 'overview':
             # Try to pre-select the current view if it's one of the main modules
             current_view_name = [m for m in module_options if m.lower().startswith(st.session_state['view']) or m.lower().startswith(st.session_state['view'].replace('_', ' '))]
             if current_view_name: selected_view_index = module_options.index(current_view_name[0])

        selected_view = st.radio(
            "Select your analysis module:",
            options=module_options,
            index=selected_view_index, 
            format_func=lambda x: x.split(' ')[0] if x != 'Select a module...' else x, 
            key='main_module_radio',
            on_change=trigger_analysis_callback 
        )
        
        if selected_view == 'Select a module...': set_view('select_module')
        elif selected_view == 'Corpus Overview': set_view('overview')
        elif selected_view == 'Concordance (KWIC)': set_view('concordance')
        elif selected_view == 'Collocation': set_view('collocation')
        elif selected_view == 'N-Gram': set_view('n_gram')
        elif selected_view == 'Dictionary': set_view('dictionary')
        
        # 3. MODULE SETTINGS (CONDITIONAL) - Now correctly gated by 'select_module'
        if st.session_state['view'] != 'select_module':
            st.header("3. Module Settings")
            
            # Concordance Settings
            if st.session_state['view'] == 'concordance':
                st.subheader("Concordance Settings")
                # ... (Concordance settings widgets)
                
            # Collocation Settings
            elif st.session_state['view'] == 'collocation':
                st.subheader("Collocation Settings")
                # ... (Collocation settings widgets)
                
            # N-Gram Settings
            elif st.session_state['view'] == 'n_gram':
                st.subheader("N-Gram Settings")
                # ... (N-Gram settings widgets)
    
            # Dictionary Settings (No settings here, just word input)
            elif st.session_state['view'] == 'dictionary':
                st.subheader("Dictionary Word Input")
                st.text_input("Enter target word/pattern:", value=st.session_state.get('dict_word_input_main', ''), key="dict_word_input_main_key", on_change=trigger_analysis_callback)
    
    else: # If df is not loaded
        st.info("Upload a corpus file or select a built-in corpus to enable analysis modules.")

# --- Main Analysis Area ---

# Check if corpus is loaded before proceeding
if df is None or df.empty:
    if uploaded_file is None and selected_corpus_name == "Select built-in corpus...":
         st.warning("Please upload a corpus file or select a built-in corpus from the sidebar.")
    elif uploaded_file is not None or selected_corpus_name != "Select built-in corpus...":
         st.warning("Loading/Processing corpus. Please wait.")
    st.stop()
    
# NEW: Initial state welcome message
if st.session_state['view'] == 'select_module':
    st.markdown("""
        ## Welcome to CORTEX Corpus Explorer
        
        Please select an **Analysis Module** from the sidebar (Step 2) to begin your corpus investigation.
        
        - **Corpus Overview**: View summary statistics and word cloud.
        - **Concordance (KWIC)**: Search for a word/pattern and view it in context.
        - **Collocation**: Find words that statistically co-occur with your search term.
        - **N-Gram**: Find frequent sequences of words.
        - **Dictionary**: Get a quick summary of a word's forms and common contexts.
        
        *If you've uploaded a sentence-aligned **Parallel Corpus** (Step 1b), the translations will be automatically included in your Concordance results.*
    """)
    st.stop()

# ... (rest of the main content remains the same, except for the KWIC display modifications) ...

# -----------------------------------------------------
# MODULE: CORPUS OVERVIEW
# -----------------------------------------------------
if st.session_state['view'] == 'overview':
    # ... (existing overview logic) ...
    pass


# -----------------------------------------------------
# MODULE: CONCORDANCE LOGIC
# -----------------------------------------------------
elif st.session_state['view'] == 'concordance':
    # ... (existing concordance input logic) ...
    
    # ... (inside run_concordance_analysis, UPDATED CALL TO generate_kwic)
    kwic_rows, total_matches, primary_target_mwu, literal_freq = generate_kwic(
        df, 
        raw_target_input, 
        kwic_left, 
        kwic_right, 
        pattern_collocate, 
        pattern_collocate_pos_input, 
        pattern_search_window,
        parallel_df=st.session_state.get('parallel_df'), # NEW ARGUMENT
        translation_col=st.session_state.get('parallel_col_name') # NEW ARGUMENT
    )
    # ... (rest of concordance logic) ...
    
    # ... (Concordance Display Logic, around line 1133) ...
    # UPDATED KWIC DISPLAY FOR TRANSLATION COLUMN
    col_kwic, col_freq = st.columns([3, 2], gap="large")
    with col_kwic:
        st.subheader(f"Concordance (KWIC) — top {len(kwic_rows)} lines (Scrollable max {KWIC_MAX_DISPLAY_LINES})")
        
        # Check if translation is available to decide on columns to drop
        translation_available = st.session_state.get('parallel_df') is not None and st.session_state.get('parallel_col_name') is not None
        
        if translation_available:
            kwic_df = pd.DataFrame(kwic_rows).drop(columns=['Collocate'])
            # Reorder columns to place Translation last
            cols = ['Left', 'Node', 'Right', 'Translation']
            kwic_df = kwic_df[cols]
        else:
            kwic_df = pd.DataFrame(kwic_rows).drop(columns=['Collocate', 'Translation']) # Drop new column if not available
            
        kwic_preview = kwic_df.copy().reset_index(drop=True)
        kwic_preview.insert(0, "No", range(1, len(kwic_preview)+1))
        
        # ... (KWIC table style remains the same, but the header changes based on columns)
        
        # Use HTML to force the column names and allow full width
        kwic_html = kwic_preview.to_html(index=False, classes=['dataframe'])
        st.markdown(f"<div class='dataframe-container-scroll'>{kwic_html}</div>", unsafe_allow_html=True)

# -----------------------------------------------------
# MODULE: N-GRAM LOGIC
# -----------------------------------------------------
elif st.session_state['view'] == 'n_gram':
    # ... (existing n-gram logic) ...
    pass


# -----------------------------------------------------
# MODULE: DICTIONARY LOGIC
# -----------------------------------------------------
elif st.session_state['view'] == 'dictionary':
    # ... (existing dictionary logic) ...
    
    # ... (inside dictionary logic, around line 1335, for Random Concordance Examples)
    with st.spinner(f"Fetching random concordance examples for '{current_dict_word}'..."):
        # UPDATED CALL TO generate_kwic
        kwic_rows, total_matches, _, _ = generate_kwic(
            df, 
            current_dict_word, 
            kwic_left, 
            kwic_right, 
            random_sample=True, 
            limit=KWIC_MAX_DISPLAY_LINES,
            parallel_df=st.session_state.get('parallel_df'), # NEW ARGUMENT
            translation_col=st.session_state.get('parallel_col_name') # NEW ARGUMENT
        )
    
    if kwic_rows:
        # ... (existing display limit) ...
        
        # UPDATED KWIC DISPLAY FOR TRANSLATION COLUMN
        translation_available = st.session_state.get('parallel_df') is not None and st.session_state.get('parallel_col_name') is not None
        
        if translation_available:
            kwic_df = pd.DataFrame(kwic_rows).drop(columns=['Collocate'])
            cols = ['Left', 'Node', 'Right', 'Translation']
            kwic_df = kwic_df[cols]
        else:
            kwic_df = pd.DataFrame(kwic_rows).drop(columns=['Collocate', 'Translation'])

        # ... (rest of dictionary KWIC display remains the same) ...
    
    # ... (inside dictionary logic, around line 1383, for Collocation Examples)
    # The dedicated function display_collocation_kwic_examples already handles the new arguments.
    display_collocation_kwic_examples(
        df_corpus=df,
        node_word=current_dict_word,
        top_collocates_df=top_collocates,
        window=coll_window,
        limit_per_collocate=1
    )


# -----------------------------------------------------
# MODULE: COLLOCATION LOGIC
# -----------------------------------------------------
elif st.session_state['view'] == 'collocation' and st.session_state.get('analyze_btn', False) and st.session_state.get('typed_target_input'):
    # ... (existing collocation logic) ...
    
    # ... (inside collocation display, around line 1600)
    # LL-Ranked KWIC Examples
    st.subheader(f"📚 Concordance Examples for Top {KWIC_COLLOC_DISPLAY_LIMIT} LL Collocates (1 example per collocate)")
    # The dedicated function display_collocation_kwic_examples already handles the new arguments.
    display_collocation_kwic_examples(
        df_corpus=df, 
        node_word=primary_target_mwu, 
        top_collocates_df=full_ll, 
        window=coll_window,
        limit_per_collocate=1 
    )
    
    st.markdown("---")
    
    # MI-Ranked KWIC Examples
    st.subheader(f"📚 Concordance Examples for Top {KWIC_COLLOC_DISPLAY_LIMIT} MI Collocates (1 example per collocate)")
    # The dedicated function display_collocation_kwic_examples already handles the new arguments.
    display_collocation_kwic_examples(
        df_corpus=df, 
        node_word=primary_target_mwu, 
        top_collocates_df=full_mi_all, 
        window=coll_window,
        limit_per_collocate=1
    )
    # ... (rest of collocation logic) ...
