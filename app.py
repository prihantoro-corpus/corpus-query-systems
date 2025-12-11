# app.py
# CORTEX -- Corpus Explorer Version Alpha (10-Dec-25)

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

# --- Re-enabled Imports ---
try:
    from wordcloud import WordCloud 
    WORDCLOUD_FEATURE_AVAILABLE = True
except ImportError:
    WORDCLOUD_FEATURE_AVAILABLE = False
    
try:
    from pyvis.network import Network
    PYVIS_FEATURE_AVAILABLE = True
except ImportError:
    PYVIS_FEATURE_AVAILABLE = False
# --------------------------
    
try:
    from cefrpy import CEFRAnalyzer
    CEFR_ANALYZER = CEFRAnalyzer()
    CEFR_FEATURE_AVAILABLE = True
except ImportError:
    CEFR_FEATURE_AVAILABLE = False
    
try:
    import eng_to_ipa as ipa
    IPA_FEATURE_AVAILABLE = True
except ImportError:
    IPA_FEATURE_AVAILABLE = False


import streamlit.components.v1 as components 
import xml.etree.ElementTree as ET # Import for XML parsing


# We explicitly exclude external LLM libraries for the free, stable version.
# The interpret_results_llm function is replaced with a placeholder.

st.set_page_config(page_title="CORTEX -- Corpus Explorer Version Alpha (10-Dec-25) by PRIHANTORO (www.prihantoro.com; prihantoro@live.undip.ac.id)", layout="wide") 

# --- CONSTANTS ---
KWIC_MAX_DISPLAY_LINES = 100
KWIC_INITIAL_DISPLAY_HEIGHT = 10 
KWIC_COLLOC_DISPLAY_LIMIT = 20 # Limit for KWIC examples below collocation tables

# Define global names for parallel mode
SOURCE_LANG_CODE = 'EN' # Default source language code
TARGET_LANG_CODE = 'ID' # Default target language code
DEFAULT_LANG_CODE = 'RAW'

# ---------------------------
# Initializing Session State
# ---------------------------
if 'view' not in st.session_state:
    st.session_state['view'] = 'overview'
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
# --- Parallel Corpus State ---
if 'parallel_mode' not in st.session_state:
    st.session_state['parallel_mode'] = False
if 'df_target_lang' not in st.session_state:
    st.session_state['df_target_lang'] = pd.DataFrame()
if 'target_sent_map' not in st.session_state:
    st.session_state['target_sent_map'] = {}
# --- Monolingual XML state ---
if 'monolingual_xml_file_upload' not in st.session_state:
    st.session_state['monolingual_xml_file_upload'] = None
# --- XML Structure Cache ---
if 'xml_structure_data' not in st.session_state:
     st.session_state['xml_structure_data'] = None
if 'xml_structure_error' not in st.session_state: # NEW: To store XML parsing errors
    st.session_state['xml_structure_error'] = None
# --- Display Settings ---
if 'show_pos_tag' not in st.session_state:
    st.session_state['show_pos_tag'] = False
if 'show_lemma' not in st.session_state:
    st.session_state['show_lemma'] = False
# --- New Language State ---
if 'user_explicit_lang_code' not in st.session_state:
     st.session_state['user_explicit_lang_code'] = 'EN' # Default to English


# ---------------------------
# Built-in Corpus Configuration (UPDATED)
# ---------------------------
BUILT_IN_CORPORA = {
    "Select built-in corpus...": None,
    "BROWN (EN XML Tagged)": "https://raw.githubusercontent.com/prihantoro-corpus/corpus-query-systems/main/BrownCorpus.xml",
    "KOSLAT (ID XML Tagged)": "https://raw.githubusercontent.com/prihantoro-corpus/corpus-query-systems/main/KOSLAT-full.xml",
}

# Define color map constants globally (used for both graph and word cloud)
POS_COLOR_MAP = {
    'N': '#33CC33',  # Noun (Green)
    'V': '#3366FF',  # Verb (Blue)
    'J': '#FF33B5',  # Adjective (Pink)
    'R': '#FFCC00',  # Adverb (Yellow)
    '#': '#AAAAAA',  # Nonsense Tag / Raw (Gray)
    'O': '#AAAAAA'   # Other (Gray)
}

PUNCTUATION = {'.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}', '"', "'", '---', '--', '-', '...', '«', '»', '—'}

# --------------------------------------------------------
# NEW FUNCTIONS: Zipf and Band Calculation (from v17.45)
# --------------------------------------------------------
def pmw_to_zipf(pmw):
    """
    Convert frequency per million (PMW) to Zipf scale.
    Formula: Zipf = log10(PMW) + 3
    """
    if pmw <= 0:
        return np.nan
    return math.log10(pmw) + 3


def zipf_to_band(zipf):
    """
    Assign 1–5 Zipf band based on score:
    Band 1: 7.0–7.9
    Band 2: 6.0–6.9
    Band 3: 5.0–5.9
    Band 4: 4.0–4.9
    Band 5: 1.0–3.9
    """
    if pd.isna(zipf):
        return np.nan
    elif zipf >= 7.0:
        return 1
    elif zipf >= 6.0:
        return 2
    elif zipf >= 5.0:
        return 3
    elif zipf >= 4.0:
        return 4
    else: 
        return 5
# --------------------------------------------------------


# --- Word Cloud Function ---
@st.cache_data
def create_word_cloud(freq_data, is_tagged_mode):
    """Generates a word cloud from frequency data with conditional POS coloring."""
    
    # Check added for robustness against sandbox environment
    if not WORDCLOUD_FEATURE_AVAILABLE:
        return None
        
    # Filter out multi-word units for visualization stability
    single_word_freq_data = freq_data[~freq_data['token'].str.contains(' ')].copy()
    if single_word_freq_data.empty:
        return None # SAFE EXIT 1

    word_freq_dict = single_word_freq_data.set_index('token')['frequency'].to_dict()
    word_to_pos = single_word_freq_data.set_index('token').get('pos', pd.Series('O')).to_dict()
    
    # We must import WordCloud here to use it from within the function
    from wordcloud import WordCloud 
    
    # Simple list of English stopwords; can be expanded.
    stopwords = set(["the", "of", "to", "and", "in", "that", "is", "a", "for", "on", "it", "with", "as", "by", "this", "be", "are", "have", "not", "will", "i", "we", "you"])
    
    wc = WordCloud(
        width=800,
        height=400,
        background_color='black',
        colormap='viridis', 
        stopwords=stopwords,
        min_font_size=10
    )
    
    try:
        wordcloud = wc.generate_from_frequencies(word_freq_dict)
    except ValueError:
        return None # SAFE EXIT 2: If the dictionary is empty after stopwords filtering

    if is_tagged_mode:
        def final_color_func(word, *args, **kwargs):
            pos_tag = word_to_pos.get(word, 'O')
            pos_code = pos_tag[0].upper() if pos_tag and len(pos_tag) > 0 else 'O'
            if pos_code not in POS_COLOR_MAP:
                pos_code = 'O'
            return POS_COLOR_MAP.get(pos_code, POS_COLOR_MAP['O'])

        wordcloud = wordcloud.recolor(color_func=final_color_func)
        
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    plt.tight_layout(pad=0)
    
    return fig

# --- NAVIGATION FUNCTIONS ---
def set_view(view_name):
    st.session_state['view'] = view_name
    st.session_state['llm_interpretation_result'] = None
    
def reset_analysis():
    # Clear all cached functions related to old data.
    st.cache_data.clear()
    
    # Reset view and flags
    st.session_state['view'] = 'overview'
    st.session_state['trigger_analyze'] = False
    st.session_state['n_gram_trigger_analyze'] = True 
    st.session_state['n_gram_results_df'] = pd.DataFrame()
    st.session_state['initial_load_complete'] = False
    st.session_state['llm_interpretation_result'] = None
    # Reset parallel corpus state
    st.session_state['parallel_mode'] = False
    st.session_state['df_target_lang'] = pd.DataFrame()
    st.session_state['target_sent_map'] = {}
    st.session_state['monolingual_xml_file_upload'] = None
    st.session_state['xml_structure_data'] = None # Clear structure data
    st.session_state['xml_structure_error'] = None # Clear structure error
    
    # --- Force a complete script rerun directly inside the callback ---
    st.rerun()
    
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
    """
    Finds all unique tokens/POS pairs sharing the target word's lemma(s).
    FIX: All token/lemma output is converted to lowercase.
    """
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
    
    # FIX 1: Convert token and lemma columns to lowercase before dropping duplicates and sorting
    all_forms_df['token'] = all_forms_df['token'].str.lower()
    all_forms_df['lemma'] = all_forms_df['lemma'].str.lower()
    
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
def generate_kwic(df_corpus, raw_target_input, kwic_left, kwic_right, pattern_collocate_input="", pattern_collocate_pos_input="", pattern_window=0, limit=KWIC_MAX_DISPLAY_LINES, random_sample=False, is_parallel_mode=False, show_pos=False, show_lemma=False):
    """
    Generalized function to generate KWIC lines based on target and optional collocate filter.
    Returns: (list_of_kwic_rows, total_matches, primary_target_mwu, literal_freq, list_of_sent_ids, breakdown_df)
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
    
    # --- NEW: Store the actual token that matched at the first position for breakdown ---
    matching_tokens_at_node_one = []
    
    # Execute Search Loop (find all positions)
    if primary_target_len == 1 and not is_structural_search:
        target_pattern = search_components[0]['pattern']
        for i, token in enumerate(tokens_lower):
            if target_pattern.fullmatch(token):
                all_target_positions.append(i)
                matching_tokens_at_node_one.append(df_corpus['token'].iloc[i]) # Store the original token
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
                matching_tokens_at_node_one.append(df_corpus['token'].iloc[i]) # Store the original token (first word in MWU)
            
    literal_freq = len(all_target_positions)
    if literal_freq == 0:
        return ([], 0, raw_target_input, 0, [], pd.DataFrame()) 
        
    # --- NEW: Generate Frequency Breakdown DataFrame ---
    breakdown_data = Counter(matching_tokens_at_node_one)
    breakdown_list = []
    total_tokens_float = float(total_tokens)
    
    for token, freq in breakdown_data.most_common():
        rel_freq = (freq / total_tokens_float) * 1_000_000
        breakdown_list.append({
            "Token Form": token, 
            "Absolute Frequency": freq, 
            "Relative Frequency (per M)": round(rel_freq, 4)
        })
        
    breakdown_df = pd.DataFrame(breakdown_list)
    
    # --- NEW: Add Zipf Metrics (MODIFIED) ---
    breakdown_df['Zipf Score'] = breakdown_df['Relative Frequency (per M)'].apply(pmw_to_zipf).round(2)
    breakdown_df['Zipf Law Frequency Band'] = breakdown_df['Zipf Score'].apply(zipf_to_band)
    # ----------------------------
    
    # ---------------------------------------------------
        
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
                token_pos = df_corpus["pos"].iloc[j] if "pos" in df_corpus.columns else '##'
                
                word_matches = collocate_word_regex is None or collocate_word_regex.fullmatch(token_lower)
                pos_matches = collocate_pos_regex is None or (collocate_pos_regex.fullmatch(token_pos) if not is_raw_mode else False)
                
                if word_matches and pos_matches:
                    found_collocate = True
                    break
            
            if found_collocate:
                final_positions.append(i)
                
    total_matches = len(final_positions)
    if total_matches == 0:
        return ([], 0, raw_target_input, 0, [], breakdown_df)

    # --- Sample Positions ---
    if random_sample:
        import random
        random.seed(42) # Consistent random sample
        sample_size = min(total_matches, limit)
        display_positions = random.sample(final_positions, sample_size)
    else:
        display_positions = final_positions[:limit]
    
    # --- Format KWIC lines (MODIFIED for T/P/L inline display) ---
    kwic_rows = []
    sent_ids = [] # List to store the sentence ID for each KWIC row
    
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

    
    # Helper to format a single token (inline format)
    def format_token_inline(token, pos, lemma, is_collocate_match=False, is_node_word=False):
        
        # --- 1. Token (main font) ---
        token_html = token
        
        # Yellow background for collocate match
        if is_collocate_match:
            token_html = f"<span style='color: black; background-color: #FFEA00;'>{token}</span>"
        
        # Bold if it's the node word
        if is_node_word:
             token_html = f"<b>{token_html}</b>"
        
        output = [token_html]
        
        # 2. POS Tagging: /TAG
        if show_pos:
            pos_val = pos if pos not in ('##', '###') else ''
            # Apply styling: smaller font, green color
            styled_pos = f"<span style='font-size: 0.8em; color: #33CC33;'>{pos_val}</span>"
            output.append("/" + styled_pos)
        
        # 3. Lemma Tagging: {lemma}
        if show_lemma:
            lemma_val = lemma if lemma not in ('##', '###') else ''
            # Apply styling: smallest font, cyan color
            styled_lemma = f"<span style='font-size: 0.7em; color: #00AAAA;'>{lemma_val}</span>"
            output.append("{" + styled_lemma + "}")
            
        return "".join(output)


    for i in display_positions:
        kwic_start = max(0, i - current_kwic_left)
        kwic_end = min(total_tokens, i + primary_target_len + current_kwic_right)
        
        # Determine the sentence ID of the node word (first token in MWU)
        if is_parallel_mode and 'sent_id' in df_corpus.columns:
            node_sent_id = df_corpus["sent_id"].iloc[i]
            sent_ids.append(node_sent_id)
        else:
            sent_ids.append(None) # Not parallel mode or no ID available
            
        formatted_line = []
        node_orig_tokens = []
        collocate_to_display = ""

        
        # Iterate over the context window
        for k in range(kwic_start, kwic_end):
            token = df_corpus["token"].iloc[k]
            token_lower = df_corpus["_token_low"].iloc[k]
            token_pos = df_corpus["pos"].iloc[k] if "pos" in df_corpus.columns else '##'
            token_lemma = df_corpus["lemma"].iloc[k] if "lemma" in df_corpus.columns else '##'

            is_node_word = (i <= k < i + primary_target_len)
            
            is_collocate_match = False
            if is_pattern_search_active and not is_node_word:
                 word_matches_highlight = collocate_word_regex_highlight is None or collocate_word_regex_highlight.fullmatch(token_lower)
                 pos_matches_highlight = collocate_pos_regex_highlight is None or (collocate_pos_regex_highlight.fullmatch(token_pos) if not is_raw_mode else False)
                 
                 if word_matches_highlight and pos_matches_highlight:
                      is_collocate_match = True
                      if collocate_to_display == "": # Capture the first matching collocate
                          collocate_to_display = token # Use the original token case
            
            if is_node_word:
                # Format node word(s) as inline, explicitly marking them as node words
                node_orig_tokens.append(format_token_inline(token, token_pos, token_lemma, is_collocate_match=False, is_node_word=True)) 
            else:
                # Format context tokens (potentially collocates)
                formatted_line.append(format_token_inline(token, token_pos, token_lemma, is_collocate_match=is_collocate_match, is_node_word=False))


        # FIX 2: Corrected slicing logic
        
        # 1. Determine the number of tokens in the left context
        # This is the index 'i' relative to 'kwic_start'
        left_context_count = i - kwic_start
        
        # 2. Slice the context tokens (formatted_line) correctly:
        left_context = formatted_line[:left_context_count]
        right_context = formatted_line[left_context_count:]

        node_orig = " ".join(node_orig_tokens) 
        
        # Join the token groups by a *single* space
        kwic_rows.append({
            "Left": " ".join(left_context), 
            "Node": node_orig, 
            "Right": " ".join(right_context),
            "Collocate": collocate_to_display # Only filled if pattern search is active
        })
        
    return (kwic_rows, total_matches, raw_target_input, literal_freq, sent_ids, breakdown_df) # Added breakdown_df

# --- N-GRAM LOGIC FUNCTION (FIXED: Added corpus_id argument for better caching) ---
@st.cache_data(show_spinner=False)
def generate_n_grams(df_corpus, n_size, n_gram_filters, is_raw_mode, corpus_id):
    """
    Generates N-grams, applies positional filters (token, POS, lemma), and calculates frequencies.
    corpus_id is included in the signature to ensure cache invalidation when the corpus changes.
    """
    total_tokens = len(df_corpus)
    if total_tokens < n_size or n_size < 1:
        return pd.DataFrame()
    
    # Pre-extract lists for faster lookup
    tokens = df_corpus["token"].tolist()
    tokens_low = df_corpus["_token_low"].tolist()
    pos = df_corpus["pos"].tolist() if "pos" in df_corpus.columns else ["##"] * total_tokens
    lemma = df_corpus["lemma"].tolist() if "lemma" in df_corpus.columns else ["##"] * total_tokens

    def matches_filter(token, token_low, pos_tag, lemma_tag, pattern_str, is_raw_mode):
        """Checks if a single token/tag set matches a positional pattern string."""
        if not pattern_str:
            return True

        pattern_str = pattern_str.strip()
        
        # 1. Structural/Lemma Query ([lemma*])
        lemma_match_re = re.search(r"\[(.*?)\]", pattern_str)
        if lemma_match_re and not is_raw_mode:
            lemma_pattern = re.escape(lemma_match_re.group(1).lower()).replace(r'\*', '.*')
            return re.fullmatch(f"^{lemma_pattern}$", lemma_tag.lower())

        # 2. POS Query (_POS*)
        pos_match_re = re.search(r"\_([\w\*|]+)", pattern_str)
        if pos_match_re and not is_raw_mode:
            pos_input = pos_match_re.group(1).strip()
            pos_patterns = [p.strip() for p in pos_input.split('|') if p.strip()]
            full_pos_regex_list = [re.escape(p).replace(r'\*', '.*') for p in pos_patterns]
            full_pos_regex = re.compile("^(" + "|".join(full_pos_regex_list) + ")$")
            return full_pos_regex.fullmatch(pos_tag)

        # 3. Simple Token/Word Query (word*)
        pattern = re.escape(pattern_str).replace(r'\*', '.*')
        return re.fullmatch(f"^{pattern}$", token_low)

        
    matched_n_grams_list = []
    
    for i in range(total_tokens - n_size + 1):
        current_tokens = tokens[i:i + n_size]
        current_tokens_low = tokens_low[i:i + n_size]
        current_pos = pos[i:i + n_size]
        current_lemma = lemma[i:i + n_size]
        
        is_match = True
        
        # Apply positional filters
        for pos_idx, pattern_str in n_gram_filters.items():
            pos_int = int(pos_idx) - 1 # Convert 1-based UI index to 0-based Python index
            if pos_int < 0 or pos_int >= n_size: continue
            
            if not matches_filter(
                current_tokens[pos_int], 
                current_tokens_low[pos_int], 
                current_pos[pos_int], 
                current_lemma[pos_int], 
                pattern_str, 
                is_raw_mode
            ):
                is_match = False
                break
        
        if is_match:
            matched_n_grams_list.append(tuple(current_tokens))
            
    if not matched_n_grams_list:
        return pd.DataFrame()
        
    # Count frequencies
    n_gram_counts = Counter(matched_n_grams_list)
    
    data = []
    total_tokens_float = float(total_tokens) # Use float for accurate calculation
    
    for n_gram, freq in n_gram_counts.items():
        n_gram_str = " ".join(n_gram)
        # Calculate relative frequency per million tokens
        rel_freq = (freq / total_tokens_float) * 1_000_000
        
        data.append({
            "N-Gram": n_gram_str,
            "Frequency": freq,
            "Relative Frequency (per M)": round(rel_freq, 4)
        })
        
    n_gram_df = pd.DataFrame(data)
    
    def is_only_punc_or_digit(n_gram_str):
        for token in n_gram_str.split():
            if token.lower() not in PUNCTUATION and not token.isdigit():
                return False
        return True
        
    n_gram_df = n_gram_df[~n_gram_df["N-Gram"].apply(is_only_punc_or_digit)]
    
    return n_gram_df.sort_values("Frequency", ascending=False).reset_index(drop=True)
# -----------------------------

# --- Statistical Helpers ---
EPS = 1e-12
def safe_log(x):
    return math.log(max(x, EPS))
def compute_ll(k11, k12, k21, k22):
    """Computes the Log-Likelihood (LL) statistic."""
    total = k11 + k12 + k21 + k22
    if total == 0: return 0.0
    e11 = (k11 + k12) * (k11 + k21) / total
    e12 = (k11 + k12) * (k12 + k22) / total
    e21 = (k21 + k22) * (k11 + k21) / total
    e22 = (k21 + k22) * (k12 + k22) / total
    s = 0.0
    for k,e in ((k11,e11),(k12,e12),(k21,e21),(k22,e22)):
        if k > 0 and e > 0: s += k * math.log(k / e)
    return 2.0 * s
def compute_mi(k11, target_freq, coll_total, corpus_size):
    """Compuutes the Mutual Information (MI) statistic."""
    expected = (target_freq * coll_total) / corpus_size
    if expected == 0 or k11 == 0: return 0.0
    return math.log2(k11 / expected)
def significance_from_ll(ll_val):
    """Converts Log-Likelihood value to significance level."""
    if ll_val >= 15.13: return '*** (p<0.001)'
    if ll_val >= 10.83: return '** (p<0.01)'
    if ll_val >= 3.84: return ' * (p<0.05)'
    return 'ns'

# --- IO / Data Helpers ---
def df_to_excel_bytes(df):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    buf.seek(0)
    return buf.getvalue()

@st.cache_data
def create_pyvis_graph(target_word, coll_df):
    if not PYVIS_FEATURE_AVAILABLE: return ""

    net = Network(height="400px", width="100%", bgcolor="#222222", font_color="white", cdn_resources='local')
    if coll_df.empty: return ""
    max_ll = coll_df['LL'].max()
    min_ll = coll_df['LL'].min()
    ll_range = max_ll - min_ll
    
    net.set_options("""
    var options = {
      "nodes": {"borderWidth": 2, "size": 15, "font": {"size": 30}},
      "edges": {"width": 5, "smooth": {"type": "dynamic"}},
      "physics": {"barnesHut": {"gravitationalConstant": -10000, "centralGravity": 0.3, "springLength": 95, "springConstant": 0.04, "damping": 0.9, "avoidOverlap": 0.5}, "minVelocity": 0.75}
    }
    """)
    
    net.add_node(target_word, label=target_word, size=40, color='#FFFF00', title=f"Target: {target_word}", x=0, y=0, fixed=True, font={'color': 'black'})
    
    LEFT_BIAS = -500; RIGHT_BIAS = 500
    all_directions = coll_df['Direction'].unique()
    if 'R' not in all_directions and 'L' in all_directions: RIGHT_BIAS = -500
    elif 'L' not in all_directions and 'R' in all_directions: LEFT_BIAS = 500

    for index, row in coll_df.iterrows():
        collocate = row['Collocate']
        ll_score = row['LL']
        observed = row['Observed']
        pos_tag = row['POS']
        direction = row.get('Direction', 'R') 
        obs_l = row.get('Obs_L', 0)
        obs_r = row.get('Obs_R', 0)
        x_position = LEFT_BIAS if direction in ('L', 'B') else RIGHT_BIAS

        pos_code = pos_tag[0].upper() if pos_tag and len(pos_tag) > 0 else 'O'
        if pos_tag.startswith('##'): pos_code = '#'
        elif pos_code not in ['N', 'V', 'J', 'R']: pos_code = 'O'
        
        color = POS_COLOR_MAP.get(pos_code, POS_COLOR_MAP['O'])
        
        node_size = 25
        if ll_range > 0:
            normalized_ll = (ll_score - min_ll) / ll_range
            node_size = 15 + normalized_ll * 25 
            
        tooltip_title = (
            f"POS: {row['POS']}\n"
            f"Obs: {observed} (Left: {obs_l}, Right: {obs_r})\n"
            f"LL: {ll_score:.2f}\n"
            f"Dominant Direction: {direction}"
        )

        net.add_node(collocate, label=collocate, size=node_size, color=color, title=tooltip_title, x=x_position)
        net.add_edge(target_word, collocate, value=ll_score, width=5, title=f"LL: {ll_score:.2f}")

    html_content = ""; temp_path = None
    try:
        temp_filename = "pyvis_graph.html"
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, temp_filename)
        net.write_html(temp_path, notebook=False)
        with open(temp_path, 'r', encoding='utf-8') as f: html_content = f.read()
    finally:
        if temp_path and os.path.exists(temp_path): os.remove(temp_path)

    return html_content

@st.cache_data
def download_file_to_bytesio(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() 
        return BytesIO(response.content)
    except Exception as e:
        st.error(f"Failed to download built-in corpus from {url}. Ensure the file is public and the URL is a RAW content link.")
        return None

# --- NEW: Robust XML Sanitization Helper ---
def sanitize_xml_content(file_source):
    """
    Reads file content, performs robust cleaning for control characters 
    and unescaped entities, and returns the cleaned string.
    """
    file_source.seek(0)
    
    try:
        xml_content_bytes = file_source.read()
        xml_content = xml_content_bytes.decode('utf-8')
        
        # 1. Remove illegal control characters (keep \t, \n, \r)
        # XML 1.0 valid chars: #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]
        # Python's re.sub below handles the main C0/C1 control blocks
        illegal_chars_re = re.compile(u'[^\u0020-\uD7FF\uE000-\uFFFD\u0009\u000A\u000D]', re.IGNORECASE)
        cleaned_xml_content = illegal_chars_re.sub('', xml_content)
        
        # 2. Fix unescaped ampersands (&) that are not part of an existing entity reference (e.g., &amp;)
        cleaned_xml_content = re.sub(r'&(?![A-Za-z0-9#]{2,5};|#)', r'&amp;', cleaned_xml_content)
        
        # 3. Aggressive: Remove leading/trailing whitespace which sometimes confuses parsers
        cleaned_xml_content = cleaned_xml_content.strip()
        
        return cleaned_xml_content
        
    except Exception as e:
        st.session_state['xml_structure_error'] = f"File Read/Decode Error: {e}"
        return None
# ----------------------------------------------


# ---------------------------------------------------------------------
# XML PARSING HELPERS (Refactored to use sanitize_xml_content)
# ---------------------------------------------------------------------

def extract_xml_structure(file_source, max_values=20):
    """
    Parses an XML file and extracts structure, using ET.fromstring on the cleaned content.
    """
    if file_source is None:
        return None
    
    if isinstance(file_source, list):
        if not file_source: return None
        file_to_analyze = file_source[0]
    else:
        file_to_analyze = file_source

    # Apply aggressive sanitization first
    cleaned_xml_content = sanitize_xml_content(file_to_analyze)
    
    if cleaned_xml_content is None:
        return None # Error already captured in session state if sanitization/read failed

    # --- AGGRESSIVE ERROR CAPTURE ---
    try:
        # Parse from string (after cleaning)
        root = ET.fromstring(cleaned_xml_content) 
        st.session_state['xml_structure_error'] = None # Clear previous error on success
    except Exception as e:
        # Store the exact parsing error string in session state
        st.session_state['xml_structure_error'] = f"XML Parsing Error: {e}"
        return None
    # --- END AGGRESSIVE ERROR CAPTURE ---

    # Structure: {tag_name: {attr_name: set_of_values, ...}, ...}
    structure = {}
    
    def process_element(element):
        tag = element.tag
        if tag not in structure:
            structure[tag] = {}
        
        # Process attributes
        for attr_name, attr_value in element.attrib.items():
            if attr_name not in structure[tag]:
                structure[tag][attr_name] = set()
            
            # Sample unique values up to max_values
            if len(structure[tag][attr_name]) < max_values:
                structure[tag][attr_name].add(attr_value)

        # Recurse through children
        for child in element:
            process_element(child)

    process_element(root)
    
    return structure

# Helper to format structure data into an indented HTML list (NEW)
def format_structure_data_hierarchical(structure_data, indent_level=0, max_values=20):
    """
    Formats the hierarchical XML structure data into an indented HTML list.
    """
    if not structure_data:
        return ""

    html_list = []
    
    # Helper for indentation and basic styling
    def get_indent(level):
        # 1.5em per level for indentation
        return f'<span style="padding-left: {level * 1.5}em;">'

    # Sort tags alphabetically for consistent display
    for tag in sorted(structure_data.keys()):
        tag_data = structure_data[tag]
        
        # Start the tag line
        tag_line = f'{get_indent(indent_level)}<span style="color: #6A5ACD; font-weight: bold;">&lt;{tag}&gt;</span></span><br>'
        html_list.append(tag_line)
        
        # Process attributes for this tag
        for attr in sorted(tag_data.keys()):
            values = sorted(list(tag_data.get(attr, set())))
            
            # Format sampled values
            sampled_values_str = ", ".join(values[:max_values])
            if len(values) > max_values:
                sampled_values_str += f", ... ({len(values) - max_values} more unique)"

            # Attribute line: indented, showing attribute name and sampled values
            attr_line = f'{get_indent(indent_level + 1)}'
            attr_line += f'<span style="color: #8B4513;">@{attr}</span> = '
            attr_line += f'<span style="color: #3CB371;">"{sampled_values_str}"</span></span><br>'
            html_list.append(attr_line)

    return "".join(html_list)


# Core function to parse XML and extract tokens (used by both monolingual and parallel loaders)
def parse_xml_content_to_df(file_source):
    """
    Parses a single XML file, extracts sentences and IDs, and tokenizes/verticalizes if needed.
    Returns: {'lang_code': str, 'df_data': list of dicts, 'sent_map': {sent_id: raw_sentence_text}}
    """
    
    cleaned_xml_content = sanitize_xml_content(file_source)
    
    if cleaned_xml_content is None:
        return None
    
    try:
        # Use fromstring for robustness after cleaning
        root = ET.fromstring(cleaned_xml_content)
        
        # 2. Extract Language Code: Check corpus > text > root attributes
        lang_code = root.get('lang')
        if not lang_code:
            # Look for lang attribute in <text> or <corpus> tag in the raw string
            lang_match = re.search(r'(<text\s+lang="([^"]+)">|<corpus\s+[^>]*lang="([^"]+)">)', cleaned_xml_content)
            if lang_match:
                # Group 2 is from <text>, Group 3 is from <corpus> (prioritize <corpus>)
                lang_code = lang_match.group(3) or lang_match.group(2)
                
        if not lang_code:
            # Default to XML if no language code is explicitly found
            lang_code = 'XML' 
        
        lang_code = lang_code.upper()
            
    except Exception as e:
        # Critical failure: XML is not well-formed even after cleaning
        file_name_label = getattr(file_source, 'name', 'Uploaded XML File')
        st.error(f"Error reading or parsing XML file {file_name_label}: {e}")
        st.session_state['xml_structure_error'] = f"Tokenization Parse Error: {e}" # Ensure tokenization error is also visible
        return None

    df_data = []
    sent_map = {}
    
    # 3. Iterate over <sent> tags (or similar, like <p> if no <sent> is found)
    sent_tags = root.findall('sent')
    if not sent_tags: # Fallback to looking at direct children if <sent> is missing (e.g., if the user uses <p>)
        sent_tags = list(root)
    
    if not sent_tags:
        # Fallback 1: Try to process *all* text content in root
        raw_sentence_text = "".join(root.itertext()).strip() 
        if raw_sentence_text:
            # Tokenize the entire raw text
            cleaned_text = re.sub(r'([^\w\s])', r' \1 ', raw_sentence_text) 
            tokens = [t.strip() for t in cleaned_text.split() if t.strip()]
            if tokens:
               for token in tokens:
                    df_data.append({"token": token, "pos": "##", "lemma": "##", "sent_id": 1})
               sent_map[1] = raw_sentence_text
            return {'lang_code': lang_code, 'df_data': df_data, 'sent_map': sent_map}
            
        file_name_label = getattr(file_source, 'name', 'Uploaded XML File')
        st.warning(f"No parseable content found in corpus file: {file_name_label}.")
        return None

    # --- Use a counter for missing/non-integer IDs for robustness ---
    sequential_id_counter = 0

    for sent_elem in sent_tags:
        
        # --- ID Extraction: Prioritize 'n' > 'id' > Sequential Counter ---
        sent_id_str = sent_elem.get('n') or sent_elem.get('id')
        
        sent_id = None
        
        if sent_id_str:
            try:
                # Try to convert to integer (required for alignment checks)
                sent_id = int(sent_id_str)
            except ValueError:
                # If the ID is a string (e.g., "s1.2") or simply non-numeric, use sequential
                sequential_id_counter += 1
                sent_id = sequential_id_counter
        else:
            # If no 'n' or 'id' attribute found, use sequential ID.
            sequential_id_counter += 1
            sent_id = sequential_id_counter

        if sent_id is None: 
            continue

        # --- Check for nested <w> tags (Vertical/Tagged Format) ---
        word_tags = sent_elem.findall('.//w') # Use findall('.//w') to search recursively
        
        raw_sentence_text = ""
        
        if word_tags:
            # Tagged XML format (e.g., TreeTagger/TEI-like)
            raw_tokens = []
            for w_elem in word_tags:
                token = w_elem.text.strip() if w_elem.text else ""
                pos = w_elem.get('pos') or w_elem.get('type') or "##"
                lemma = w_elem.get('lemma') or "##"
                
                if not token: continue
                
                df_data.append({"token": token, "pos": pos, "lemma": lemma, "sent_id": sent_id})
                raw_tokens.append(token)
            
            raw_sentence_text = " ".join(raw_tokens)
            
        else:
            # Raw Text XML format (Linear) - content is inside the <sent> tag itself
            
            # FIX: Use itertext() to extract all text content robustly, ignoring child tags/attributes
            raw_sentence_text = "".join(sent_elem.itertext()).strip() 
            inner_content = raw_sentence_text
            
            # Check for embedded vertical format (multi-line, multi-column data *inside* the tag)
            normalized_content = inner_content.replace('\r\n', '\n').replace('\r', '\n')
            lines = [line.strip() for line in normalized_content.split('\n') if line.strip()]
            # FIX: Corrected Syntax Error in re.split pattern
            is_vertical_format = sum(line.count('\t') > 0 or len(re.split(r'\s+', line.strip())) >= 3 for line in lines) / len(lines) > 0.5
            
            if is_vertical_format:
                raw_tokens = []
                for line in lines:
                    parts = re.split(r'\s+', line.strip(), 2) 
                    token = parts[0]
                    pos = parts[1] if len(parts) > 1 and parts[1] else "##"
                    lemma = parts[2] if len(parts) > 2 and parts[2] else "##"
                    
                    if not token: continue
                    
                    df_data.append({"token": token, "pos": pos, "lemma": lemma, "sent_id": sent_id})
                    raw_tokens.append(token)
                # Keep raw_sentence_text as is (extracted via itertext()) for the sent_map
            
            else:
                # Pure Horizontal text (raw) - requires tokenization
                
                # Use the clean raw_sentence_text derived from itertext()
                raw_text_to_tokenize = raw_sentence_text.replace('\n', ' ').replace('\t', ' ')
                
                # --- FIXED TOKENIZATION ---
                # 1. Add spaces around punctuation/symbols 
                cleaned_text = re.sub(r'([^\w\s])', r' \1 ', raw_text_to_tokenize) 
                # 2. Split by any whitespace that remains
                tokens = [t.strip() for t in cleaned_text.split() if t.strip()] 
                # --------------------------

                for token in tokens:
                    df_data.append({"token": token, "pos": "##", "lemma": "##", "sent_id": sent_id})
        
        # Store raw sentence for the target map
        if raw_sentence_text:
            sent_map[sent_id] = raw_sentence_text.strip()
            
    if not df_data:
        file_name_label = getattr(file_source, 'name', 'Uploaded XML File')
        st.warning(f"No tokenized data was extracted from the XML file: {file_name_label}.")
        return None
        
    return {'lang_code': lang_code, 'df_data': df_data, 'sent_map': sent_map}


# ---------------------------------------------------------------------
# Monolingual XML Loader 
# ---------------------------------------------------------------------
@st.cache_data
def load_monolingual_corpus_files(file_sources, explicit_lang_code, selected_format):
    global SOURCE_LANG_CODE, TARGET_LANG_CODE
    
    st.session_state['parallel_mode'] = False
    st.session_state['df_target_lang'] = pd.DataFrame()
    st.session_state['target_sent_map'] = {}
    st.session_state['xml_structure_data'] = None # Reset old structure
    st.session_state['xml_structure_error'] = None # Reset old error

    if not file_sources:
        return None
        
    all_df_data = []
    
    # Set the global language code from the user's explicit selection initially (This is fine for initial state before XML parsing)
    SOURCE_LANG_CODE = explicit_lang_code
    TARGET_LANG_CODE = 'NA'
    
    is_tagged_format = 'verticalised' in selected_format or 'TreeTagger' in selected_format
    
    # Track the language detected by the XML parser if explicit code was 'OTHER'
    xml_detected_lang_code = None

    
    for file_source in file_sources:
        file_source.seek(0)
        
        if file_source.name.lower().endswith('.xml'):
            result = parse_xml_content_to_df(file_source)
            if result:
                # If user chose 'OTHER', we cache the XML detected code for the global update later
                if explicit_lang_code == 'OTHER' and result['lang_code'] not in ('XML', 'OTHER'):
                    xml_detected_lang_code = result['lang_code'] 
                    
                all_df_data.extend(result['df_data'])
                st.session_state['monolingual_xml_file_upload'] = file_source 
        
        else: # TXT, CSV, or assumed RAW (non-XML)
            try:
                file_bytes = file_source.read()
                file_content_str = file_bytes.decode('utf-8', errors='ignore')
                clean_lines = [line for line in file_content_str.splitlines() if line and not line.strip().startswith('#')]
                clean_content = "\n".join(clean_lines)
            except Exception as e:
                st.error(f"Error reading raw file content: {e}")
                continue

            # Check if it is assumed to be a vertical T/P/L file
            if is_tagged_format:
                file_buffer_for_pandas = StringIO(clean_content)
                df_attempt = None
                for sep_char in ['\t', r'\s+']: 
                    try:
                        file_buffer_for_pandas.seek(0)
                        df_attempt = pd.read_csv(
                            file_buffer_for_pandas, 
                            sep=sep_char, 
                            header=None, 
                            engine="python", 
                            dtype=str, 
                            skipinitialspace=True,
                            usecols=[0, 1, 2], 
                            names=['token', 'pos', 'lemma']
                        )
                        if df_attempt is not None and df_attempt.shape[1] >= 3:
                            break 
                        df_attempt = None 
                    except Exception:
                        df_attempt = None 
                
                if df_attempt is not None and df_attempt.shape[1] >= 3:
                    df_file = df_attempt.copy()
                    df_file["token"] = df_file["token"].fillna("").astype(str).str.strip() 
                    df_file["pos"] = df_file["pos"].fillna("###").astype(str)
                    df_file["lemma"] = df_file["lemma"].fillna("###").astype(str)
                    df_file['sent_id'] = 0 
                    all_df_data.extend(df_file.to_dict('records'))
                else:
                    st.warning(f"File {file_source.name} was expected to be a vertical format but could not be parsed as 3+ columns. Falling back to raw text.")
                    is_tagged_format = False # Fallback to raw for this file
            
            if not is_tagged_format or selected_format == '.txt': # Raw Text Processing
                raw_text = clean_content
                cleaned_text = re.sub(r'([^\w\s])', r' \1 ', raw_text)
                tokens = [t.strip() for t in cleaned_text.split() if t.strip()] 
                
                df_raw_file = pd.DataFrame({
                    "token": tokens,
                    "pos": ["##"] * len(tokens),
                    "lemma": ["##"] * len(tokens),
                    "sent_id": [0] * len(tokens)
                })
                all_df_data.extend(df_raw_file.to_dict('records'))

    if not all_df_data:
        return None
        
    df_src = pd.DataFrame(all_df_data)
    df_src["_token_low"] = df_src["token"].str.lower()
    
    # If XML detection occurred and user chose 'OTHER', update SOURCE_LANG_CODE
    if xml_detected_lang_code:
        SOURCE_LANG_CODE = xml_detected_lang_code # Update global var inside the cached function if auto-detected
    
    # Structure extraction for the first file (if XML)
    if st.session_state['monolingual_xml_file_upload']:
        st.session_state['xml_structure_data'] = extract_xml_structure(st.session_state['monolingual_xml_file_upload'])
    else:
        st.session_state['xml_structure_data'] = None
    
    return df_src


# ---------------------------------------------------------------------
# Parallel XML Loader
# ---------------------------------------------------------------------
@st.cache_data
def load_xml_parallel_corpus(src_file, tgt_file, src_lang_code, tgt_lang_code):
    global SOURCE_LANG_CODE, TARGET_LANG_CODE

    st.session_state['parallel_mode'] = False
    st.session_state['df_target_lang'] = pd.DataFrame()
    st.session_state['target_sent_map'] = {}
    st.session_state['monolingual_xml_file_upload'] = None # Clear mono XML state
    st.session_state['xml_structure_data'] = None # Reset old structure
    st.session_state['xml_structure_error'] = None # Reset old error
    
    if src_file is None or tgt_file is None: return None

    # Reset file pointers before parsing
    src_file.seek(0)
    tgt_file.seek(0)
    
    src_result = parse_xml_content_to_df(src_file)
    tgt_result = parse_xml_content_to_df(tgt_file)
    
    if src_result is None or tgt_result is None:
        return None
        
    df_src = pd.DataFrame(src_result['df_data'])
    df_tgt = pd.DataFrame(tgt_result['df_data'])

    # 1. Check for Alignment (Sentence IDs)
    src_sent_ids = set(df_src['sent_id'].unique())
    tgt_sent_ids = set(df_tgt['sent_id'].unique())
    
    if src_sent_ids != tgt_sent_ids:
        missing_in_tgt = src_sent_ids - tgt_sent_ids
        missing_in_src = tgt_sent_ids - src_sent_ids
        
        error_msg = f"Alignment Check Failed: Sentence IDs mismatch."
        if missing_in_tgt:
            error_msg += f" Source ({src_result['lang_code']}) is missing sentence IDs: {sorted(list(missing_in_tgt))[:5]}..."
        if missing_in_src:
            error_msg += f" Target ({tgt_result['lang_code']}) is missing sentence IDs: {sorted(list(missing_in_src))[:5]}..."
        
        st.error(error_msg)
        return None
        
    # 2. Finalize Session State
    # For Parallel mode, we use the codes provided in the text inputs
    SOURCE_LANG_CODE = src_lang_code 
    TARGET_LANG_CODE = tgt_lang_code 

    df_src["_token_low"] = df_src["token"].str.lower()
    
    st.session_state['parallel_mode'] = True
    st.session_state['df_target_lang'] = df_tgt
    st.session_state['target_sent_map'] = tgt_result['sent_map'] 
    
    # --- XML Structure Extraction for Overview (Combining structures) ---
    # We rely on the structure extraction to use a fresh read/copy inside.
    src_file.seek(0)
    tgt_file.seek(0)
    src_structure = extract_xml_structure(src_file)
    tgt_file.seek(0)
    tgt_structure = extract_xml_structure(tgt_file)
    
    combined_structure = {}
    if src_structure:
        combined_structure.update(src_structure)
    if tgt_structure:
        # Merge target structure, prioritizing source if tags clash, but merging attributes
        for tag, attrs in tgt_structure.items():
            if tag not in combined_structure:
                combined_structure[tag] = attrs
            else:
                for attr, values in attrs.items():
                    if attr not in combined_structure[tag]:
                        combined_structure[tag][attr] = values
                    else:
                        combined_structure[tag][attr].update(values)
                        # Keep only 20 unique samples
                        combined_structure[tag][attr] = set(list(combined_structure[tag][attr])[:20])

    st.session_state['xml_structure_data'] = combined_structure
    
    return df_src


# ---------------------------------------------------------------------
# EXISTING: Excel Parallel Corpus Loading
# ---------------------------------------------------------------------
@st.cache_data
def load_excel_parallel_corpus_file(file_source, excel_format):
    global SOURCE_LANG_CODE, TARGET_LANG_CODE
    
    st.session_state['parallel_mode'] = False
    st.session_state['df_target_lang'] = pd.DataFrame()
    st.session_state['target_sent_map'] = {}
    st.session_state['monolingual_xml_file_upload'] = None # Clear mono XML state
    st.session_state['xml_structure_data'] = None # Clear structure data
    st.session_state['xml_structure_error'] = None # Clear structure error
    
    if file_source is None: return None
    
    try:
        # Reset file pointer
        file_source.seek(0)
        df_raw = pd.read_excel(file_source, engine='openpyxl')
    except Exception as e:
        st.error(f"Failed to read Excel file: {e}")
        return None

    if df_raw.shape[1] < 2:
        st.error("Excel file must contain at least two columns for source and target language.")
        return None
    
    src_lang = df_raw.columns[0]
    tgt_lang = df_raw.columns[1]
    
    # These set the global codes for Excel file headers
    SOURCE_LANG_CODE = src_lang
    TARGET_LANG_CODE = tgt_lang
    
    data_src = []
    target_sent_map = {}
    sent_id_counter = 0
    
    for index, row in df_raw.iterrows():
        sent_id_counter += 1
        src_text = str(row.iloc[0]).strip()
        tgt_text = str(row.iloc[1]).strip()
        
        # --- FIXED TOKENIZATION ---
        cleaned_text = re.sub(r'([^\w\s])', r' \1 ', src_text)
        src_tokens = [t.strip() for t in cleaned_text.split() if t.strip()]
        # --------------------------
        
        target_sent_map[sent_id_counter] = tgt_text 
        
        for token in src_tokens:
            data_src.append({
                "token": token,
                "pos": "##",
                "lemma": "##",
                "sent_id": sent_id_counter
            })
            
    if not data_src:
        st.error("No valid sentences found in the parallel corpus.")
        return None

    df_src = pd.DataFrame(data_src)
    df_src["_token_low"] = df_src["token"].str.lower()

    st.session_state['parallel_mode'] = True
    st.session_state['target_sent_map'] = target_sent_map
    
    # Handle XML within Excel format (if tagged) - placeholder as it's complex
    if 'with XML' in excel_format:
        st.info("Note: 'Excel with XML' format is currently treated as standard Excel parallel text for tokenization purposes.")
        
    return df_src


# --- Monolingual File Dispatcher (Updated for Built-in) ---
@st.cache_data
def load_corpus_file_built_in(file_source, corpus_name, explicit_lang_code):
    # This is a specific loader for built-in text files (old logic simplified)
    global SOURCE_LANG_CODE, TARGET_LANG_CODE
    
    st.session_state['parallel_mode'] = False
    st.session_state['df_target_lang'] = pd.DataFrame()
    st.session_state['target_sent_map'] = {}
    st.session_state['monolingual_xml_file_upload'] = None
    st.session_state['xml_structure_data'] = None 
    st.session_state['xml_structure_error'] = None # Reset old error
    
    # Set the global language code from the user's explicit selection initially
    SOURCE_LANG_CODE = explicit_lang_code
    TARGET_LANG_CODE = 'NA'
        
    if file_source is None: return None
    
    # Check if the corpus name/URL suggests XML (KOSLAT-ID uses XML)
    is_xml_corpus_name = "xml" in BUILT_IN_CORPORA.get(corpus_name, "").lower() or "xml" in corpus_name.lower()

    if is_xml_corpus_name:
        # --- Handle Built-in XML Corpus ---
        try:
            # We must use a copy of the stream for parsing and structure extraction
            file_source.seek(0)
            file_copy_for_parsing = BytesIO(file_source.read())
            file_copy_for_parsing.seek(0)
            
            xml_result = parse_xml_content_to_df(file_copy_for_parsing) # Use the robust XML parser
            
            if xml_result:
                df = pd.DataFrame(xml_result['df_data'])
                df["_token_low"] = df["token"].str.lower()
                
                # If user explicitly selected 'OTHER', update SOURCE_LANG_CODE with the detected code
                if explicit_lang_code == 'OTHER' and xml_result['lang_code'] not in ('XML', 'OTHER'):
                    SOURCE_LANG_CODE = xml_result['lang_code']
                
                # IMPORTANT: Extract structure from the copy before it's garbage collected
                file_copy_for_parsing.seek(0)
                st.session_state['xml_structure_data'] = extract_xml_structure(file_copy_for_parsing) 
                
                return df
            # If XML parsing fails, fall through to raw text processing as a last resort.
        except Exception as e:
            st.warning(f"Failed to parse built-in XML corpus '{corpus_name}': {e}. Falling back to raw text processing.")
            # Clear file_source to re-read as raw text below
            file_source.seek(0) 

    # --- Prepare content string for non-XML or failed XML built-ins ---
    try:
        file_source.seek(0)
        file_bytes = file_source.read()

        try:
            file_content_str = file_bytes.decode('utf-8')
            file_content_str = re.sub(r'(\s+\n|\n\s+)', '\n', file_content_str)
        except UnicodeDecodeError:
            file_content_str = file_bytes.decode('utf-8', errors='ignore')
        
        clean_lines = [line for line in file_content_str.splitlines() if line and not line.strip().startswith('#')]
        raw_text = "\n".join(clean_lines)
    except Exception as e: 
        st.error(f"Error reading built-in file content: {e}")
        return None

    df = pd.DataFrame()
    
    # --- Built-in T/P/L logic (for Europarl, etc.) ---
    is_vertical_format = ("europarl" in corpus_name.lower()) or ("corpus-query-systems" in BUILT_IN_CORPORA.get(corpus_name, "").lower() and not is_xml_corpus_name)

    if is_vertical_format:
        try:
            file_buffer_for_pandas = StringIO(raw_text)
            
            df = pd.read_csv(
                file_buffer_for_pandas, 
                sep=r'\s+', 
                header=None, 
                engine="python", 
                dtype=str, 
                skipinitialspace=True,
                usecols=[0, 1, 2], 
                names=['token', 'pos', 'lemma']
            )
            
            df["token"] = df["token"].fillna("").astype(str).str.strip() 
            df["pos"] = df["pos"].fillna("###").astype(str)
            df["lemma"] = df["lemma"].fillna("###").astype(str)
            df['sent_id'] = 0 
            
        except Exception as e:
            st.warning(f"Failed to parse built-in corpus '{corpus_name}' as T/P/L: {e}. Falling back to raw tokenization.")
            df = pd.DataFrame() 
    
    if df.empty:
        # Final Raw Text Processing 
        cleaned_text = re.sub(r'([^\w\s])', r' \1 ', raw_text)
        tokens = [t.strip() for t in cleaned_text.split() if t.strip()] 
        df = pd.DataFrame({
            "token": tokens,
            "pos": ["##"] * len(tokens),
            "lemma": ["##"] * len(tokens)
        })
        df['sent_id'] = 0 
            
    df["_token_low"] = df["token"].str.lower()
    return df 

# -----------------------------------------------------
# Function to display KWIC examples for collocates (MODIFIED FOR STYLING)
# -----------------------------------------------------
def display_collocation_kwic_examples(df_corpus, node_word, top_collocates_df, window, limit_per_collocate=1, is_parallel_mode=False, target_sent_map=None, show_pos=False, show_lemma=False):
    """
    Generates and displays KWIC examples for a list of top collocates.
    Displays up to KWIC_COLLOC_DISPLAY_LIMIT total examples.
    """
    if top_collocates_df.empty:
        st.info("No collocates to display examples for.")
        return

    colloc_list = top_collocates_df.head(KWIC_COLLOC_DISPLAY_LIMIT)
    collex_rows_total = []
    
    # Custom KWIC table style (Now includes flexible width for columns)
    collocate_example_table_style = f"""
    <style>
    .collex-table-container-fixed {{
        max-height: 400px; /* Fixed height for scrollable view */
        overflow-y: auto;
        margin-bottom: 1rem;
        width: 100%;
    }}
    .collex-table-inner table {{ 
        width: 100%; 
        table-layout: fixed; /* Fixed layout for proportional columns */
        font-family: monospace; 
        color: white; 
        font-size: 0.9em;
    }}
    .collex-table-inner th {{ font-weight: bold; text-align: center; background-color: #383838; white-space: nowrap; }}
    
    /* Apply explicit proportional widths to Left, Node, Right, and optionally Translation */
    .collex-table-inner td:nth-child(1) {{ width: 8%; text-align: left; font-weight: bold; border-right: 1px solid #444; white-space: nowrap; }} /* Collocate Column */
    .collex-table-inner td:nth-child(2) {{ width: 35%; text-align: right; white-space: normal; vertical-align: top; padding: 5px 10px; }} /* Left Context */
    .collex-table-inner td:nth-child(3) {{ 
          width: 15%; 
          text-align: center; 
          font-weight: bold; 
          background-color: #f0f0f0; 
          color: black; 
          white-space: normal; vertical-align: top; padding: 5px 10px;
    }} /* Node */
    .collex-table-inner td:nth-child(4) {{ width: 35%; text-align: left; white-space: normal; vertical-align: top; padding: 5px 10px; }} /* Right Context */
    .collex-table-inner td:nth-child(5) {{ text-align: left; color: #CCFFCC; width: 7%; font-family: sans-serif; font-size: 0.8em; white-space: normal; }} /* Translation Column (if present, takes remaining width) */

    </style>
    """
    st.markdown(collocate_example_table_style, unsafe_allow_html=True)
    
    
    with st.spinner(f"Generating concordance examples for top {len(colloc_list)} collocates..."):
        for rank, (index, row) in enumerate(colloc_list.iterrows()):
            collocate_word = row['Collocate']
            
            # KWIC returns (kwic_rows, total_matches, raw_target_input, literal_freq, sent_ids, breakdown_df)
            kwic_rows, total_matches, _, _, sent_ids, _ = generate_kwic(
                df_corpus, node_word, window, window, 
                pattern_collocate_input=collocate_word, 
                pattern_collocate_pos_input="", 
                pattern_window=window, # Use collocation window for context
                limit=limit_per_collocate, # Show 1 example max
                is_parallel_mode=is_parallel_mode, # Pass parallel flag
                show_pos=show_pos, # Pass display flags
                show_lemma=show_lemma # Pass display flags
            )
            
            if kwic_rows:
                # Assuming limit=1, we only take the first row
                kwic_row = kwic_rows[0]
                sent_id = sent_ids[0]
                
                translation = ""
                if is_parallel_mode and sent_id is not None and target_sent_map:
                        translation = target_sent_map.get(sent_id, "TRANSLATION N/A")
                
                collex_rows_total.append({
                    "Collocate": f"{rank+1}. {collocate_word}",
                    "Left Context": kwic_row['Left'],
                    "Node": kwic_row['Node'],
                    "Right Context": kwic_row['Right'],
                    "Translation": translation if is_parallel_mode else None # New column
                })
        
    if collex_rows_total:
        collex_df = pd.DataFrame(collex_rows_total)
        # Drop translation column if not in parallel mode
        if not is_parallel_mode:
            collex_df = collex_df.drop(columns=['Translation'])

        # Manually create header for the collocate example table
        header = "<tr><th>Collocate (Rank)</th><th>Left Context</th><th>Node</th><th>Right Context</th>"
        if is_parallel_mode:
            header += f"<th>Translation ({TARGET_LANG_CODE})</th>"
        header += "</tr>"
        
        # Use HTML table and escape=False to preserve the HTML formatting (inline styles)
        html_table = collex_df.to_html(header=False, escape=False, classes=['collex-table-inner'], index=False)
        # Insert the custom header before the table body
        html_table = html_table.replace("<thead></thead>", f"<thead>{header}</thead>", 1)
        
        scrollable_html = f"<div class='collex-table-container-fixed'>{html_table}</div>"
        st.markdown(scrollable_html, unsafe_allow_html=True)
        st.caption(f"Context window is set to **±{window} tokens** (Collocation window). Matching collocate is **bolded and highlighted bright yellow**. POS/Lemma display: **{show_pos}**/**{show_lemma}**.")
    else:
        st.info(f"No specific KWIC examples found for the top {len(colloc_list)} collocates within the ±{window} window.")
# -----------------------------------------------------


# -----------------------------------------------------
# COLLOCATION LOGIC (omitted for brevity)
# -----------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_collocation_results(df_corpus, raw_target_input, coll_window, mi_min_freq, max_collocates, is_raw_mode, collocate_regex="", collocate_pos_regex_input="", selected_pos_tags=None, collocate_lemma=""):
    """
    Generalized function to run collocation analysis.
    Returns: (stats_df_sorted, freq, primary_target_mwu)
    """
    
    total_tokens = len(df_corpus)
    tokens_lower = df_corpus["_token_low"].tolist()
    
    # --- MWU/WILDCARD/STRUCTURAL RESOLUTION (reused from KWIC logic) ---
    search_terms = raw_target_input.split()
    primary_target_len = len(search_terms)
    
    def create_structural_matcher(term):
        lemma_pattern = None; pos_pattern = None
        lemma_match = re.search(r"\[(.*?)\]", term)
        if lemma_match:
            lemma_input = lemma_match.group(1).strip().lower()
            if lemma_input: lemma_pattern = re.compile(f"^{re.escape(lemma_input).replace(r'\*', '.*')}$")
        pos_match = re.search(r"\_([\w\*|]+)", term)
        if pos_match:
            pos_input = pos_match.group(1).strip()
            if pos_input:
                pos_patterns = [p.strip() for p in pos_input.split('|') if p.strip()]
                pos_pattern = re.compile("^(" + "|".join([re.escape(p).replace(r'\*', '.*') for p in pos_patterns]) + ")$")
        if lemma_pattern or pos_pattern: return {'type': 'structural', 'lemma_pattern': lemma_pattern, 'pos_pattern': pos_pattern}
        pattern = re.escape(term.lower()).replace(r'\*', '.*')
        return {'type': 'word', 'pattern': re.compile(f"^{pattern}$")}
        
    search_components = [create_structural_matcher(term) for term in search_terms]
    all_target_positions = []
    
    # Execute Search Loop
    if primary_target_len == 1 and not any('structural' == c['type'] for c in search_components):
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
                    if not component['pattern'].fullmatch(tokens_lower[corpus_index]): match = False; break
                elif component['type'] == 'structural':
                    current_lemma = df_corpus["lemma"].iloc[corpus_index].lower()
                    current_pos = df_corpus["pos"].iloc[corpus_index]
                    lemma_match = component['lemma_pattern'] is None or component['lemma_pattern'].fullmatch(current_lemma)
                    pos_match = component['pos_pattern'] is None or component['pos_pattern'].fullmatch(current_pos)
                    if not (lemma_match and pos_match): match = False; break
            if match: all_target_positions.append(i)
            
    primary_target_positions = all_target_positions 
    freq = len(primary_target_positions)
    primary_target_mwu = raw_target_input

    if freq == 0:
        return (pd.DataFrame(), 0, raw_target_input)

    # --- COLLOCATION COUNTING ---
    collocate_directional_counts = Counter() 
    
    PUNCTUATION_COLLOCATES = PUNCTUATION # Defined globally
    
    for i in primary_target_positions:
        start_index = max(0, i - coll_window)
        end_index = min(total_tokens, i + primary_target_len + coll_window) 
        
        for j in range(start_index, end_index):
            if i <= j < i + primary_target_len: continue
            
            w = tokens_lower[j]
            
            # FIX: Filter out punctuation collocates here
            if w in PUNCTUATION_COLLOCATES or w.isdigit():
                 continue
                
            p = df_corpus["pos"].iloc[j]
            l = df_corpus["lemma"].iloc[j].lower() if "lemma" in df_corpus.columns else "##"
            direction = 'L' if j < i else 'R'
            
            collocate_directional_counts[(w, p, l, direction)] += 1
    
    raw_stats_data = {} 
    token_counts_unfiltered = Counter(tokens_lower) 

    for (w, p, l, direction), observed_dir in collocate_directional_counts.items():
        key_tuple = (w, p, l)
        if key_tuple not in raw_stats_data:
            raw_stats_data[key_tuple] = {'L': 0, 'R': 0, 'Total': 0, 'w': w, 'p': p, 'l': l}
            
        raw_stats_data[key_tuple][direction] += observed_dir
        raw_stats_data[key_tuple]['Total'] += observed_dir

    stats_list = []
    for key_tuple, data in raw_stats_data.items():
        w, p, l = key_tuple
        observed = data['Total']
        dominant_direction = 'R' if data['R'] > data['L'] else ('L' if data['L'] > data['R'] else 'B')
        total_freq = token_counts_unfiltered.get(w, 0)
        
        k11 = observed
        k12 = freq - k11
        k21 = total_freq - k11
        k22 = total_tokens - (k11 + k12 + k21)
        
        ll = compute_ll(k11, k12, k21, k22)
        mi = compute_mi(k11, freq, total_freq, total_tokens)
        
        stats_list.append({
            "Collocate": w, "POS": p, "Lemma": l, "Observed": observed,
            "Total_Freq": total_freq, "LL": round(ll,6), "MI": round(mi,6),
            "Significance": significance_from_ll(ll), "Direction": dominant_direction, 
            "Obs_L": data['L'], "Obs_R": data['R'] 
        })

    stats_df = pd.DataFrame(stats_list)
    
    # --- APPLY FILTERS ---
    filtered_df = stats_df.copy()
    
    if collocate_regex:
        pattern = re.escape(collocate_regex).replace(r'\*', '.*').replace(r'\|', '|').replace(r'\.', '.')
        try:
            filtered_df = filtered_df[filtered_df['Collocate'].str.fullmatch(pattern, case=True, na=False)]
        except re.error:
            filtered_df = pd.DataFrame() 
            
    if collocate_pos_regex_input and not is_raw_mode:
        pos_patterns = [p.strip() for p in collocate_pos_regex_input.split('|') if p.strip()]
        full_pos_regex_list = [re.escape(p).replace(r'\*', '.*') for p in pos_patterns]
        if full_pos_regex_list:
            full_pos_regex = "^(" + "|".join(full_pos_regex_list) + ")$"
            try:
                filtered_df = filtered_df[filtered_df['POS'].str.contains(full_pos_regex, case=True, na=False, regex=True)]
            except re.error:
                filtered_df = pd.DataFrame()
        
    if selected_pos_tags and not is_raw_mode and not collocate_pos_regex_input:
        filtered_df = filtered_df[filtered_df['POS'].isin(selected_pos_tags)]
        
    if collocate_lemma and 'Lemma' in filtered_df.columns and not is_raw_mode: 
        lemma_pattern = re.escape(collocate_lemma).replace(r'\*', '.*').replace(r'\|', '|').replace(r'\.', '.')
        try:
            filtered_df = filtered_df[filtered_df['Lemma'].str.fullmatch(lemma_pattern, case=True, na=False)]
        except re.error:
             filtered_df = pd.DataFrame()
    
    stats_df_filtered = filtered_df
    
    if stats_df_filtered.empty:
        return (pd.DataFrame(), freq, primary_target_mwu)
        
    stats_df_sorted = stats_df_filtered.sort_values("LL", ascending=False)
    
    return (stats_df_sorted, freq, primary_target_mwu)

# ---------------------------
# UI: header
# ---------------------------
st.title("CORTEX - Corpus Texts Explorer ")
st.caption("Upload vertical corpus (**token POS lemma**) or **raw horizontal text**, or **Parallel Corpus (Excel/XML)**.")

# ---------------------------
# Panel: upload and corpus info
# ---------------------------
corpus_source = None
corpus_name = "Uploaded File"
df_source_lang_for_analysis = None
parallel_uploaded = False

# --- SIDEBAR: CORPUS SELECTION, NAVIGATION, & MODULE SETTINGS ---
with st.sidebar:
    
    # 1. CORPUS SELECTION (TOP)
    st.header("1. Corpus Source")
    
    # --- A. BUILT-IN SELECTION ---
    st.markdown("##### 📦 Built-in Corpus")
    selected_corpus_name = st.selectbox(
        "Select a pre-loaded corpus:", 
        options=list(BUILT_IN_CORPORA.keys()),
        key="corpus_select", 
        on_change=reset_analysis
    )
    
    st.markdown("---")
    
    # --- C. GLOBAL LANGUAGE SELECTION (NEW) ---
    st.markdown("##### 🌐 Language Setting")
    
    # Set the initial index based on the session state's explicit code
    initial_lang_index = 0
    if st.session_state.get('user_explicit_lang_code') == 'ID':
        initial_lang_index = 1
    elif st.session_state.get('user_explicit_lang_code') == 'OTHER':
        initial_lang_index = 2

    selected_lang_name = st.selectbox(
        "Corpus Language (Explicit Selection):",
        options=["English (EN)", "Indonesian (ID)", "Other (RAW/XML Tag)"],
        key="global_lang_select",
        index=initial_lang_index, 
        on_change=reset_analysis # Reset analysis completely when language changes
    )
    
    # Map selected language to code and store explicitly
    lang_code_map_sidebar = {"English (EN)": "EN", "Indonesian (ID)": "ID", "Other (RAW/XML Tag)": "OTHER"}
    explicit_lang_code = lang_code_map_sidebar.get(selected_lang_name, 'OTHER')
    st.session_state['user_explicit_lang_code'] = explicit_lang_code
    
    st.markdown("---")
    
    # --- B. CUSTOM CORPUS SELECTION MODE ---
    corpus_mode = st.radio(
        "Choose Corpus Type:",
        options=["Monolingual Corpus", "Parallel Corpus"],
        key="corpus_mode_radio",
        on_change=reset_analysis
    )

    # --- B1. MONOLINGUAL CORPUS UPLOAD ---
    if corpus_mode == "Monolingual Corpus":
        st.markdown("##### 📁 Monolingual File(s) Upload")
        
        # Format Selection (Language selection is now explicit)
        selected_format_mono = st.selectbox(
            "1. Choose Format:",
            options=[
                ".txt (Raw Text/Linear)",
                ".xml (Raw Text/Linear)",
                ".txt verticalised (T/P/L columns)",
                ".xml verticalised (XML with <w> tags)",
                ".txt TreeTagger format",
                ".xml TreeTagger format"
            ],
            key="mono_format_select"
        )
        
        # File Uploader (Allow multiple)
        uploaded_files_mono = st.file_uploader(
            "2. Upload Corpus File(s):", 
            type=["txt","xml", "csv"], 
            accept_multiple_files=True,
            key="mono_file_upload",
            on_change=reset_analysis
        )
        
        # Custom Monolingual Loading Logic
        if uploaded_files_mono:
             with st.spinner(f"Processing Monolingual Corpus ({len(uploaded_files_mono)} file(s))..."):
                 df_source_lang_for_analysis = load_monolingual_corpus_files(
                     uploaded_files_mono, 
                     explicit_lang_code, # Use the new explicit selection
                     selected_format_mono
                 )
                 if df_source_lang_for_analysis is not None:
                     corpus_name = f"Monolingual ({SOURCE_LANG_CODE}, {selected_format_mono})"
    
    # --- B2. PARALLEL CORPUS UPLOAD ---
    else: # Parallel Corpus
        st.markdown("##### 🔗 Parallel Corpus Upload")
        
        parallel_file_mode = st.radio(
            "1. Choose File Structure:",
            options=["One corpus file", "Two corpus files (aligned IDs required)"],
            key="parallel_file_mode_radio"
        )
        
        if parallel_file_mode == "One corpus file":
            excel_format = st.radio(
                "2. Choose Format:",
                options=[".xlsx (Col 1: Source, Col 2: Target)", ".xlsx with XML (Aligned Text/Tags)"],
                key="excel_format_radio"
            )
            parallel_excel_file = st.file_uploader(
                "3. Upload Excel File:", 
                type=["xlsx"],
                key="parallel_excel_file_upload",
                on_change=reset_analysis
            )
            
            if parallel_excel_file is not None:
                with st.spinner("Processing Excel Parallel Corpus..."):
                    df_source_lang_for_analysis = load_excel_parallel_corpus_file(parallel_excel_file, excel_format)
                    if df_source_lang_for_analysis is not None:
                        corpus_name = f"Parallel (Excel) ({SOURCE_LANG_CODE}/{TARGET_LANG_CODE})"
                        parallel_uploaded = True

        else: # Two corpus files
            xml_format = st.radio(
                "2. Choose Format:",
                options=[".xml verticalised", ".xml TreeTagger format"],
                key="xml_format_parallel_radio"
            )
            
            src_lang_input = st.text_input("Source Language Code (e.g., EN)", value=st.session_state.get('src_lang_code', 'EN'), key='src_lang_code_input')
            tgt_lang_input = st.text_input("Target Language Code (e.g., ID)", value=st.session_state.get('tgt_lang_code', 'ID'), key='tgt_lang_code_input')
            st.session_state['src_lang_code'] = src_lang_input
            st.session_state['tgt_lang_code'] = tgt_lang_input

            xml_src_file = st.file_uploader(
                f"3. Upload Source Language XML ({src_lang_input})", 
                type=["xml"],
                key="xml_src_file_upload",
                on_change=reset_analysis
            )
            xml_tgt_file = st.file_uploader(
                f"4. Upload Target Language XML ({tgt_lang_input})", 
                type=["xml"],
                key="xml_tgt_file_upload",
                on_change=reset_analysis
            )
            
            if xml_src_file is not None and xml_tgt_file is not None:
                with st.spinner("Processing XML Parallel Corpus..."):
                    df_source_lang_for_analysis = load_xml_parallel_corpus(xml_src_file, xml_tgt_file, src_lang_input, tgt_lang_input)
                    if df_source_lang_for_analysis is not None:
                        corpus_name = f"Parallel (XML) ({SOURCE_LANG_CODE}/{TARGET_LANG_CODE})"
                        parallel_uploaded = True

    # --- C. BUILT-IN FALLBACK (Only executes if no custom file was loaded) ---
    if df_source_lang_for_analysis is None and selected_corpus_name != "Select built-in corpus...":
        corpus_url = BUILT_IN_CORPORA[selected_corpus_name] 
        # Check if we need to download/load the file
        if 'initial_load_complete' not in st.session_state or st.session_state['initial_load_complete'] == False:
            with st.spinner(f"Downloading {selected_corpus_name}..."):
                corpus_source = download_file_to_bytesio(corpus_url)
        else:
             corpus_source = download_file_to_bytesio(corpus_url) 
        corpus_name = selected_corpus_name
        df_source_lang_for_analysis = load_corpus_file_built_in(corpus_source, corpus_name, explicit_lang_code)
    
    # Use the loaded DF for the rest of the sidebar logic
    df_sidebar = df_source_lang_for_analysis
    
    # Determine tagging mode safely for filter visibility
    is_raw_mode_sidebar = True
    if df_sidebar is not None and 'pos' in df_sidebar.columns and len(df_sidebar) > 0:
        # Check if 99% or more of tags are the default "##" or "###"
        count_of_raw_tags = df_sidebar['pos'].str.contains('##|###', na=False).sum()
        is_raw_mode_sidebar = (count_of_raw_tags / len(df_sidebar)) > 0.99
    
    # 2. NAVIGATION
    st.markdown("---")
    st.subheader("2. Navigation (TOOLS)")
    
    is_active_o = st.session_state['view'] == 'overview'
    st.button("📖 Overview", key='nav_overview', on_click=set_view, args=('overview',), use_container_width=True, type="primary" if is_active_o else "secondary")
    
    # Removed Corpus Structure Navigation Button
    
    is_active_d = st.session_state['view'] == 'dictionary' 
    st.button("📘 Dictionary", key='nav_dictionary', on_click=set_view, args=('dictionary',), use_container_width=True, type="primary" if is_active_d else "secondary")
    
    is_active_c = st.session_state['view'] == 'concordance'
    st.button("📚 Concordance", key='nav_concordance', on_click=set_view, args=('concordance',), use_container_width=True, type="primary" if is_active_c else "secondary")
    
    is_active_n = st.session_state['view'] == 'n_gram' # NEW N-GRAM BUTTON
    st.button("🔢 N-Gram", key='nav_n_gram', on_click=set_view, args=('n_gram',), use_container_width=True, type="primary" if is_active_n else "secondary")

    is_active_l = st.session_state['view'] == 'collocation'
    st.button("🔗 Collocation", key='nav_collocation', on_click=set_view, args=('collocation',), use_container_width=True, type="primary" if is_active_l else "secondary")

    # 3. TOOL SETTINGS (Conditional Block)
    if st.session_state['view'] != 'overview':
        st.markdown("---")
        st.subheader("3. Tool Settings")
        
        # --- UNIVERSAL DISPLAY SETTINGS (NEW) ---
        st.markdown("##### KWIC/Context Display")
        
        has_pos_lemma_data = not is_raw_mode_sidebar
        
        if not has_pos_lemma_data:
            st.info("POS/Lemma display requires a tagged corpus.")
            st.session_state['show_pos_tag'] = False
            st.session_state['show_lemma'] = False
        
        show_pos_tag = st.checkbox(
            "Show POS Tag", 
            value=st.session_state.get('show_pos_tag', False), 
            key='show_pos_tag_checkbox', 
            disabled=not has_pos_lemma_data
        )
        st.session_state['show_pos_tag'] = show_pos_tag

        show_lemma = st.checkbox(
            "Show Lemma", 
            value=st.session_state.get('show_lemma', False), 
            key='show_lemma_checkbox',
            disabled=not has_pos_lemma_data
        )
        st.session_state['show_lemma'] = show_lemma
        
        if show_pos_tag or show_lemma:
            st.caption("Context displays in the format: **token/TAG{lemma}**")

        st.markdown("---")
        
        # --- CONCORDANCE SETTINGS ---
        if st.session_state['view'] == 'concordance':
            st.subheader("Concordance Parameters")
            st.write("KWIC Context (Display)")
            kwic_left = st.number_input("Left Context (tokens)", min_value=1, max_value=20, value=st.session_state.get('kwic_left', 7), step=1, help="Number of tokens shown to the left of the node word.", key="concordance_kwic_left")
            kwic_right = st.number_input("Right Context (tokens)", min_value=1, max_value=20, value=st.session_state.get('kwic_right', 7), step=1, help="Number of tokens shown to the right of the node word.", key="concordance_kwic_right")
            st.session_state['kwic_left'] = kwic_left
            st.session_state['kwic_right'] = kwic_right
            
            st.markdown("---")
            st.subheader("Pattern Search Filter")
            
            st.caption("The **Node Word** is set by the primary search input above.")
            
            pattern_search_window = st.number_input(
                "Search Window (tokens, each side)", 
                min_value=1, max_value=10, value=st.session_state.get('pattern_search_window', 5), step=1, 
                key="pattern_search_window_input", 
                on_change=trigger_analysis_callback 
            )
            
            pattern_collocate = st.text_input(
                "Collocate Word/Pattern (* for wildcard)", 
                value=st.session_state.get('pattern_collocate_input', ''),
                key="pattern_collocate_input", 
                on_change=trigger_analysis_callback 
            )
            
            if df_sidebar is not None and 'pos' in df_sidebar.columns and not is_raw_mode_sidebar:
                pattern_collocate_pos_input = st.text_input(
                    "Collocate POS Tag Pattern (Wildcard/Concatenation)", 
                    value=st.session_state.get('pattern_collocate_pos_input', ''),
                    key="pattern_collocate_pos_input",
                    help="E.g., V* (Verbs), *G (Gerunds), NNS|NNP (Plural/Proper Nouns). Filters collocates by POS tag.",
                    on_change=trigger_analysis_callback 
                )
                st.session_state['pattern_collocate_pos'] = pattern_collocate_pos_input
            else:
                st.info("POS filtering for collocates requires a tagged corpus.")
                st.session_state['pattern_collocate_pos'] = ''

            st.session_state['pattern_search_window'] = pattern_search_window
            st.session_state['pattern_collocate'] = pattern_collocate
            
        # --- N-GRAM SETTINGS ---
        elif st.session_state['view'] == 'n_gram':
            st.subheader("N-Gram Parameters")
            
            # N-Gram size slider
            n_gram_size = st.slider(
                "N-Gram Size (N)", 
                min_value=1, max_value=5, 
                value=st.session_state.get('n_gram_size', 2), 
                step=1, 
                key="n_gram_size_slider",
                on_change=trigger_n_gram_analysis_callback,
                help="Select the size of the token sequence (unigram, bigram, trigram, etc.)"
            )
            st.session_state['n_gram_size'] = n_gram_size
            
            st.markdown("---")
            st.subheader("Positional Filters")

            help_text = "Enter a pattern for filtering this position (wildcard `*` supported):\n\n"
            help_text += "1. **Token/Word:** `govern*` or `the` (default regex match).\n"
            if not is_raw_mode_sidebar:
                help_text += "2. **POS Tag:** `_N*` (matches all tags starting with N).\n"
                help_text += "3. **Lemma:** `[have]` (matches 'have', 'has', 'having', etc.)."
            else:
                st.info("⚠️ Tagged corpus required for Lemma/POS filtering.")
            
            # Dynamic Positional Filter Boxes
            current_n_gram_filters = st.session_state.get('n_gram_filters', {})
            
            # Ensure filter list is correctly sized (up to N)
            new_filters = {}
            for i in range(1, n_gram_size + 1):
                default_val = current_n_gram_filters.get(str(i), '')
                
                input_key = f"n_gram_filter_{i}"
                
                filter_input = st.text_input(
                    f"Position {i} Filter", 
                    value=default_val, 
                    key=input_key,
                    on_change=trigger_n_gram_analysis_callback,
                    args=(),
                    help=help_text
                )
                new_filters[str(i)] = filter_input.strip()

            # Update session state filters, keeping only up to N-gram size
            st.session_state['n_gram_filters'] = {k: v for k, v in new_filters.items() if int(k) <= n_gram_size}
            
        # --- COLLOCATION SETTINGS ---
        elif st.session_state['view'] == 'collocation':
            st.subheader("Collocation Parameters")
            
            max_collocates = st.number_input("Max Collocates to Show (Network/Tables)", min_value=5, max_value=100, value=st.session_state.get('max_collocates', 20), step=5, help="Maximum number of collocates displayed.", key="coll_max_collocates")
            coll_window = st.number_input("Collocation window (tokens each side)", min_value=1, max_value=10, value=st.session_state.get('coll_window', 5), step=1, help="Window used for collocation counting (default ±5).", key="coll_window_input")
            mi_min_freq = st.number_input("MI minimum observed freq", min_value=1, max_value=100, value=st.session_state.get('mi_min_freq', 1), step=1, key="coll_mi_min_freq")
            
            st.session_state['max_collocates'] = max_collocates
            st.session_state['coll_window'] = coll_window
            st.session_state['mi_min_freq'] = mi_min_freq

            st.markdown("---")
            st.subheader("Collocate Filters")
            
            collocate_regex = st.text_input("Filter by Word/Regex (* for wildcard)", value=st.session_state.get('collocate_regex_input', ''), key="collocate_regex_input_coll")
            st.session_state['collocate_regex'] = collocate_regex
            
            if df_sidebar is not None and 'pos' in df_sidebar.columns and not is_raw_mode_sidebar:
                collocate_pos_regex_input = st.text_input(
                    "Filter by POS Tag Pattern (Wildcard/Concatenation)", 
                    value=st.session_state.get('collocate_pos_regex_input_coll', ''), 
                    key="collocate_pos_regex_input_coll_tag",
                    help="E.g., V* (Verbs), NNS|NNP (Plural/Proper Nouns)."
                )
                st.session_state['collocate_pos_regex'] = collocate_pos_regex_input
                
                all_pos_tags = sorted([tag for tag in df_sidebar['pos'].unique() if tag != '##' and tag != '###'])
                if all_pos_tags:
                    selected_pos_tags = st.multiselect(
                        "OR Filter by specific POS Tag(s)",
                        options=all_pos_tags,
                        default=st.session_state.get('selected_pos_tags_input', None),
                        key="selected_pos_tags_input",
                        help="Only shows collocates matching one of the selected POS tags. Ignored if Pattern is also set."
                    )
                    st.session_state['selected_pos_tags'] = selected_pos_tags
                else:
                    st.info("POS filtering requires a tagged corpus.")
                    st.session_state['collocate_pos_regex'] = ''
                    st.session_state['selected_pos_tags'] = None

            if df_sidebar is not None and 'lemma' in df_sidebar.columns and not is_raw_mode_sidebar:
                collocate_lemma_input = st.text_input("Filter by Lemma (case-insensitive, * for wildcard)", value=st.session_state.get('collocate_lemma_input', ''), key="collocate_lemma_input_coll")
                st.session_state['collocate_lemma'] = collocate_lemma_input
            else:
                st.info("Lemma filtering requires a lemmatized corpus.")
                st.session_state['collocate_lemma'] = ''
            
        # --- DICTIONARY SETTINGS (Placeholder) ---
        elif st.session_state['view'] == 'dictionary':
            st.info("Dictionary module currently uses global Collocation Window/Filter settings for collocation analysis, accessible in the Collocation view.")

    # --- 4. FEATURE STATUS CHECK (NEW) ---
    st.markdown("---")
    st.subheader("4. Feature Status")

    # CEFR Status Check
    cefr_status = f"**CEFR Categorization:** {'✅ ACTIVE' if CEFR_FEATURE_AVAILABLE else '❌ DISABLED (Install `cefrpy`)'}"
    if explicit_lang_code != 'EN' and CEFR_FEATURE_AVAILABLE:
        cefr_status += f" (Disabled for **{explicit_lang_code}** corpus)"
    st.markdown(cefr_status)

    # IPA Status Check
    ipa_status = f"**IPA Transcription:** {'✅ ACTIVE' if IPA_FEATURE_AVAILABLE else '❌ DISABLED (Install `eng-to-ipa`)'}"
    if explicit_lang_code != 'EN' and IPA_FEATURE_AVAILABLE:
        ipa_status += f" (Disabled for non-English corpus)"
    st.markdown(ipa_status)
# --- END SIDEBAR ---

# --- NEW ADDITION: AUTHOR INFO AND DOCUMENTATION LINK ---
    st.markdown("---")
    st.subheader("App Info")
    st.markdown(
        "This app is written by **Prihantoro** "
        "([prihantoro@live.undip.ac.id](mailto:prihantoro@live.undip.ac.id); [www.prihantoro.com](http://www.prihantoro.com))"
    )
    # Add the clickable book icon and link
    st.markdown(
        """
        <a href="https://docs.google.com/document/d/1rqrj3X_uoKWL_5P2QBlSQMW06R3EoknxqmpIcxTRrKI/edit?usp=sharing" target="_blank" style="text-decoration: none;">
            <button style="background-color: #333333; color: white; border: none; padding: 10px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 4px;">
                <span style="font-size: 1.2em;">📖</span> App Documentation
            </button>
        </a>
        """, 
        unsafe_allow_html=True
    )
# -----------------------------------------------------

# load corpus (cached) for main body access - Use the result from the sidebar
df = df_source_lang_for_analysis

# --- Check for initial load failure and display better message ---
if df is None:
    st.header("👋 Welcome to CORTEX!")
    st.markdown("---")
    st.markdown("## Get Started")
    st.markdown("**Choose a preloaded corpus or upload your own corpus** in the sidebar to begin analysis.")
    st.error(f"❌ **CORPUS LOAD FAILED** or **NO CORPUS SELECTED**. Please check the sidebar selection and ensure files are correctly formatted/aligned.")
    st.stop()
# ---------------------------------------------------------------------

# --- Define Language Suffix for Headers ---
is_parallel_mode_active = st.session_state.get('parallel_mode', False)

# --- Language Code Override for Monolingual Mode (v17.42 FIX) ---
if not is_parallel_mode_active:
     # Force global code to match explicit user selection state for dictionary logic
     # This is CRITICAL to ensure the dictionary module uses the correct language features
     SOURCE_LANG_CODE = st.session_state.get('user_explicit_lang_code', 'EN')
# --------------------------------------------------------------------

lang_display_suffix = f" in **{SOURCE_LANG_CODE}**" if is_parallel_mode_active else ""
lang_input_suffix = f" in **{SOURCE_LANG_CODE}**" if is_parallel_mode_active else ""
is_monolingual_xml_loaded = st.session_state.get('monolingual_xml_file_upload') is not None
if st.session_state.get('xml_structure_data', None) is not None and not is_parallel_mode_active:
     lang_display_suffix = f" in **{SOURCE_LANG_CODE} (XML Monolingual)**"
     lang_input_suffix = f" in **{SOURCE_LANG_CODE} (XML Monolingual)**"
# ---------------------------------------------------------------


# --- CRITICAL STATUS MESSAGE FOR DEBUGGING (SUCCESS PATH) ---
is_raw_mode = True
if 'pos' in df.columns and len(df) > 0:
    # Check if 99% or more of tags are the default "##" or "###"
    count_of_raw_tags = df['pos'].str.contains('##|###', na=False).sum()
    is_raw_mode = (count_of_raw_tags / len(df)) > 0.99
    
if is_parallel_mode_active:
    app_mode = f"Analyzing Parallel Corpus: {corpus_name} (Source: {SOURCE_LANG_CODE})"
    st.info(f"✅ Parallel Corpus loaded successfully. Total tokens ({SOURCE_LANG_CODE}): **{len(df):,}**. Total sentences: **{len(st.session_state['target_sent_map']):,}**.")
elif st.session_state.get('xml_structure_data', None) is not None and not is_parallel_mode_active:
     # This case handles both uploaded XML and built-in XML (like KOSLAT-ID)
    app_mode = f"Analyzing Corpus: {corpus_name} (XML Monolingual - {SOURCE_LANG_CODE})"
    st.info(f"✅ Monolingual XML Corpus **'{corpus_name}'** loaded successfully. Total tokens: **{len(df):,}**.")
else:
    app_mode = f"Analyzing Corpus: {corpus_name} ({'RAW/LINEAR MODE' if is_raw_mode else 'TAGGED MODE'})"
    st.info(f"✅ Corpus **'{corpus_name}'** loaded successfully. Total tokens: **{len(df):,}**.")

st.markdown("---")
    
# --- CORPUS STATS CALCULATION (SHARED) ---
total_tokens = len(df)
tokens_lower = df["_token_low"].tolist()
tokens_lower_filtered = [t for t in tokens_lower if t not in PUNCTUATION and not t.isdigit()]
token_counts = Counter(tokens_lower) 
unique_types = len(set(tokens_lower_filtered))
unique_lemmas = df["lemma"].nunique() if "lemma" in df.columns else "###"

freq_df_filtered = df[~df['_token_low'].isin(PUNCTUATION) & ~df['_token_low'].str.isdigit()].copy()
# Only include POS in the frequency table if it's a tagged corpus
if not is_raw_mode:
    freq_df = freq_df_filtered[freq_df_filtered['token'] != ''].groupby(["token","pos"]).size().reset_index(name="frequency").sort_values("frequency", ascending=False).reset_index(drop=True)
else:
     freq_df = freq_df_filtered[freq_df_filtered['token'] != ''].groupby(["token"]).size().reset_index(name="frequency").sort_values("frequency", ascending=False).reset_index(drop=True)
     
# -------------------------------------------------------------------------------------------------------


st.header(app_mode)

# -----------------------------------------------------
# MODULE: CORPUS OVERVIEW
# -----------------------------------------------------
if st.session_state['view'] == 'overview':
    
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Corpus Summary")
        # STTR calculation omitted for brevity but can be easily added back
        info_data = {
            "Metric": [f"Corpus size ({SOURCE_LANG_CODE} tokens)", "Unique types (w/o punc)", "Lemma count"],
            "Value": [f"{total_tokens:,}", unique_types, unique_lemmas]
        }
        if st.session_state.get('parallel_mode', False):
            info_data["Metric"].append("Aligned Sentences")
            info_data["Value"].append(f"{len(st.session_state['target_sent_map']):,}")

        info_df = pd.DataFrame(info_data)
        st.dataframe(info_df, use_container_width=True, hide_index=True) 

        st.subheader("Word Cloud (Top Words - Stopwords Filtered)")
        
        # --- Word Cloud Display logic ---
        if not freq_df.empty:
            
            if not WORDCLOUD_FEATURE_AVAILABLE:
                st.info("⚠️ **Word Cloud Feature Disabled:** Visualization requires the external `wordcloud` library, which could not be initialized. Please ensure it is installed correctly in your local environment.")
            else:
                wordcloud_fig = create_word_cloud(freq_df, not is_raw_mode)
                
                if wordcloud_fig is not None: 
                    if not is_raw_mode:
                        st.markdown(
                            """
                            **Word Cloud Color Key (POS):** | <span style="color:#33CC33;">**Green**</span> Noun | <span style="color:#3366FF;">**Blue**</span> Verb | <span style="color:#FF33B5;">**Pink**</span> Adjective | <span style="color:#FFCC00;">**Yellow**</span> Adverb |
                            """
                        , unsafe_allow_html=True)
                        
                    st.pyplot(wordcloud_fig)
                else:
                    st.info("Not enough single tokens remaining to generate a word cloud.")

        else:
            st.info("No tokens to generate a word cloud.")
        # ---------------------------------------------------------------------------------

    with col2:
        st.subheader("Top frequency")
        
        # --- MODIFICATION START ---
        if not freq_df.empty:
            freq_head = freq_df.head(10).copy()
            
            # Calculate Relative Frequency (per M) for the Overview table
            total_tokens_float = float(total_tokens)
            freq_head['Relative Frequency (per M)'] = freq_head['frequency'].apply(lambda f: round((f / total_tokens_float) * 1_000_000, 4))
            
            # Reorder columns for display
            display_cols = ["token", "frequency", "Relative Frequency (per M)"]
            if 'pos' in freq_head.columns:
                display_cols.insert(1, 'pos')
            
            freq_head = freq_head[display_cols]
            
            freq_head.insert(0,"No", range(1, len(freq_head)+1))
            freq_head.rename(columns={'token': 'Token', 'frequency': 'Abs. Freq'}, inplace=True)
            
            st.dataframe(freq_head, use_container_width=True, hide_index=True) 
            st.download_button("⬇ Download full frequency list (xlsx)", data=df_to_excel_bytes(freq_df), file_name="full_frequency_list_filtered.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
             st.info("Frequency data not available.")
        # --- MODIFICATION END ---
        
    st.markdown("---")
    
    # --- XML CORPUS STRUCTURE DISPLAY (NEW HIERARCHICAL DISPLAY) ---
    structure_data = st.session_state.get('xml_structure_data')
    structure_error = st.session_state.get('xml_structure_error')

    # Display the section only if an XML file was involved (either monoline XML or parallel mode)
    if is_monolingual_xml_loaded or is_parallel_mode_active or selected_corpus_name == "KOSLAT-ID (XML Tagged)":
        
        # Use a large, always visible expander for structure details
        with st.expander("📊 XML Corpus Structure (Details)", expanded=True):
        
            if structure_error:
                st.error(f"❌ **XML Parsing Failed in Parser Function.** The underlying Python `xml.etree.ElementTree.fromstring()` function raised the following error: \n\n`{structure_error}`")
                st.info("This usually indicates severe malformation, an illegal XML character, or a missing closing tag in the raw corpus file.")
                
            if structure_data:
                st.subheader("Structure and Attributes (Hierarchical View)")
                
                file_label = f"Source ({SOURCE_LANG_CODE})" if is_parallel_mode_active else f"Monolingual ({SOURCE_LANG_CODE})"
                if is_parallel_mode_active:
                     st.caption(f"Showing combined structure from **{SOURCE_LANG_CODE}** and **{TARGET_LANG_CODE}**.")
                else:
                    st.caption(f"Showing structure from **{file_label}**. Attributes are sampled up to 20 unique values.")
                    
                try:
                    # Use the hierarchical function to generate HTML
                    structure_html = format_structure_data_hierarchical(structure_data)

                    st.markdown(
                        f"""
                        <div style="font-family: monospace; font-size: 0.9em; padding: 10px; background-color: #282828; border-radius: 5px;">
                        {structure_html}
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    # AGGRESSIVE DEBUGGING - Show error message if rendering fails (not parsing)
                    st.error(f"❌ **XML Hierarchical Display FAILED (Rendering Error: {e})**. Showing raw data structure below for diagnosis.")
                    
                    # --- DIAGNOSTIC/FALLBACK RAW TEXT DISPLAY ---
                    with st.expander("Show Raw Python Data (for diagnosis)"):
                         st.info("The data below is the Python dictionary successfully produced by the XML parser.")
                         
                         # Show the raw Python dictionary object
                         st.json(structure_data)
                         
                         # Fallback 2: Show the unstyled raw text output if json fails or for comparison
                         def format_structure_data_raw_text(structure_data, max_values=20):
                              lines = []
                              for tag in sorted(structure_data.keys()):
                                  lines.append(f"\n<{tag}>")
                                  for attr in sorted(structure_data[tag].keys()):
                                      values = sorted(list(structure_data[tag][attr]))
                                      sampled_values_str = ", ".join(values[:max_values])
                                      if len(values) > max_values:
                                          sampled_values_str += f", ... ({len(values) - max_values} more unique)"
                                      lines.append(f"    @{attr}: [{sampled_values_str}]")
                              return "\n".join(lines)
                              
                         st.code(format_structure_data_raw_text(structure_data))
                         # --- END DIAGNOSTIC/FALLBACK ---
                    
            elif not structure_error:
                # Only show this if no error occurred AND no data was returned (i.e., parser ran but found nothing)
                st.info("XML structure not found in the loaded corpus. The corpus must be an XML file and well-formed.")
             
    # -----------------------------------------------------

    st.markdown("---")

# -----------------------------------------------------
# MODULE: SEARCH INPUT (SHARED FOR CONCORDANCE/COLLOCATION)
# -----------------------------------------------------

if st.session_state['view'] != 'overview' and st.session_state['view'] != 'dictionary' and st.session_state['view'] != 'n_gram':
    
    # --- SEARCH INPUT (SHARED) ---
    # FIX: Use conditional language suffix
    st.subheader(f"Search Input: {st.session_state['view'].capitalize()}{lang_display_suffix}")
    
    # The input field that controls analysis for Concordance/Collocation
    # FIX: Use conditional language suffix
    typed_target = st.text_input(
        f"Type a primary token/MWU (word* or 'in the') or Structural Query ([lemma*]_POS*){lang_input_suffix}", 
        value=st.session_state.get('typed_target_input', ''), 
        key="typed_target_input",
        on_change=trigger_analysis_callback # Triggers analysis on Enter/change
    )
    
    primary_input = typed_target.strip()
    target_input = primary_input
    
    use_pattern_search = False
    if st.session_state['view'] == 'concordance':
        if primary_input and (st.session_state.get('pattern_collocate_input', '').strip() or st.session_state.get('pattern_collocate_pos_input', '').strip()):
            use_pattern_search = True

    if not target_input and not use_pattern_search and st.session_state['view'] not in ('dictionary', 'n_gram'):
        st.info(f"Type a term or pattern for {st.session_state['view'].capitalize()} analysis.")
    
    # The explicit button for manual initiation
    analyze_btn_explicit = st.button("🔎 Analyze")
    
    analyze_btn = analyze_btn_explicit or st.session_state['trigger_analyze']
    st.session_state['analyze_btn'] = analyze_btn # Store for downstream check
    st.session_state['trigger_analyze'] = False
    
    st.markdown("---")


# -----------------------------------------------------
# MODULE: N-GRAM LOGIC
# -----------------------------------------------------
if st.session_state['view'] == 'n_gram':
    
    # FIX: Use conditional language suffix
    st.subheader(f"🔢 N-Gram Frequency Analysis (N={st.session_state['n_gram_size']}){lang_display_suffix}")
    
    # Check if a rerun was triggered by changing a filter/size, OR if the analysis was reset due to corpus change.
    analyze_n_gram = st.session_state['n_gram_trigger_analyze'] or st.session_state['n_gram_results_df'].empty
    st.session_state['n_gram_trigger_analyze'] = False # Reset the trigger immediately
    
    # Force re-analysis if manual button is pressed
    manual_analyze_btn = st.button("🔎 Re-Analyze N-Grams")
    if manual_analyze_btn:
        analyze_n_gram = True
    
    if analyze_n_gram:
        with st.spinner(f"Generating and filtering {st.session_state['n_gram_size']}-grams..."):
            # FIX: Passed corpus_name as a unique ID to break the cache when corpus changes
            n_gram_df = generate_n_grams(
                df, 
                st.session_state['n_gram_size'],
                st.session_state['n_gram_filters'],
                is_raw_mode,
                corpus_name # <-- Unique ID for cache invalidation
            )
            st.session_state['n_gram_results_df'] = n_gram_df.copy()
            
    n_gram_df = st.session_state['n_gram_results_df']
    
    if n_gram_df.empty:
        st.warning("No N-grams found matching the criteria. Adjust the N-Gram size or clear filters in the sidebar.")
        st.stop()
        
    st.success(f"Found **{len(n_gram_df):,}** unique {st.session_state['n_gram_size']}-grams matching the criteria.")
    
    # --- Display Table ---
    st.markdown("---")
    st.subheader(f"Top N-Grams (Showing {min(100, len(n_gram_df))})")
    
    n_gram_display_df = n_gram_df.head(100).copy()
    
    # Custom CSS for scrollable tables (Max 100 entries)
    scroll_style = f"""
    <style>
    .scrollable-table {{
        max-height: 400px; /* Fixed height for 100 entries max */
        overflow-y: auto;
    }}
    </style>
    """
    st.markdown(scroll_style, unsafe_allow_html=True)
    
    # Use a scrollable container
    html_table = n_gram_display_df.to_html(index=False, classes=['n-gram-table'])
    # FIX 3.1: Change to triple quotes for robustness
    st.markdown(f"""<div class='scrollable-table'>{html_table}</div>""", unsafe_allow_html=True)

    # --- Download Button ---
    st.markdown("---")
    st.subheader("Download Full Results")
    
    download_label = (
        f"⬇ Download Full {st.session_state['n_gram_size']}-Gram List "
        f"({len(n_gram_df):,} entries) (xlsx)"
    )
    
    st.download_button(
        download_label,
        data=df_to_excel_bytes(n_gram_df), 
        file_name=f"{st.session_state['n_gram_size']}-gram_full_list.xlsx", 
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# -----------------------------------------------------
# MODULE: CONCORDANCE LOGIC
# -----------------------------------------------------
if st.session_state['view'] == 'concordance' and st.session_state.get('analyze_btn', False) and st.session_state.get('typed_target_input'):
    
    # Get current parameters
    kwic_left = st.session_state.get('kwic_left', 7)
    kwic_right = st.session_state.get('kwic_right', 7)
    pattern_collocate = st.session_state.get('pattern_collocate_input', '').lower().strip()
    pattern_collocate_pos = st.session_state.get('pattern_collocate_pos_input', '').strip() 
    pattern_window = st.session_state.get('pattern_search_window', 0)
    
    is_pattern_search_active = pattern_collocate or pattern_collocate_pos
    is_parallel_mode = st.session_state.get('parallel_mode', False)
    target_sent_map = st.session_state.get('target_sent_map', {})
    
    # Display settings
    show_pos_tag = st.session_state['show_pos_tag']
    show_lemma = st.session_state['show_lemma']
    
    # Generate KWIC lines using the reusable function
    with st.spinner("Searching corpus and generating concordance..."):
        # KWIC returns (kwic_rows, total_matches, raw_target_input, literal_freq, list_of_sent_ids, breakdown_df)
        kwic_rows, total_matches, raw_target_input, literal_freq, sent_ids, breakdown_df = generate_kwic(
            df, st.session_state['typed_target_input'], kwic_left, kwic_right, 
            pattern_collocate if is_pattern_search_active else "", 
            pattern_collocate_pos if is_pattern_search_active else "", 
            pattern_window if is_pattern_search_active else 0,
            limit=KWIC_MAX_DISPLAY_LINES,
            is_parallel_mode=is_parallel_mode, # Pass parallel flag
            show_pos=show_pos_tag, 
            show_lemma=show_lemma
        )
    
    if literal_freq == 0: # Check literal_freq here, not total_matches, for consistency with breakdown
        st.warning(f"Target '{raw_target_input}' not found in corpus.")
        st.stop()
        
    # Prepare metadata for display
    rel_freq = (literal_freq / total_tokens) * 1_000_100
    
    # --- MODIFICATION: Renamed column and updated value ---
    wildcard_freq_df = pd.DataFrame([{"Query Result": raw_target_input, "Absolute Frequency": literal_freq, "Relative Frequency (per M)": f"{rel_freq:.4f}"}])
    results_df = wildcard_freq_df 

    # --- KWIC Display ---
    st.subheader("📚 Concordance Results")
    
    if is_pattern_search_active:
        st.success(f"Pattern search successful! Found **{total_matches}** filtered instances of '{raw_target_input}' co-occurring with the specified criteria. POS/Lemma Display: **{show_pos_tag}**/**{show_lemma}**.")
    else:
        st.success(f"Found **{literal_freq}** total occurrences of the primary target word matching the criteria. POS/Lemma Display: **{show_pos_tag}**/**{show_lemma}**.")
    
    # --- LLM INTERPRETATION BUTTON/EXPANDER ---
    if st.button("🧠 Interpret Concordance Results (LLM)", key="llm_concordance_btn"):
        kwic_df_for_llm = pd.DataFrame(kwic_rows).head(10).copy().drop(columns=['Collocate'])
        interpret_results_llm(raw_target_input, "Concordance", "KWIC Context Sample (Max 10 lines)", kwic_df_for_llm)

    if st.session_state['llm_interpretation_result']:
        with st.expander("LLM Interpretation (Feature Disabled)", expanded=True):
            st.markdown(st.session_state['llm_interpretation_result'])
        st.markdown("---")
    # ----------------------------------------
    
    # --- NEW POSITION: TARGET FREQUENCY (Full Width) ---
    st.subheader(f"Target Query Summary")
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    st.markdown("---") # Separator

    # --- NEW: Breakdown of Matching Forms (v17.50: Finalized User-Specified Dark Theme Styling) ---
    if not breakdown_df.empty:
        st.subheader(f"Token Breakdown for Query '{raw_target_input}'")
        
        # Display max 100 entries
        breakdown_display_df = breakdown_df.head(100).copy()
        
        # Use a scrollable container and apply specific table styling
        scroll_style_breakdown = f"""
        <style>
        /* Container for scroll */
        .scrollable-breakdown-table {{
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #444444; /* Use header background color for border */
        }}
        /* Style the table and cells */
        .breakdown-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }}
        .breakdown-table th {{
            background-color: #444444; /* User's Header Background */
            color: #FAFAFA;           /* User's Text Color */
            padding: 8px;
            text-align: left;
        }}
        .breakdown-table td {{
            background-color: #1F1F1F; /* *** FIXED ROW BACKGROUND: Matches KWIC/App BG *** */
            color: #FAFAFA;           /* User's Text Color */
            padding: 8px;
            border-bottom: 1px solid #333;
        }}
        /* *** CRITICAL FIX: Forces Text Color and Background on ALL 4 columns, including the new Zipf column (4th) *** */
        .breakdown-table td:nth-child(1), .breakdown-table td:nth-child(2), .breakdown-table td:nth-child(3), .breakdown-table td:nth-child(4) {{
            background-color: #1F1F1F !important; /* Forces dark background to hide Streamlit numeric column style */
            color: #FAFAFA !important; /* Forces visible white text */
        }}
        </style>
        """
        st.markdown(scroll_style_breakdown, unsafe_allow_html=True)

        # Apply the CSS class 'breakdown-table' to the generated HTML
        html_table_breakdown = breakdown_display_df.to_html(index=False, classes=['breakdown-table'])
        st.markdown(f"""<div class='scrollable-breakdown-table'>{html_table_breakdown}</div>""", unsafe_allow_html=True)
        # --- END MODIFIED CSS ---
        
        st.download_button(
            "⬇ Download full token breakdown (xlsx)", 
            data=df_to_excel_bytes(breakdown_df), 
            file_name=f"{raw_target_input.replace(' ', '_')}_token_breakdown.xlsx", 
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.markdown("---")
    # -----------------------------------------

    # --- KWIC Display (Now Full Width) ---
    # NOTE: The KWIC display logic uses total_matches, which accounts for the collocate pattern filter.
    st.subheader(f"Concordance (KWIC) — top {len(kwic_rows)} lines (Scrollable max {KWIC_MAX_DISPLAY_LINES})")
        
    kwic_df = pd.DataFrame(kwic_rows).drop(columns=['Collocate'])
    kwic_preview = kwic_df.copy().reset_index(drop=True)
    kwic_preview.insert(0, "No", range(1, len(kwic_preview)+1))
    
    # --- NEW: Add Translation Column ---
    if is_parallel_mode:
        translations = [st.session_state['target_sent_map'].get(sent_id, "TRANSLATION N/A") for sent_id in sent_ids]
        kwic_preview[f'Translation ({TARGET_LANG_CODE})'] = translations
    
    # --- KWIC Table Style (REVISED FOR EXPLICIT FLEXIBLE COLUMN WIDTHS) ---
    kwic_table_style = f"""
           <style>
           .dataframe-container-scroll {{
               max-height: 400px; /* Fixed vertical height */
               overflow-y: auto;
               margin-bottom: 1rem;
               width: 100%;
           }}
           .dataframe table {{ 
               width: 100%; 
               table-layout: fixed; /* Use fixed layout to enforce proportional width */
               font-family: monospace; 
               color: white;
               font-size: 0.9em;
           }}
           .dataframe th {{ font-weight: bold; text-align: center; white-space: nowrap; }}
           
           /* KWIC Width Fix: Set proportional column widths (ensures full width is used even without POS/Lemma) */
           .dataframe td:nth-child(1) {{ width: 5%; }} /* No column */
           .dataframe td:nth-child(2) {{ width: 40%; text-align: right; }} /* Left context */
           .dataframe td:nth-child(3) {{ 
               width: 15%; /* Node */
               text-align: center; 
               font-weight: bold; 
               background-color: #f0f0f0; 
               color: black; 
           }} 
           .dataframe td:nth-child(4) {{ width: 40%; text-align: left; }} /* Right context */
           
           /* Ensure content can wrap */
           .dataframe td:nth-child(2), .dataframe td:nth-child(3), .dataframe td:nth-child(4) {{ 
               white-space: normal;
               vertical-align: top;
               padding: 5px 10px;
               line-height: 1.5; 
           }}
           
           /* Adjust for Translation column if present (total is 100%) */
           .dataframe th:nth-last-child(1) {{ width: 10%; }} /* Translation Column */
           .dataframe td:nth-last-child(1) {{ text-align: left; color: #CCFFCC; font-family: sans-serif; font-size: 0.8em; white-space: normal; }}

           </style>
    """
    st.markdown(kwic_table_style, unsafe_allow_html=True)
    
    # Use HTML table and escape=False to preserve the HTML formatting (inline styles)
    html_table = kwic_preview.to_html(escape=False, classes=['dataframe'], index=False)
    scrollable_html = f"""<div class='dataframe-container-scroll'>{html_table}</div>"""

    st.markdown(scrollable_html, unsafe_allow_html=True)

    st.caption("Note: Pattern search collocates are **bolded and highlighted bright yellow**.")
    st.download_button("⬇ Download full concordance (xlsx)", data=df_to_excel_bytes(kwic_preview), file_name=f"{raw_target_input.replace(' ', '_')}_full_concordance.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# -----------------------------------------------------
# MODULE: DICTIONARY
# -----------------------------------------------------
if st.session_state['view'] == 'dictionary':
    
    # FIX: Use conditional language suffix
    st.subheader(f"📘 Dictionary Lookup{lang_display_suffix}")
    
    # --- Input and Analysis Trigger (Automatic on Change) ---
    # FIX: Use conditional language suffix
    current_dict_word = st.text_input(
        f"Enter a Token/Word to lookup (e.g., 'sessions'){lang_input_suffix}:", 
        value=st.session_state.get('dict_word_input_main', ''),
        key="dict_word_input_main",
    ).strip()
    
    if not current_dict_word:
        st.info("Enter a word to view its linguistic summary, examples, and collocates. Analysis runs automatically.")
        st.button("🔎 Manual Re-Analyze", key="manual_dict_analyze_disabled", disabled=True)
        st.stop()
    
    st.button("🔎 Manual Re-Analyze", key="manual_dict_analyze")
        
    st.markdown("---")
    
    # --- 1. Consolidated Word Forms by Lemma ---
    if is_raw_mode or 'lemma' not in df.columns:
        st.warning("Lemma and POS analysis is disabled because the corpus is not tagged/lemmatized. The corpus needs to be uploaded as a T/P/L vertical file.")
        forms_list = pd.DataFrame()
        unique_lemma_list = []
    else:
        forms_list, unique_pos_list, unique_lemma_list = get_all_lemma_forms_details(df, current_dict_word)

    # MODIFIED: Changed header to "Word Forms" globally
    st.subheader(f"Word Forms")

    # --------------------------------------------------------
    # IPA/CEFR/Pronunciation Feature Logic (REVISED for KBBI/Cambridge)
    # --------------------------------------------------------
    corpus_lang = SOURCE_LANG_CODE.upper() 
    
    english_langs = ('EN', 'ENG', 'ENGLISH') 
    indonesian_langs = ('ID', 'INDONESIAN')
    
    is_english_corpus = corpus_lang in english_langs
    is_indonesian_corpus = corpus_lang in indonesian_langs
    
    # IPA/CEFR are enabled ONLY for English.
    ipa_active = IPA_FEATURE_AVAILABLE and is_english_corpus
    cefr_active = CEFR_FEATURE_AVAILABLE and is_english_corpus
    
    # Calculate initial frequency to determine if the token exists at all
    word_freq = token_counts.get(current_dict_word.lower(), 0)

    # --- GLOBAL FIX: Force Forms List Generation in Raw Mode ---
    # This section ensures the table appears if the token is found, regardless of POS/Lemma tags.
    if forms_list.empty and word_freq > 0:
        # Manufacturing the form list for display (as requested)
        forms_list = pd.DataFrame([{
            'token': current_dict_word,
            'pos': '##',
            'lemma': '##'
        }])
    # ------------------------------------------------------------------

    # If the word wasn't found at all (freq is 0 and the manufactured list is still empty)
    if forms_list.empty:
        st.warning(f"Token **'{current_dict_word}'** not found in the corpus.")
        st.stop()

    # --- Table Generation (Runs for Tagged or Forced Raw Mode) ---
    forms_list.rename(columns={
        'token': 'Token', 
        'pos': 'POS Tag', 
        'lemma': 'Lemma'
    }, inplace=True)
        
    # --- ADD FREQUENCY COLUMNS (Absolute and Relative) ---
    forms_list.insert(forms_list.shape[1], 'Absolute Frequency', forms_list['Token'].apply(lambda t: token_counts.get(t.lower(), 0)))
    forms_list.insert(forms_list.shape[1], 'Relative Frequency (per M)', forms_list['Absolute Frequency'].apply(lambda f: round((f / total_tokens) * 1_000_000, 4)))
    
    
    # ------------------ ZIPF BAND CALCULATION (NEW) ------------------
    # Calculate Zipf score from Relative Frequency (which is PMW)
    forms_list.insert(forms_list.shape[1], 'Zipf Score', forms_list['Relative Frequency (per M)'].apply(pmw_to_zipf).round(2))
    
    # Assign Zipf Band (1-5)
    forms_list.insert(forms_list.shape[1], 'Zipf Band (1-5)', forms_list['Zipf Score'].apply(zipf_to_band))
    # -----------------------------------------------------------
    
    # ------------------ CEFR Column Insertion (FIXED) ------------------
    if cefr_active: # This runs only if is_english_corpus is True
        
        def safe_get_cefr(token):
            """Safely calls CEFR_ANALYZER and catches exceptions from cefrpy library."""
            # Use 'NA' as the standard placeholder for uncategorized words
            if not CEFR_ANALYZER:
                return 'NA'
            try:
                # The token is already lowercased by the main frequency counter, but applying lower() again for robustness
                level = CEFR_ANALYZER.get_cefr_level(token).upper()
                return level if level != 'NA' else 'NA' # Ensure NA is used for uncategorized
            except Exception:
                # Catch any error (e.g., word not found, internal library error)
                return 'NA'

        forms_list.insert(forms_list.shape[1], 'CEFR', forms_list['Token'].apply(safe_get_cefr))
        
    else: # Ensure column is present with 'NA' placeholder
        forms_list.insert(forms_list.shape[1], 'CEFR', 'NA')
    # -----------------------------------------------------------
    
    # ------------------ IPA Column Insertion ------------------
    if ipa_active: # This runs only if is_english_corpus is True
        try:
            def get_ipa_transcription(token):
                try:
                    import eng_to_ipa as ipa 
                    return ipa.convert(token)
                except Exception:
                    return "NA" 
            forms_list.insert(forms_list.shape[1], 'IPA Transcription', forms_list['Token'].apply(get_ipa_transcription))
            
        except Exception as e:
            st.error(f"Error during IPA transcription: {e}")
            ipa_active = False 
            
    # Ensure column is present with 'NA' placeholder if not active/failed
    if not ipa_active:
        forms_list.insert(forms_list.shape[1], 'IPA Transcription', 'NA')
    # -----------------------------------------------------------
            
    # --- Pronunciation Link Logic (REVISED for KBBI/Cambridge) ---
    if is_indonesian_corpus: # This runs if corpus_lang is ID
        # Use KBBI for Indonesian dictionary
        pronunciation_url = lambda token: f"https://kbbi.kemdikbud.go.id/entri/{token.lower()}"
        pronunciation_label = f"Dictionary ({corpus_lang} - KBBI)"
    elif is_english_corpus:
        # Use Cambridge Dictionary for English
        pronunciation_url = lambda token: f"https://dictionary.cambridge.org/dictionary/english/{token.lower()}"
        pronunciation_label = "Dictionary (EN - Cambridge)"
    else:
        # Generic link for other languages / Fallback
        pronunciation_url = lambda token: f"https://forvo.com/word/{token}/#{corpus_lang.lower()}"
        pronunciation_label = f"Pronunciation/Dictionary ({corpus_lang})"


    # Create a new column with the clickable link HTML
    # The column name is the 'pronunciation_label' determined above (e.g., 'Dictionary (ID - KBBI)')
    forms_list.insert(forms_list.shape[1], pronunciation_label, forms_list['Token'].apply(
        lambda token: f"<a href='{pronunciation_url(token)}' target='_blank'>Link</a>" # Changed to 'Link' for brevity
    ))
    
    # Define table styling for cleaner look with markdown
    html_style = """
    <style>
    .forms-list-table {
        width: 100%;
        border-collapse: collapse;
        font-family: Arial, sans-serif;
        font-size: 0.9em;
    }
    .forms-list-table th {
        background-color: #383838;
        color: white;
        padding: 8px;
        text-align: left;
    }
    .forms-list-table td {
        padding: 8px;
        border-bottom: 1px solid #444444;
    }
    .forms-list-table tr:hover {
        background-color: #333333;
    }
    </style>
    """
    st.markdown(html_style, unsafe_allow_html=True)

    st.markdown(
        forms_list.to_html(index=False, escape=False, classes=['forms-list-table']), 
        unsafe_allow_html=True
    )
    
    # Feature info messages based on current language
    if not IPA_FEATURE_AVAILABLE and is_english_corpus:
        st.info("💡 **Phonetic Transcription (IPA) feature requires the `eng-to-ipa` library to be installed** (`pip install eng-to-ipa`).")
    elif not is_english_corpus and IPA_FEATURE_AVAILABLE:
        st.info(f"💡 Phonetic Transcription (IPA) is disabled for non-English corpus ({SOURCE_LANG_CODE}).")
        
    if not CEFR_FEATURE_AVAILABLE and is_english_corpus:
        st.info("💡 **CEFR Categorization feature requires the `cefrpy` library to be installed** (`pip install cefrpy`).")
        
    st.markdown("---")

    # --- 2. Related Forms (by Regex) ---
    st.subheader("Related Forms (by Regex)")
    
    related_regex_forms = get_related_forms_by_regex(df, current_dict_word)
    
    if related_regex_forms:
        st.markdown(f"**Tokens matching the pattern `*.{current_dict_word}.*` (case insensitive):**")
        st.text_area(
            "Related Forms (by regex)", 
            ", ".join(related_regex_forms), 
            height=100, 
            key=f"regex_forms_output_{current_dict_word}" 
        )
    else:
        st.info(f"No related tokens found matching the regex pattern.")
        
    st.markdown("---")
    
    # --- 3. Random Concordance Examples ---
    # FIX: Use conditional language suffix
    st.subheader(f"Random Examples (Concordance{lang_display_suffix})")
    
    kwic_left = st.session_state.get('kwic_left', 7)
    kwic_right = st.session_state.get('kwic_right', 7)
    is_parallel_mode = st.session_state.get('parallel_mode', False)
    target_sent_map = st.session_state.get('target_sent_map', {})
    
    # Display settings
    show_pos_tag = st.session_state['show_pos_tag']
    show_lemma = st.session_state['show_lemma']

    with st.spinner(f"Fetching random concordance examples for '{current_dict_word}'..."):
        # KWIC returns (kwic_rows, total_matches, raw_target_input, literal_freq, sent_ids, breakdown_df)
        kwic_rows, total_matches, _, _, sent_ids, _ = generate_kwic(
            df, current_dict_word, kwic_left, kwic_right, 
            random_sample=True, limit=KWIC_MAX_DISPLAY_LINES,
            is_parallel_mode=is_parallel_mode,
            show_pos=show_pos_tag,
            show_lemma=show_lemma
        )
    
    if kwic_rows:
        display_limit = min(5, len(kwic_rows))
        st.success(f"Showing {display_limit} random examples from {total_matches:,} total matches. POS/Lemma Display: **{show_pos_tag}**/**{show_lemma}**.")
        
        kwic_df = pd.DataFrame(kwic_rows).drop(columns=['Collocate'])
        kwic_preview = kwic_df.copy().reset_index(drop=True)
        kwic_preview.insert(0, "No", range(1, len(kwic_preview)+1))
        
        # --- NEW: Add Translation Column ---
        if is_parallel_mode:
            translations = [st.session_state['target_sent_map'].get(sent_id, "TRANSLATION N/A") for sent_id in sent_ids]
            kwic_preview[f'Translation ({TARGET_LANG_CODE})'] = translations

        kwic_table_style = f"""
            <style>
            .dictionary-kwic-container {{
                max-height: 250px; /* Fixed vertical height */
                overflow-y: auto;
                margin-bottom: 1rem;
                width: 100%;
            }}
            .dict-kwic-table table {{ 
                width: 100%; 
                table-layout: fixed; /* Fixed layout to enforce proportional width */
                font-family: monospace; 
                color: white;
                font-size: 0.9em;
            }}
            .dict-kwic-table th {{ font-weight: bold; text-align: center; white-space: nowrap; }}
            
            /* KWIC Width Fix: Set proportional column widths */
            .dict-kwic-table td:nth-child(1) {{ width: 5%; }} /* No column */
            .dict-kwic-table td:nth-child(2) {{ width: 40%; text-align: right; }} /* Left context */
            .dict-kwic-table td:nth-child(3) {{ 
                width: 15%; /* Node */
                text-align: center; 
                font-weight: bold; 
                background-color: #f0f0f0; 
                color: black; 
            }} 
            .dict-kwic-table td:nth-child(4) {{ width: 40%; text-align: left; }} /* Right context */
            
            /* Ensure content can wrap */
            .dict-kwic-table td:nth-child(2), .dict-kwic-table td:nth-child(3), .dict-kwic-table td:nth-child(4) {{ 
                white-space: normal;
                vertical-align: top;
                padding: 5px 10px;
                line-height: 1.5;
            }}
            
            /* Adjust for Translation column if present (total is 100%) */
            .dict-kwic-table th:nth-last-child(1) {{ width: 10%; }} /* Translation Column */
            .dict-kwic-table td:nth-last-child(1) {{ text-align: left; color: #CCFFCC; font-family: sans-serif; font-size: 0.8em; white-space: normal; }}

            </style>
        """
        st.markdown(kwic_table_style, unsafe_allow_html=True)
        
        html_table = kwic_preview.to_html(escape=False, classes=['dict-kwic-table'], index=False)
        # FIX 2: Change to triple quotes for robustness
        scrollable_html = f"""<div class='dictionary-kwic-container'>{html_table}</div>"""
        st.markdown(scrollable_html, unsafe_allow_html=True)
    else:
        st.info("No examples found.")
        
    st.markdown("---")

    # --- 4. Collocates and Collocate Examples ---
    # FIX: Use conditional language suffix
    st.subheader(f"Collocation Analysis{lang_display_suffix}")
    
    coll_window = st.session_state.get('coll_window', 5)
    mi_min_freq = st.session_state.get('mi_min_freq', 1)
    max_collocates = st.session_state.get('max_collocates', 20)
    
    collocate_regex = st.session_state.get('collocate_regex_input', '').lower().strip()
    collocate_pos_regex_input = st.session_state.get('collocate_pos_regex_input_coll', '').strip()
    selected_pos_tags = st.session_state.get('selected_pos_tags_input', [])
    collocate_lemma = st.session_state.get('collocate_lemma_input', '').lower().strip()
    
    with st.spinner(f"Running collocation analysis (window ±{coll_window})..."):
        stats_df_sorted, freq, primary_target_mwu = generate_collocation_results(
            df, current_dict_word, coll_window, mi_min_freq, max_collocates, is_raw_mode,
            collocate_regex, collocate_pos_regex_input, selected_pos_tags, collocate_lemma
        )
    
    if stats_df_sorted.empty:
        st.warning("No collocates found matching the criteria.")
        # Only stop if the primary target itself wasn't found (handled above). Continue to show the rest of the dictionary info.
    else:
        top_collocates = stats_df_sorted.head(20)
        
        # 3a. Top Collocates List
        collocate_list = ", ".join(top_collocates['Collocate'].tolist())
        st.markdown(f"**Top {len(top_collocates)} Collocates (LL-ranked):**")
        st.text_area("Collocate List", collocate_list, height=100)
        
        st.markdown("---")
        st.subheader(f"Collocate Examples (Top {len(top_collocates)} LL Collocates)")
        
        # Use the dedicated KWIC display function (which now handles parallel mode)
        display_collocation_kwic_examples(
            df_corpus=df, 
            node_word=current_dict_word, 
            top_collocates_df=top_collocates, 
            window=coll_window,
            limit_per_collocate=1,
            is_parallel_mode=is_parallel_mode,
            target_sent_map=st.session_state['target_sent_map'],
            show_pos=show_pos_tag,
            show_lemma=show_lemma
        )


# -----------------------------------------------------
# MODULE: COLLOCATION LOGIC
# -----------------------------------------------------
if st.session_state['view'] == 'collocation' and st.session_state.get('analyze_btn', False) and st.session_state.get('typed_target_input'):
    
    # Get Collocation Settings
    coll_window = st.session_state.get('coll_window', 5)
    mi_min_freq = st.session_state.get('mi_min_freq', 1)
    max_collocates = st.session_state.get('max_collocates', 20) 
    
    # Get Filter Settings
    collocate_regex = st.session_state.get('collocate_regex_input', '').lower().strip()
    collocate_pos_regex_input = st.session_state.get('collocate_pos_regex_input_coll', '').strip()
    selected_pos_tags = st.session_state.get('selected_pos_tags_input', [])
    collocate_lemma = st.session_state.get('collocate_lemma_input', '').lower().strip()
    
    raw_target_input = st.session_state.get('typed_target_input')
    
    # Display settings
    show_pos_tag = st.session_state['show_pos_tag']
    show_lemma = st.session_state['show_lemma']
    
    with st.spinner("Running collocation analysis..."):
        stats_df_sorted, freq, primary_target_mwu = generate_collocation_results(
            df, raw_target_input, coll_window, mi_min_freq, max_collocates, is_raw_mode,
            collocate_regex, collocate_pos_regex_input, selected_pos_tags, collocate_lemma
        )

    if freq == 0:
        st.warning(f"Target '{raw_target_input}' not found in corpus.")
        st.stop()
        
    primary_rel_freq = (freq / total_tokens) * 1_000_100
    
    # FIX: Use conditional language suffix
    st.subheader(f"🔗 Collocation Analysis Results{lang_display_suffix}")
    st.success(f"Analyzing target '{primary_target_mwu}'. Frequency: **{freq:,}**, Relative Frequency: **{primary_rel_freq:.4f}** per million. POS/Lemma Display: **{show_pos_tag}**/**{show_lemma}**.")

    if stats_df_sorted.empty:
        st.warning("No collocates found after applying filters.")
        st.stop()
        
    # --- LLM INTERPRETATION BUTTON/EXPANDER ---
    if st.button("🧠 Interpret Collocation Results (LLM)", key="llm_collocation_btn"):
        # This LLM feature is still disabled/placeholder
        interpret_results_llm(
            target_word=raw_target_input,
            analysis_type="Collocation",
            data_description="Top Log-Likelihood Collocates",
            data=stats_df_sorted[['Collocate', 'POS', 'Observed', 'LL', 'Direction']].head(10)
        )
            
    if st.session_state['llm_interpretation_result']:
        with st.expander("LLM Interpretation (Feature Disabled)", expanded=True):
            st.markdown(st.session_state['llm_interpretation_result'])
        st.markdown("---")
    
    # --- Graph Data ---
    top_collocates_for_graphs = stats_df_sorted.head(max_collocates)
    left_directional_df = top_collocates_for_graphs[top_collocates_for_graphs['Direction'].isin(['L', 'B'])].copy()
    right_directional_df = top_collocates_for_graphs[top_collocates_for_graphs['Direction'].isin(['R', 'B'])].copy()

    # --- DISPLAY GRAPHS SIDE BY SIDE ---
    st.markdown("---")
    st.subheader("Interactive Collocation Networks (Directional)")
    
    col_left_graph, col_right_graph = st.columns(2)
    
    # Only try to display if pyvis is available
    if PYVIS_FEATURE_AVAILABLE:
        with col_left_graph:
            st.subheader(f"Left Collocates Only (Top {len(left_directional_df)} LL)")
            if not left_directional_df.empty:
                network_html_left = create_pyvis_graph(primary_target_mwu, left_directional_df)
                components.html(network_html_left, height=450)
            else:
                st.info("No Left-dominant collocates found.")

        with col_right_graph:
            st.subheader(f"Right Collocates Only (Top {len(right_directional_df)} LL)")
            if not right_directional_df.empty:
                network_html_right = create_pyvis_graph(primary_target_mwu, right_directional_df)
                components.html(network_html_right, height=450)
            else:
                st.info("No Right-dominant collocates found.")
    else:
        st.warning("⚠️ **Network Graph Disabled:** The `pyvis` library is not available.")
        st.markdown("---")
    
    st.markdown(
        """
        **General Graph Key:** | Central Node (Target): **Yellow** | Collocate Node Color: Noun (N) **Green**, Verb (V) **Blue**, Adjective (J) **Pink**, Adverb (R) **Yellow**. | Bubble Size: Scales with Log-Likelihood (LL).
        """
    )
    st.markdown("---")
    
    # --- Full Tables (Max 100 entries, scrollable) ---
    st.subheader(f"Collocation Tables — Top {min(100, len(stats_df_sorted))} LL/MI")
    
    # Filter to max 100 entries for display
    full_ll = stats_df_sorted.head(100).copy().reset_index(drop=True)
    full_ll.insert(0, "Rank", range(1, len(full_ll)+1))
    
    full_mi_all = stats_df_sorted[stats_df_sorted["Observed"] >= mi_min_freq].sort_values("MI", ascending=False).reset_index(drop=True)
    full_mi = full_mi_all.head(100).copy()
    full_mi.insert(0, "Rank", range(1, len(full_mi)+1))
    
    col_ll_table, col_mi_table = st.columns(2, gap="large")
    
    # --- Custom CSS for scrollable tables (Max 100 entries) ---
    scroll_style = f"""
    <style>
    .scrollable-table {{
        max-height: 400px; /* Fixed height for 100 entries max */
        overflow-y: auto;
    }}
    </style>
    """
    st.markdown(scroll_style, unsafe_allow_html=True)
    
    with col_ll_table:
        st.markdown(f"**Log-Likelihood (LL) (Top {len(full_ll)})**")
        
        # Display table with relevant columns
        ll_display_df = full_ll[['Rank', 'Collocate', 'LL', 'Direction', 'Significance']].copy()
        
        # Use a scrollable container for the main table
        html_table = ll_display_df.to_html(index=False, classes=['collocate-table'])
        st.markdown(f"""<div class='scrollable-table'>{html_table}</div>""", unsafe_allow_html=True)
        
    with col_mi_table:
        st.markdown(f"**Mutual Information (MI) (obs ≥ {mi_min_freq}, Top {len(full_mi)})**")
        
        # Display table with relevant columns
        mi_display_df = full_mi[['Rank', 'Collocate', 'MI', 'Direction', 'Significance']].copy()
        
        # Use a scrollable container for the main table
        html_table = mi_display_df.to_html(index=False, classes=['collocate-table'])
        st.markdown(f"""<div class='scrollable-table'>{html_table}</div>""", unsafe_allow_html=True)

    # ---------- Download Buttons ----------
    st.markdown("---")
    st.subheader("Download Full Results")
    
    st.download_button(
        f"⬇ Download full LL results (xlsx)", 
        data=df_to_excel_bytes(stats_df_sorted), 
        file_name=f"{primary_target_mwu.replace(' ', '_')}_LL_full_filtered.xlsx", 
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.download_button(
        f"⬇ Download full MI results (obs≥{mi_min_freq}) (xlsx)", 
        data=df_to_excel_bytes(full_mi_all), 
        file_name=f"{primary_target_mwu.replace(' ', '_')}_MI_full_obsge{mi_min_freq}_filtered.xlsx", 
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    # -----------------------------------------------------
    # DEDICATED KWIC DISPLAY FOR TOP LL AND MI COLLOCATES
    # -----------------------------------------------------
    is_parallel_mode = st.session_state.get('parallel_mode', False)
    target_sent_map = st.session_state.get('target_sent_map', {})

    st.markdown("---")
    
    # LL-Ranked KWIC Examples
    st.subheader(f"📚 Concordance Examples for Top {KWIC_COLLOC_DISPLAY_LIMIT} LL Collocates (1 example per collocate)")
    display_collocation_kwic_examples(
        df_corpus=df, 
        node_word=primary_target_mwu, 
        top_collocates_df=full_ll, 
        window=coll_window,
        limit_per_collocate=1,
        is_parallel_mode=is_parallel_mode,
        target_sent_map=st.session_state['target_sent_map'],
        show_pos=show_pos_tag,
        show_lemma=show_lemma
    )
    
    st.markdown("---")
    
    # MI-Ranked KWIC Examples
    st.subheader(f"📚 Concordance Examples for Top {KWIC_COLLOC_DISPLAY_LIMIT} MI Collocates (1 example per collocate)")
    display_collocation_kwic_examples(
        df_corpus=df, 
        node_word=primary_target_mwu, 
        top_collocates_df=full_mi, 
        window=coll_window,
        limit_per_collocate=1,
        is_parallel_mode=is_parallel_mode,
        target_sent_map=st.session_state['target_sent_map'],
        show_pos=show_pos_tag,
        show_lemma=show_lemma
    )


st.caption("Tip: This app handles pre-tagged, raw, and now **Excel-based parallel corpora**.")
