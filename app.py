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
    st.session_state['view'] = 'overview'
    st.session_state['trigger_analyze'] = False
    st.session_state['n_gram_trigger_analyze'] = False
    st.session_state['initial_load_complete'] = False
    st.session_state['llm_interpretation_result'] = None
    
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
def generate_kwic(df_corpus, raw_target_input, kwic_left, kwic_right, pattern_collocate_input="", pattern_collocate_pos_input="", pattern_window=0, limit=KWIC_MAX_DISPLAY_LINES, random_sample=False):
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
        
        kwic_rows.append({
            "Left": " ".join(left_context), 
            "Node": node_orig, 
            "Right": " ".join(right_context),
            "Collocate": collocate_to_display # Only filled if pattern search is active
        })
        
    return (kwic_rows, total_matches, raw_target_input, literal_freq)

# --- Word Cloud Function ---
@st.cache_data
def create_word_cloud(freq_data, is_tagged_mode):
    """Generates a word cloud from frequency data with conditional POS coloring."""
    
    # Filter out multi-word units for visualization stability
    single_word_freq_data = freq_data[~freq_data['token'].str.contains(' ')].copy()
    if single_word_freq_data.empty:
        return None

    word_freq_dict = single_word_freq_data.set_index('token')['frequency'].to_dict()
    word_to_pos = single_word_freq_data.set_index('token').get('pos', pd.Series('O')).to_dict()
    
    stopwords = set(["the", "of", "to", "and", "in", "that", "is", "a", "for", "on", "it", "with", "as", "by", "this", "be", "are"])
    
    wc = WordCloud(
        width=800,
        height=400,
        background_color='black',
        colormap='viridis', 
        stopwords=stopwords,
        min_font_size=10
    )
    
    wordcloud = wc.generate_from_frequencies(word_freq_dict)

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
    if ll_val >= 3.84: return '* (p<0.05)'
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

@st.cache_data
def load_corpus_file(file_source, sep=r"\s+"):
    if file_source is None: return None
    try:
        file_source.seek(0)
        file_bytes = file_source.read()

        try:
            file_content_str = file_bytes.decode('utf-8')
            file_content_str = re.sub(r'(\s+\n|\n\s+)', '\n', file_content_str)
        except UnicodeDecodeError:
            try:
                file_content_str = file_bytes.decode('iso-8859-1')
            except Exception:
                file_content_str = file_bytes.decode('utf-8', errors='ignore')
        
        clean_lines = [line for line in file_content_str.splitlines() if line and not line.strip().startswith('#')]
        clean_content = "\n".join(clean_lines)
        file_buffer_for_pandas = StringIO(clean_content)
    except Exception as e: pass

    try:
        file_buffer_for_pandas.seek(0) 
        try:
            df_attempt = pd.read_csv(file_buffer_for_pandas, sep='\t', header=None, engine="python", dtype=str)
        except Exception:
            file_buffer_for_pandas.seek(0)
            df_attempt = pd.read_csv(file_buffer_for_pandas, sep=sep, header=None, engine="python", dtype=str)
            
        if df_attempt is not None and df_attempt.shape[1] >= 3:
            df = df_attempt.iloc[:, :3]
            df.columns = ["token", "pos", "lemma"]
            
            df["token"] = df["token"].fillna("").astype(str).str.strip() 
            df["pos"] = df["pos"].fillna("###").astype(str)
            df["lemma"] = df["lemma"].fillna("###").astype(str)
            df["_token_low"] = df["token"].str.lower()
            return df
            
    except Exception: pass 

    try:
        raw_text = file_content_str
        tokens = re.findall(r'\b\w+\b|[^\w\s]+', raw_text)
        tokens = [t.strip() for t in tokens if t.strip()] 
        nonsense_tag = "##"
        nonsense_lemma = "##"
        
        pos_tags = [nonsense_tag] * len(tokens)
        lemmas = [nonsense_lemma] * len(tokens)
        
        df = pd.DataFrame({
            "token": tokens,
            "pos": pos_tags,
            "lemma": lemmas
        })
        
        df["_token_low"] = df["token"].str.lower()
        return df
        
    except Exception as raw_e: return None 

# -----------------------------------------------------
# N-GRAM LOGIC (UPDATED FOR STRUCTURAL FILTERING & RELATIVE FREQUENCY)
# -----------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_n_grams(df_corpus, n, positional_filters, is_raw_mode, max_display_lines=10000):
    total_tokens = len(df_corpus)
    
    # 1. Prepare match data lists
    if '_token_low' not in df_corpus.columns:
        df_corpus['_token_low'] = df_corpus['token'].str.lower()
        
    match_data_lists = {
        'token': df_corpus['token'].tolist(),
        '_token_low': df_corpus['_token_low'].tolist(),
        'pos': df_corpus['pos'].tolist() if 'pos' in df_corpus.columns else [],
        'lemma': df_corpus['lemma'].str.lower().tolist() if 'lemma' in df_corpus.columns else [],
    }
    
    # 2. Compile Filters
    compiled_filters = {}

    for pos_str, pattern_str in positional_filters.items():
        pattern_str = pattern_str.strip()
        if not pattern_str:
            continue

        # --- Structural Pattern Detection ---
        match_column_key = '_token_low' # Default is case-insensitive token match (regex)
        cleaned_pattern = pattern_str
        
        # Check for Lemma Pattern: [pattern]
        lemma_match = re.fullmatch(r"\[(.*)\]", pattern_str)
        if lemma_match and not is_raw_mode and 'lemma' in df_corpus.columns:
            match_column_key = 'lemma'
            cleaned_pattern = lemma_match.group(1).strip().lower()
        # Check for POS Pattern: _pattern
        elif pattern_str.startswith('_') and not is_raw_mode and 'pos' in df_corpus.columns:
            match_column_key = 'pos'
            cleaned_pattern = pattern_str[1:].strip()
        
        # --- Final Pattern compilation ---
        if cleaned_pattern:
            # Escape regex special characters, then replace * with .* for wildcard
            safe_pattern = re.escape(cleaned_pattern).replace(r'\*', '.*')
            
            # POS matches need to be case sensitive; Token/Lemma are case insensitive
            flags = 0 if match_column_key == 'pos' else re.IGNORECASE
            
            # Store tuple: (compiled_regex, column_key)
            compiled_filters[int(pos_str)] = (re.compile(f"^{safe_pattern}$", flags), match_column_key)

    # 3. Generate and Count N-Grams
    n_gram_counts = Counter()
    
    for i in range(total_tokens - n + 1):
        n_gram_raw = [df_corpus['token'].iloc[i+j] for j in range(n)]
        
        # Positional Filtering check
        is_match = True
        for pos_index, (pattern, column_key) in compiled_filters.items():
            # pos_index is 1-based, list index is 0-based
            if pos_index <= n and pos_index > 0:
                list_index = i + pos_index - 1
                match_data = match_data_lists.get(column_key)
                
                # Check bounds and data availability
                if not match_data or list_index >= len(match_data):
                    is_match = False # Should not happen with well-formed corpus, but safe check
                    break
                    
                item_to_check = match_data[list_index]
                
                if not pattern.fullmatch(item_to_check):
                    is_match = False
                    break
        
        if is_match:
            # Filter out n-grams composed solely of punctuation/numbers if no explicit filters are set
            if not compiled_filters:
                is_only_noise = all(t.strip().lower() in PUNCTUATION or t.strip().isdigit() for t in n_gram_raw)
                if is_only_noise:
                    continue

            # Key for counting is the raw token n-gram (space-joined string)
            n_gram_key = " ".join(n_gram_raw)
            n_gram_counts[n_gram_key] += 1

    # 4. Format Output
    n_gram_list = []
    total_tokens_float = float(total_tokens) # Use float for accurate calculation
    
    for n_gram, freq in n_gram_counts.most_common(max_display_lines):
        # Calculate Relative Frequency per million tokens
        rel_freq = (freq / total_tokens_float) * 1_000_000
        
        n_gram_list.append({
            'N-Gram': n_gram, 
            'Frequency': freq,
            'Relative Frequency (per mil)': round(rel_freq, 4) # Round to 4 decimal places
        })

    df_n_gram = pd.DataFrame(n_gram_list)
    df_n_gram.insert(0, 'Rank', range(1, len(df_n_gram) + 1))
    
    return df_n_gram
# -----------------------------------------------------
# Function to display KWIC examples for collocates
# -----------------------------------------------------
def display_collocation_kwic_examples(df_corpus, node_word, top_collocates_df, window, limit_per_collocate=1):
    """
    Generates and displays KWIC examples for a list of top collocates.
    Displays up to KWIC_COLLOC_DISPLAY_LIMIT total examples.
    """
    if top_collocates_df.empty:
        st.info("No collocates to display examples for.")
        return

    colloc_list = top_collocates_df.head(KWIC_COLLOC_DISPLAY_LIMIT)
    collex_rows_total = []
    
    # Custom KWIC table style for 4 columns (Collocate, Left, Node, Right)
    collocate_example_table_style = f"""
        <style>
        .collex-table-container-fixed {{
            max-height: 400px; /* Fixed height for scrollable view */
            overflow-y: auto;
            margin-bottom: 1rem;
        }}
        .collex-table-inner {{ font-family: monospace; color: white; width: 100%; }}
        .collex-table-inner table {{ width: 100%; table-layout: fixed; word-wrap: break-word; }}
        .collex-table-inner th {{ font-weight: bold; text-align: center; background-color: #383838; }}
        /* Collocate Column */
        .collex-table-inner td:nth-child(1) {{ text-align: left; font-weight: bold; width: 15%; border-right: 1px solid #444; }} 
        /* Left Context Column */
        .collex-table-inner td:nth-child(2) {{ text-align: right; color: white; width: 40%; }}
        /* Node Column */
        .collex-table-inner td:nth-child(3) {{ text-align: center; font-weight: bold; background-color: #f0f0f0; color: black; }} 
        /* Right Context Column */
        .collex-table-inner td:nth-child(4) {{ text-align: left; color: white; width: 35%; }}
        </style>
    """
    st.markdown(collocate_example_table_style, unsafe_allow_html=True)
    
    
    with st.spinner(f"Generating concordance examples for top {len(colloc_list)} collocates..."):
        for rank, (index, row) in enumerate(colloc_list.iterrows()):
            collocate_word = row['Collocate']
            
            # Generate KWIC lines filtered by this specific collocate (show limit_per_collocate max)
            kwic_rows, total_matches, _, _ = generate_kwic(
                df_corpus, node_word, window, window, 
                pattern_collocate_input=collocate_word, 
                pattern_collocate_pos_input="", 
                pattern_window=window, # Use collocation window for context
                limit=limit_per_collocate # Show 1 example max
            )
            
            if kwic_rows:
                # Assuming limit=1, we only take the first row
                kwic_row = kwic_rows[0]
                collex_rows_total.append({
                    "Collocate": f"{rank+1}. {collocate_word}",
                    "Left Context": kwic_row['Left'],
                    "Node": kwic_row['Node'],
                    "Right Context": kwic_row['Right'],
                })
        
    if collex_rows_total:
        collex_df = pd.DataFrame(collex_rows_total)
        # Manually create header for the collocate example table
        header = "<tr><th>Collocate (Rank)</th><th>Left Context</th><th>Node</th><th>Right Context</th></tr>"
        
        html_table = collex_df.to_html(header=False, escape=False, classes=['collex-table-inner'], index=False)
        # Insert the custom header before the table body
        html_table = html_table.replace("<thead></thead>", f"<thead>{header}</thead>", 1)
        
        scrollable_html = f"<div class='collex-table-container-fixed'>{html_table}</div>"
        st.markdown(scrollable_html, unsafe_allow_html=True)
        st.caption(f"Context window is set to **±{window} tokens** (Collocation window). Matching collocate is **bolded and highlighted bright yellow**.")
    else:
        st.info(f"No specific KWIC examples found for the top {len(colloc_list)} collocates within the ±{window} window.")
# -----------------------------------------------------


# -----------------------------------------------------
# COLLOCATION LOGIC (Extracted to reusable function)
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
    
    # FIX: Define PUNCTUATION locally for filtering
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
st.title("CORTEX - Corpus Texts Explorer v17.16")
st.caption("Upload vertical corpus (**token POS lemma**) or **raw horizontal text**.")

# ---------------------------
# Panel: upload and corpus info
# ---------------------------
corpus_source = None
corpus_name = "Uploaded File"

# --- SIDEBAR: CORPUS SELECTION, NAVIGATION, & MODULE SETTINGS ---
with st.sidebar:
    
    # 1. CORPUS SELECTION (TOP)
    st.header("1. Corpus Source")
    
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
        # Display download status if it's the initial run
        if 'initial_load_complete' not in st.session_state or st.session_state['initial_load_complete'] == False:
            with st.spinner(f"Downloading {selected_corpus_name}..."):
                corpus_source = download_file_to_bytesio(corpus_url)
        else:
             corpus_source = download_file_to_bytesio(corpus_url) 
        corpus_name = selected_corpus_name
    
    
    # 2. NAVIGATION
    st.markdown("---")
    st.subheader("2. Navigation (TOOLS)")
    
    is_active_o = st.session_state['view'] == 'overview'
    st.button("📖 Overview", key='nav_overview', on_click=set_view, args=('overview',), use_container_width=True, type="primary" if is_active_o else "secondary")
    
    is_active_d = st.session_state['view'] == 'dictionary' 
    st.button("📘 Dictionary", key='nav_dictionary', on_click=set_view, args=('dictionary',), use_container_width=True, type="primary" if is_active_d else "secondary")
    
    is_active_c = st.session_state['view'] == 'concordance'
    st.button("📚 Concordance", key='nav_concordance', on_click=set_view, args=('concordance',), use_container_width=True, type="primary" if is_active_c else "secondary")
    
    is_active_n = st.session_state['view'] == 'n_gram' # NEW N-GRAM BUTTON
    st.button("🔢 N-Gram", key='nav_n_gram', on_click=set_view, args=('n_gram',), use_container_width=True, type="primary" if is_active_n else "secondary")

    is_active_l = st.session_state['view'] == 'collocation'
    st.button("🔗 Collocation", key='nav_collocation', on_click=set_view, args=('collocation',), use_container_width=True, type="primary" if is_active_l else "secondary")

    # Load corpus inside sidebar to get df for filtering logic (safe execution)
    df_sidebar = load_corpus_file(corpus_source)
    
    # Determine tagging mode safely for filter visibility
    is_raw_mode_sidebar = True
    if df_sidebar is not None and 'pos' in df_sidebar.columns and len(df_sidebar) > 0:
        count_of_raw_tags = df_sidebar['pos'].str.contains('##', na=False).sum()
        is_raw_mode_sidebar = (count_of_raw_tags / len(df_sidebar)) > 0.99

    # 3. TOOL SETTINGS (Conditional Block)
    if st.session_state['view'] != 'overview':
        st.markdown("---")
        st.subheader("3. Tool Settings")
        
        # --- CONCORDANCE SETTINGS ---
        if st.session_state['view'] == 'concordance':
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
                help_text += "⚠️ Tagged corpus required for Lemma/POS filtering."
            
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


# load corpus (cached) for main body access
df = load_corpus_file(corpus_source)

# --- Check for initial load failure and display better message ---
if df is None:
    st.header("👋 Welcome to CORTEX!")
    st.markdown("---")
    st.markdown("## Get Started")
    st.markdown("**Choose a preloaded corpus or upload your own corpus** in the sidebar to begin analysis.")
    st.error(f"❌ **CORPUS LOAD FAILED** or **NO CORPUS SELECTED**. Please check the sidebar selection.")
    st.stop()
# ---------------------------------------------------------------------

# --- CRITICAL STATUS MESSAGE FOR DEBUGGING (SUCCESS PATH) ---
st.info(f"✅ Corpus **'{corpus_name}'** loaded successfully. Total tokens: **{len(df):,}**.")
st.markdown("---")
    
# --- CORPUS STATS CALCULATION (SHARED) ---
if 'pos' in df.columns and len(df) > 0:
    count_of_raw_tags = df['pos'].str.contains('##', na=False).sum()
    is_raw_mode = (count_of_raw_tags / len(df)) > 0.99
else:
    is_raw_mode = True 

total_tokens = len(df)
tokens_lower = df["_token_low"].tolist()
tokens_lower_filtered = [t for t in tokens_lower if t not in PUNCTUATION and not t.isdigit()]
token_counts = Counter(tokens_lower) 
unique_types = len(set(tokens_lower_filtered))
unique_lemmas = df["lemma"].nunique() if "lemma" in df.columns else "###"

freq_df_filtered = df[~df['_token_low'].isin(PUNCTUATION) & ~df['_token_low'].str.isdigit()].copy()
if is_raw_mode:
    freq_df_filtered['pos'] = '##'
# FIX: Filter freq_df for non-empty tokens before grouping
freq_df = freq_df_filtered[freq_df_filtered['token'] != ''].groupby(["token","pos"]).size().reset_index(name="frequency").sort_values("frequency", ascending=False).reset_index(drop=True)


if is_raw_mode:
    app_mode = f"Analyzing Corpus: {corpus_name} (RAW/LINEAR MODE)"
else:
    app_mode = f"Analyzing Corpus: {corpus_name} (TAGGED MODE)"
st.header(app_mode)

# -----------------------------------------------------
# MODULE: CORPUS OVERVIEW
# -----------------------------------------------------
if st.session_state['view'] == 'overview':
    
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Corpus Summary")
        # STTR calculation omitted for brevity but can be easily added back
        info_df = pd.DataFrame({
            "Metric": ["Corpus size (tokens)", "Unique types (w/o punc)", "Lemma count"],
            "Value": [f"{total_tokens:,}", unique_types, unique_lemmas]
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True) 

        st.subheader("Word Cloud (Top Words - Stopwords Filtered)")
        # FIX: Ensure freq_df is not empty before creating word cloud
        if not freq_df.empty:
            wordcloud_fig = create_word_cloud(freq_df, not is_raw_mode)
            
            if not is_raw_mode:
                st.markdown(
                    """
                    **Word Cloud Color Key (POS):** | <span style="color:#33CC33;">**Green**</span> Noun | <span style="color:#3366FF;">**Blue**</span> Verb | <span style="color:#FF33B5;">**Pink**</span> Adjective | <span style="color:#FFCC00;">**Yellow**</span> Adverb |
                    """
                , unsafe_allow_html=True)
                
            st.pyplot(wordcloud_fig)
        else:
            st.info("Not enough tokens to generate a word cloud.")

    with col2:
        st.subheader("Top frequency")
        # FIX: Ensure freq_df is not empty before slicing
        if not freq_df.empty:
            freq_head = freq_df.head(10).copy()
            freq_head.insert(0,"No", range(1, len(freq_head)+1))
            st.dataframe(freq_head, use_container_width=True, hide_index=True) 
            st.download_button("⬇ Download full frequency list (xlsx)", data=df_to_excel_bytes(freq_df), file_name="full_frequency_list_filtered.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
             st.info("Frequency data not available.")

    st.markdown("---")
    
# -----------------------------------------------------
# MODULE: SEARCH INPUT (SHARED FOR CONCORDANCE/COLLOCATION)
# -----------------------------------------------------

if st.session_state['view'] != 'overview' and st.session_state['view'] != 'dictionary' and st.session_state['view'] != 'n_gram':
    
    # --- SEARCH INPUT (SHARED) ---
    st.subheader(f"Search Input: {st.session_state['view'].capitalize()}")
    
    # The input field that controls analysis for Concordance/Collocation
    typed_target = st.text_input(
        "Type a primary token/MWU (word* or 'in the') or Structural Query ([lemma*]_POS*)", 
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
    
    st.subheader(f"🔢 N-Gram Frequency Analysis (N={st.session_state['n_gram_size']})")
    
    # Check if a rerun was triggered by changing a filter/size
    analyze_n_gram = st.session_state['n_gram_trigger_analyze'] or st.session_state['n_gram_results_df'].empty
    st.session_state['n_gram_trigger_analyze'] = False
    
    # Force re-analysis if manual button is pressed
    manual_analyze_btn = st.button("🔎 Re-Analyze N-Grams")
    if manual_analyze_btn:
        analyze_n_gram = True
    
    if analyze_n_gram:
        with st.spinner(f"Generating and filtering {st.session_state['n_gram_size']}-grams..."):
            n_gram_df = generate_n_grams(
                df, 
                st.session_state['n_gram_size'],
                st.session_state['n_gram_filters'],
                is_raw_mode
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
    st.markdown(f"<div class='scrollable-table'>{html_table}</div>", unsafe_allow_html=True)

    # --- Download Button ---
    st.markdown("---")
    st.subheader("Download Full Results")
    
    # Download button tied to the latest analysis result
    
    # FIX: Concatenate the f-string parts to avoid termination error
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
    # Use the session_state from the input keys
    pattern_collocate = st.session_state.get('pattern_collocate_input', '').lower().strip()
    pattern_collocate_pos = st.session_state.get('pattern_collocate_pos_input', '').strip() 
    pattern_window = st.session_state.get('pattern_search_window', 0)
    
    is_pattern_search_active = pattern_collocate or pattern_collocate_pos
    
    # Generate KWIC lines using the reusable function
    with st.spinner("Searching corpus and generating concordance..."):
        kwic_rows, total_matches, raw_target_input, literal_freq = generate_kwic(
            df, st.session_state['typed_target_input'], kwic_left, kwic_right, 
            pattern_collocate if is_pattern_search_active else "", 
            pattern_collocate_pos if is_pattern_search_active else "", 
            pattern_window if is_pattern_search_active else 0,
            limit=KWIC_MAX_DISPLAY_LINES
        )
    
    if total_matches == 0:
        st.warning(f"Target '{raw_target_input}' not found or matched 0 instances after filtering.")
        st.stop()
        
    # Prepare metadata for display
    rel_freq = (literal_freq / total_tokens) * 1_000_000
    wildcard_freq_df = pd.DataFrame([{"Query Result": raw_target_input, "Raw Frequency": literal_freq, "Relative Frequency": f"{rel_freq:.4f}"}])
    results_df = wildcard_freq_df.rename(columns={"Relative Frequency": "Expected Frequency"})

    # --- KWIC Display ---
    st.subheader("📚 Concordance Results")
    
    if is_pattern_search_active:
        st.success(f"Pattern search successful! Found **{total_matches}** instances of '{raw_target_input}' co-occurring with the specified criteria.")
    else:
        st.success(f"Found **{total_matches}** occurrences of the primary target word matching the criteria.")
    
    # --- LLM INTERPRETATION BUTTON/EXPANDER ---
    if st.button("🧠 Interpret Concordance Results (LLM)", key="llm_concordance_btn"):
        kwic_df_for_llm = pd.DataFrame(kwic_rows).head(10).copy().drop(columns=['Collocate'])
        interpret_results_llm(raw_target_input, "Concordance", "KWIC Context Sample (Max 10 lines)", kwic_df_for_llm)

    if st.session_state['llm_interpretation_result']:
        with st.expander("LLM Interpretation (Feature Disabled)", expanded=True):
            st.markdown(st.session_state['llm_interpretation_result'])
        st.markdown("---")
    # ----------------------------------------
    
    col_kwic, col_freq = st.columns([3, 2], gap="large")

    with col_kwic:
        st.subheader(f"Concordance (KWIC) — top {len(kwic_rows)} lines (Scrollable max {KWIC_MAX_DISPLAY_LINES})")
        
        kwic_df = pd.DataFrame(kwic_rows).drop(columns=['Collocate'])
        kwic_preview = kwic_df.copy().reset_index(drop=True)
        kwic_preview.insert(0, "No", range(1, len(kwic_preview)+1))
        
        # --- KWIC Table Style (Extracted for cleaner re-use) ---
        kwic_table_style = f"""
             <style>
             .dataframe-container-scroll {{
                 max-height: 400px; /* Fixed height for scrollable view */
                 overflow-y: auto;
                 margin-bottom: 1rem;
             }}
             .dataframe {{ font-family: monospace; color: white; width: 100%; }}
             .dataframe table {{ width: 100%; table-layout: fixed; word-wrap: break-word; }}
             .dataframe th {{ font-weight: bold; text-align: center; }}
             .dataframe td:nth-child(2) {{ text-align: right; color: white; }}
             .dataframe td:nth-child(3) {{ text-align: center; font-weight: bold; background-color: #f0f0f0; color: black; }}
             .dataframe td:nth-child(4) {{ text-align: left; color: white; }}
             .dataframe thead th:first-child {{ width: 30px; }}
             </style>
        """
        st.markdown(kwic_table_style, unsafe_allow_html=True)
        
        html_table = kwic_preview.to_html(escape=False, classes=['dataframe'], index=False)
        scrollable_html = f"<div class='dataframe-container-scroll'>{html_table}</div>"

        st.markdown(scrollable_html, unsafe_allow_html=True)

        st.caption("Note: Pattern search collocates are **bolded and highlighted bright yellow**.")
        st.download_button("⬇ Download full concordance (xlsx)", data=df_to_excel_bytes(kwic_df), file_name=f"{raw_target_input.replace(' ', '_')}_full_concordance.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with col_freq:
        st.subheader(f"Target Frequency")
        st.dataframe(results_df, use_container_width=True, hide_index=True)


# -----------------------------------------------------
# MODULE: DICTIONARY
# -----------------------------------------------------
if st.session_state['view'] == 'dictionary':
    
    st.subheader("📘 Dictionary Lookup")
    
    # --- Input and Analysis Trigger (Automatic on Change) ---
    current_dict_word = st.text_input(
        "Enter a Token/Word to lookup (e.g., 'sessions'):", 
        value=st.session_state.get('dict_word_input_main', ''),
        key="dict_word_input_main",
        # Reruns automatically on change, triggering analysis below.
    ).strip()
    
    if not current_dict_word:
        st.info("Enter a word to view its linguistic summary, examples, and collocates. Analysis runs automatically.")
        # Keep a manual button as a fallback/re-analyze option
        st.button("🔎 Manual Re-Analyze", key="manual_dict_analyze_disabled", disabled=True)
        st.stop()
    
    # Manual button for re-analysis
    st.button("🔎 Manual Re-Analyze", key="manual_dict_analyze")
        
    st.markdown("---")
    
    # --- 1. Consolidated Word Forms by Lemma ---
    
    if is_raw_mode or 'lemma' not in df.columns:
        st.warning("Lemma and POS analysis is disabled because the corpus is not tagged/lemmatized.")
        forms_list = pd.DataFrame()
        unique_lemma_list = []
    else:
        forms_list, unique_pos_list, unique_lemma_list = get_all_lemma_forms_details(df, current_dict_word)

    st.subheader(f"Word Forms (Based on Lemma: **{', '.join(unique_lemma_list) if unique_lemma_list else 'N/A'}**)")

    if forms_list.empty and not is_raw_mode: # If not raw mode, failure to find forms is a significant issue
        st.warning(f"Token **'{current_dict_word}'** not found in the corpus or no lemma data available.")
        st.stop()
    elif not forms_list.empty:
        # Display the consolidated forms table
        st.dataframe(
            forms_list.rename(columns={'token': 'Token', 'pos': 'POS Tag', 'lemma': 'Lemma'}),
            hide_index=True,
            use_container_width=True
        )
    
    # --- 2. Related Forms (by Regex) ---
    st.markdown("---")
    st.subheader("Related Forms (by Regex)")
    
    related_regex_forms = get_related_forms_by_regex(df, current_dict_word)
    
    if related_regex_forms:
        st.markdown(f"**Tokens matching the pattern `*.{current_dict_word}.*` (case insensitive):**")
        # Use a dynamic key based on the word to ensure the widget itself is updated.
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
    st.subheader("Random Examples (Concordance)")
    
    kwic_left = st.session_state.get('kwic_left', 7)
    kwic_right = st.session_state.get('kwic_right', 7)
    
    with st.spinner(f"Fetching random concordance examples for '{current_dict_word}'..."):
        kwic_rows, total_matches, _, _ = generate_kwic(
            df, current_dict_word, kwic_left, kwic_right, 
            random_sample=True, limit=KWIC_MAX_DISPLAY_LINES
        )
    
    if kwic_rows:
        display_limit = min(5, len(kwic_rows))
        st.success(f"Showing {display_limit} random examples from {total_matches:,} total matches.")
        
        kwic_df = pd.DataFrame(kwic_rows).drop(columns=['Collocate'])
        kwic_preview = kwic_df.copy().reset_index(drop=True)
        kwic_preview.insert(0, "No", range(1, len(kwic_preview)+1))
        
        kwic_table_style = f"""
            <style>
            .dictionary-kwic-container {{
                max-height: 250px; /* Fixed height for scrollable view */
                overflow-y: auto;
                margin-bottom: 1rem;
            }}
            .dict-kwic-table {{ font-family: monospace; color: white; width: 100%; }}
            .dict-kwic-table table {{ width: 100%; table-layout: fixed; word-wrap: break-word; }}
            .dict-kwic-table th {{ font-weight: bold; text-align: center; }}
            .dict-kwic-table td:nth-child(2) {{ text-align: right; color: white; }}
            .dict-kwic-table td:nth-child(3) {{ text-align: center; font-weight: bold; background-color: #f0f0f0; color: black; }}
            .dict-kwic-table td:nth-child(4) {{ text-align: left; color: white; }}
            .dict-kwic-table thead th:first-child {{ width: 30px; }}
            </style>
        """
        st.markdown(kwic_table_style, unsafe_allow_html=True)
        
        html_table = kwic_preview.to_html(escape=False, classes=['dict-kwic-table'], index=False)
        scrollable_html = f"<div class='dictionary-kwic-container'>{html_table}</div>"
        st.markdown(scrollable_html, unsafe_allow_html=True)
    else:
        st.info("No examples found.")
        
    st.markdown("---")

    # --- 4. Collocates and Collocate Examples ---
    st.subheader("Collocation Analysis")
    
    coll_window = st.session_state.get('coll_window', 5)
    mi_min_freq = st.session_state.get('mi_min_freq', 1)
    max_collocates = st.session_state.get('max_collocates', 20)
    
    # Collocation filter settings must be read from session state keys (even if empty in raw mode)
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
        st.stop()
        
    # Get top 20 collocates for display/examples
    top_collocates = stats_df_sorted.head(20)
    
    # 3a. Top Collocates List
    collocate_list = ", ".join(top_collocates['Collocate'].tolist())
    st.markdown(f"**Top {len(top_collocates)} Collocates (LL-ranked):**")
    st.text_area("Collocate List", collocate_list, height=100)
    
    st.markdown("---")
    st.subheader(f"Collocate Examples (Top {len(top_collocates)} LL Collocates)")
    
    
    # Use the dedicated KWIC display function
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
    
    with st.spinner("Running collocation analysis..."):
        stats_df_sorted, freq, primary_target_mwu = generate_collocation_results(
            df, raw_target_input, coll_window, mi_min_freq, max_collocates, is_raw_mode,
            collocate_regex, collocate_pos_regex_input, selected_pos_tags, collocate_lemma
        )

    if freq == 0:
        st.warning(f"Target '{raw_target_input}' not found in corpus.")
        st.stop()
        
    primary_rel_freq = (freq / total_tokens) * 1_000_000
    
    st.subheader("🔗 Collocation Analysis Results")
    st.success(f"Analyzing target '{primary_target_mwu}'. Frequency: **{freq:,}**, Relative Frequency: **{primary_rel_freq:.4f}** per million.")

    if stats_df_sorted.empty:
        st.warning("No collocates found after applying filters.")
        st.stop()
        
    # --- LLM INTERPRETATION BUTTON/EXPANDER ---
    if st.button("🧠 Interpret Collocation Results (LLM)", key="llm_collocation_btn"):
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
        st.markdown(f"<div class='scrollable-table'>{html_table}</div>", unsafe_allow_html=True)
        
    with col_mi_table:
        st.markdown(f"**Mutual Information (MI) (obs ≥ {mi_min_freq}, Top {len(full_mi)})**")
        
        # Display table with relevant columns
        mi_display_df = full_mi[['Rank', 'Collocate', 'MI', 'Direction', 'Significance']].copy()
        
        # Use a scrollable container for the main table
        html_table = mi_display_df.to_html(index=False, classes=['collocate-table'])
        st.markdown(f"<div class='scrollable-table'>{html_table}</div>", unsafe_allow_html=True)

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
    st.markdown("---")
    
    # LL-Ranked KWIC Examples
    st.subheader(f"📚 Concordance Examples for Top {KWIC_COLLOC_DISPLAY_LIMIT} LL Collocates (1 example per collocate)")
    display_collocation_kwic_examples(
        df_corpus=df, 
        node_word=primary_target_mwu, 
        top_collocates_df=full_ll, 
        window=coll_window,
        limit_per_collocate=1 # Exactly 1 example per collocate
    )
    
    st.markdown("---")
    
    # MI-Ranked KWIC Examples
    st.subheader(f"📚 Concordance Examples for Top {KWIC_COLLOC_DISPLAY_LIMIT} MI Collocates (1 example per collocate)")
    display_collocation_kwic_examples(
        df_corpus=df, 
        node_word=primary_target_mwu, 
        top_collocates_df=full_mi, 
        window=coll_window,
        limit_per_collocate=1 # Exactly 1 example per collocate
    )


st.caption("Tip: This app handles both pre-tagged vertical corpora and raw linear text.")
