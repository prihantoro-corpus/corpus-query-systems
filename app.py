# app.py
# handles non tagged and non lemmatised text
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
from matplotlib.colors import to_rgb, rgb2hex 

# Import for Pyvis Network Graph
from pyvis.network import Network
import streamlit.components.v1 as components # Import for HTML embedding

st.set_page_config(page_title="CORTEX - Corpus Explorer v13", layout="wide")

# Initialize Session State for View Management
if 'view' not in st.session_state:
    st.session_state['view'] = 'overview'

# ---------------------------
# Built-in Corpus Configuration
# ---------------------------

BUILT_IN_CORPORA = {
    "Select built-in corpus...": None,
    "Europarl 1M Only": "https://raw.githubusercontent.com/prihantoro-corpus/corpus-query-systems/main/europarl_en-1M-only.txt",
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

# --- NEW FUNCTION: Cache Reset ---
def reset_analysis():
    # Clear the entire Streamlit cache to force re-reading and re-analysis
    st.cache_data.clear()
    st.session_state['view'] = 'overview' # Reset view to overview on corpus change
    
# ---------------------------
# Helpers: stats, IO utilities
# ---------------------------
EPS = 1e-12

def safe_log(x):
    return math.log(max(x, EPS))

def compute_ll(k11, k12, k21, k22):
    """Computes the Log-Likelihood (LL) statistic."""
    total = k11 + k12 + k21 + k22
    if total == 0:
        return 0.0
    
    e11 = (k11 + k12) * (k11 + k21) / total
    e12 = (k11 + k12) * (k12 + k22) / total
    e21 = (k21 + k22) * (k11 + k21) / total
    e22 = (k21 + k22) * (k12 + k22) / total
    
    s = 0.0
    for k,e in ((k11,e11),(k12,e12),(k21,e21),(k22,e22)):
        if k > 0 and e > 0:
            s += k * math.log(k / e)
            
    return 2.0 * s

def compute_mi(k11, target_freq, coll_total, corpus_size):
    """Computes the Mutual Information (MI) statistic."""
    expected = (target_freq * coll_total) / corpus_size
    if expected == 0 or k11 == 0:
        return 0.0
    return math.log2(k11 / expected)

def significance_from_ll(ll_val):
    """Converts Log-Likelihood value to significance level."""
    if ll_val >= 15.13:
        return '*** (p<0.001)'
    if ll_val >= 10.83:
        return '** (p<0.01)'
    if ll_val >= 3.84:
        return '* (p<0.05)'
    return 'ns'

def df_to_excel_bytes(df):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    buf.seek(0)
    return buf.getvalue()

# --- MISSING FUNCTION RESTORED: Standardized Type-Token Ratio ---
def compute_sttr_tokens(tokens_list, chunk=1000):
    if len(tokens_list) < chunk:
        return (len(set(tokens_list)) / len(tokens_list)) if len(tokens_list) > 0 else 0.0
    ttrs = []
    for i in range(0, len(tokens_list), chunk):
        c = tokens_list[i:i+chunk]
        if not c: continue
        ttrs.append(len(set(c)) / len(c))
    
    return (sum(ttrs)/len(ttrs)) if ttrs else 0.0

# --- Word Cloud Function ---
@st.cache_data
def create_word_cloud(freq_data, is_tagged_mode):
    """Generates a word cloud from frequency data with conditional POS coloring."""
    
    # Filter out multi-word units for visualization stability
    single_word_freq_data = freq_data[~freq_data['token'].str.contains(' ')].copy()
    if single_word_freq_data.empty:
        return None

    # Convert frequency data to a dictionary for WordCloud
    word_freq_dict = single_word_freq_data.set_index('token')['frequency'].to_dict()
    
    # Map from word to its POS tag (used by the custom color function)
    word_to_pos = single_word_freq_data.set_index('token').get('pos', pd.Series('O')).to_dict()
    
    # Define stopwords (must be in lowercase)
    stopwords = set(["the", "of", "to", "and", "in", "that", "is", "a", "for", "on", "it", "with", "as", "by", "this", "be", "are"])
    
    wc = WordCloud(
        width=800,
        height=400,
        background_color='black',
        colormap='viridis', 
        stopwords=stopwords,
        min_font_size=10
    )
    
    # 1. Generate the WordCloud first to calculate layout
    wordcloud = wc.generate_from_frequencies(word_freq_dict)

    if is_tagged_mode:
        # Define the custom color function (must be defined inside the cacheable function)
        def final_color_func(word, *args, **kwargs):
            # This function returns the HEX color string based on POS
            pos_tag = word_to_pos.get(word, 'O')
            pos_code = pos_tag[0].upper() if pos_tag and len(pos_tag) > 0 else 'O'
            if pos_code not in POS_COLOR_MAP:
                pos_code = 'O'
            return POS_COLOR_MAP.get(pos_code, POS_COLOR_MAP['O'])

        # 2. Recolor the generated WordCloud
        wordcloud = wordcloud.recolor(color_func=final_color_func)
        
    # Plot the WordCloud image
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    plt.tight_layout(pad=0)
    
    return fig

@st.cache_data
def create_pyvis_graph(target_word, coll_df):
    """
    Creates a Pyvis interactive network graph. 
    Node colors are based on the first letter of the POS tag, if available.
    """
    net = Network(height="400px", width="100%", bgcolor="#222222", font_color="white", cdn_resources='local')
    
    max_ll = coll_df['LL'].max()
    min_ll = coll_df['LL'].min()
    ll_range = max_ll - min_ll
    
    # JSON MUST NOT contain // comments
    net.set_options(f"""
    var options = {{
      "nodes": {{
        "borderWidth": 2,
        "size": 15,
        "font": {{
            "size": 30
        }}
      }},
      "edges": {{
        "width": 5,
        "smooth": {{
          "type": "dynamic"
        }}
      }},
      "physics": {{
        "barnesHut": {{
          "gravitationalConstant": -10000,
          "centralGravity": 0.3,
          "springLength": 95,
          "springConstant": 0.04,
          "damping": 0.9,
          "avoidOverlap": 0.5
        }},
        "minVelocity": 0.75
      }}
    }}
    """)
    
    # 1. Add Target Node 
    net.add_node(target_word, label=target_word, size=40, color='#FFFF00', title=f"Target: {target_word}", font={'color': 'black'})
    
    # 2. Add Collocate Nodes and Edges
    for _, row in coll_df.iterrows():
        collocate = row['Collocate']
        ll_score = row['LL']
        observed = row['Observed']
        
        # Get the POS tag and determine the color
        pos_tag = row['POS']
        
        # Determine color based on POS prefix
        pos_code = pos_tag[0].upper() if pos_tag and len(pos_tag) > 0 else 'O'
        
        if pos_tag.startswith('##'):
            pos_code = '#'
        elif pos_code not in ['N', 'V', 'J', 'R']:
            pos_code = 'O'
        
        color = POS_COLOR_MAP.get(pos_code, POS_COLOR_MAP['O'])
        
        # Node size based on LL score
        if ll_range > 0:
            normalized_ll = (ll_score - min_ll) / ll_range
            node_size = 15 + normalized_ll * 25 
        else:
            node_size = 25
            
        net.add_node(collocate, label=collocate, size=node_size, color=color, title=f"POS: {row['POS']}\nObs: {observed}\nLL: {ll_score:.2f}")
        net.add_edge(target_word, collocate, value=ll_score, width=5, title=f"LL: {ll_score:.2f}")

    # --- Use a Temporary HTML File ---
    html_content = ""
    temp_path = None
    try:
        temp_filename = "pyvis_graph.html"
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, temp_filename)
        net.write_html(temp_path, notebook=False)
        with open(temp_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    return html_content
    # --- END OF FIX ---


@st.cache_data
def download_file_to_bytesio(url):
    """Downloads a file from a URL and returns its content as BytesIO."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() 
        return BytesIO(response.content)
    except Exception as e:
        st.error(f"Failed to download built-in corpus from {url}. Ensure the file is public and the URL is a RAW content link.")
        return None

# ---------------------------
# Cached loading (Robust Encoding & Parsing Implemented)
# ---------------------------
@st.cache_data
def load_corpus_file(file_source, sep=r"\s+"):
    """
    Loads corpus either from an uploaded file handle or a BytesIO object,
    prioritizing structured tab-separated format.
    """
    
    if file_source is None:
        return None

    # --- Step 1: Robustly read and clean the content string ---
    try:
        file_source.seek(0)
        file_bytes = file_source.read()

        # Try robust decoding to handle BOM/encoding issues
        try:
            file_content_str = file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            try:
                # Fallback to Latin-1/ISO-8859-1 (common for older/non-standard files)
                file_content_str = file_bytes.decode('iso-8859-1')
            except Exception:
                # Last resort: decode using errors='ignore'
                file_content_str = file_bytes.decode('utf-8', errors='ignore')
        
        # Filter out potential comments/metadata lines before parsing
        clean_lines = [line for line in file_content_str.splitlines() if line and not line.strip().startswith('#')]
        clean_content = "\n".join(clean_lines)
        
        file_buffer_for_pandas = StringIO(clean_content)

    except Exception as e:
        st.error(f"Error reading file stream: {e}")
        return None
    
    # --- Step 2: Try Structured Parsing (Tab-separated) ---
    df_attempt = None
    
    try:
        # Attempt 1: Tab Separator (most common for tagged corpora)
        file_buffer_for_pandas.seek(0)
        try:
            df_attempt = pd.read_csv(file_buffer_for_pandas, sep='\t', header=None, engine="python", dtype=str)
        except Exception:
            file_buffer_for_pandas.seek(0) 
            # Attempt 2: Default Separator (Whitespace)
            df_attempt = pd.read_csv(file_buffer_for_pandas, sep=sep, header=None, engine="python", dtype=str)
            
        # Check only for the number of columns (3+) to detect a tagged file
        if df_attempt is not None and df_attempt.shape[1] >= 3:
            # --- PROCESS TAGGED/VERTICAL FILE ---
            df = df_attempt.iloc[:, :3]
            df.columns = ["token", "pos", "lemma"]
            # Preserve original tags and lemmas for visualization
            df["token"] = df["token"].fillna("").astype(str)
            df["pos"] = df["pos"].fillna("###").astype(str)
            df["lemma"] = df["lemma"].fillna("###").astype(str)
            df["_token_low"] = df["token"].str.lower()
            return df
            
    except Exception:
         # If structured reading fails entirely, proceed to raw text processing
         pass

    # --- Step 3: Fallback to Raw Horizontal Text ---
    try:
        raw_text = file_content_str
        
        # Regex to split tokens based on space and punctuation, keeping punctuation as separate tokens
        tokens = re.findall(r'\b\w+\b|[^\w\s]+', raw_text)
        tokens = [t.strip() for t in tokens if t.strip()] # Clean up any empty strings

        # Assign nonsense tag and lemma
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
        
    except Exception as raw_e:
        st.error(f"Error processing corpus data: {raw_e}")
        return None 


# ---------------------------
# UI: header
# ---------------------------
st.title("CORTEX - Corpus Texts Explorer v13")
st.caption("Upload vertical corpus (**token POS lemma**) or **raw horizontal text**. Raw text is analyzed quickly using basic tokenization and generic tags (`##`).")

# ---------------------------
# Panel: upload and corpus info
# ---------------------------
corpus_source = None
corpus_name = "Uploaded File"

# --- SIDEBAR: CORPUS SELECTION (STATIC) ---
with st.sidebar:
    st.header("Upload & Options")
    
    st.subheader("1. Choose Corpus Source")
    
    # Added callback to reset cache when built-in corpus selection changes
    selected_corpus_name = st.selectbox(
        "Select a pre-loaded corpus:", 
        options=list(BUILT_IN_CORPORA.keys()),
        key="corpus_select", 
        on_change=reset_analysis
    )
    
    # Added callback to reset cache when file is uploaded
    uploaded_file = st.file_uploader(
        "OR Upload your own corpus file", 
        type=["txt","csv"],
        key="file_upload",
        on_change=reset_analysis
    )
    
    st.markdown("---")
    
    # Determine the corpus source
    if uploaded_file is not None:
        corpus_source = uploaded_file
        corpus_name = uploaded_file.name
    elif selected_corpus_name != "Select built-in corpus...":
        corpus_url = BUILT_IN_CORPORA[selected_corpus_name]
        with st.spinner(f"Downloading {selected_corpus_name}..."):
            corpus_source = download_file_to_bytesio(corpus_url)
        corpus_name = selected_corpus_name
    
    if corpus_source is None:
        st.info("Please select a corpus or upload a file to proceed.")
        st.stop()
        
    # --- SIDEBAR: MODULE SETTINGS (DYNAMIC) ---
    st.subheader("2. Settings")

    if st.session_state['view'] == 'concordance':
        st.write("KWIC Context (Display)")
        kwic_left = st.number_input("Left Context (tokens)", min_value=1, max_value=20, value=7, step=1, help="Number of tokens shown to the left of the node word.")
        kwic_right = st.number_input("Right Context (tokens)", min_value=1, max_value=20, value=7, step=1, help="Number of tokens shown to the right of the node word.")
        st.session_state['kwic_left'] = kwic_left
        st.session_state['kwic_right'] = kwic_right
        
    elif st.session_state['view'] == 'collocation':
        coll_window = st.number_input("Collocation window (tokens each side)", min_value=1, max_value=10, value=5, step=1, help="Window used for collocation counting (default ±5).")
        st.write("MI minimum observed frequency (for MI tables).")
        mi_min_freq = st.number_input("MI min observed freq", min_value=1, max_value=100, value=1, step=1)
        st.session_state['coll_window'] = coll_window
        st.session_state['mi_min_freq'] = mi_min_freq

    st.markdown("---")
    st.write("Shareable deployment tip:")
    st.info("Deploy this app on Streamlit Cloud or HuggingFace Spaces for free sharing.")


# load corpus (cached)
df = load_corpus_file(corpus_source)

if df is None:
    st.error("Corpus failed to load. Please check the file format or download source.")
    st.stop()
    
# --- CORPUS STATS CALCULATION (SHARED) ---
if 'pos' in df.columns and len(df) > 0:
    count_of_raw_tags = df['pos'].str.contains('##', na=False).sum()
    is_raw_mode = (count_of_raw_tags / len(df)) > 0.99
else:
    is_raw_mode = True 

total_tokens = len(df)
tokens_lower = df["_token_low"].tolist()
PUNCTUATION = {'.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}', '"', "'", '---', '--', '-', '...', '«', '»', '—'}
tokens_lower_filtered = [t for t in tokens_lower if t not in PUNCTUATION and not t.isdigit()]
token_counts = Counter(tokens_lower_filtered)
unique_types = len(set(tokens_lower_filtered))
unique_lemmas = df["lemma"].nunique() if "lemma" in df.columns else "###"
sttr_score = compute_sttr_tokens(tokens_lower_filtered)

freq_df_filtered = df[~df['_token_low'].isin(PUNCTUATION) & ~df['_token_low'].str.isdigit()].copy()
if is_raw_mode:
    freq_df_filtered['pos'] = '##'
freq_df = freq_df_filtered.groupby(["token","pos"]).size().reset_index(name="frequency").sort_values("frequency", ascending=False).reset_index(drop=True)

if is_raw_mode:
    app_mode = f"Analyzing Corpus: {corpus_name} (RAW/LINEAR MODE)"
else:
    app_mode = f"Analyzing Corpus: {corpus_name} (TAGGED MODE)"
st.header(app_mode)

# --- NAVIGATION BUTTONS (OVERVIEW/TOP LEVEL) ---
if st.session_state['view'] == 'overview':
    st.subheader("Corpus Explorer Modules")
    col_nav = st.columns(2)
    with col_nav[0]:
        if st.button("📚 Concordance (KWIC)", use_container_width=True):
            st.session_state['view'] = 'concordance'
            st.rerun()
    with col_nav[1]:
        if st.button("🔗 Collocation Network", use_container_width=True):
            st.session_state['view'] = 'collocation'
            st.rerun()
    st.markdown("---")


# -----------------------------------------------------
# MODULE: CORPUS OVERVIEW
# -----------------------------------------------------
if st.session_state['view'] == 'overview':
    
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Corpus Summary")
        info_df = pd.DataFrame({
            "Metric": ["Corpus size (tokens)", "Unique types (w/o punc)", "Lemma count", "STTR (w/o punc, chunk=1000)"],
            "Value": [f"{total_tokens:,}", unique_types, unique_lemmas, round(sttr_score,4)]
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True) 

        # --- Word Cloud Display (Conditional Coloring) ---
        st.subheader("Word Cloud (Top Words - Stopwords Filtered)")
        if not freq_df.empty:
            wordcloud_fig = create_word_cloud(freq_df, not is_raw_mode)
            
            # Display the legend key if in tagged mode
            if not is_raw_mode:
                st.markdown(
                    """
                    **Word Cloud Color Key (POS):**
                    | Color | POS Prefix |
                    | :--- | :--- |
                    | <span style="color:#33CC33;">**Green**</span> | Noun (N*) |
                    | <span style="color:#3366FF;">**Blue**</span> | Verb (V*) |
                    | <span style="color:#FF33B5;">**Pink**</span> | Adjective (J*) |
                    | <span style="color:#FFCC00;">**Yellow**</span> | Adverb (R*) |
                    """
                , unsafe_allow_html=True)
                
            st.pyplot(wordcloud_fig)
        else:
            st.info("Not enough tokens to generate a word cloud.")

    with col2:
        st.subheader("Top frequency (token / POS / freq) (Punctuation skipped)")
        
        freq_head = freq_df.head(10).copy()
        freq_head.insert(0,"No", range(1, len(freq_head)+1))
        st.dataframe(freq_head, use_container_width=True, hide_index=True) 
        
        st.download_button("⬇ Download full frequency list (xlsx)", data=df_to_excel_bytes(freq_df), file_name="full_frequency_list_filtered.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.markdown("---")
    
# -----------------------------------------------------
# MODULE: CONCORDANCE / COLLOCATION (SHARED SEARCH INPUT)
# -----------------------------------------------------

if st.session_state['view'] != 'overview':
    
    # --- SEARCH INPUT (SHARED) ---
    st.subheader(f"Search Input: {st.session_state['view'].capitalize()}")
    col_a, col_b = st.columns(2)
    with col_a:
        typed_target = st.text_input("Type a token, MWU ('in the'), or wildcard ('in*')", value="")
    with col_b:
        uploaded_targets = st.file_uploader("Or upload list of tokens (one per line)", type=["txt","csv"], key="targets_upload")

    selected_target = None
    if uploaded_targets is not None:
        try:
            target_list = pd.read_csv(uploaded_targets, header=None, squeeze=True, engine="python")[0].astype(str).str.strip().tolist()
        except Exception:
            uploaded_targets.seek(0)
            target_list = uploaded_targets.read().decode('utf-8').splitlines()
            target_list = [t.strip() for t in target_list if t.strip()]
        if target_list:
            selected_target = st.selectbox("Select target from uploaded list", options=target_list)
    target_input = (selected_target if selected_target else typed_target).strip()

    contains_wildcard = '*' in target_input

    if not target_input:
        st.info(f"Type a term or pattern for {st.session_state['view'].capitalize()} analysis. Then click Analyze.")
    analyze_btn = st.button("🔎 Analyze")
    
    st.markdown("---")


# -----------------------------------------------------
# MODULE: CONCORDANCE LOGIC
# -----------------------------------------------------
if st.session_state['view'] == 'concordance' and analyze_btn and target_input:
    
    kwic_left = st.session_state.get('kwic_left', 7)
    kwic_right = st.session_state.get('kwic_right', 7)
    target = target_input.lower()
    
    # --- MWU/WILDCARD RESOLUTION ---
    
    primary_target_mwu = None
    primary_target_tokens = []
    primary_target_len = 0
    wildcard_freq_df = pd.DataFrame()
    
    # FIX: Initialize primary target variables to prevent NameError
    
    if contains_wildcard:
        
        # Single-token wildcard (in*, *as, pi*e)
        if ' ' not in target:
            pattern = re.escape(target).replace(r'\*', '.*')
            wildcard_matches = [token for token in token_counts if re.fullmatch(pattern, token)]
            
            match_counts = {token: token_counts[token] for token in wildcard_matches if token in token_counts}
            sorted_matches = sorted(match_counts.items(), key=lambda item: item[1], reverse=True)
            
            wildcard_freq_list = []
            for term, count in sorted_matches:
                rel_freq = (count / total_tokens) * 1_000_000
                wildcard_freq_list.append({"Query Result": term, "Raw Frequency": count, "Relative Frequency": f"{rel_freq:.4f}"})
            wildcard_freq_df = pd.DataFrame(wildcard_freq_list)
            
        # Multi-word wildcard (in *)
        else:
            target_pattern_parts = target.split(' ')
            num_parts = len(target_pattern_parts)
            mwu_matches = []
            
            for i in range(len(tokens_lower) - num_parts + 1):
                match = True
                for k, part in enumerate(target_pattern_parts):
                    part_pattern = re.escape(part).replace(r'\*', '.*')
                    if not re.fullmatch(part_pattern, tokens_lower[i + k]):
                        match = False
                        break
                if match:
                    mwu_string = " ".join(tokens_lower[i:i + num_parts])
                    mwu_matches.append(mwu_string)
                    
            match_counts = Counter(mwu_matches)
            
            wildcard_freq_list = []
            for term, count in match_counts.most_common():
                rel_freq = (count / total_tokens) * 1_000_000
                wildcard_freq_list.append({"Query Result": term, "Raw Frequency": count, "Relative Frequency": f"{rel_freq:.4f}"})
            wildcard_freq_df = pd.DataFrame(wildcard_freq_list)
            
        # Set primary target
        if not wildcard_freq_df.empty:
            primary_target_mwu = wildcard_freq_df.iloc[0]["Query Result"]
            primary_target_tokens = primary_target_mwu.split()
            primary_target_len = len(primary_target_tokens)
            target_display = f"'{target_input}' (Most Frequent Match: '{primary_target_mwu}')"
        else:
            st.warning(f"Target pattern '{target_input}' not found in corpus.")
            st.stop()
            
    # Literal Search
    else:
        primary_target_mwu = target
        primary_target_tokens = target.split()
        primary_target_len = len(primary_target_tokens)
        target_display = f"'{target_input}'"
        
        literal_freq = 0
        for i in range(len(tokens_lower) - primary_target_len + 1):
             if tokens_lower[i:i + primary_target_len] == primary_target_tokens:
                 literal_freq += 1
        
        if literal_freq == 0:
            st.warning(f"Target '{target_input}' not found in corpus.")
            st.stop()
            
        rel_freq = (literal_freq / total_tokens) * 1_000_000
        wildcard_freq_df = pd.DataFrame([{"Query Result": primary_target_mwu, "Raw Frequency": literal_freq, "Relative Frequency": f"{rel_freq:.4f}"}])

    # 2. Concordance Generation (Sampling by Variation for Wildcards)
    
    kwic_rows = []
    
    if contains_wildcard and not wildcard_freq_df.empty:
        max_kwic_lines = 10
        total_kwic_lines = 0
        
        # Iterate over the top query results (MWU variations)
        for _, row in wildcard_freq_df.iterrows():
            if total_kwic_lines >= max_kwic_lines:
                break
                
            mwu = row["Query Result"]
            mwu_tokens = mwu.split()
            mwu_len = len(mwu_tokens)
            
            # Find all positions for this specific MWU variation
            mwu_positions = []
            for i in range(len(tokens_lower) - mwu_len + 1):
                if tokens_lower[i:i + mwu_len] == mwu_tokens:
                    mwu_positions.append(i)
            
            # Determine how many lines to take from this MWU (min of its freq, and lines remaining)
            lines_to_take = min(1, max_kwic_lines - total_kwic_lines, len(mwu_positions))
            
            # Take a sample (the first 'lines_to_take' occurrences)
            for i in mwu_positions[:lines_to_take]:
                # i is the index of the first token of the MWU
                left = tokens_lower[max(0, i - kwic_left):i]
                right = tokens_lower[i + mwu_len:i + mwu_len + kwic_right]
                
                # Node is the sequence of original tokens
                node_orig_tokens = df["token"].iloc[i:i + mwu_len].tolist()
                node_orig = " ".join(node_orig_tokens)
                
                kwic_rows.append({"Left": " ".join(left), "Node": node_orig, "Right": " ".join(right)})
                total_kwic_lines += 1
                
    else:
        # Literal search: standard position finding and sampling
        positions = []
        for i in range(len(tokens_lower) - primary_target_len + 1):
            if tokens_lower[i:i + primary_target_len] == primary_target_tokens:
                positions.append(i)
        
        for i in positions[:10]:
            left = tokens_lower[max(0, i - kwic_left):i]
            right = tokens_lower[i + primary_target_len:i + primary_target_len + kwic_right]
            
            node_orig_tokens = df["token"].iloc[i:i + primary_target_len].tolist()
            node_orig = " ".join(node_orig_tokens)
            
            kwic_rows.append({"Left": " ".join(left), "Node": node_orig, "Right": " ".join(right)})
    
    # --- KWIC Display ---
    st.subheader("📚 Concordance Results")
    st.success(f"Found {wildcard_freq_df['Raw Frequency'].sum()} total occurrences matching '{target_input}'.")
    
    col_kwic, col_freq = st.columns([3, 2], gap="large")

    with col_kwic:
        st.subheader(f"Concordance (KWIC) — top {len(kwic_rows)} (Sampled by Variation)")
        
        kwic_df = pd.DataFrame(kwic_rows)
        kwic_preview = kwic_df.copy().reset_index(drop=True)
        kwic_preview.insert(0, "No", range(1, len(kwic_preview)+1))
        st.dataframe(kwic_preview, use_container_width=True, hide_index=True)

        st.download_button("⬇ Download full concordance (xlsx)", data=df_to_excel_bytes(kwic_df), file_name=f"{target.replace(' ', '_')}_full_concordance.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with col_freq:
        if contains_wildcard:
            st.subheader(f"Wildcard Results: '{target_input}' (Top 10)")
        else:
            st.subheader(f"Target Frequency")
            
        freq_results_preview = wildcard_freq_df.head(10).copy()
        st.dataframe(freq_results_preview, use_container_width=True, hide_index=True)
        
        st.download_button(
            "⬇ Download full result frequency (xlsx)", 
            data=df_to_excel_bytes(wildcard_freq_df), 
            file_name=f"{target.replace(' ', '_')}_wildcard_frequency_full.xlsx", 
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# -----------------------------------------------------
# MODULE: COLLOCATION LOGIC
# -----------------------------------------------------
if st.session_state['view'] == 'collocation' and analyze_btn and target_input:
    
    coll_window = st.session_state.get('coll_window', 5)
    mi_min_freq = st.session_state.get('mi_min_freq', 1)
    target = target_input.lower()

    # --- MWU/WILDCARD RESOLUTION (Focus on single primary target) ---
    
    # Re-run MWU/Wildcard resolution to establish primary_target_mwu and freq
    primary_target_mwu = None
    primary_target_tokens = []
    primary_target_len = 0
    
    if contains_wildcard:
        wildcard_matches = []
        
        # Run resolution logic to get the most frequent match
        if ' ' not in target:
            pattern = re.escape(target).replace(r'\*', '.*')
            wildcard_matches = [token for token in token_counts if re.fullmatch(pattern, token)]
            match_counts = {token: token_counts[token] for token in wildcard_matches if token in token_counts}
            sorted_matches = sorted(match_counts.items(), key=lambda item: item[1], reverse=True)
            
            if sorted_matches:
                primary_target_mwu = sorted_matches[0][0]
                freq = sorted_matches[0][1]
                primary_target_tokens = [primary_target_mwu]
                primary_target_len = 1
            
        else:
            target_pattern_parts = target.split(' ')
            num_parts = len(target_pattern_parts)
            mwu_matches = []
            
            for i in range(len(tokens_lower) - num_parts + 1):
                match = True
                for k, part in enumerate(target_pattern_parts):
                    part_pattern = re.escape(part).replace(r'\*', '.*')
                    if not re.fullmatch(part_pattern, tokens_lower[i + k]):
                        match = False
                        break
                if match:
                    mwu_matches.append(" ".join(tokens_lower[i:i + num_parts]))
            
            match_counts = Counter(mwu_matches)
            if match_counts:
                primary_target_mwu, freq = match_counts.most_common(1)[0]
                primary_target_tokens = primary_target_mwu.split()
                primary_target_len = len(primary_target_tokens)
            
        if not primary_target_mwu:
            st.warning(f"Target pattern '{target_input}' not found in corpus.")
            st.stop()
        
        target_display = f"'{target_input}' (Analysis on Most Frequent Match: '{primary_target_mwu}')"

    # Literal Search
    else:
        primary_target_mwu = target
        primary_target_tokens = target.split()
        primary_target_len = len(primary_target_tokens)
        target_display = f"'{target_input}'"
        
        # Calculate literal frequency
        freq = 0
        for i in range(len(tokens_lower) - primary_target_len + 1):
             if tokens_lower[i:i + primary_target_len] == primary_target_tokens:
                 freq += 1
        
        if freq == 0:
            st.warning(f"Target '{target_input}' not found in corpus.")
            st.stop()
    
    # Calculate global target frequency and display info
    primary_target_positions = []
    for i in range(len(tokens_lower) - primary_target_len + 1):
        if tokens_lower[i:i + primary_target_len] == primary_target_tokens:
            primary_target_positions.append(i)

    primary_rel_freq = (freq / total_tokens) * 1_000_000
    
    st.subheader("🔗 Collocation Analysis Results")
    st.success(f"Analyzing target {target_display}. Frequency: **{freq:,}**, Relative Frequency: **{primary_rel_freq:.4f}** per million.")

    # --- COLLOCATION CALCULATION ---
    coll_pairs = []
    for i in primary_target_positions:
        start = max(0, i - coll_window)
        end = min(total_tokens, i + primary_target_len + coll_window) 
        
        for j in range(start, end):
            if i <= j < i + primary_target_len:
                continue
            coll_pairs.append((tokens_lower[j], df["pos"].iloc[j]))

    coll_df = pd.DataFrame(coll_pairs, columns=["collocate", "pos"])
    coll_counts = coll_df.groupby(["collocate","pos"]).size().reset_index(name="Observed")

    stats_list = []
    token_counts_unfiltered = Counter(tokens_lower) 
    
    for _, row in coll_counts.iterrows():
        w = row["collocate"]
        p = row["pos"]
        observed = int(row["Observed"])
        total_freq = token_counts_unfiltered.get(w, 0)
        k11 = observed
        k12 = freq - k11
        k21 = total_freq - k11
        k22 = total_tokens - (k11 + k12 + k21)
        ll = compute_ll(k11, k12, k21, k22)
        mi = compute_mi(k11, freq, total_freq, total_tokens)
        sig = significance_from_ll(ll)
        stats_list.append({
            "Collocate": w,
            "POS": p,
            "Observed": observed,
            "Total_Freq": total_freq,
            "LL": round(ll,6),
            "MI": round(mi,6),
            "Significance": sig
        })

    stats_df = pd.DataFrame(stats_list)
    if stats_df.empty:
        st.warning("No collocates found.")
        st.stop()
        
    stats_df_sorted = stats_df.sort_values("LL", ascending=False)
    network_df = stats_df_sorted.head(20).copy()

    full_ll = stats_df_sorted.head(10).copy()
    full_ll.insert(0, "Rank", range(1, len(full_ll)+1))
    
    full_mi_all = stats_df[stats_df["Observed"] >= mi_min_freq].sort_values("MI", ascending=False).reset_index(drop=True)
    full_mi = full_mi_all.head(10).copy()
    full_mi.insert(0, "Rank", range(1, len(full_mi)+1))

    
    # --- Collocation Network Graph ---
    st.markdown("---")
    st.subheader(f"Interactive Collocation Network (Top {len(network_df)} LL)")
    
    network_html = create_pyvis_graph(primary_target_mwu, network_df)
    components.html(network_html, height=450)
    
    st.markdown(
        """
        **Graph Key (POS Tags Restored):**
        * Central Node (Target): **Yellow**
        * Collocate Node Color: Noun (Prefix 'N') **Green**, Verb (Prefix 'V') **Blue**, Adjective (Prefix 'J') **Pink**, Adverb (Prefix 'R') **Yellow**. All other tags/raw text (`##`) are **Gray**.
        * Edge Thickness: **All Thick** (Uniform)
        * Collocate Bubble Size: Scales with Log-Likelihood (LL) score (**Bigger Bubble** = Stronger Collocation)
        * Hover over nodes for details (POS, Observed Freq, LL score).
        """
    )
    st.markdown("---")
    
    # --- Conditional POS Table Display ---
    if not is_raw_mode:
        def category_df(prefixes):
            mask = pd.Series(False, index=stats_df.index)
            for pref in prefixes:
                mask = mask | stats_df["POS"].str.startswith(pref, na=False)
            sub = stats_df[mask].copy()
            ll_sub = sub.sort_values("LL", ascending=False).reset_index(drop=True).head(10).copy()
            ll_sub.insert(0, "Rank", range(1, len(ll_sub)+1))
            mi_sub = sub[sub["Observed"] >= mi_min_freq].sort_values("MI", ascending=False).reset_index(drop=True).head(10).copy()
            mi_sub.insert(0, "Rank", range(1, len(mi_sub)+1))
            return ll_sub, mi_sub

        ll_N, mi_N = category_df(("N",))
        ll_V, mi_V = category_df(("V",))
        ll_J, mi_J = category_df(("J",))
        ll_R, mi_R = category_df(("R",))
        
        st.subheader("Log-Likelihood — Top 10 by POS")
        cols = st.columns(4, gap="small")
        
        with cols[0]:
            st.markdown("**N (N*) — LL**")
            st.dataframe(ll_N, use_container_width=True, hide_index=True)
        with cols[1]:
            st.markdown("**V (V*) — LL**")
            st.dataframe(ll_V, use_container_width=True, hide_index=True)
        with cols[2]:
            st.markdown("**J (J*) — LL**")
            st.dataframe(ll_J, use_container_width=True, hide_index=True)
        with cols[3]:
            st.markdown("**R (R*) — LL**")
            st.dataframe(ll_R, use_container_width=True, hide_index=True)
        
        st.markdown("---")

    else:
        st.info("POS-specific collocation tables (N, V, J, R) are skipped in RAW/LINEAR mode.")
    
    # ---------- Full Collocation Tables (All Tags) ----------
    st.subheader("Top 10 Collocations (All Tags)")
    
    cols_full = st.columns(2, gap="large")
    
    with cols_full[0]:
        st.markdown("**Log-Likelihood (LL)**")
        st.dataframe(full_ll, use_container_width=True, hide_index=True)
    
    with cols_full[1]:
        st.markdown(f"**Mutual Information (MI) (obs ≥ {mi_min_freq})**")
        st.dataframe(full_mi, use_container_width=True, hide_index=True)

    # ---------- Download Buttons ----------
    st.markdown("---")
    st.subheader("Download Full Results")
    
    # Download logic for Collocation results... 
    st.download_button(
        "⬇ Download full LL results (xlsx)", 
        data=df_to_excel_bytes(stats_df_sorted), 
        file_name=f"{primary_target_mwu.replace(' ', '_')}_LL_full.xlsx", 
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.download_button(
        f"⬇ Download full MI results (obs≥{mi_min_freq}) (xlsx)", 
        data=df_to_excel_bytes(full_mi_all), 
        file_name=f"{primary_target_mwu.replace(' ', '_')}_MI_full_obsge{mi_min_freq}.xlsx", 
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.caption("Tip: This app handles both pre-tagged vertical corpora and raw linear text, adjusting analysis depth automatically.")
