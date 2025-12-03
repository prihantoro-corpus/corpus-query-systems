# app.py
# CORTEX Corpus Explorer v15.0 - Integrated LLM Interpretation & Debugging Tool
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

# --- LLM Imports ---
try:
    from google import genai
    from google.genai import types
except ImportError:
    # If the library import fails (e.g., due to requirements.txt missing the package)
    pass 

st.set_page_config(page_title="CORTEX - Corpus Explorer v15.0", layout="wide") # Updated title

# Initialize Session State for View Management
if 'view' not in st.session_state:
    st.session_state['view'] = 'overview'
    
# Initialize Analysis Trigger States
if 'last_target_input' not in st.session_state:
    st.session_state['last_target_input'] = ''
if 'last_pattern_collocate' not in st.session_state:
    st.session_state['last_pattern_collocate'] = ''
if 'trigger_analyze' not in st.session_state:
    st.session_state['trigger_analyze'] = False
if 'initial_load_complete' not in st.session_state:
    st.session_state['initial_load_complete'] = False
if 'last_pattern_search_window' not in st.session_state:
    st.session_state['last_pattern_search_window'] = 0
if 'collocate_pos_regex' not in st.session_state: 
    st.session_state['collocate_pos_regex'] = ''
if 'pattern_collocate_pos' not in st.session_state: 
    st.session_state['pattern_collocate_pos'] = ''
if 'collocate_lemma' not in st.session_state: 
    st.session_state['collocate_lemma'] = ''
if 'llm_interpretation_result' not in st.session_state:
    st.session_state['llm_interpretation_result'] = None


# --- CONSTANTS ---
KWIC_MAX_DISPLAY_LINES = 100
KWIC_INITIAL_DISPLAY_HEIGHT = 10 # Approximate lines for initial view

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

# --- NAVIGATION FUNCTIONS ---
def set_view(view_name):
    st.session_state['view'] = view_name
    st.session_state['llm_interpretation_result'] = None # Clear interpretation on view change
    
def reset_analysis():
    st.cache_data.clear()
    st.session_state['view'] = 'overview'
    st.session_state['last_target_input'] = ''
    st.session_state['last_pattern_collocate'] = ''
    st.session_state['trigger_analyze'] = False
    st.session_state['initial_load_complete'] = False
    st.session_state['llm_interpretation_result'] = None # Clear interpretation on reset
    
# --- Analysis Trigger Callback ---
def trigger_analysis_callback():
    # Only trigger if the input value is non-empty, otherwise it triggers on clearing the input
    if st.session_state.get('pattern_collocate_input', '').strip() or st.session_state.get('typed_target_input', '').strip():
        st.session_state['trigger_analyze'] = True
        st.session_state['llm_interpretation_result'] = None

# -----------------------------------------------------
# LLM INTERPRETATION (Live API Implementation - with robust error trapping)
# -----------------------------------------------------

def interpret_results_llm(target_word, analysis_type, data_description, data):
    """
    Calls the Gemini 2.5 Flash API to get a linguistic interpretation of the corpus results.
    """
    
    if data is None or data.empty:
        return f"Analysis results are empty. Cannot generate an interpretation for '{target_word}'."

    # 1. Initialize Client - relies on GEMINI_API_KEY being set in Streamlit Secrets
    try:
        if 'genai' not in globals() or not os.environ.get("GEMINI_API_KEY"):
             return "LLM API Error: **GEMINI_API_KEY** environment variable is not set in Streamlit Secrets, or the 'google-genai' library failed to import. Please check your app's Secrets settings."
             
        client = genai.Client()
    except Exception as e:
        return f"LLM Client Initialization Failed. Please ensure 'google-genai' is installed and API key is valid. Error: {e}"

    # 2. Construct the Prompt with Data and Instructions
    data_sample = data.head(20).to_markdown(index=False, numalign="left", stralign="left")
    
    system_instruction = (
        "You are an expert Corpus Linguist and Lexicographer. "
        "Your task is to analyze the provided raw linguistic data (KWIC or Collocation) "
        "and provide a concise, professional interpretation. Focus on semantic prosody, "
        "typical syntactic patterns, and the functional or register-specific usage of the target word. "
        "Your interpretation must be in clean Markdown, maximum 4 paragraphs."
    )
    
    user_prompt = f"""
    Analyze the following {analysis_type} results for the target word: **{target_word}**.
    
    --- Data Type: {data_description} ---
    {data_sample}
    
    Provide your expert interpretation based *only* on the provided data.
    """
    
    # 3. Call the API
    try:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.3, # Low temperature for factual, reliable analysis
        )
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[user_prompt],
            config=config
        )
        
        st.session_state['llm_interpretation_result'] = response.text
        return response.text
        
    except Exception as e:
        # Aggressive error reporting to prevent silent failure
        error_message = f"Gemini API FAILED. The key might be invalid, expired, or the request exceeded the rate limit. Full Error: {e}"
        st.session_state['llm_interpretation_result'] = error_message
        return f"LLM API Error: {error_message}"

# ---------------------------
# Helpers: stats, IO utilities (rest of the helper functions remain unchanged)
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
    """Compuutes the Mutual Information (MI) statistic."""
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

# --- Standardized Type-Token Ratio ---
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


@st.cache_data
def create_pyvis_graph(target_word, coll_df):
    """
    Creates a Pyvis interactive network graph. 
    Node colors are based on the first letter of the POS tag, if available.
    Nodes are positioned left/right based on dominant direction.
    """
    net = Network(height="400px", width="100%", bgcolor="#222222", font_color="white", cdn_resources='local')
    
    max_ll = coll_df['LL'].max()
    min_ll = coll_df['LL'].min()
    ll_range = max_ll - min_ll
    
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
    
    # 1. Add Target Node (Central)
    # Target node is placed centrally (fixed x, y for stability)
    net.add_node(target_word, label=target_word, size=40, color='#FFFF00', title=f"Target: {target_word}", x=0, y=0, fixed=True, font={'color': 'black'})
    
    # Define directional bias: large absolute X value forces initial position
    # The bias depends on the direction of the nodes in the input DataFrame
    
    LEFT_BIAS = -500 
    RIGHT_BIAS = 500

    if not coll_df.empty:
        # Determine if the graph is predominantly Left or Right for consistent positioning
        all_directions = coll_df['Direction'].unique()
        
        if 'R' not in all_directions and 'L' in all_directions:
             RIGHT_BIAS = -500 # If only Left nodes, bias them all to the left
        elif 'L' not in all_directions and 'R' in all_directions:
             LEFT_BIAS = 500   # If only Right nodes, bias them all to the right


    # 2. Add Collocate Nodes and Edges (Directionally placed)
    for index, row in coll_df.iterrows():
        collocate = row['Collocate']
        ll_score = row['LL']
        observed = row['Observed']
        pos_tag = row['POS']
        
        # New directional data for placement and tooltip
        direction = row.get('Direction', 'R') 
        obs_l = row.get('Obs_L', 0)
        obs_r = row.get('Obs_R', 0)
        
        # Determine initial X position based on dominant direction, using the calculated bias
        x_position = LEFT_BIAS if direction in ('L', 'B') else RIGHT_BIAS

        # Determine color and size
        pos_code = pos_tag[0].upper() if pos_tag and len(pos_tag) > 0 else 'O'
        if pos_tag.startswith('##'):
            pos_code = '#'
        elif pos_code not in ['N', 'V', 'J', 'R']:
            pos_code = 'O'
        
        color = POS_COLOR_MAP.get(pos_code, POS_COLOR_MAP['O'])
        
        if ll_range > 0:
            normalized_ll = (ll_score - min_ll) / ll_range
            node_size = 15 + normalized_ll * 25 
        else:
            node_size = 25
            
        # Tooltip includes directional counts
        tooltip_title = (
            f"POS: {row['POS']}\n"
            f"Obs: {observed} (Left: {obs_l}, Right: {obs_r})\n"
            f"LL: {ll_score:.2f}\n"
            f"Dominant Direction: {direction}"
        )

        # Add node with forced x position (fixed=False allows physics to arrange it nearby)
        net.add_node(collocate, label=collocate, size=node_size, color=color, title=tooltip_title, x=x_position)
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

    try:
        file_source.seek(0)
        file_bytes = file_source.read()

        try:
            file_content_str = file_bytes.decode('utf-8')
            # FIX: Robust cleaning to handle various newline/whitespace issues in vertical files
            file_content_str = re.sub(r'(\s+\n|\n\s+)', '\n', file_content_str)
            
        except UnicodeDecodeError:
            try:
                file_content_str = file_bytes.decode('iso-8859-1')
            except Exception:
                file_content_str = file_bytes.decode('utf-8', errors='ignore')
        
        clean_lines = [line for line in file_content_str.splitlines() if line and not line.strip().startswith('#')]
        clean_content = "\n".join(clean_lines)
        
        file_buffer_for_pandas = StringIO(clean_content)

    except Exception as e:
        # Fallback to raw text parsing attempt if structured parsing fails
         pass

    try:
        file_buffer_for_pandas.seek(0) 
        try:
            # Try structured format first
            df_attempt = pd.read_csv(file_buffer_for_pandas, sep='\t', header=None, engine="python", dtype=str)
        except Exception:
            # Fallback to space/general delimiter
            file_buffer_for_pandas.seek(0)
            df_attempt = pd.read_csv(file_buffer_for_pandas, sep=sep, header=None, engine="python", dtype=str)
            
        if df_attempt is not None and df_attempt.shape[1] >= 3:
            df = df_attempt.iloc[:, :3]
            df.columns = ["token", "pos", "lemma"]
            
            # FIX: Explicitly strip all surrounding whitespace (including newlines) from tokens
            df["token"] = df["token"].fillna("").astype(str).str.strip() 
            
            df["pos"] = df["pos"].fillna("###").astype(str)
            df["lemma"] = df["lemma"].fillna("###").astype(str)
            df["_token_low"] = df["token"].str.lower()
            return df
            
    except Exception:
         pass # Continue to raw text tokenization if all else fails

    try:
        # Raw text tokenization as a final fallback
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
        
    except Exception as raw_e:
        # Final failure point
        return None 


# ---------------------------
# UI: header
# ---------------------------
st.title("CORTEX - Corpus Texts Explorer v15.0")
st.caption("Upload vertical corpus (**token POS lemma**) or **raw horizontal text**. Raw text is analyzed quickly using basic tokenization and generic tags (`##`).")

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
        with st.spinner(f"Downloading {selected_corpus_name}..."):
            corpus_source = download_file_to_bytesio(corpus_url)
        corpus_name = selected_corpus_name
    
    
    # 2. NAVIGATION (MOVED UP)
    st.markdown("---")
    st.subheader("2. Navigation (TOOLS)")
    
    is_active_o = st.session_state['view'] == 'overview'
    st.button("📖 Overview", key='nav_overview', on_click=set_view, args=('overview',), use_container_width=True, type="primary" if is_active_o else "secondary")
    
    is_active_c = st.session_state['view'] == 'concordance'
    st.button("📚 Concordance", key='nav_concordance', on_click=set_view, args=('concordance',), use_container_width=True, type="primary" if is_active_c else "secondary")

    is_active_l = st.session_state['view'] == 'collocation'
    st.button("🔗 Collocation", key='nav_collocation', on_click=set_view, args=('collocation',), use_container_width=True, type="primary" if is_active_l else "secondary")

    st.markdown("---")
    
    # 3. MODULE SETTINGS (MOVED UP)
    st.subheader("3. Tool Settings")
    
    # Load corpus inside sidebar to get df for filtering logic (safe execution)
    df_sidebar = load_corpus_file(corpus_source)
    
    # Determine tagging mode safely for filter visibility
    is_raw_mode_sidebar = True
    if df_sidebar is not None and 'pos' in df_sidebar.columns and len(df_sidebar) > 0:
        count_of_raw_tags = df_sidebar['pos'].str.contains('##', na=False).sum()
        is_raw_mode_sidebar = (count_of_raw_tags / len(df_sidebar)) > 0.99
    
    if st.session_state['view'] == 'concordance':
        st.write("KWIC Context (Display)")
        kwic_left = st.number_input("Left Context (tokens)", min_value=1, max_value=20, value=7, step=1, help="Number of tokens shown to the left of the node word.")
        kwic_right = st.number_input("Right Context (tokens)", min_value=1, max_value=20, value=7, step=1, help="Number of tokens shown to the right of the node word.")
        st.session_state['kwic_left'] = kwic_left
        st.session_state['kwic_right'] = kwic_right
        
        # --- NEW: Concordance Pattern Search Settings ---
        st.markdown("---")
        st.subheader("Pattern Search Filter")
        
        st.caption("The **Node Word** is set by the primary search input above.")
        
        # Check if window is changed separately (implicitly triggers re-run)
        pattern_search_window = st.number_input(
            "Search Window (tokens, each side)", 
            min_value=1, max_value=10, value=5, step=1, 
            key="pattern_search_window_input", 
            help="The maximum distance (L/R) the collocate can be from the Node Word. This also sets the KWIC display context when active.",
            on_change=trigger_analysis_callback # Trigger analysis if window changes
        )
        
        # Collocate Word/Pattern Input
        pattern_collocate = st.text_input(
            "Collocate Word/Pattern (* for wildcard)", 
            value="", 
            key="pattern_collocate_input", 
            help="The specific word or pattern required to be in the context window (e.g., 'approach' or '*ly'). Press Enter/Click Away to search.",
            on_change=trigger_analysis_callback # Use callback to set the analysis flag when value changes/Enter is pressed
        )
        
        # Collocate POS Pattern Input (NEW for Concordance)
        if df_sidebar is not None and 'pos' in df_sidebar.columns and not is_raw_mode_sidebar:
            pattern_collocate_pos_input = st.text_input(
                "Collocate POS Tag Pattern (Wildcard/Concatenation)", 
                value="", 
                key="pattern_collocate_pos_input",
                help="E.g., V* (Verbs), *G (Gerunds), NNS|NNP (Plural/Proper Nouns). Filters collocates by POS tag."
            )
            st.session_state['pattern_collocate_pos'] = pattern_collocate_pos_input
        else:
            st.info("POS filtering for collocates requires a tagged corpus.")
            st.session_state['pattern_collocate_pos'] = ''


        st.session_state['pattern_search_window'] = pattern_search_window
        st.session_state['pattern_collocate'] = pattern_collocate
        
    elif st.session_state['view'] == 'collocation':
        # Collocation Settings
        max_collocates = st.number_input("Max Collocates to Show (Network/Tables)", min_value=5, max_value=100, value=20, step=5, help="Maximum number of collocates displayed in the network graph and top tables.")
        coll_window = st.number_input("Collocation window (tokens each side)", min_value=1, max_value=10, value=5, step=1, help="Window used for collocation counting (default ±5).")
        mi_min_freq = st.number_input("MI minimum observed freq", min_value=1, max_value=100, value=1, step=1)
        
        st.session_state['max_collocates'] = max_collocates
        st.session_state['coll_window'] = coll_window
        st.session_state['mi_min_freq'] = mi_min_freq

        st.markdown("---")
        st.subheader("Collocate Filters")
        
        # 1. Word/Regex Filter
        collocate_regex = st.text_input("Filter by Word/Regex (* for wildcard)", value="")
        st.session_state['collocate_regex'] = collocate_regex
        
        # --- POS Filters (Requires Tagged Corpus) ---
        if df_sidebar is not None and 'pos' in df_sidebar.columns and not is_raw_mode_sidebar:
            
            # 2a. POS Tag Pattern Filter 
            collocate_pos_regex_input = st.text_input(
                "Filter by POS Tag Pattern (Wildcard/Concatenation)", 
                value=st.session_state['collocate_pos_regex'], # Use session state value
                key="collocate_pos_regex_input_coll",
                help="E.g., V* (Verbs), *G (Gerunds), NNS|NNP (Plural/Proper Nouns). Uses regex on POS tags."
            )
            st.session_state['collocate_pos_regex'] = collocate_pos_regex_input
            
            # 2b. POS Multiselect Filter (Retained for specific tag selection)
            all_pos_tags = sorted([tag for tag in df_sidebar['pos'].unique() if tag != '##' and tag != '###'])
            
            if all_pos_tags:
                selected_pos_tags = st.multiselect(
                    "OR Filter by specific POS Tag(s)",
                    options=all_pos_tags,
                    default=None,
                    help="Only shows collocates matching one of the selected POS tags. If none selected, this filter is ignored."
                )
                st.session_state['selected_pos_tags'] = selected_pos_tags
            else:
                st.session_state['selected_pos_tags'] = None
        else:
            st.info("POS filtering requires a tagged corpus.")
            # Set to empty string for safety when raw mode is active
            st.session_state['collocate_pos_regex'] = ''
            st.session_state['pos_wildcard_regex'] = '' 
            st.session_state['selected_pos_tags'] = None

        # 3. Lemma Filter (Requires Lemmatized Corpus)
        if df_sidebar is not None and 'lemma' in df_sidebar.columns and not is_raw_mode_sidebar:
            collocate_lemma_input = st.text_input("Filter by Lemma (case-insensitive, * for wildcard)", value="", help="Enter the base form (e.g., 'approach'). Uses wildcard/regex logic on the lemma.")
            st.session_state['collocate_lemma'] = collocate_lemma_input
        else:
            st.info("Lemma filtering requires a lemmatized corpus.")
            # Set to empty string for safety when raw mode is active
            st.session_state['collocate_lemma'] = ''

    st.markdown("---")
    st.write("Shareable deployment tip:")
    st.info("Deploy this app on Streamlit Cloud or HuggingFace Spaces for free sharing.")
    
    # -----------------------------------------------------------------
    # TEMPORARY DEBUG CHECK (ADDED CODE BLOCK)
    # -----------------------------------------------------------------
    st.markdown("---")
    if st.button("DEBUG: Check API Key Status", key="debug_key_status"):
        key_is_set = os.environ.get("GEMINI_API_KEY") is not None
        if key_is_set:
            st.sidebar.success("✅ **SUCCESS:** GEMINI_API_KEY is loaded!")
        else:
            st.sidebar.error("❌ **FAILURE:** GEMINI_API_KEY is NOT loaded. Check Secrets format and Restart app.")
    # -----------------------------------------------------------------


# load corpus (cached) for main body access
df = load_corpus_file(corpus_source)

# --- Check for initial load failure and display better message ---
if df is None:
    # POSITIVE AND CLEAR MESSAGE
    st.header("👋 Welcome to CORTEX!")
    st.markdown("---")
    st.markdown("## Get Started")
    st.markdown("**Choose a preloaded corpus or upload your own corpus** in the sidebar to begin analysis.")
    st.stop()
# ---------------------------------------------------------------------
    
# --- CORPUS STATS CALCULATION (SHARED) ---
# df is guaranteed to be non-None here
if 'pos' in df.columns and len(df) > 0:
    count_of_raw_tags = df['pos'].str.contains('##', na=False).sum()
    is_raw_mode = (count_of_raw_tags / len(df)) > 0.99
else:
    is_raw_mode = True 

total_tokens = len(df)
tokens_lower = df["_token_low"].tolist()
PUNCTUATION = {'.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}', '"', "'", '---', '--', '-', '...', '«', '»', '—'}
tokens_lower_filtered = [t for t in tokens_lower if t not in PUNCTUATION and not t.isdigit()]
token_counts = Counter(tokens_lower) # Use unfiltered lower tokens for total counts
token_counts_filtered = Counter(tokens_lower_filtered)
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
    
    # Full-width primary search input (used for v14.8/v15.0)
    typed_target = st.text_input(
        "Type a primary token/MWU (word* or 'in the') or Structural Query ([lemma*]_POS*)", 
        value="", 
        key="typed_target_input",
        on_change=trigger_analysis_callback # Trigger analysis if primary input changes
    )
    
    # Determine the primary search input
    primary_input = typed_target.strip()
    target_input = primary_input
    contains_wildcard = '*' in target_input
    
    # FIX: Initialize use_pattern_search for all views
    use_pattern_search = False
    
    if st.session_state['view'] == 'concordance':
        # Check if we should use the Pattern Search parameters instead
        if primary_input and (st.session_state.get('pattern_collocate', '').strip() or st.session_state.get('pattern_collocate_pos', '').strip()):
            if st.session_state.get('pattern_collocate', '').strip() or st.session_state.get('pattern_collocate_pos', '').strip():
                use_pattern_search = True

    if not target_input and not use_pattern_search:
        st.info(f"Type a term or pattern for {st.session_state['view'].capitalize()} analysis.")
    
    # Explicit Analyze Button
    analyze_btn_explicit = st.button("🔎 Analyze")
    
    # --- Auto-Analysis Trigger Logic ---
    analyze_btn = analyze_btn_explicit or st.session_state['trigger_analyze']
    
    # Reset the implicit trigger flag after checking it
    st.session_state['trigger_analyze'] = False
    
    st.markdown("---")


# -----------------------------------------------------
# MODULE: CONCORDANCE LOGIC
# -----------------------------------------------------
if st.session_state['view'] == 'concordance' and analyze_btn and target_input:
    
    # --- Check for redundant analysis and early exit ---
    current_inputs = (target_input, st.session_state.get('pattern_collocate_input'), st.session_state.get('pattern_search_window_input'), st.session_state.get('pattern_collocate_pos'))
    last_inputs = (st.session_state['last_target_input'], st.session_state['last_pattern_collocate'], st.session_state['last_pattern_search_window'], st.session_state.get('last_pattern_collocate_pos_run', ''))

    if not analyze_btn_explicit and current_inputs == last_inputs:
         if st.session_state.get('initial_load_complete'):
             st.stop()
    
    # Store current inputs for next run comparison
    st.session_state['last_target_input'] = target_input
    st.session_state['last_pattern_collocate'] = st.session_state.get('pattern_collocate_input')
    st.session_state['last_pattern_search_window'] = st.session_state.get('pattern_search_window_input')
    st.session_state['last_pattern_collocate_pos_run'] = st.session_state.get('pattern_collocate_pos')
    st.session_state['initial_load_complete'] = True
    
    # --- Start Analysis ---
    kwic_left = st.session_state.get('kwic_left', 7)
    kwic_right = st.session_state.get('kwic_right', 7)
    
    # Use raw input for structural parsing to preserve case
    raw_target_input = target_input
    target = raw_target_input.lower()
    
    # --- PATTERN SEARCH VARIABLES ---
    pattern_collocate = st.session_state.get('pattern_collocate', '').lower().strip()
    pattern_collocate_pos = st.session_state.get('pattern_collocate_pos', '').strip() 
    pattern_window = st.session_state.get('pattern_search_window', 0)
    
    is_pattern_search_active = use_pattern_search and (pattern_collocate or pattern_collocate_pos) and pattern_window > 0

    # --- MWU/WILDCARD/STRUCTURAL RESOLUTION (Unified Search) ---
    
    # 1. Parse Query Components
    search_terms = raw_target_input.split()
    primary_target_len = len(search_terms)
    is_structural_search = not is_raw_mode and any('[' in t or '_' in t for t in search_terms)
    
    def create_structural_matcher(term):
        # Structural Parsing
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
        
        # Fallback: simple word/wildcard match
        pattern = re.escape(term.lower()).replace(r'\*', '.*')
        return {'type': 'word', 'pattern': re.compile(f"^{pattern}$")}

    search_components = [create_structural_matcher(term) for term in search_terms]
    
    # 2. Execute Search Loop
    all_target_positions = []
    
    if primary_target_len == 1 and not is_structural_search:
        # Optimization: Simple token/wildcard search
        target_pattern = search_components[0]['pattern']
        for i, token in enumerate(tokens_lower):
            if target_pattern.fullmatch(token):
                all_target_positions.append(i)
        
    else:
        # MWU or Structural Search (Unified Loop)
        for i in range(len(tokens_lower) - primary_target_len + 1):
            match = True
            
            for k, component in enumerate(search_components):
                corpus_index = i + k
                
                # Should not happen given the range loop bounds, but safe
                if corpus_index >= len(df):
                    match = False
                    break
                    
                if component['type'] == 'word':
                    # Standard word/wildcard match (case-insensitive)
                    current_token = tokens_lower[corpus_index]
                    if not component['pattern'].fullmatch(current_token):
                        match = False
                        break
                        
                elif component['type'] == 'structural':
                    # Structural match (lemma and/or POS)
                    current_lemma = df["lemma"].iloc[corpus_index].lower()
                    current_pos = df["pos"].iloc[corpus_index]
                    
                    lemma_match = True
                    if component['lemma_pattern']:
                        if not component['lemma_pattern'].fullmatch(current_lemma):
                            lemma_match = False
                            
                    pos_match = True
                    if component['pos_pattern']:
                        if not component['pos_pattern'].fullmatch(current_pos):
                            pos_match = False
                            
                    if not (lemma_match and pos_match):
                        match = False
                        break
                        
            if match:
                all_target_positions.append(i)
                
    # 3. Finalize Primary Target Details
    literal_freq = len(all_target_positions)
    primary_target_mwu = raw_target_input
    primary_target_tokens = raw_target_input.split()
    target_display = f"'{raw_target_input}'"
    
    if literal_freq == 0:
        st.warning(f"Target '{raw_target_input}' not found in corpus.")
        st.stop()
        
    # Recalculate wildcard_freq_df (only meaningful if single token/wildcard, otherwise just displays the MWU result)
    if primary_target_len == 1 and contains_wildcard and not is_structural_search:
        # Use existing single-token wildcard logic for displaying frequency breakdown
        pattern = re.escape(target).replace(r'\*', '.*')
        wildcard_matches = [token for token in token_counts if re.fullmatch(pattern, token)]
        match_counts = {token: token_counts[token] for token in wildcard_matches if token in token_counts}
        sorted_matches = sorted(match_counts.items(), key=lambda item: item[1], reverse=True)
        wildcard_freq_list = []
        for term, count in sorted_matches:
            rel_freq = (count / total_tokens) * 1_000_000
            wildcard_freq_list.append({"Query Result": term, "Raw Frequency": count, "Relative Frequency": f"{rel_freq:.4f}"})
        wildcard_freq_df = pd.DataFrame(wildcard_freq_list)
        # Re-set primary_target_mwu to the highest frequency single token for simple display.
        primary_target_mwu = wildcard_freq_df.iloc[0]["Query Result"] if not wildcard_freq_df.empty else raw_target_input
    else:
         # Use the MWU for display, regardless of structural components
        rel_freq = (literal_freq / total_tokens) * 1_000_000
        wildcard_freq_df = pd.DataFrame([{"Query Result": primary_target_mwu, "Raw Frequency": literal_freq, "Relative Frequency": f"{rel_freq:.4f}"}])

    # ------------------------------------------------------------------------------------------
    
    # --- 2. Concordance Generation (Pattern Search or Standard) ---
    
    kwic_rows = []
    kwic_rows_for_llm = [] # For LLM prompt (plain text)
    
    # 2b. Apply pattern filtering if active 
    final_positions = []
    collocate_count_in_context = 0
    
    if is_pattern_search_active:
        
        # --- 1. Prepare Collocate Word/Pattern Regex ---
        collocate_word_regex = None
        collocate_pattern_str = None
        if pattern_collocate:
            collocate_pattern_str = re.escape(pattern_collocate).replace(r'\*', '.*')
            try:
                 collocate_word_regex = re.compile(collocate_pattern_str)
            except re.error as e:
                 st.error(f"Invalid Collocate Word/Pattern Regex: {e}")
                 st.stop()
                 
        # --- 2. Prepare Collocate POS Pattern Regex ---
        collocate_pos_regex = None
        if pattern_collocate_pos and not is_raw_mode:
            pos_patterns = [p.strip() for p in pattern_collocate_pos.split('|') if p.strip()]
            full_pos_regex_list = []
            for pattern in pos_patterns:
                escaped_pattern = re.escape(pattern).replace(r'\*', '.*')
                full_pos_regex_list.append(escaped_pattern)
            
            if full_pos_regex_list:
                # Anchor needed for full tag match
                full_pos_regex = re.compile("^(" + "|".join(full_pos_regex_list) + ")$")
                collocate_pos_regex = full_pos_regex

        
        st.info(f"Pattern Search Active: Node='{raw_target_input}', Collocate Criteria: Word='{pattern_collocate}', POS='{pattern_collocate_pos}', Window=±{pattern_window} tokens.")
        
        for i in all_target_positions:
            # Note: The search window is centered around the full MWU now
            start_index = max(0, i - pattern_window)
            end_index = min(len(tokens_lower), i + primary_target_len + pattern_window)
            
            found_collocate = False
            for j in range(start_index, end_index):
                if i <= j < i + primary_target_len:
                    continue # Skip the node word(s) itself
                
                token_index_in_corpus = j
                token_lower = tokens_lower[j]
                token_pos = df["pos"].iloc[token_index_in_corpus]
                
                # Check 1: Word/Pattern Match (Required if provided)
                word_matches = (collocate_word_regex is None) or collocate_word_regex.fullmatch(token_lower)
                
                # Check 2: POS Tag Match (Required if provided)
                pos_matches = (collocate_pos_regex is None) or (collocate_pos_regex.fullmatch(token_pos) if not is_raw_mode else False)

                # Collocate must match ALL active criteria
                if word_matches and pos_matches:
                    found_collocate = True
                    break
            
            # ONLY append positions where collocate was found
            if found_collocate:
                final_positions.append(i)
                collocate_count_in_context += 1 # Count of successful co-occurrences
                
        if not final_positions:
            st.warning(f"Pattern search found 0 instances of '{raw_target_input}' co-occurring with the specified collocate criteria within the window.")
            st.stop()
            
    else:
        # Standard search: all instances found earlier
        final_positions = all_target_positions
        collocate_count_in_context = 0 # Not relevant for standard search

    # 2c. Format KWIC lines (applies to filtered or unfiltered positions)
    
    # --- Maximum lines to display in the scrollable table ---
    max_kwic_display = min(len(final_positions), KWIC_MAX_DISPLAY_LINES)
    
    # --- Synchronize KWIC Display Window with Pattern Window (FIX) ---
    current_kwic_left = kwic_left
    current_kwic_right = kwic_right
    
    if is_pattern_search_active and pattern_window > 0:
        # Override KWIC display to match the restrictive search window
        current_kwic_left = pattern_window
        current_kwic_right = pattern_window
    # -----------------------------------------------------------------
    

    for i in final_positions[:max_kwic_display]: # Use max_kwic_display here
        
        # Determine KWIC window based on user settings OR synchronized pattern window
        kwic_start = max(0, i - current_kwic_left)
        kwic_end = min(len(df), i + primary_target_len + current_kwic_right)
        
        full_line_tokens = df["token"].iloc[kwic_start:kwic_end].tolist()
        
        # --- KWIC Line Formatting (HTML Insertion) ---
        formatted_line = []
        node_orig_tokens = []
        
        # Re-initialize collocate regex objects for highlighting purposes
        collocate_word_regex_highlight = None
        if pattern_collocate:
            collocate_pattern_str = re.escape(pattern_collocate).replace(r'\*', '.*')
            collocate_word_regex_highlight = re.compile(collocate_pattern_str)
            
        collocate_pos_regex_highlight = None
        if pattern_collocate_pos and not is_raw_mode:
            pos_patterns = [p.strip() for p in pattern_collocate_pos.split('|') if p.strip()]
            full_pos_regex_list = [re.escape(p).replace(r'\*', '.*') for p in pos_patterns]
            if full_pos_regex_list:
                full_pos_regex = re.compile("^(" + "|".join(full_pos_regex_list) + ")$")
                collocate_pos_regex_highlight = full_pos_regex

        for k, token in enumerate(full_line_tokens):
            token_index_in_corpus = kwic_start + k
            token_lower = token.lower()
            token_pos = df["pos"].iloc[token_index_in_corpus]
            
            is_node_word = (i <= token_index_in_corpus < i + primary_target_len)
            
            is_collocate_match = False
            # Check for collocate match *only* within the displayed KWIC window, if pattern search is active
            if is_pattern_search_active and not is_node_word:
                 
                 # Check 1: Word/Pattern Match (Required if provided)
                 word_matches_highlight = (collocate_word_regex_highlight is None) or collocate_word_regex_highlight.fullmatch(token_lower)
                
                 # Check 2: POS Tag Match (Required if provided)
                 pos_matches_highlight = (collocate_pos_regex_highlight is None) or (collocate_pos_regex_highlight.fullmatch(token_pos) if not is_raw_mode else False)
                 
                 # Collocate is highlighted if it matches ALL active criteria
                 if word_matches_highlight and pos_matches_highlight:
                    is_collocate_match = True
            
            if is_node_word:
                # Node word remains unformatted in the context columns (but collected for the Node column)
                node_orig_tokens.append(token)
                # FIX: Add a placeholder to formatted_line to maintain index alignment for slicing
                formatted_line.append("") 
                
            elif is_collocate_match:
                # Collocate must be BOLDED and BRIGHT YELLOW HIGHLIGHTED
                html_token = f"<b><span style='color: black; background-color: #FFEA00;'>{token}</span></b>"
                formatted_line.append(html_token)
            else:
                formatted_line.append(token)

        # Split the formatted line back into Left and Right context based on KWIC window
        node_start_rel = i - kwic_start
        node_end_rel = node_start_rel + primary_target_len

        # Slicing now works correctly because the placeholder maintains the indices
        left_context = formatted_line[:node_start_rel]
        right_context = formatted_line[node_end_rel:]
        
        node_orig = " ".join(node_orig_tokens)
        
        kwic_rows.append({
            "Left": " ".join(left_context), 
            "Node": node_orig, 
            "Right": " ".join(right_context)
        })

        # Prepare plain text rows for LLM only (removing HTML for clean context)
        kwic_rows_for_llm.append({
            "Left_Context": " ".join(left_context).replace("<b><span style='color: black; background-color: #FFEA00;'>", "").replace("</span></b>", ""), 
            "Node": node_orig, 
            "Right_Context": " ".join(right_context).replace("<b><span style='color: black; background-color: #FFEA00;'>", "").replace("</span></b>", "")
        })
    
    # Prepare DataFrame for LLM interpretation (plain text)
    kwic_df_for_llm = pd.DataFrame(kwic_rows_for_llm).head(10).copy()
    
    # --- Prepare Collocate Frequency Data ---
    results_panel_data = []
    
    if is_pattern_search_active:
        
        # Calculate expected frequency (total frequency of the collocate pattern/POS in the whole corpus)
        expected_metric = "Collocate Pattern/POS"
        collocate_pattern_total_freq = 0
        
        if pattern_collocate: # Use word frequency as baseline if word pattern is present
            collocate_pattern_str_for_freq = re.escape(pattern_collocate).replace(r'\*', '.*')
            collocate_regex_for_freq = re.compile(collocate_pattern_str_for_freq)
            collocate_pattern_total_freq = sum(1 for token in tokens_lower if collocate_regex_for_freq.fullmatch(token))
            expected_metric = "Collocate Word Pattern"

        elif pattern_collocate_pos and not is_raw_mode: # Use POS frequency if only POS is present
            pos_patterns = [p.strip() for p in pattern_collocate_pos.split('|') if p.strip()]
            full_pos_regex_list = [re.escape(p).replace(r'\*', '.*') for p in pos_patterns]
            if full_pos_regex_list:
                 full_pos_regex = re.compile("^(" + "|".join(full_pos_regex_list) + ")$")
                 collocate_pattern_total_freq = sum(1 for pos in df["pos"] if full_pos_regex.fullmatch(pos))
                 expected_metric = "Collocate POS Pattern"
                 
        # 1. Node Word/Pattern Frequency
        results_panel_data.append({
            "Metric": "Node Word/Pattern Frequency",
            # Use the input string for the value if it's a structural search, otherwise the found token/mwu.
            "Value": raw_target_input,
            "Frequency": literal_freq
        })
        # 2. Observed Collocate Frequency (The actual target count in the co-occurrence window)
        results_panel_data.append({
            "Metric": "Observed Collocate Frequency (Co-occurrence)",
            "Value": f"Word='{pattern_collocate}' POS='{pattern_collocate_pos}'",
            "Frequency": collocate_count_in_context 
        })
        # 3. Expected Collocate Frequency (Total frequency of the most restrictive part of the Collocate Pattern in the corpus)
        results_panel_data.append({
            "Metric": f"Expected Collocate Frequency (Total Corpus: {expected_metric})",
            "Value": pattern_collocate if pattern_collocate else pattern_collocate_pos,
            "Frequency": collocate_pattern_total_freq
        })
        
        results_df = pd.DataFrame(results_panel_data)
        results_df["Frequency"] = results_df["Frequency"].apply(lambda x: f"{x:,}")

    else:
        # Standard display for non-pattern searches
        results_df = wildcard_freq_df.rename(columns={"Relative Frequency": "Expected Frequency"})
        
    
    # --- KWIC Display ---
    st.subheader("📚 Concordance Results")
    total_matches = len(final_positions)
    
    # --- Success message reflects pattern search ---
    if is_pattern_search_active:
        st.success(f"Pattern search successful! Found **{total_matches}** instances of '{raw_target_input}' co-occurring with the specified criteria within ±{current_kwic_left} tokens.")
    elif is_structural_search:
        st.success(f"Structural search successful! Found **{total_matches}** occurrences matching '{raw_target_input}'.")
    else:
        st.success(f"Found **{total_matches}** occurrences of the primary target word matching the criteria.")
    # -----------------------------------------------------

    # --- LLM INTERPRETATION BUTTON/EXPANDER ---
    if st.button("🧠 Interpret Concordance Results (LLM)", key="llm_concordance_btn"):
        with st.spinner("Requesting linguistic interpretation from LLM..."):
            result = interpret_results_llm(
                target_word=raw_target_input,
                analysis_type="Concordance",
                data_description="KWIC Context Sample (Max 10 lines)",
                data=kwic_df_for_llm
            )
            if "LLM API Error" in result:
                 st.error(result)

    
    if st.session_state['llm_interpretation_result']:
        with st.expander("LLM Interpretation", expanded=True):
            st.markdown(st.session_state['llm_interpretation_result'])
        st.markdown("---")
    # ----------------------------------------
    
    col_kwic, col_freq = st.columns([3, 2], gap="large")

    with col_kwic:
        st.subheader(f"Concordance (KWIC) — top {max_kwic_display} lines (Scrollable max {KWIC_MAX_DISPLAY_LINES})")
        
        kwic_df = pd.DataFrame(kwic_rows)
        kwic_preview = kwic_df.copy().reset_index(drop=True)
        kwic_preview.insert(0, "No", range(1, len(kwic_preview)+1))
        
        # Calculate approximate height for 10 visible lines
        row_height = 30 # approximate height of a single row in pixels
        scrollable_height = KWIC_INITIAL_DISPLAY_HEIGHT * row_height
        
        # 1. Custom CSS for table appearance (Alignment and Font + SCROLLING CONTAINER)
        kwic_table_style = f"""
             <style>
             .dataframe-container-scroll {{
                 height: {scrollable_height}px; /* Fixed height for scrollable view */
                 overflow-y: auto;
                 margin-bottom: 1rem;
             }}
             .dataframe {{
                 font-family: monospace;
                 color: white; /* Default text color for context */
                 width: 100%;
             }}
             .dataframe table {{
                width: 100%;
                table-layout: fixed;
                word-wrap: break-word;
             }}
             .dataframe th {{
                font-weight: bold;
                text-align: center;
             }}
             .dataframe td:nth-child(2) {{ /* Left context */
                text-align: right;
                color: white; /* Explicitly set context text color */
             }}
             .dataframe td:nth-child(3) {{ /* Node */
                text-align: center;
                font-weight: bold;
                background-color: #f0f0f0; /* Light Gray Highlight for Node */
                color: black;
             }}
             .dataframe td:nth-child(4) {{ /* Right context */
                text-align: left;
                color: white; /* Explicitly set context text color */
             }}
             /* Remove indexing column width constraint */
             .dataframe thead th:first-child {{ 
                 width: 30px; 
             }}
             </style>
        """
        st.markdown(kwic_table_style, unsafe_allow_html=True)
        
        # 2. Render the DataFrame inside a custom scrollable HTML container
        html_table = kwic_preview.to_html(
            escape=False,
            classes=['dataframe'],
            index=False
        )
        
        # Wrap the table in a scrollable div
        scrollable_html = f"<div class='dataframe-container-scroll'>{html_table}</div>"

        st.markdown(scrollable_html, unsafe_allow_html=True)

        st.caption("Note: Pattern search collocates are **bolded and highlighted bright yellow** for maximum visibility.")
        st.download_button("⬇ Download full concordance (xlsx)", data=df_to_excel_bytes(kwic_df), file_name=f"{target.replace(' ', '_')}_full_concordance.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with col_freq:
        if is_pattern_search_active:
            st.subheader(f"Pattern Search Frequencies")
            # Custom display for pattern search
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            st.caption(f"The total corpus frequency of the collocate pattern is listed above.")
        else:
            if contains_wildcard or is_structural_search:
                st.subheader(f"Search Results: '{raw_target_input}' (Top 10)")
            else:
                st.subheader(f"Target Frequency")
                
            freq_results_preview = wildcard_freq_df.rename(columns={"Relative Frequency": "Expected Frequency"}).head(10).copy()
            st.dataframe(freq_results_preview, use_container_width=True, hide_index=True)
        
        # Download button handles either case (kwic_df or wildcard_freq_df)
        download_df = kwic_df if is_pattern_search_active else wildcard_freq_df
        download_filename = f"{target.replace(' ', '_')}_pattern_results.xlsx" if is_pattern_search_active else f"{target.replace(' ', '_')}_wildcard_frequency_full.xlsx"
        
        st.download_button(
            "⬇ Download frequency results (xlsx)", 
            data=df_to_excel_bytes(download_df), 
            file_name=download_filename, 
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# -----------------------------------------------------
# MODULE: COLLOCATION LOGIC
# -----------------------------------------------------
if st.session_state['view'] == 'collocation' and analyze_btn and target_input:
    
    # Get Collocation Settings
    coll_window = st.session_state.get('coll_window', 5)
    mi_min_freq = st.session_state.get('mi_min_freq', 1)
    max_collocates = st.session_state.get('max_collocates', 20) 
    
    # Get Filter Settings
    collocate_regex = st.session_state.get('collocate_regex', '').lower().strip()
    
    # Robust POS retrieval
    collocate_pos_regex_raw = st.session_state.get('collocate_pos_regex', '')
    collocate_pos_regex_input = str(collocate_pos_regex_raw or '').strip()

    selected_pos_tags = st.session_state.get('selected_pos_tags', [])
    
    # Robust Lemma retrieval
    collocate_lemma_raw = st.session_state.get('collocate_lemma', '')
    collocate_lemma = str(collocate_lemma_raw or '').lower().strip()
    
    # Use raw input for structural parsing to preserve case
    raw_target_input = target_input
    target = raw_target_input.lower()
    
    # --- MWU/WILDCARD/STRUCTURAL RESOLUTION (Unified Search) ---
    
    # 1. Parse Query Components
    search_terms = raw_target_input.split()
    primary_target_len = len(search_terms)
    is_structural_search_coll = not is_raw_mode and any('[' in t or '_' in t for t in search_terms)
    
    def create_structural_matcher(term):
        # Structural Parsing
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
        
        # Fallback: simple word/wildcard match
        pattern = re.escape(term.lower()).replace(r'\*', '.*')
        return {'type': 'word', 'pattern': re.compile(f"^{pattern}$")}

    search_components = [create_structural_matcher(term) for term in search_terms]
    
    # 2. Execute Search Loop
    all_target_positions = []
    
    if primary_target_len == 1 and not is_structural_search_coll:
        # Optimization: Simple token/wildcard search
        target_pattern = search_components[0]['pattern']
        for i, token in enumerate(tokens_lower):
            if target_pattern.fullmatch(token):
                all_target_positions.append(i)
        
    else:
        # MWU or Structural Search (Unified Loop)
        for i in range(len(tokens_lower) - primary_target_len + 1):
            match = True
            
            for k, component in enumerate(search_components):
                corpus_index = i + k
                
                # Should not happen given the range loop bounds, but safe
                if corpus_index >= len(df):
                    match = False
                    break
                    
                if component['type'] == 'word':
                    # Standard word/wildcard match (case-insensitive)
                    current_token = tokens_lower[corpus_index]
                    if not component['pattern'].fullmatch(current_token):
                        match = False
                        break
                        
                elif component['type'] == 'structural':
                    # Structural match (lemma and/or POS)
                    current_lemma = df["lemma"].iloc[corpus_index].lower()
                    current_pos = df["pos"].iloc[corpus_index]
                    
                    lemma_match = True
                    if component['lemma_pattern']:
                        if not component['lemma_pattern'].fullmatch(current_lemma):
                            lemma_match = False
                            
                    pos_match = True
                    if component['pos_pattern']:
                        if not component['pos_pattern'].fullmatch(current_pos):
                            pos_match = False
                            
                    if not (lemma_match and pos_match):
                        match = False
                        break
                        
            if match:
                all_target_positions.append(i)
    
    # 3. Finalize Primary Target Details
    primary_target_positions = all_target_positions 
    freq = len(primary_target_positions)
    primary_target_mwu = raw_target_input
    target_display = f"'{raw_target_input}'"

    if freq == 0:
        st.warning(f"Target '{raw_target_input}' not found in corpus.")
        st.stop()
        
    primary_rel_freq = (freq / total_tokens) * 1_000_000
    
    st.subheader("🔗 Collocation Analysis Results")
    st.success(f"Analyzing target {target_display}. Frequency: **{freq:,}**, Relative Frequency: **{primary_rel_freq:.4f}** per million.")

    # --- COLLOCATION COUNTING (UPDATED FOR DIRECTION) ---
    
    # collocate_directional_counts: stores { (w, p, l, direction): count }
    collocate_directional_counts = Counter() 
    
    for i in primary_target_positions:
        start_index = max(0, i - coll_window)
        end_index = min(total_tokens, i + primary_target_len + coll_window) 
        
        for j in range(start_index, end_index):
            if i <= j < i + primary_target_len:
                continue
            
            w = tokens_lower[j]
            p = df["pos"].iloc[j]
            l = df["lemma"].iloc[j].lower() if "lemma" in df.columns else "##"
            
            # Determine Direction
            if j < i:
                direction = 'L' # Left collocate
            else:
                direction = 'R' # Right collocate
            
            key = (w, p, l, direction)
            collocate_directional_counts[key] += 1
    
    # --- Aggregate Counts and Determine Dominant Direction ---
    
    # raw_stats_data: { (w, p, l): {'L': count, 'R': count, 'Total': count, 'w': w, 'p': p, 'l': l} }
    raw_stats_data = {} 
    
    for (w, p, l, direction), observed_dir in collocate_directional_counts.items():
        key_tuple = (w, p, l)
        
        if key_tuple not in raw_stats_data:
            raw_stats_data[key_tuple] = {'L': 0, 'R': 0, 'Total': 0, 'w': w, 'p': p, 'l': l}
            
        raw_stats_data[key_tuple][direction] += observed_dir
        raw_stats_data[key_tuple]['Total'] += observed_dir

    stats_list = []
    token_counts_unfiltered = Counter(tokens_lower) 

    for key_tuple, data in raw_stats_data.items():
        w, p, l = key_tuple
        observed = data['Total']
        
        # Determine Dominant Direction
        if data['R'] > data['L']:
            dominant_direction = 'R'
        elif data['L'] > data['R']:
            dominant_direction = 'L'
        else:
            dominant_direction = 'B' # Both/Equal
        
        total_freq = token_counts_unfiltered.get(w, 0)
        
        # Calculate LL/MI based on total observed frequency
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
            "Lemma": l,
            "Observed": observed,
            "Total_Freq": total_freq,
            "LL": round(ll,6),
            "MI": round(mi,6),
            "Significance": sig,
            "Direction": dominant_direction, 
            "Obs_L": data['L'],             
            "Obs_R": data['R']              
        })

    stats_df = pd.DataFrame(stats_list)
    
    # --- APPLY FILTERS ---
    if not stats_df.empty:
        filtered_df = stats_df.copy()
        
        # 1. Word/Regex Filter
        if collocate_regex:
            pattern = re.escape(collocate_regex).replace(r'\*', '.*').replace(r'\|', '|').replace(r'\.', '.')
            try:
                filtered_df = filtered_df[filtered_df['Collocate'].str.fullmatch(pattern, case=True, na=False)]
            except re.error:
                st.error(f"Invalid regular expression for Word/Regex filter: '{collocate_regex}'")
                filtered_df = pd.DataFrame() 
                
        # 2a. POS Pattern Filter 
        if collocate_pos_regex_input and not is_raw_mode:
            # Convert user input like 'V*|NN*' into full regex: 'V.*|NN.*'
            pos_patterns = [p.strip() for p in collocate_pos_regex_input.split('|') if p.strip()]
            
            full_pos_regex_list = []
            for pattern in pos_patterns:
                # Ensure '|' is handled by split, then escape except for '*', replace '*' with '.*'
                escaped_pattern = re.escape(pattern).replace(r'\*', '.*')
                full_pos_regex_list.append(escaped_pattern)
            
            if full_pos_regex_list:
                # Use alternation for multiple patterns, ensuring anchor for full match on the tag: ^(V.*|NN.*)$
                full_pos_regex = "^(" + "|".join(full_pos_regex_list) + ")$"
                
                try:
                    # Filtered df based on POS tag matching the user's pattern
                    filtered_df = filtered_df[filtered_df['POS'].str.contains(full_pos_regex, case=True, na=False, regex=True)]
                except re.error:
                    st.error(f"Invalid POS pattern/regex: '{collocate_pos_regex_input}'")
                    filtered_df = pd.DataFrame()
            
        # 2b. POS Multiselect Filter (Only applied if pattern is empty)
        if selected_pos_tags and not is_raw_mode and not collocate_pos_regex_input:
            filtered_df = filtered_df[filtered_df['POS'].isin(selected_pos_tags)]
            
        # 3. Lemma Filter
        # Check if corpus is fully tagged/lemmatized AND a lemma filter was provided
        if collocate_lemma and 'Lemma' in filtered_df.columns and not is_raw_mode: 
            lemma_pattern = re.escape(collocate_lemma).replace(r'\*', '.*').replace(r'\|', '|').replace(r'\.', '.')
            try:
                filtered_df = filtered_df[filtered_df['Lemma'].str.fullmatch(lemma_pattern, case=True, na=False)]
            except re.error:
                 st.error(f"Invalid regular expression for Lemma filter: '{collocate_lemma}'")
                 filtered_df = pd.DataFrame()
        
        stats_df_filtered = filtered_df
        
        if stats_df_filtered.empty:
            st.warning("No collocates found after applying filters.")
            st.stop()
            
        stats_df_sorted = stats_df_filtered.sort_values("LL", ascending=False)

    else:
        st.warning("No collocates found.")
        st.stop()
        
    # --- LLM INTERPRETATION BUTTON/EXPANDER ---
    if st.button("🧠 Interpret Collocation Results (LLM)", key="llm_collocation_btn"):
        with st.spinner("Requesting linguistic interpretation from LLM..."):
            result = interpret_results_llm(
                target_word=raw_target_input,
                analysis_type="Collocation",
                data_description="Top Log-Likelihood Collocates",
                data=stats_df_sorted[['Collocate', 'POS', 'Observed', 'LL', 'Direction']]
            )
            if "LLM API Error" in result:
                 st.error(result)
            
    if st.session_state['llm_interpretation_result']:
        with st.expander("LLM Interpretation", expanded=True):
            st.markdown(st.session_state['llm_interpretation_result'])
        st.markdown("---")
    # ----------------------------------------
        
    # --- Prepare Directional DataFrames for Graphing (Top N) ---

    # Left Collocates: Dominant Left or Equal (B), ranked by LL
    left_directional_df = stats_df_sorted[stats_df_sorted['Direction'].isin(['L', 'B'])].head(max_collocates).copy()
    
    # Right Collocates: Dominant Right or Equal (B), ranked by LL
    right_directional_df = stats_df_sorted[stats_df_sorted['Direction'].isin(['R', 'B'])].head(max_collocates).copy()


    # --- DISPLAY GRAPHS SIDE BY SIDE ---
    st.markdown("---")
    st.subheader("Interactive Collocation Networks (Directional)")
    
    col_left_graph, col_right_graph = st.columns(2)

    with col_left_graph:
        # --- LEFT COLLOCATE GRAPH ---
        st.subheader(f"Left Collocates Only (Top {len(left_directional_df)} LL)")
        
        if not left_directional_df.empty:
            # Use the input string for the graph display if structural search was used
            graph_target_word = raw_target_input
            network_html_left = create_pyvis_graph(graph_target_word, left_directional_df)
            components.html(network_html_left, height=450)
            st.markdown(
                """
                **Left Collocates Key:** Shows collocates that **precede** the target word (Direction 'L' or 'B'), placed on the left side.
                """
            )
        else:
            st.info("No Left-dominant collocates found that meet the frequency or filter criteria.")

    with col_right_graph:
        # --- RIGHT COLLOCATE GRAPH ---
        st.subheader(f"Right Collocates Only (Top {len(right_directional_df)} LL)")
        
        if not right_directional_df.empty:
            # Use the input string for the graph display if structural search was used
            graph_target_word = raw_target_input
            network_html_right = create_pyvis_graph(graph_target_word, right_directional_df)
            components.html(network_html_right, height=450)
            st.markdown(
                """
                **Right Collocates Key:** Shows collocates that **follow** the target word (Direction 'R' or 'B'), placed on the right side.
                """
            )
        else:
            st.info("No Right-dominant collocates found that meet the frequency or filter criteria.")
    
    # --- Graph General Key (Placed below the columns) ---
    st.markdown("---")
    st.markdown(
        """
        **General Graph Key:**
        * Central Node (Target): **Yellow**, fixed at the center.
        * Collocate Node Color: Noun (N) **Green**, Verb (V) **Blue**, Adjective (J) **Pink**, Adverb (R) **Yellow**. Others/Raw are **Gray**.
        * Collocate Bubble Size: Scales with Log-Likelihood (LL) score (**Bigger Bubble** = Stronger Collocation).
        * Hover over collocate nodes for details (POS, Total Observed Freq, Directional Counts, LL score).
        """
    )
    st.markdown("---")
    
    # --- Prepare Full Tables ---
    network_df = stats_df_sorted.head(max_collocates).copy()
    full_ll = stats_df_sorted.copy()
    full_ll.insert(0, "Rank", range(1, len(full_ll)+1))
    
    full_mi_all = stats_df_filtered[stats_df_filtered["Observed"] >= mi_min_freq].sort_values("MI", ascending=False).reset_index(drop=True)
    full_mi = full_mi_all.head(max_collocates).copy()
    full_mi.insert(0, "Rank", range(1, len(full_mi)+1))

    # ---------- Full Collocation Tables (All Tags) ----------
    display_rows = min(10, max_collocates)
    st.subheader(f"Top {display_rows} Collocations (All Tags - Directional Detail)")
    
    cols_full = st.columns(2, gap="large")
    
    with cols_full[0]:
        st.markdown(f"**Log-Likelihood (LL) (Top {display_rows})**")
        st.dataframe(full_ll.head(display_rows), use_container_width=True, hide_index=True)
    
    with cols_full[1]:
        st.markdown(f"**Mutual Information (MI) (obs ≥ {mi_min_freq}, Top {display_rows})**")
        st.dataframe(full_mi.head(display_rows), use_container_width=True, hide_index=True)

    # ---------- Download Buttons ----------
    st.markdown("---")
    st.subheader("Download Full Results")
    
    # Download logic for Collocation results... 
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

st.caption("Tip: This app handles both pre-tagged vertical corpora and raw linear text, adjusting analysis depth automatically.")
