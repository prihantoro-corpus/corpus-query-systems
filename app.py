# app.py
# handles non tagged and non lemmatised text
import streamlit as st
import pandas as pd
import numpy as np
import math
from collections import Counter
from io import BytesIO, StringIO # Added StringIO
import tempfile 
import os       
import re       
import requests 

# Import for Pyvis Network Graph
from pyvis.network import Network
import streamlit.components.v1 as components # Import for HTML embedding

st.set_page_config(page_title="Corpus Explorer Version 12 Dec 25", layout="wide")

# ---------------------------
# Built-in Corpus Configuration
# ---------------------------

BUILT_IN_CORPORA = {
    "Select built-in corpus...": None,
    "Europarl 1M Only": "https://raw.githubusercontent.com/prihantoro-corpus/corpus-query-systems/main/europarl_en-1M-only.txt",
    "sample speech 13kb only": "https://raw.githubusercontent.com/prihantoro-corpus/corpus-query-systems/main/Speech%20address.txt",
}


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
    
    # Define colors for POS categories (Restored for tagged files)
    pos_colors = {
        'N': '#33CC33',  # Noun (Green)
        'V': '#3366FF',  # Verb (Blue)
        'J': '#FF33B5',  # Adjective (Pink)
        'R': '#FFCC00',  # Adverb (Yellow)
        '#': '#AAAAAA',  # Nonsense Tag / Raw (Gray)
        'O': '#AAAAAA'   # Other (Gray)
    }
    
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
        
        color = pos_colors.get(pos_code, pos_colors['O'])
        
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
# Cached loading (Conditional Logic Implemented)
# ---------------------------
@st.cache_data
def load_corpus_file(file_source, sep=r"\s+"):
    """
    Loads corpus either from an uploaded file handle or a BytesIO object,
    prioritizing structured tab-separated format.
    """
    
    if file_source is None:
        return None

    # 1. Try to read as structured 3-column data (Vertical Corpus)
    try:
        file_source.seek(0)
        file_content_str = file_source.read().decode('utf-8')
        
        # Use StringIO to handle the string content for reliable pandas parsing
        # Filter out potential comments/metadata lines before parsing
        clean_lines = [line for line in file_content_str.splitlines() if line and not line.strip().startswith('#')]
        clean_content = "\n".join(clean_lines)
        
        file_buffer_for_pandas = StringIO(clean_content)

        # Attempt 1: Tab Separator (most common for tagged corpora)
        try:
            df_attempt = pd.read_csv(file_buffer_for_pandas, sep='\t', header=None, engine="python", dtype=str)
        except Exception:
            file_buffer_for_pandas.seek(0) # Reset buffer
            # Attempt 2: Default Separator (Whitespace)
            df_attempt = pd.read_csv(file_buffer_for_pandas, sep=sep, header=None, engine="python", dtype=str)
            
        # Continue with structural check
        is_vertical = False
        if df_attempt.shape[1] >= 3:
            df_check = df_attempt.iloc[:, :3].copy()
            # Heuristic: if unique tokens are significantly less than total rows, it's structured
            if df_check.iloc[:, 0].nunique() < len(df_check) * 0.95 and len(df_check) > 5:
                is_vertical = True
            
        if is_vertical:
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

    # 2. Fallback: Treat as Raw Horizontal Text (Fast Regex Tokenizer + Nonsense Tags)
    try:
        # We already have file_content_str from the failed attempt, or need to re-read
        if 'file_content_str' not in locals():
            file_source.seek(0)
            file_content_str = file_source.read().decode('utf-8')
            
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
st.title("CORTEX -- Corpus Texts Explorer version 12-Dec-25 (Fast Mode)")
st.caption("Upload vertical corpus (**token POS lemma**) or **raw horizontal text**. Raw text is analyzed quickly using basic tokenization and generic tags (`##`).")

# ---------------------------
# Panel: upload and corpus info
# ---------------------------
corpus_source = None
corpus_name = "Uploaded File"

with st.sidebar:
    st.header("Upload & Options")
    
    st.subheader("1. Choose Corpus Source")
    
    # Option A: Built-in Corpus Selection
    selected_corpus_name = st.selectbox(
        "Select a pre-loaded corpus:", 
        options=list(BUILT_IN_CORPORA.keys())
    )
    
    # Option B: Upload File
    uploaded_file = st.file_uploader("OR Upload your own corpus file", type=["txt","csv"])
    
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
        
    st.subheader("2. Settings")

    # 1. Collocation window control
    coll_window = st.number_input("Collocation window (tokens each side)", min_value=1, max_value=10, value=5, step=1, help="Window used for collocation counting (default ±5).")
    st.markdown("---")
    
    # 2. Concordance context controls (New Widgets)
    st.subheader("KWIC Context (Display)")
    kwic_left = st.number_input("Left Context (tokens)", min_value=1, max_value=20, value=7, step=1, help="Number of tokens shown to the left of the node word.")
    kwic_right = st.number_input("Right Context (tokens)", min_value=1, max_value=20, value=7, step=1, help="Number of tokens shown to the right of the node word.")
    st.markdown("---")
    
    # 3. MI min frequency control
    st.write("MI minimum observed frequency (for MI tables).")
    mi_min_freq = st.number_input("MI min observed freq", min_value=1, max_value=100, value=1, step=1)
    st.markdown("---")
    st.write("Shareable deployment tip:")
    st.info("Deploy this app on Streamlit Cloud or HuggingFace Spaces for free sharing.")


# load corpus (cached)
df = load_corpus_file(corpus_source)

if df is None:
    st.error("Corpus failed to load. Please check the file format or download source.")
    st.stop()
    
# Determine if raw mode was used by checking for the nonsense tag in the output DataFrame
is_raw_mode = 'pos' in df.columns and any(df['pos'].str.contains('##', na=False))

if is_raw_mode:
    st.header(f"Analyzing Corpus: {corpus_name} (RAW/LINEAR MODE)")
else:
    st.header(f"Analyzing Corpus: {corpus_name} (TAGGED MODE)")

total_tokens = len(df)

# corpus info and frequency list
tokens_lower = df["_token_low"].tolist()
pos_tags = df["pos"].tolist()
PUNCTUATION = {'.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}', '"', "'", '---', '--', '-', '...', '«', '»', '—'}
tokens_lower_filtered = [t for t in tokens_lower if t not in PUNCTUATION and not t.isdigit()]

token_counts = Counter(tokens_lower_filtered)
unique_types = len(set(tokens_lower_filtered))
unique_lemmas = df["lemma"].nunique() if "lemma" in df.columns else "###"

# STTR per 1000
def compute_sttr_tokens(tokens_list, chunk=1000):
    if len(tokens_list) < chunk:
        return (len(set(tokens_list)) / len(tokens_list)) if len(tokens_list) > 0 else 0.0
    ttrs = []
    for i in range(0, len(tokens_list), chunk):
        c = tokens_list[i:i+chunk]
        if not c: continue
        ttrs.append(len(set(c)) / len(c))
    
    return (sum(ttrs)/len(ttrs)) if ttrs else 0.0

sttr_score = compute_sttr_tokens(tokens_lower_filtered)

col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Corpus summary")
    info_df = pd.DataFrame({
        "Metric": ["Corpus size (tokens)", "Unique types (w/o punc)", "Lemma count", "STTR (w/o punc, chunk=1000)"],
        "Value": [f"{total_tokens:,}", unique_types, unique_lemmas, round(sttr_score,4)]
    })
    st.dataframe(info_df, use_container_width=True, hide_index=True) 

with col2:
    st.subheader("Top frequency (token / POS / freq) (Punctuation skipped)")
    
    freq_df_filtered = df[~df['_token_low'].isin(PUNCTUATION) & ~df['_token_low'].str.isdigit()].copy()
    
    # Only suppress POS tags if RAW mode was detected
    if is_raw_mode:
         freq_df_filtered['pos'] = '##'

    freq_df = freq_df_filtered.groupby(["token","pos"]).size().reset_index(name="frequency").sort_values("frequency", ascending=False).reset_index(drop=True)
    
    freq_head = freq_df.head(10).copy()
    freq_head.insert(0,"No", range(1, len(freq_head)+1))
    st.dataframe(freq_head, use_container_width=True, hide_index=True) 
    
    st.download_button("⬇ Download full frequency list (xlsx)", data=df_to_excel_bytes(freq_df), file_name="full_frequency_list_filtered.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")

# ---------------------------
# Target input 
# ---------------------------
st.subheader("Target word input")
col_a, col_b = st.columns(2)
with col_a:
    typed_target = st.text_input("Type a token (case-insensitive)", value="")
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
if not target_input:
    st.info("Type a token or upload a target list and choose one. Then click Analyze.")
analyze_btn = st.button("🔎 Analyze")

# ---------------------------
# Main analysis
# ---------------------------
if analyze_btn and target_input:
    target = target_input.lower()
    positions = [i for i, t in enumerate(tokens_lower) if t == target]
    freq = len(positions)
    if freq == 0:
        st.warning(f"Token '{target_input}' not found in corpus.")
    else:
        st.success(f"Found {freq} occurrences of '{target_input}' (case-insensitive).")
        rel_freq = (freq / total_tokens) * 1_000_000
        st.write(f"Relative frequency: **{rel_freq:.2f}** per million")

        # ---------- CONCORDANCE: KWIC ----------
        st.subheader(f"Concordance (KWIC) — top 10 ({kwic_left} left & {kwic_right} right)")
        kwic_rows = []
        for i in positions:
            left = tokens_lower[max(0, i - kwic_left):i]
            right = tokens_lower[i + 1:i + 1 + kwic_right]
            node_orig = df["token"].iloc[i]
            kwic_rows.append({"Left": " ".join(left), "Node": node_orig, "Right": " ".join(right)})
        kwic_df = pd.DataFrame(kwic_rows)
        kwic_preview = kwic_df.head(10).copy().reset_index(drop=True)
        kwic_preview.insert(0, "No", range(1, len(kwic_preview)+1))
        st.dataframe(kwic_preview, use_container_width=True, hide_index=True)

        st.download_button("⬇ Download full concordance (xlsx)", data=df_to_excel_bytes(kwic_df), file_name=f"{target}_full_concordance.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # ---------- COLLATION: compute stats ----------
        coll_pairs = []
        for i in positions:
            start = max(0, i - coll_window)
            end = min(total_tokens, i + coll_window + 1)
            for j in range(start, end):
                if j == i:
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
        else:
            stats_df_sorted = stats_df.sort_values("LL", ascending=False)
            network_df = stats_df_sorted.head(20).copy()

            # full LL top10
            full_ll = stats_df_sorted.head(10).copy()
            full_ll.insert(0, "Rank", range(1, len(full_ll)+1))
            
            # full MI (apply MI min freq)
            full_mi_all = stats_df[stats_df["Observed"] >= mi_min_freq].sort_values("MI", ascending=False).reset_index(drop=True)
            full_mi = full_mi_all.head(10).copy()
            full_mi.insert(0, "Rank", range(1, len(full_mi)+1))

            
            # --- Collocation Network Graph ---
            st.markdown("---")
            st.subheader(f"Interactive Collocation Network (Top {len(network_df)} LL)")
            
            network_html = create_pyvis_graph(target_input, network_df)
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
                # --- Category Tables Restoration (Only for Tagged Files) ---
                def category_df(prefixes):
                    mask = pd.Series(False, index=stats_df.index)
                    for pref in prefixes:
                        # Check if POS starts with the prefix (case-insensitive)
                        mask = mask | stats_df["POS"].str.startswith(pref, na=False)
                    sub = stats_df[mask].copy()
                    ll_sub = sub.sort_values("LL", ascending=False).reset_index(drop=True).head(10).copy()
                    ll_sub.insert(0, "Rank", range(1, len(ll_sub)+1))
                    mi_sub = sub[sub["Observed"] >= mi_min_freq].sort_values("MI", ascending=False).reset_index(drop=True).head(10).copy()
                    mi_sub.insert(0, "Rank", range(1, len(mi_sub)+1))
                    return ll_sub, mi_sub

                # Assuming standard tag prefixes (e.g., N, V, J, R, but will match Europarl tags like NN, VVP, JJ, etc.)
                ll_N, mi_N = category_df(("N",))    # Noun
                ll_V, mi_V = category_df(("V",))    # Verb
                ll_J, mi_J = category_df(("J",))    # Adjective
                ll_R, mi_R = category_df(("R",))    # Adverb
                
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
                
                st.markdown("---") # Separate POS tables from full tables

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
            
            if not is_raw_mode:
                # Download categories re-added
                st.markdown("**LL Top 10 Downloads (POS-Filtered)**")
                ll_dl_cols = st.columns(5)
                ll_mapping = {
                    "Full": full_ll, "N": ll_N, "V": ll_V, "J": ll_J, "R": ll_R
                }
                for i, (cat, df_tab) in enumerate(ll_mapping.items()):
                    bname = f"LL {cat} top10"
                    ll_dl_cols[i].download_button(f"⬇ {bname} (xlsx)", data=df_to_excel_bytes(df_tab), file_name=f"{target}_LL_{cat}_top10.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                st.markdown("**MI Top 10 Downloads (POS-Filtered)**")
                mi_dl_cols = st.columns(5)
                mi_mapping = {
                    "Full": full_mi, "N": mi_N, "V": mi_V, "J": mi_J, "R": mi_R
                }
                for i, (cat, df_tab) in enumerate(mi_mapping.items()):
                    bname = f"MI {cat} top10"
                    mi_dl_cols[i].download_button(f"⬇ {bname} (xlsx)", data=df_to_excel_bytes(df_tab), file_name=f"{target}_MI_{cat}_top10.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                st.markdown("---")
            
            # Full Results Download
            st.download_button(
                "⬇ Download full LL results (xlsx)", 
                data=df_to_excel_bytes(stats_df_sorted), 
                file_name=f"{target}_LL_full.xlsx", 
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.download_button(
                f"⬇ Download full MI results (obs≥{mi_min_freq}) (xlsx)", 
                data=df_to_excel_bytes(full_mi_all), 
                file_name=f"{target}_MI_full_obsge{mi_min_freq}.xlsx", 
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

st.caption("Tip: This app handles both pre-tagged vertical corpora and raw linear text, adjusting analysis depth automatically.")
