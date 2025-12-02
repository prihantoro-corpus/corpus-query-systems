# app.py
import streamlit as st
import pandas as pd
import numpy as np
import math
from collections import Counter
from io import BytesIO
import tempfile 
import os       
import re       
import spacy    # Using spaCy for tagging raw text
import nltk     # Keeping import for tokenization fallback

# --- Load spaCy Model ---
# NOTE: The spaCy model MUST be installed in your environment before deployment.
# Typically done with: python -m spacy download en_core_web_sm
@st.cache_resource
def load_spacy_model():
    try:
        # Load the small English model
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("Error: spaCy model 'en_core_web_sm' not found. Please install it using 'python -m spacy download en_core_web_sm' or run the app in a pre-configured environment.")
        return None

NLP_MODEL = load_spacy_model()
# ---------------------------

st.set_page_config(page_title="Corpus Explorer Version 12 Dec 25", layout="wide")

# ---------------------------
# Helpers: stats, IO utilities
# ---------------------------
EPS = 1e-12

def safe_log(x):
    return math.log(max(x, EPS))

def compute_ll(k11, k12, k21, k22):
    """
    Computes the Log-Likelihood (LL) statistic based on a 2x2 contingency table.
    """
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
    """
    Computes the Mutual Information (MI) statistic.
    """
    expected = (target_freq * coll_total) / corpus_size
    if expected == 0 or k11 == 0:
        return 0.0
    return math.log2(k11 / expected)

def significance_from_ll(ll_val):
    """
    Converts Log-Likelihood value to significance level.
    """
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
    Creates a Pyvis interactive network graph with custom styling.
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
    
    # Define colors for POS categories
    pos_colors = {
        'N': '#33CC33',  # Noun
        'V': '#3366FF',  # Verb
        'J': '#FF33B5',  # Adjective (J*)
        'R': '#FFCC00',  # Adverb (R*)
        'O': '#AAAAAA',  # Other (from spaCy: other parts, punct, etc.)
        'RAW': '#AAAAAA', # Placeholder/Raw tag
        '###': '#AAAAAA' # Placeholder/Raw tag
    }
    
    # 2. Add Collocate Nodes and Edges
    for _, row in coll_df.iterrows():
        collocate = row['Collocate']
        ll_score = row['LL']
        observed = row['Observed']
        pos_code = row['POS'][0].upper() if row['POS'] else 'O'
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


# ---------------------------
# Cached loading (Modified to handle horizontal raw text using spaCy)
# ---------------------------
@st.cache_data
def load_corpus_file(file_bytes, sep=r"\s+"):
    # 1. Try to read as structured 3-column data (Vertical Corpus)
    try:
        file_bytes.seek(0)
        df_attempt = pd.read_csv(file_bytes, sep=sep, header=None, engine="python", dtype=str)
        
        # Heuristic to check if it's a vertical corpus
        is_vertical = False
        if df_attempt.shape[1] >= 3:
            df_check = df_attempt.iloc[:, :3].copy()
            # If the number of unique tokens in the first column is significantly less than total rows, it's likely structured (not raw run-on text)
            if df_check.iloc[:, 0].nunique() < len(df_check) * 0.95 and len(df_check) > 5:
                is_vertical = True
            
        if is_vertical:
            # Use the structured file
            df = df_attempt.iloc[:, :3]
            df.columns = ["token", "pos", "lemma"]
            df["token"] = df["token"].fillna("").astype(str)
            df["pos"] = df["pos"].fillna("###").astype(str)
            df["lemma"] = df["lemma"].fillna("###").astype(str)
            df["_token_low"] = df["token"].str.lower()
            return df
            
        file_bytes.seek(0) # Reset pointer for raw reading

    except Exception:
         file_bytes.seek(0) 
         pass # Proceed to raw text processing

    # 2. Fallback: Treat as Raw Horizontal Text (Tagging using spaCy if model loaded)
    try:
        raw_text = file_bytes.read().decode('utf-8')
        
        if NLP_MODEL:
            # Use spaCy for accurate tokenization, POS, and lemmatization
            doc = NLP_MODEL(raw_text)
            tokens = [token.text for token in doc if token.text.strip()]
            pos_tags = [token.pos_ for token in doc if token.text.strip()]
            lemmas = [token.lemma_ for token in doc if token.text.strip()]
            
            st.sidebar.info("Raw text successfully processed and tagged by **spaCy**.")
            
        else:
            # Fallback to simple regex tokenization if spaCy model failed to load
            tokens = re.findall(r'\b\w+\b|[^\w\s]+', raw_text)
            tokens = [t.strip() for t in tokens if t.strip()]
            pos_tags = ["RAW"] * len(tokens)
            lemmas = ["###"] * len(tokens)
            st.sidebar.warning("Raw text processed using basic tokenization. POS/Lemma columns are placeholders ('RAW'/'###').")
        
        df = pd.DataFrame({
            "token": tokens,
            "pos": pos_tags,
            "lemma": lemmas
        })
        
        df["_token_low"] = df["token"].str.lower()
        return df
        
    except Exception as raw_e:
        st.error(f"Error processing raw text: {raw_e}")
        raise raw_e


# ---------------------------
# UI: header
# ---------------------------
st.title("CORTEX -- Corpus Texts Explorer version 12-Dec-25")
st.caption("Upload vertical corpus (**token POS lemma**) or **raw horizontal text**. Uses spaCy for tagging untagged text.")

# ---------------------------
# Panel: upload and corpus info
# ---------------------------
with st.sidebar:
    st.header("Upload & Options")
    uploaded_file = st.file_uploader("Upload corpus file", type=["txt","csv"])
    st.markdown("---")
    
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

if uploaded_file is None:
    st.info("Please upload a corpus file (vertical: token pos lemma, or raw text).")
    st.stop()

# load corpus (cached)
df = load_corpus_file(uploaded_file)
total_tokens = len(df)

# corpus info and frequency list
tokens_lower = df["_token_low"].tolist()
pos_tags = df["pos"].tolist()
# Define punctuation tokens to filter out (simple approach)
PUNCTUATION = {'.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}', '"', "'", '---', '--', '-', '...', '«', '»', '—'}
# Filter tokens for frequency list
tokens_lower_filtered = [t for t in tokens_lower if t not in PUNCTUATION and not t.isdigit()]

token_counts = Counter(tokens_lower_filtered)
unique_types = len(set(tokens_lower_filtered))
unique_lemmas = df["lemma"].nunique() if "lemma" in df.columns else "###"

# STTR per 1000
def compute_sttr_tokens(tokens_list, chunk=1000):
    """
    Computes the Standardized Type-Token Ratio (STTR) by averaging TTR over fixed chunks (ratio 0-1).
    """
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
    
    freq_df_filtered = df[~df['_token_low'].isin(PUNCTUATION) & ~df['_token_low'].str.isdigit()]
    
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

            full_ll = stats_df_sorted.head(10).copy()
            full_ll.insert(0, "Rank", range(1, len(full_ll)+1))
            full_mi_all = stats_df[stats_df["Observed"] >= mi_min_freq].sort_values("MI", ascending=False).reset_index(drop=True)
            full_mi = full_mi_all.head(10).copy()
            full_mi.insert(0, "Rank", range(1, len(full_mi)+1))

            def category_df(prefixes):
                mask = pd.Series(False, index=stats_df.index)
                for pref in prefixes:
                    mask = mask | stats_df["POS"].str.startswith(pref, na=False)
                # Also include "RAW" and "###" in a separate 'Other' category if needed, but here we focus on major POS tags
                sub = stats_df[mask].copy()
                ll_sub = sub.sort_values("LL", ascending=False).reset_index(drop=True).head(10).copy()
                ll_sub.insert(0, "Rank", range(1, len(ll_sub)+1))
                mi_sub = sub[sub["Observed"] >= mi_min_freq].sort_values("MI", ascending=False).reset_index(drop=True).head(10).copy()
                mi_sub.insert(0, "Rank", range(1, len(mi_sub)+1))
                return ll_sub, mi_sub

            # Using standard spaCy/Universal POS conventions for better compatibility
            ll_N, mi_N = category_df(("NOUN", "PROPN"))    # Noun
            ll_V, mi_V = category_df(("VERB", "AUX"))      # Verb/Auxiliary
            ll_J, mi_J = category_df(("ADJ",))             # Adjective
            ll_R, mi_R = category_df(("ADV",))             # Adverb

            st.download_button("⬇ Download full LL (all collocates) (xlsx)", data=df_to_excel_bytes(stats_df_sorted), file_name=f"{target}_LL_full.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.download_button(f"⬇ Download full MI (obs≥{mi_min_freq}) (xlsx)", data=df_to_excel_bytes(full_mi_all), file_name=f"{target}_MI_full_obsge{mi_min_freq}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            # ---------- Collocation Network Graph ----------
            st.markdown("---")
            st.subheader(f"Interactive Collocation Network (Top {len(network_df)} LL)")
            
            network_html = create_pyvis_graph(target_input, network_df)
            components.html(network_html, height=450)
            
            st.markdown(
                """
                **Graph Key (Revised):**
                * Central Node (Target): **Yellow**
                * Collocate Node Color: Noun (**Green**), Verb (**Blue**), Adjective (**Pink**), Adverb (**Yellow**), Other (**Gray**)
                * Edge Thickness: **All Thick** (Uniform)
                * Collocate Bubble Size: Scales with Log-Likelihood (LL) score (**Bigger Bubble** = Stronger Collocation)
                * Hover over nodes for details (POS, Observed Freq, LL score).
                """
            )
            st.markdown("---")

            # ---------- Side-by-side display for LL: N | V | J | R ----------
            st.subheader("Log-Likelihood — Top 10 by POS")
            cols = st.columns(4, gap="small")
            
            with cols[0]:
                st.markdown("**N (NOUN/PROPN) — LL**")
                st.dataframe(ll_N, use_container_width=True, hide_index=True)
            with cols[1]:
                st.markdown("**V (VERB/AUX) — LL**")
                st.dataframe(ll_V, use_container_width=True, hide_index=True)
            with cols[2]:
                st.markdown("**J (ADJ) — LL**")
                st.dataframe(ll_J, use_container_width=True, hide_index=True)
            with cols[3]:
                st.markdown("**R (ADV) — LL**")
                st.dataframe(ll_R, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("Download Top 10 Tables")
            
            st.markdown("**LL Top 10 Downloads**")
            ll_dl_cols = st.columns(5)
            ll_mapping = {
                "Full": full_ll, "N": ll_N, "V": ll_V, "J": ll_J, "R": ll_R
            }
            for i, (cat, df_tab) in enumerate(ll_mapping.items()):
                bname = f"LL {cat} top10"
                ll_dl_cols[i].download_button(f"⬇ {bname} (xlsx)", data=df_to_excel_bytes(df_tab), file_name=f"{target}_LL_{cat}_top10.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            st.markdown("---")
            st.markdown(f"**MI Top 10 Downloads (obs ≥ {mi_min_freq})**")
            mi_dl_cols = st.columns(5)
            mi_mapping = {
                "Full": full_mi, "N": mi_N, "V": mi_V, "J": mi_J, "R": mi_R
            }
            for i, (cat, df_tab) in enumerate(mi_mapping.items()):
                bname = f"MI {cat} top10"
                mi_dl_cols[i].download_button(f"⬇ {bname} (xlsx)", data=df_to_excel_bytes(df_tab), file_name=f"{target}_MI_{cat}_top10.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                
            st.markdown("---")

            # ---------- Side-by-side display for MI: Full | N | V | J | R ----------
            st.subheader(f"Mutual Information — Top 10 (obs ≥ {mi_min_freq})")
            cols_mi = st.columns(5, gap="small")
            with cols_mi[0]:
                st.markdown("**Full (MI)**")
                st.dataframe(full_mi, use_container_width=True, hide_index=True)
            with cols_mi[1]:
                st.markdown("**N (NOUN/PROPN) — MI**")
                st.dataframe(mi_N, use_container_width=True, hide_index=True)
            with cols_mi[2]:
                st.markdown("**V (VERB/AUX) — MI**")
                st.dataframe(mi_V, use_container_width=True, hide_index=True)
            with cols_mi[3]:
                st.markdown("**J (ADJ) — MI**")
                st.dataframe(mi_J, use_container_width=True, hide_index=True)
            with cols_mi[4]:
                st.markdown("**R (ADV) — MI**")
                st.dataframe(mi_R, use_container_width=True, hide_index=True)

st.caption("Tip: Deploy this file to Streamlit Cloud or HuggingFace Spaces to share with others.")
